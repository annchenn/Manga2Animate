import os
import glob
from PIL import Image, ImageOps
import torch
import json
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.ops import box_convert
import insightface
from groundingdino.util.inference import load_model, load_image, predict, annotate
from .inception_resnet_v1 import InceptionResnetV1
import sys
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import scipy.optimize
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from pathlib import Path

from vistorybench.bench.base_evaluator import BaseEvaluator
from vistorybench.dataset_loader.read_outputs import load_outputs

# Dynamically add AdaFace submodule to Python path to resolve imports
# within the submodule, as it's not a standard package.
adaface_path = os.path.join(os.path.dirname(__file__), 'AdaFace')
if adaface_path not in sys.path:
    sys.path.append(adaface_path)

from .AdaFace.inference import load_pretrained_model, to_input, adaface_models

class MultiModelFeatures:
    """Container class for saving multi-model features"""
    def __init__(self, features_dict):
        """
        Args:
            features_dict: Dictionary, keys are model names, values are feature tensors
        """
        self.features_dict = features_dict
        self.model_names = list(features_dict.keys())
        
    def to(self, device):
        """Move all features to specified device"""
        new_dict = {}
        for name, feat in self.features_dict.items():
            if feat is not None:
                new_dict[name] = feat.to(device)
            else:
                new_dict[name] = feat
        return MultiModelFeatures(new_dict)
        
    def cuda(self):
        """Move all features to GPU"""
        return self.to('cuda')
        
    def cpu(self):
        """Move all features to CPU"""
        return self.to('cpu')
        
    def __getitem__(self, key):
        return self.features_dict[key]
        
    def keys(self):
        return self.features_dict.keys()
        
    def values(self):
        return self.features_dict.values()
    
    def get_concatenated_features(self):
        """Get concatenated features"""
        valid_features = [feat for feat in self.features_dict.values() if feat is not None]
        if valid_features:
            # Ensure all features are 2-dimensional
            normalized_features = []
            for feat in valid_features:
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                # Normalize features
                feat = F.normalize(feat, p=2, dim=1)
                normalized_features.append(feat)
            
            # Concatenate features
            if len(normalized_features) > 1:
                # Take the first row (first face)
                face_features = [feat[0] for feat in normalized_features]
                combined_feature = torch.cat(face_features, dim=0)
            else:
                combined_feature = normalized_features[0][0]
                
            return combined_feature.unsqueeze(0)  # Return [1, combined_dim]
        return None
    
    def get_single_model_features(self, model_name):
        """Get features from a single model"""
        if model_name in self.features_dict and self.features_dict[model_name] is not None:
            feat = self.features_dict[model_name]
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            return F.normalize(feat, p=2, dim=1)
        return None

class CIDSEvaluator(BaseEvaluator):
    _shared_model_cache = {}

    def _apply_model_bundle(self, bundle):
        """Attach cached heavy models to this evaluator instance."""
        self.dino = bundle.get('dino')
        self.clip = bundle.get('clip')
        self.clip_processor = bundle.get('clip_processor')
        self.siglip = bundle.get('siglip')
        self.siglip_processor = bundle.get('siglip_processor')
        self.arcface = bundle.get('arcface')
        self.arcface_provider = bundle.get('arcface_provider')
        self.facenet = bundle.get('facenet')
        self.adaface = bundle.get('adaface')

    def __init__(self, config: dict, timestamp: str, mode: str, language: str, outputs_timestamp=None):
        super().__init__(config, timestamp, mode, language, outputs_timestamp)

        # Device
        self.device = torch.device(self.get_device())

        # Defaults
        self.BOX_TRESHOLD = 0.25
        self.TEXT_TRESHOLD = 0.25
        self.PRE_TEXT_PROMPT = {}

        # Evaluator-specific config
        self.cids_cfg = self.get_evaluator_config('cids')
        self.ref_mode = self.cids_cfg.get('ref_mode', 'origin')
        self.use_multi_face_encoder = self.cids_cfg.get('use_multi_face_encoder', True)
        self.ensemble_method = self.cids_cfg.get('ensemble_method', 'average')

        # Which image encoder scores character identity. SigLIP replaces CLIP as the
        # default: the characters here are drawn, so the face encoders never fire and
        # SigLIP separates near-identical anime designs better than CLIP does.
        encoders_cfg = self.cids_cfg.get('encoders', {}) or {}
        self.image_encoder = encoders_cfg.get('image', 'siglip')
        # When set, every character uses this encoder regardless of its tag.
        self.encoder_override = encoders_cfg.get('override') or None
        
        dino = self.cids_cfg.get('detection', {}).get('dino', {})
        if 'box_threshold' in dino:
            self.BOX_TRESHOLD = dino['box_threshold']
        if 'text_threshold' in dino:
            self.TEXT_TRESHOLD = dino['text_threshold']

        match = self.cids_cfg.get('matching', {})
        self.superfluous_threshold = match.get('superfluous_threshold', 0.5)
        self.topk_per_nochar = match.get('topk_per_nochar', 3)
        self.ensemble_weights = self.cids_cfg.get('ensemble_weights', {})
        # Copy-paste metric parameters for chordal metric (arccos-free, range [0,1])
        self.copy_paste_eps = float(self.cids_cfg.get('copy_paste_eps', 1e-6))
        self.copy_paste_tau_chord = float(self.cids_cfg.get('copy_paste_tau_chord', 1e-6))

        # ArcFace detection thresholds stepping (constants)
        self.arcface_det_thresh_initial = 0.45
        self.arcface_det_thresh_min = 0.1
        self.arcface_det_thresh_step = 0.1

        # Cosine similarity function
        self.cos = F.cosine_similarity

        # Prompt Align GPT-V config for single_character_action integration
        pa_cfg = self.get_evaluator_config('prompt_align')
        gpt_cfg = pa_cfg.get('gpt', {}) if isinstance(pa_cfg, dict) else {}
        base_url = self.get_base_url()
        api_key = self.get_api_key()
        model = self.get_model_id() or 'gpt-4.1'
        self.gpt_api_pkg = (model, api_key, base_url)
        self.gpt_workers = int(gpt_cfg.get('workers', 1) or 1)
        self._pa_skip_warned = False
        self._pa_prompt_path = 'vistorybench/bench/prompt_align/user_prompts/user_prompt_single_character_text_align.txt'
        # Repeated VLM scoring, matching the prompt_align evaluator.
        self.pa_repeats = int(pa_cfg.get('repeats', 5) or 1)
        self.pa_temperature = float(pa_cfg.get('temperature', 1.0))
        self.pa_top_p = float(pa_cfg.get('top_p', 1.0))

        # OCCM detection prompt: deliberately plain, so the same words are used on the
        # ground-truth manga panel and on the generated shot.
        occm_cfg = self.cids_cfg.get('occm', {}) or {}
        self.OCCM_PROMPT = occm_cfg.get('prompt', 'character')

        # Load models on initialization
        self._load_models(self.pretrain_path, self.device)

    def _load_models(self, pretrain_path, device='cuda'):
        """Load all models required for CIDS evaluation."""
        pretrain_path = Path(pretrain_path)

        clip_model_id = "openai/clip-vit-large-patch14"
        siglip_model_id = "google/siglip-so400m-patch14-384"
        if isinstance(self.cids_cfg, dict):
            enc = self.cids_cfg.get('encoders')
            if isinstance(enc, dict) and 'clip' in enc and 'model_id' in enc['clip']:
                clip_model_id = enc['clip']['model_id']
            if isinstance(enc, dict) and 'siglip' in enc and 'model_id' in enc['siglip']:
                siglip_model_id = enc['siglip']['model_id']

        cache_key = (str(pretrain_path.resolve()), str(device), clip_model_id, siglip_model_id)
        cached_bundle = self._shared_model_cache.get(cache_key)

        if cached_bundle:
            self._apply_model_bundle(cached_bundle)
            print(f"Reusing cached CIDS models for device {device} (pretrain: {pretrain_path})")
        else:
            bundle = {}

            # Load GroundingDINO
            try:
                gd_config = os.path.join(os.path.dirname(__file__), 'GroundingDINO_SwinT_OGC.py')
                gd_weights = pretrain_path / 'groundingdino/weights/groundingdino_swint_ogc.pth'
                bundle['dino'] = load_model(str(gd_config), str(gd_weights)).to(device)
                print("GroundingDINO model loaded successfully")
            except Exception as e:
                print(f"Could not load GroundingDINO model: {e}")
                bundle['dino'] = None

            # Load CLIP
            try:
                clip_path = pretrain_path / clip_model_id
                bundle['clip'] = CLIPModel.from_pretrained(clip_path).to(device)
                bundle['clip_processor'] = CLIPProcessor.from_pretrained(clip_path)
                print("CLIP model loaded successfully")
            except Exception as e:
                print(f"Could not load CLIP model: {e}")
                bundle['clip'] = None
                bundle['clip_processor'] = None

            # Load SigLIP
            try:
                siglip_path = pretrain_path / siglip_model_id
                bundle['siglip'] = AutoModel.from_pretrained(siglip_path).to(device).eval()
                # Image processor only: the full AutoProcessor drags in SiglipTokenizer,
                # which needs sentencepiece, and we never encode text with it.
                bundle['siglip_processor'] = AutoImageProcessor.from_pretrained(siglip_path)
                print("SigLIP model loaded successfully")
            except Exception as e:
                print(f"Could not load SigLIP model: {e}")
                bundle['siglip'] = None
                bundle['siglip_processor'] = None

            # Load ArcFace
            arcface_model = None
            arcface_provider = None

            def _init_arcface(providers, ctx_id):
                model = insightface.app.FaceAnalysis(
                    root=pretrain_path / 'insightface',
                    name="antelopev2",
                    providers=providers
                )
                model.prepare(ctx_id=ctx_id, det_thresh=self.arcface_det_thresh_initial)
                return model

            try:
                arcface_model = _init_arcface(['CUDAExecutionProvider'], ctx_id=0)
                arcface_provider = 'cuda'
                print("ArcFace model loaded successfully")
            except Exception as e:
                print(f"Could not load ArcFace model on CUDA: {e}")
                try:
                    arcface_model = _init_arcface(['CPUExecutionProvider'], ctx_id=-1)
                    arcface_provider = 'cpu'
                    print("ArcFace model loaded on CPU fallback")
                except Exception as cpu_e:
                    print(f"Could not load ArcFace model on CPU either: {cpu_e}")
                    arcface_model = None
                    arcface_provider = None

            bundle['arcface'] = arcface_model
            bundle['arcface_provider'] = arcface_provider

            # Load FaceNet
            try:
                bundle['facenet'] = InceptionResnetV1(pretrained='vggface2', pretrained_path=pretrain_path).eval().to(device)
                print("FaceNet model loaded successfully")
            except Exception as e:
                print(f"Could not load FaceNet model: {e}")
                bundle['facenet'] = None

            # Load AdaFace
            try:
                model_key = 'ir_101'
                ada_ckpt = str(pretrain_path / 'adaface/adaface_ir101_webface12m.ckpt')
                adaface_models[model_key] = ada_ckpt
                bundle['adaface'] = load_pretrained_model(model_key).to(device)
                print("AdaFace model loaded successfully")
            except Exception as e:
                print(f"Could not load AdaFace model: {e}")
                bundle['adaface'] = None

            self._shared_model_cache[cache_key] = bundle
            self._apply_model_bundle(bundle)

        # Initialize FaceRestoreHelper
        try:
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=112,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='jpg',
                use_parse=False
            )
            print("Face helper initialized successfully")
        except Exception as e:
            print(f"Could not initialize Face helper: {e}")
            self.face_helper = None

    def _get_encoder_name(self, char_tag):
        """Select appropriate encoder based on character tag and configuration"""
        if self.encoder_override:
            return self.encoder_override
        if char_tag == "realistic_human":
            return "face_mix" if self.use_multi_face_encoder else "arcface"
        return self.image_encoder

    def _get_arcface_features(self, img_np, det_thresh=None):
        """Extract facial features using ArcFace model"""
        if self.arcface is None:
            return None

        ctx_id = 0 if self.arcface_provider != 'cpu' else -1
        curr_thresh = det_thresh if det_thresh is not None else self.arcface_det_thresh_initial
        while curr_thresh >= self.arcface_det_thresh_min:
            self.arcface.prepare(ctx_id=ctx_id, det_thresh=curr_thresh)
            faces = self.arcface.get(img_np)
            if len(faces) > 0:
                return torch.from_numpy(faces[0].embedding)
            curr_thresh -= self.arcface_det_thresh_step
        return None

    def _get_adaface_features(self, img_pil):
        """Extract facial features using AdaFace model"""
        if self.face_helper is None:
            return None
            
        try:
            self.face_helper.clean_all()
            img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            self.face_helper.read_image(img_np)
            self.face_helper.get_face_landmarks_5(only_center_face=False)
            self.face_helper.align_warp_face()

            aligned_faces = self.face_helper.cropped_faces
            if len(aligned_faces) > 0:
                aligned_face = aligned_faces[0]  # Take the first face
                aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                bgr_input = to_input(aligned_rgb)
                bgr_input = bgr_input.to(self.device)
                
                with torch.no_grad():
                    feature, _ = self.adaface(bgr_input)
                return feature.detach().cpu().squeeze()
        except Exception as e:
            print(f"AdaFace feature extraction failed: {e}")
        return None

    def _get_facenet_features(self, img_pil):
        """Extract facial features using FaceNet model"""
        if self.face_helper is None:
            return None
            
        try:
            self.face_helper.clean_all()
            img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            self.face_helper.read_image(img_np)
            self.face_helper.get_face_landmarks_5(only_center_face=False)
            self.face_helper.align_warp_face()

            aligned_faces = self.face_helper.cropped_faces
            if len(aligned_faces) > 0:
                aligned_face = aligned_faces[0]  # Take the first face
                aligned_face = torch.tensor(aligned_face).float() / 255.0
                aligned_face = aligned_face.permute(2, 0, 1)
                
                with torch.no_grad():
                    feature = self.facenet(aligned_face.unsqueeze(0).to(self.device))
                return feature.detach().cpu().squeeze()
        except Exception as e:
            print(f"FaceNet feature extraction failed: {e}")
        return None

    def get_char_feat(
        self,
        img: Image.Image | list[Image.Image],
        encoder_name="arcface",
        det_thresh=None,
        return_indices: bool = False,
    ) -> torch.Tensor:
        if not isinstance(img, list):
            img = [img]

        if len(img) == 0:
            print("\033[33mWarning: Empty image list provided to get_char_feat\033[0m")
            return None

        total_imgs = len(img)
        valid_indices = list(range(total_imgs))

        if encoder_name in ("clip", "siglip"):
            if encoder_name == "siglip":
                model, processor = self.siglip, self.siglip_processor
            else:
                model, processor = self.clip, self.clip_processor
            if model is None or processor is None:
                raise RuntimeError(
                    f"{encoder_name} encoder requested but the model failed to load; "
                    f"check data/pretrain for it."
                )
            inputs = processor(images=img, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            image_features = F.normalize(image_features.float(), p=2, dim=1)

        elif encoder_name == "arcface":
            image_features = []
            valid_indices = []

            if self.arcface is None:
                print("ArcFace model is unavailable; skipping arcface feature extraction.")
                image_features = None
            else:
                for idx, _img in enumerate(img):
                    _img_np = cv2.cvtColor(np.array(_img), cv2.COLOR_RGB2BGR)
                    feat = self._get_arcface_features(_img_np, det_thresh)
                    if feat is not None:
                        image_features.append(feat)
                        valid_indices.append(idx)
                    else:
                        try:
                            _img.save(f"{self.mid_result_dir}/bad_case.png")
                        except Exception:
                            pass
                        print("No face detected for current image even after lowering thresholds.")

                if image_features:
                    image_features = torch.stack(image_features, dim=0)
                    image_features = F.normalize(image_features, p=2, dim=1)
                else:
                    image_features = None

        elif encoder_name == "face_mix":
            all_multi_features = []
            valid_indices = []
            
            for idx, _img in enumerate(img):
                _img_np = cv2.cvtColor(np.array(_img), cv2.COLOR_RGB2BGR)
                model_features = {}
                
                arcface_features = self._get_arcface_features(_img_np, det_thresh)
                if arcface_features is not None:
                    if arcface_features.dim() == 1:
                        arcface_features = arcface_features.unsqueeze(0)
                    model_features['arcface'] = F.normalize(arcface_features, p=2, dim=1)
                
                if self.adaface is not None and self.face_helper is not None:
                    adaface_features = self._get_adaface_features(_img)
                    if adaface_features is not None:
                        if adaface_features.dim() == 1:
                            adaface_features = adaface_features.unsqueeze(0)
                        model_features['adaface'] = F.normalize(adaface_features, p=2, dim=1)
                
                if self.facenet is not None and self.face_helper is not None:
                    facenet_features = self._get_facenet_features(_img)
                    if facenet_features is not None:
                        if facenet_features.dim() == 1:
                            facenet_features = facenet_features.unsqueeze(0)
                        model_features['facenet'] = F.normalize(facenet_features, p=2, dim=1)
                
                if model_features:
                    multi_feat = MultiModelFeatures(model_features)
                    all_multi_features.append(multi_feat)
                    valid_indices.append(idx)
            
            if all_multi_features:
                image_features = all_multi_features
            else:
                image_features = None
        else:
            raise NotImplementedError

        if return_indices:
            indices = valid_indices if image_features is not None else []
            return image_features, indices

        return image_features

    def _compute_ensemble_similarity(self, multi_feat1, multi_feat2, method="average"):
        if not isinstance(multi_feat1, MultiModelFeatures) or not isinstance(multi_feat2, MultiModelFeatures):
            if hasattr(multi_feat1, 'get_concatenated_features'):
                feat1 = multi_feat1.get_concatenated_features()
            else:
                feat1 = multi_feat1
            if hasattr(multi_feat2, 'get_concatenated_features'):
                feat2 = multi_feat2.get_concatenated_features()
            else:
                feat2 = multi_feat2
                
            if feat1 is None or feat2 is None:
                return 0.0
            return F.cosine_similarity(feat1, feat2, dim=1).item()
        
        if method == "concatenate":
            feat1 = multi_feat1.get_concatenated_features()
            feat2 = multi_feat2.get_concatenated_features()
            if feat1 is None or feat2 is None:
                return 0.0
            return F.cosine_similarity(feat1, feat2, dim=1).item()
            
        elif method == "average":
            similarities = []
            common_models = set(multi_feat1.keys()) & set(multi_feat2.keys())
            
            for model_name in common_models:
                feat1 = multi_feat1[model_name]
                feat2 = multi_feat2[model_name]
                if feat1 is not None and feat2 is not None:
                    sim_matrix = F.cosine_similarity(feat1.unsqueeze(1), feat2.unsqueeze(0), dim=2)
                    sim_matrix_np = sim_matrix.detach().cpu().numpy()
                    cost_matrix = -sim_matrix_np
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
                    
                    matched_scores = sim_matrix_np[row_ind, col_ind]
                    avg_sim = matched_scores.mean()
                    similarities.append(avg_sim)
            
            if similarities:
                return sum(similarities) / len(similarities)
            else:
                return 0.0
                
        elif method == "weighted":
            similarities = []
            weights = []
            common_models = set(multi_feat1.keys()) & set(multi_feat2.keys())
            
            model_weights = self.ensemble_weights
            
            for model_name in common_models:
                feat1 = multi_feat1[model_name]
                feat2 = multi_feat2[model_name]
                if feat1 is not None and feat2 is not None:
                    sim_matrix = F.cosine_similarity(feat1.unsqueeze(1), feat2.unsqueeze(0), dim=2)
                    sim_matrix_np = sim_matrix.detach().cpu().numpy()
                    cost_matrix = -sim_matrix_np
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
                    
                    matched_scores = sim_matrix_np[row_ind, col_ind]
                    avg_sim = matched_scores.mean()
                    weight = model_weights.get(model_name, 1.0)
                    similarities.append(avg_sim * weight)
                    weights.append(weight)
            
            if similarities and weights:
                return sum(similarities) / sum(weights)
            else:
                return 0.0
        else:
            return self._compute_ensemble_similarity(multi_feat1, multi_feat2, "average")
    
    def _compute_multimodel_similarity_matrix(self, output_feats, ref_feats, method="average"):
        is_multi_output = isinstance(output_feats, list) and len(output_feats) > 0 and isinstance(output_feats[0], MultiModelFeatures)
        is_multi_ref = isinstance(ref_feats, list) and len(ref_feats) > 0 and isinstance(ref_feats[0], MultiModelFeatures)
        
        if is_multi_output or is_multi_ref:
            if is_multi_output and not is_multi_ref:
                similarities = []
                for out_feat in output_feats:
                    ref_multi = MultiModelFeatures({'single': ref_feats})
                    sim = self._compute_ensemble_similarity(out_feat, ref_multi, method)
                    similarities.append([sim])
                return torch.tensor(similarities)
                
            elif not is_multi_output and is_multi_ref:
                similarities = []
                out_multi = MultiModelFeatures({'single': output_feats})
                for ref_feat in ref_feats:
                    sim = self._compute_ensemble_similarity(out_multi, ref_feat, method)
                    similarities.append([sim])
                return torch.tensor(similarities).T
                
            elif is_multi_output and is_multi_ref:
                similarities = []
                for out_feat in output_feats:
                    row_sims = []
                    for ref_feat in ref_feats:
                        sim = self._compute_ensemble_similarity(out_feat, ref_feat, method)
                        row_sims.append(sim)
                    similarities.append(row_sims)
                return torch.tensor(similarities)
        
        if not isinstance(output_feats, torch.Tensor):
            output_feats = torch.tensor(output_feats)
        if not isinstance(ref_feats, torch.Tensor):
            ref_feats = torch.tensor(ref_feats)
        return (output_feats @ ref_feats.T)

    def dino_detect(self, inp_img: str, inp_cap: str, box_threshold=None, text_threshold=None):
        if box_threshold is None:
            box_threshold = self.BOX_TRESHOLD
        if text_threshold is None:
            text_threshold = self.TEXT_TRESHOLD
        
        try:
            if not os.path.exists(inp_img):
                print(f"\033[31mError: Image file not found for DINO detection: {inp_img}\033[0m")
                empty_boxes = torch.empty(0, 4)
                empty_logits = torch.empty(0)
                empty_phrases = []
                return empty_boxes, empty_logits, empty_phrases, None
                
            image_source, image = load_image(inp_img)
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.dino,
                    image=image,
                    caption=inp_cap,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            return boxes, logits, phrases, annotated_frame
        except Exception as e:
            print(f"\033[31mError in DINO detection for {inp_img}: {str(e)}\033[0m")
            empty_boxes = torch.empty(0, 4)
            empty_logits = torch.empty(0)
            empty_phrases = []
            return empty_boxes, empty_logits, empty_phrases, None

    def crop_img(self, img_src: str, boxes, return_boxes: bool = False):
        try:
            if not os.path.exists(img_src):
                print(f"\033[31mError: Image file not found for cropping: {img_src}\033[0m")
                return ([] , []) if return_boxes else []
            
            if len(boxes) == 0:
                return ([] , []) if return_boxes else []
                
            img_source = Image.open(img_src)
            w, h = img_source.size
            boxes = boxes.to('cpu')
            orig_boxes = boxes.clone()
            boxes = (boxes * torch.Tensor([w, h, w, h])).int()
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            cropped_images = []
            filtered_boxes = []
            for idx, box in enumerate(xyxy):
                try:
                    x_min, y_min, x_max, y_max = box.tolist()
                    x_min = max(0, min(x_min, w))
                    y_min = max(0, min(y_min, h))
                    x_max = max(0, min(x_max, w))
                    y_max = max(0, min(y_max, h))
                    
                    if x_max > x_min and y_max > y_min:
                        cropped = img_source.crop((x_min, y_min, x_max, y_max))
                        cropped_images.append(cropped)
                        if return_boxes:
                            filtered_boxes.append(orig_boxes[idx].tolist())
                except Exception as e:
                    print(f"\033[33mWarning: Failed to crop box {box}: {str(e)}\033[0m")
                    continue
            if return_boxes:
                return cropped_images, filtered_boxes
            return cropped_images
        except Exception as e:
            print(f"\033[31mError in crop_img for {img_src}: {str(e)}\033[0m")
            return ([] , []) if return_boxes else []

    # ------------------------------------------------------------------
    # OCCM: how well the number of characters on screen matches the source
    # ------------------------------------------------------------------
    PANEL_DIR_NAME = 'manga_panels'

    def _panel_path(self, story_id: str, shot_index: int):
        """Ground-truth manga panel for a shot, or None when the story has no panels."""
        panel_dir = os.path.join(self.dataset_path, 'ViStory', str(story_id), self.PANEL_DIR_NAME)
        if not os.path.isdir(panel_dir):
            return None
        for stem in (f'panel{shot_index}', f'panel_{shot_index}', f'shot_{shot_index}', str(shot_index)):
            for ext in ('.png', '.jpg', '.jpeg', '.webp'):
                candidate = os.path.join(panel_dir, stem + ext)
                if os.path.exists(candidate):
                    return candidate
        return None

    def _expected_character_count(self, story_id: str, shot_index: int, script_count: int) -> int:
        """
        Reference character count for OCCM.

        Counted by running GroundingDINO with the same plain "character" prompt over
        the ground-truth manga panel, so the reference and the generated shot are
        measured by the same detector at the same threshold. The script's character
        list used to be the reference, but it lists characters the panel only implies
        (named but off-panel, or drawn too small to read), which charged the generator
        for characters the source art does not actually show.

        Falls back to the script count when the story ships no panel for this shot.
        """
        cache = getattr(self, '_panel_count_cache', None)
        if cache is None:
            cache = self._panel_count_cache = {}
        key = (str(story_id), int(shot_index))
        if key in cache:
            return cache[key]

        panel = self._panel_path(story_id, shot_index)
        if panel is None:
            if not getattr(self, '_panel_missing_warned', set()) or str(story_id) not in self._panel_missing_warned:
                if not hasattr(self, '_panel_missing_warned'):
                    self._panel_missing_warned = set()
                self._panel_missing_warned.add(str(story_id))
                print(f"\033[33mOCCM: no {self.PANEL_DIR_NAME}/ for story {story_id}; "
                      f"falling back to the script's character count.\033[0m")
            cache[key] = int(script_count)
            return cache[key]

        _, logits, _, _ = self.dino_detect(panel, self.OCCM_PROMPT)
        count = int(len(logits))
        print(f"OCCM: story {story_id} shot {shot_index}: {count} character(s) detected in {os.path.basename(panel)}")
        cache[key] = count
        return count

    @staticmethod
    def _occm_score(detected: int, expected: int) -> float:
        """
        100 at an exact match, decaying with the relative miscount.

        `expected` can now be 0 (a panel GroundingDINO finds nobody in), so the
        denominator is floored at 1 instead of collapsing the score to 0 for any
        detection at all.
        """
        epsilon = 1e-6
        denom = epsilon + max(int(expected), 1)
        return 100 * float(np.exp(-float(abs(int(detected) - int(expected))) / denom))

    # ------------------------------------------------------------------
    # Per-frame character detection and matching
    # ------------------------------------------------------------------
    def _match_frame(self, frame_path, shot_chars, Characters, ref_feats_by_char, prompt):
        """
        Detect characters in one frame and assign them to the shot's cast.

        Returns (matches, detected_count) where `matches` maps character key ->
        {"box", "cross_sim", "crop"} for the characters that got a box, and
        `detected_count` is how many boxes GroundingDINO produced (what OCCM counts).
        """
        boxes, logits, _, _ = self.dino_detect(frame_path, prompt)
        candidate_boxes, candidate_crops = [], []
        if len(logits) > 0:
            # Use all detected boxes (sorted by confidence) to reflect actual on-screen characters
            _, indices = torch.sort(logits, descending=True)
            cropped_chunk, filtered_boxes = self.crop_img(frame_path, boxes[indices], return_boxes=True)
            keep_n = min(len(filtered_boxes), len(cropped_chunk))
            candidate_crops = cropped_chunk[:keep_n]
            candidate_boxes = filtered_boxes[:keep_n]
        detected_count = len(candidate_boxes)

        if not candidate_crops:
            return {}, detected_count

        boxes_t = torch.tensor(candidate_boxes, dtype=torch.float32)
        num_candidates = len(candidate_crops)
        num_chars = len(shot_chars)
        sim_scores = torch.full((num_chars, num_candidates), -1e6, dtype=torch.float32)
        char_feat_cache = {}
        char_sim_cache = {}
        valid_rows = [False] * num_chars

        # Build similarity matrix row for each character with its dedicated encoder
        for row_idx, char in enumerate(shot_chars):
            enc_name = self._get_encoder_name(Characters[char]["tag"])
            output_feats, valid_indices = self.get_char_feat(
                candidate_crops, encoder_name=enc_name, return_indices=True
            )
            char_feat_cache[char] = {"features": output_feats, "indices": valid_indices}
            if output_feats is None or len(valid_indices) == 0:
                continue

            out_feats_for_sim = output_feats.to(self.device) if isinstance(output_feats, torch.Tensor) else output_feats
            ref = ref_feats_by_char[char]
            ref_feats_for_sim = ref.to(self.device) if isinstance(ref, torch.Tensor) else ref
            similarity_matrix = self._compute_multimodel_similarity_matrix(
                out_feats_for_sim, ref_feats_for_sim, method=self.ensemble_method
            )
            if similarity_matrix is None or not isinstance(similarity_matrix, torch.Tensor) or similarity_matrix.numel() == 0:
                continue

            similarity_matrix = similarity_matrix.detach().cpu()
            char_sim_cache[char] = {cand_idx: similarity_matrix[idx] for idx, cand_idx in enumerate(valid_indices)}
            best_scores = similarity_matrix.max(dim=1).values
            for local_idx, cand_idx in enumerate(valid_indices):
                sim_scores[row_idx, cand_idx] = best_scores[local_idx]
            valid_rows[row_idx] = True

        if not any(valid_rows):
            return {}, detected_count

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-sim_scores.cpu().numpy())
        matches = {}
        for r, c in zip(row_ind, col_ind):
            if r >= num_chars or c >= num_candidates or not valid_rows[r]:
                continue
            char = shot_chars[r]
            # only accept candidate boxes that have features for the current character, so none are missing later
            if c not in (char_feat_cache.get(char, {}).get("indices") or []):
                continue
            sim_row = char_sim_cache.get(char, {}).get(c)
            cross_sim = round(sim_row.mean().item(), 4) if sim_row is not None else 0.0
            matches[char] = {
                "box": [round(x, 3) for x in boxes_t[c].tolist()],
                "cross_sim": cross_sim,
                "crop": candidate_crops[c],
            }
        return matches, detected_count

    def _score_single_character_action(self, system_prompt, shot, character_name, crop):
        """VLM score for one character's action, repeated so the spread is measurable."""
        from vistorybench.bench.prompt_align.gptv_utils import (
            load_img_content, repeat_score, summarize_scores,
        )
        prompt_text = shot.get('script') or shot.get('character_action') or shot.get('action') or shot.get('description') or ''
        transcript = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "text", "text": f"Evaluated Character Name is {character_name}"},
                load_img_content(crop, image_mode='pil'),
            ]},
        ]
        try:
            samples = repeat_score(
                transcript, self.gpt_api_pkg, repeats=self.pa_repeats,
                temp=self.pa_temperature, top_p=self.pa_top_p
            )
        except Exception as e:
            print(f"\033[33mWarning: single_character_action scoring failed for {character_name}: {e}\033[0m")
            samples = []
        return summarize_scores(samples)

    def evaluate(self, method: str, story_id: str, **kwargs):

        # Create a directory for intermediate results (aligned to unified result schema)
        run_dir = os.path.join(self.result_path, method, self.mode, self.language, self.timestamp)
        self.mid_result_dir = os.path.join(run_dir, 'metrics', 'cids', 'mid_results', story_id)
        os.makedirs(self.mid_result_dir, exist_ok=True)

        story_data = self.story_dataset.load_story(story_id)
        shots = story_data['shots']
        Characters = story_data['characters']

        all_outputs = load_outputs(
            outputs_root=self.output_path,
            methods=[method],
            modes=[self.mode],
            languages=[self.language],
            timestamps=self.outputs_timestamp,
            return_latest=False
        )
        story_outputs = all_outputs.get(story_id)
        if not story_outputs:
            print(f"Warning: No outputs found for story {story_id}, method {method}")
            return None

        CHARACTER = list(Characters.keys())
        REF_PATH = {char: os.path.join(self.dataset_path, 'ViStory', story_id, "image", char) for char in CHARACTER}
        TEXT_PROMPT = {char: "character" for char in CHARACTER} if not self.PRE_TEXT_PROMPT else self.PRE_TEXT_PROMPT

        if self.ref_mode == 'mid-gen':
            MID_GEN_REF_PATH = {char: None for char in CHARACTER}
            MID_GEN_CHAR_LIST = story_outputs.get('chars', [])

        # Load references
        ref_clip_feats = {}
        ref_copy_paste_feats = {}
        for char in CHARACTER:
            enc_name = self._get_encoder_name(Characters[char]["tag"])
            ch_name = Characters[char]['name']
            input_ref_imgs = []
            ref_feats = None
            ref_indices = []

            if self.ref_mode == 'origin':
                ref_imgs = sorted(glob.glob(os.path.join(REF_PATH[char], '**/*.jpg'), recursive=True))
            elif self.ref_mode == 'mid-gen':
                ref_imgs = []
                for img_path in MID_GEN_CHAR_LIST:
                    char_file_name = os.path.basename(img_path).split(".")[0]
                    if char_file_name in MID_GEN_REF_PATH.keys() and char == char_file_name:
                        MID_GEN_REF_PATH[char] = img_path
                        break
                    elif ch_name in char_file_name:
                        MID_GEN_REF_PATH[char] = img_path
                        break
                if MID_GEN_REF_PATH[char] is None:
                    print(f'[WARN] Missing character: {char}, Assign an empty value [] (available character: {CHARACTER})')
                else:
                    ref_imgs = sorted(glob.glob(MID_GEN_REF_PATH[char], recursive=True))

            for img in ref_imgs:
                boxes, logits, _, _ = self.dino_detect(img, TEXT_PROMPT[char])
                if len(logits) == 0:
                    print(f"\033[33mNo objects detected in reference image {img} for {char} with prompt: {TEXT_PROMPT[char]}\033[0m")
                    continue
                _, indices = torch.topk(logits, 1)
                boxes = boxes[indices]
                cropped_imgs = self.crop_img(img, boxes)
                if len(cropped_imgs) != 0:
                    input_ref_imgs.append(cropped_imgs[0])
                else:
                    print(f"\033[33mNo char: {char} found in {img} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")

            if len(input_ref_imgs) > 0:
                ref_feats, ref_indices = self.get_char_feat(
                    input_ref_imgs,
                    encoder_name=enc_name,
                    return_indices=True,
                )
                if ref_feats is None:
                    input_ref_imgs = [Image.open(x) for x in ref_imgs]
                    if len(input_ref_imgs) > 0:
                        ref_feats, ref_indices = self.get_char_feat(
                            input_ref_imgs,
                            encoder_name=enc_name,
                            return_indices=True,
                        )
            else:
                print(f"\033[33mNo valid cropped reference images for {char}, trying original images\033[0m")
                input_ref_imgs = [Image.open(x) for x in ref_imgs]
                if len(input_ref_imgs) > 0:
                    ref_feats, ref_indices = self.get_char_feat(
                        input_ref_imgs,
                        encoder_name=enc_name,
                        return_indices=True,
                    )
                else:
                    ref_feats = None

            assert ref_feats is not None, f"Cant get ref char: {char}, please check. No valid reference images found."
            ref_clip_feats[char] = ref_feats
            primary_idx = ref_indices[0] if ref_indices else 0
            primary_ref_img = input_ref_imgs[primary_idx]
            cp_feats = self.get_char_feat(
                [primary_ref_img, ImageOps.mirror(primary_ref_img)],
                encoder_name=enc_name,
            )
            if cp_feats is None:
                raise AssertionError(f"Failed to build copy-paste reference features for {char}.")
            if isinstance(cp_feats, list):
                if len(cp_feats) == 1:
                    cp_feats = [cp_feats[0], cp_feats[0]]
            elif isinstance(cp_feats, torch.Tensor):
                if cp_feats.dim() == 1:
                    cp_feats = cp_feats.unsqueeze(0)
                if cp_feats.shape[0] == 1:
                    cp_feats = torch.cat([cp_feats, cp_feats], dim=0)
            ref_copy_paste_feats[char] = cp_feats

        results = {"cids": {}}
        # All matched crops (every sampled frame) -> cross similarity and copy-paste.
        char_pil_imgs = {}
        # One crop per shot (the shot's representative frame) -> self consistency, so
        # self-CIDS keeps meaning "how stable is this character across shots" and stays
        # comparable between a storyboard run and a 10-frame-per-shot video run.
        char_rep_imgs = {}
        occm_scores = []
        occm_details = {}
        new_shot_indices = []

        # Single-character action alignment (per-shot, per-character)
        pa_detailed_scores = {}
        pa_story_scores = []
        pa_story_stds = []
        pa_tasks = []
        pa_shot_scores = {}
        model_type, api_key, base_url = self.gpt_api_pkg if hasattr(self, 'gpt_api_pkg') else (None, '', '')
        pa_available = bool(api_key) and bool(base_url)
        if not pa_available and not getattr(self, '_pa_skip_warned', False):
            print("\033[33mWarning: single_character_action skipped: missing GPT-V base_url or api_key in config.\033[0m")
            self._pa_skip_warned = True
        single_prompt_text = None
        if pa_available:
            try:
                with open(self._pa_prompt_path, 'r', encoding='utf-8', errors='replace') as _pf:
                    single_prompt_text = _pf.read()
            except Exception as _e:
                print(f"\033[33mWarning: failed to read single_character_action prompt file: {_e}\033[0m")
                pa_available = False

        for shot in shots:
            shot_results = {}
            pa_char_scores = {}
            pa_char_stds = {}
            shot_id = int(shot["index"])

            if not story_outputs['shots'].get(shot_id):
                print(f"Warning: No output image for shot_id {shot_id}. Skipping...")
                continue

            frames = self.get_shot_frames(story_outputs, shot_id)
            frames = [f for f in frames if f and os.path.exists(f)]

            shot_chars = shot['character_key']
            if not shot_chars:
                continue

            # OCCM reference: characters GroundingDINO finds in the ground-truth manga
            # panel, falling back to the script's character list when no panel exists.
            expected_chars = self._expected_character_count(story_id, shot_id, len(shot_chars))

            if not frames:
                print(f"\033[31mError: No frames found for shot {shot_id}. Skipping shot.\033[0m")
                for char in shot_chars:
                    shot_results.update({char: {"box": "null", "cross_sim": 0.0}})
                occm_scores.append(self._occm_score(0, expected_chars))
                results.update({f"shot-{shot_id}": shot_results})
                continue

            for char in shot_chars:
                char_pil_imgs.setdefault(char, [])
                char_rep_imgs.setdefault(char, [])

            common_prompt = TEXT_PROMPT.get(shot_chars[0], "character") if isinstance(TEXT_PROMPT, dict) else "character"
            rep_idx = len(frames) // 2

            # Score every sampled frame independently, then average across frames.
            frame_matches = []
            detected_counts = []
            for frame_path in frames:
                matches, detected = self._match_frame(
                    frame_path, shot_chars, Characters, ref_clip_feats, common_prompt
                )
                frame_matches.append(matches)
                detected_counts.append(detected)

            occm_frame_scores = [self._occm_score(d, expected_chars) for d in detected_counts]
            occm_shot = sum(occm_frame_scores) / len(occm_frame_scores)
            occm_scores.append(occm_shot)
            occm_details[shot_id] = {
                "expected": expected_chars,
                "detected_per_frame": detected_counts,
                "score": round(occm_shot, 4),
            }

            for char in shot_chars:
                sims = []
                boxes_seen = []
                for f_i, matches in enumerate(frame_matches):
                    hit = matches.get(char)
                    if hit is None:
                        sims.append(0.0)
                        continue
                    sims.append(hit["cross_sim"])
                    boxes_seen.append(hit["box"])
                    char_pil_imgs[char].append(hit["crop"])
                    hit["crop"].save(f"{self.mid_result_dir}/shot{shot_id:02d}-{char}-f{f_i:02d}.png")

                rep_hit = frame_matches[rep_idx].get(char) or next(
                    (m[char] for m in frame_matches if m.get(char)), None
                )
                if rep_hit is not None:
                    char_rep_imgs[char].append(rep_hit["crop"])

                mean_sim = round(sum(sims) / len(sims), 4) if sims else 0.0
                shot_results.update({
                    char: {
                        "box": rep_hit["box"] if rep_hit else "null",
                        "cross_sim": mean_sim,
                        "num_frames": len(frames),
                        "num_frames_matched": len(boxes_seen),
                    }
                })

                # Single-character action alignment via GPT-V, scored on the
                # representative crop and repeated so we get a mean and a spread.
                # Queued rather than called here: at 5 repeats per character these are
                # the slowest part of the story by far, and they have no reason to run
                # one at a time.
                if pa_available and single_prompt_text:
                    if rep_hit is not None:
                        pa_tasks.append((shot_id, Characters[char]['name'], shot, rep_hit["crop"]))
                    else:
                        pa_char_scores[Characters[char]['name']] = 0
                        pa_char_stds[Characters[char]['name']] = 0.0

            if pa_available:
                pa_shot_scores[shot_id] = (pa_char_scores, pa_char_stds)

            results.update({f"shot-{shot_id}": shot_results})
            new_shot_indices.append(shot_id)

        # Run the queued single_character_action scorings in parallel, then fold them
        # back into their shots.
        if pa_available and pa_tasks:
            def score_task(task):
                shot_id, char_name, shot, crop = task
                return shot_id, char_name, self._score_single_character_action(
                    single_prompt_text, shot, char_name, crop
                )

            workers = max(1, int(getattr(self, 'gpt_workers', 1) or 1))
            print(f"CIDS: scoring {len(pa_tasks)} character action item(s) x {self.pa_repeats} "
                  f"repeats for story {story_id}, {workers} workers.")
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    task_results = list(ex.map(score_task, pa_tasks))
            else:
                task_results = [score_task(t) for t in pa_tasks]

            for shot_id, char_name, stats in task_results:
                scores, stds = pa_shot_scores.setdefault(shot_id, ({}, {}))
                scores[char_name] = round(stats["mean"], 4)
                stds[char_name] = round(stats["std"], 4)

        if pa_available:
            for shot_id in sorted(pa_shot_scores):
                scores, stds = pa_shot_scores[shot_id]
                shot_pa_mean = sum(scores.values()) / len(scores) if scores else 0.0
                shot_pa_std = sum(stds.values()) / len(stds) if stds else 0.0
                pa_detailed_scores[shot_id] = {
                    "single_character_action": round(shot_pa_mean, 4),
                    "single_character_action_std": round(shot_pa_std, 4),
                    "single_character_action_char": dict(scores),
                    "single_character_action_char_std": dict(stds),
                }
                pa_story_scores.append(shot_pa_mean)
                pa_story_stds.append(shot_pa_std)

        # Aggregate per-character and story metrics using char_pil_imgs (from processed + new)
        copy_paste_cnt = 0
        shot_copy_paste_score = 0.0
        for char in CHARACTER:
            enc_name = self._get_encoder_name(Characters[char]["tag"])
            if char in char_pil_imgs and len(char_pil_imgs[char]) > 0:
                char_feats = self.get_char_feat(char_pil_imgs[char], encoder_name=enc_name)
                if char_feats is None:
                    results["cids"].update({char: {"cross": 0.0, "self": 0.0}})
                    continue

                # Cross uses every sampled frame; self uses one crop per shot so it
                # keeps measuring shot-to-shot drift rather than within-clip stillness.
                cross_similarity_matrix = self._compute_multimodel_similarity_matrix(char_feats, ref_clip_feats[char], method=self.ensemble_method)
                cross_sim = cross_similarity_matrix.mean().item()

                rep_imgs = char_rep_imgs.get(char) or char_pil_imgs[char]
                rep_feats = self.get_char_feat(rep_imgs, encoder_name=enc_name)
                if rep_feats is None:
                    rep_feats = char_feats
                self_similarity_matrix = self._compute_multimodel_similarity_matrix(rep_feats, rep_feats, method=self.ensemble_method)
                if self_similarity_matrix.shape[0] > 1:
                    indices = torch.triu_indices(self_similarity_matrix.shape[0], self_similarity_matrix.shape[1], offset=1)
                    self_sim = self_similarity_matrix[indices[0], indices[1]].mean().item()
                else:
                    self_sim = 1.0
                results["cids"].update({char: {"cross": round(cross_sim, 4), "self": round(self_sim, 4)}})

                copy_paste_cnt += 1
                cross_sims = self._compute_multimodel_similarity_matrix(
                    char_feats,
                    ref_copy_paste_feats[char],
                    method=self.ensemble_method,
                )  # [G, 2]
                s_orig = torch.clamp(cross_sims[:, 0], -1.0, 1.0)  # [G]
                s_mirror = torch.clamp(cross_sims[:, 1], -1.0, 1.0)  # [G]
                all_sims = torch.stack([s_orig, s_mirror], dim=1)  # [G, 2]
                T = 0.01
                probs = torch.softmax(all_sims / T, dim=1)  # [G, 2]
                copy_scores = probs[:, 0]  # [G]
                char_copy_rate = copy_scores.mean().item()
                shot_copy_paste_score += char_copy_rate
            else:
                results["cids"].update({char: {"cross": 0.0, "self": 0.0}})

        if copy_paste_cnt > 0:
            shot_copy_paste_score /= copy_paste_cnt
            results.update({"copy-paste-score": shot_copy_paste_score})
        else:
            results.update({"copy-paste-score": "null"})

        try:
            cids_map = results.get("cids", {}) if isinstance(results, dict) else {}
            self_vals = []
            cross_vals = []
            for _char, _rec in cids_map.items():
                if isinstance(_rec, dict):
                    sv = _rec.get("self")
                    cv = _rec.get("cross")
                    if isinstance(sv, (int, float)):
                        self_vals.append(float(sv))
                    if isinstance(cv, (int, float)):
                        cross_vals.append(float(cv))
            cids_self_mean = round(sum(self_vals) / len(self_vals), 4) if self_vals else 0.0
            cids_cross_mean = round(sum(cross_vals) / len(cross_vals), 4) if cross_vals else 0.0
            occm_mean = round(sum(occm_scores) / len(occm_scores), 4) if occm_scores else 0.0
            cp = results.get("copy-paste-score", None)
            copy_paste_score = None if cp == "null" else float(cp) if isinstance(cp, (int, float)) else None

            results["metrics"] = {
                "cids_self_mean": cids_self_mean,
                "cids_cross_mean": cids_cross_mean,
                "copy_paste_score": copy_paste_score if copy_paste_score is not None else 0.0,
                "occm": occm_mean
            }
            if pa_available and pa_story_scores:
                results["metrics"]["single_character_action"] = round(sum(pa_story_scores) / len(pa_story_scores), 4)
                results["metrics"]["single_character_action_std"] = round(sum(pa_story_stds) / len(pa_story_stds), 4)
        except Exception as _e:
            print(f"\033[33mWarning: failed to compute CIDS story metrics for {story_id}: {_e}\033[0m")

        if pa_available:
            results["detailed_scores"] = pa_detailed_scores
        results["occm_details"] = occm_details
        results["scoring"] = {"repeats": self.pa_repeats, "temperature": self.pa_temperature,
                              "top_p": self.pa_top_p,
                              "encoder": self.encoder_override or self.image_encoder,
                              "occm_prompt": self.OCCM_PROMPT,
                              "box_threshold": self.BOX_TRESHOLD,
                              "text_threshold": self.TEXT_TRESHOLD}
        results["new_shot_indices"] = new_shot_indices
        print(f"CIDS evaluation complete for story: {story_id}")
        return results

    def build_item_records(self, method: str, story_id: str, story_result):
        items = []
        try:
            if isinstance(story_result, dict):
                new_only = set(story_result.get("new_shot_indices") or [])
                # Existing CIDS item records
                for k, shot_res in story_result.items():
                    if not isinstance(k, str) or not k.startswith("shot-"):
                        continue
                    try:
                        shot_idx = int(k.split("-")[1])
                    except Exception:
                        shot_idx = k
                    if new_only and shot_idx not in new_only:
                        continue
                    if isinstance(shot_res, dict):
                        for char_key, v in shot_res.items():
                            if not isinstance(v, dict):
                                continue
                            cross_val = v.get("cross_sim", 0.0)
                            box = v.get("box", None)
                            if cross_val is None:
                                continue
                            item = {
                                "metric": {"name": "cids", "submetric": "cross_sim"},
                                "scope": {"level": "item", "story_id": str(story_id), "shot_index": shot_idx, "character_key": char_key},
                                "value": cross_val,
                                "unit": "cosine_similarity",
                                "extras": {"box": box},
                                "status": "complete",
                            }
                            items.append(item)
                # OCCM item records: what the panel showed vs what the shot showed
                for shot_idx, det in (story_result.get("occm_details") or {}).items():
                    items.append({
                        "metric": {"name": "cids", "submetric": "occm"},
                        "scope": {"level": "item", "story_id": str(story_id), "shot_index": shot_idx},
                        "value": det.get("score"),
                        "extras": {"expected": det.get("expected"),
                                   "detected_per_frame": det.get("detected_per_frame")},
                        "status": "complete",
                    })
                # New PromptAlign single_character_action item records per character (if available)
                detailed = story_result.get("detailed_scores")
                if isinstance(detailed, dict):
                    for shot_idx, dims in detailed.items():
                        char_map = (dims or {}).get("single_character_action_char")
                        if isinstance(char_map, dict):
                            for char_name, score in char_map.items():
                                item = {
                                    "metric": {"name": "prompt_align", "submetric": "single_character_action"},
                                    "scope": {"level": "item", "story_id": str(story_id), "shot_index": shot_idx, "character_name": char_name},
                                    "value": score,
                                    "extras": {"std": ((dims or {}).get("single_character_action_char_std") or {}).get(char_name)},
                                    "status": "complete",
                                }
                                items.append(item)
        except Exception as _e:
            print(f"\033[33mWarning: build_item_records failed for CIDS story {story_id}: {_e}\033[0m")
        return items
