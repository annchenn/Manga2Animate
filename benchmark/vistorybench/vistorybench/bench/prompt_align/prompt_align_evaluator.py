from concurrent.futures import ThreadPoolExecutor

from vistorybench.bench.base_evaluator import BaseEvaluator
from vistorybench.dataset_loader.read_outputs import load_outputs
from vistorybench.bench.prompt_align.gptv_utils import (
    load_img_content,
    repeat_score,
    summarize_scores,
)

DIMENSIONS = ("scene", "character_action", "camera")


class PromptAlignEvaluator(BaseEvaluator):
    """
    VLM prompt alignment.

    Every (shot, dimension, frame) is scored `repeats` times rather than once. The
    reported value is the mean over those samples and `<dim>_std` is how far they
    spread, which is what tells us whether a gap between two methods is larger than
    the judge's own noise. A video shot contributes all of its sampled frames to the
    same pool, so its mean is the average over frames as well as over repeats.
    """

    def __init__(self, config: dict, timestamp: str, mode: str, language: str, outputs_timestamp=None):
        super().__init__(config, timestamp, mode, language, outputs_timestamp)

        # Get evaluator-specific config
        pa_cfg = self.get_evaluator_config('prompt_align')
        gpt_cfg = pa_cfg.get('gpt', {}) if isinstance(pa_cfg, dict) else {}
        # CLI overrides have priority but do not mutate YAML structure
        model = self.get_model_id() or 'gpt-4.1'
        base_url = self.get_base_url()
        api_key = self.get_api_key()

        self.gpt_api_pkg = (model, api_key, base_url)
        self.workers = int(gpt_cfg.get('workers', 1) or 1)

        # Repeated scoring: 5 draws per item, at a temperature that lets them differ.
        self.repeats = int(pa_cfg.get('repeats', 5) or 1)
        self.temperature = float(pa_cfg.get('temperature', 1.0))
        self.top_p = float(pa_cfg.get('top_p', 1.0))
        # None => score every frame the shot provides.
        max_frames = pa_cfg.get('max_frames_per_shot', None)
        self.max_frames_per_shot = int(max_frames) if max_frames else None

        # Built-in prompts (no YAML overrides)
        vlm_bench_path = 'vistorybench/bench/prompt_align'
        self.txt_prompt_list = {
            "scene": f"{vlm_bench_path}/user_prompts/user_prompt_environment_text_align.txt",
            "character_action": f"{vlm_bench_path}/user_prompts/user_prompt_character_text_align.txt",
            "camera": f"{vlm_bench_path}/user_prompts/user_prompt_camera_text_align.txt",
        }
        self._prompt_cache = {}

    def _dimension_prompt(self, eval_dimension):
        if eval_dimension not in self._prompt_cache:
            with open(self.txt_prompt_list[eval_dimension], "r") as f:
                self._prompt_cache[eval_dimension] = f.read()
        return self._prompt_cache[eval_dimension]

    def _subsample(self, frames):
        """Evenly thin a frame list down to `max_frames_per_shot`, keeping first and last."""
        if not self.max_frames_per_shot or len(frames) <= self.max_frames_per_shot:
            return frames
        n = self.max_frames_per_shot
        step = (len(frames) - 1) / (n - 1) if n > 1 else 0
        return [frames[round(i * step)] for i in range(n)]

    def _eval_single_dimension(self, eval_dimension, prompt, image_path, character_name=None, image_mode='path'):
        """Score one (dimension, frame) `repeats` times; returns the raw samples."""
        user_content = [
            {"type": "text", "text": self._dimension_prompt(eval_dimension)},
            {"type": "text", "text": prompt},
        ]
        if character_name:
            user_content.append({"type": "text", "text": f"Evaluated Character Name is {character_name}"})
        user_content.append(load_img_content(image_path, image_mode))

        transcript = [{"role": "user", "content": user_content}]
        return repeat_score(
            transcript,
            self.gpt_api_pkg,
            repeats=self.repeats,
            temp=self.temperature,
            top_p=self.top_p,
        )

    def evaluate(self, method: str, story_id: str, **kwargs):
        story_data = self.story_dataset.load_story(story_id)
        all_outputs = load_outputs(
            outputs_root=self.output_path,
            methods=[method],
            modes=[self.mode],
            languages=[self.language],
            timestamps=self.outputs_timestamp,
            return_latest=False
        )
        story_outputs = all_outputs.get(story_id)

        if not story_outputs or not story_outputs.get("shots"):
            print(f"Skipping prompt alignment for {story_id}: missing outputs.")
            return None

        image_paths = story_outputs["shots"]
        if not story_data or not image_paths:
            print(f"Skipping prompt alignment for {story_id}: data mismatch or missing.")
            return None

        shots = [shot for shot in story_data['shots'] if image_paths.get(int(shot['index']))]

        # One task per (shot, dimension, frame). Fanning out this finely keeps the
        # pool busy even when a story has few shots but many frames per shot.
        tasks = []
        for shot in shots:
            shot_index = int(shot['index'])
            frames = self._subsample(self.get_shot_frames(story_outputs, shot_index))
            prompts = {
                "scene": shot['scene'],
                "character_action": shot['script'],
                "camera": shot['camera'],
            }
            for dim in DIMENSIONS:
                for frame in frames:
                    tasks.append((shot_index, dim, frame, prompts[dim]))

        def run_task(task):
            shot_index, dim, frame, prompt = task
            try:
                return shot_index, dim, self._eval_single_dimension(dim, prompt, frame, image_mode='path')
            except Exception as e:
                print(f"Error evaluating {dim} for {story_id} shot {shot_index}: {e}")
                return shot_index, dim, []

        worker_count = max(1, getattr(self, "workers", 1) or 1)
        print(f"PromptAlign: {len(tasks)} (shot, dimension, frame) items x {self.repeats} repeats "
              f"for story {story_id}, {worker_count} workers.")
        if worker_count > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                task_results = list(ex.map(run_task, tasks))
        else:
            task_results = [run_task(t) for t in tasks]

        # Pool the samples per (shot, dimension)
        pooled = {}
        for shot_index, dim, samples in task_results:
            pooled.setdefault(shot_index, {}).setdefault(dim, []).extend(samples)

        all_shots_scores = {}
        shot_means = {dim: [] for dim in DIMENSIONS}
        shot_stds = {dim: [] for dim in DIMENSIONS}
        for shot_index, dims in pooled.items():
            shot_scores = {}
            for dim in DIMENSIONS:
                stats = summarize_scores(dims.get(dim, []))
                if stats["n"] == 0:
                    continue
                shot_scores[dim] = round(stats["mean"], 4)
                shot_scores[f"{dim}_std"] = round(stats["std"], 4)
                shot_scores[f"{dim}_samples"] = dims.get(dim, [])
                shot_means[dim].append(stats["mean"])
                shot_stds[dim].append(stats["std"])
            if shot_scores:
                all_shots_scores[shot_index] = shot_scores

        avg_scores = {}
        for dim in DIMENSIONS:
            means, stds = shot_means[dim], shot_stds[dim]
            avg_scores[dim] = round(sum(means) / len(means), 4) if means else 0
            # Mean within-item spread: how noisy the judge is on a typical shot.
            avg_scores[f"{dim}_std"] = round(sum(stds) / len(stds), 4) if stds else 0

        final_result = {
            'metrics': avg_scores,
            'detailed_scores': all_shots_scores,
            'scoring': {'repeats': self.repeats, 'temperature': self.temperature, 'top_p': self.top_p},
        }

        print(f"PromptAlign evaluation complete for story: {story_id}. Average scores: {avg_scores}")
        return final_result

    def build_item_records(self, method: str, story_id: str, story_result):
        items = []
        try:
            if isinstance(story_result, dict):
                detailed = story_result.get("detailed_scores")
                if isinstance(detailed, dict):
                    for shot_idx, dims in detailed.items():
                        for dim in DIMENSIONS:
                            if dim not in (dims or {}):
                                continue
                            item = {
                                "metric": {"name": "prompt_align", "submetric": dim},
                                "scope": {"level": "item", "story_id": str(story_id), "shot_index": shot_idx},
                                "value": dims[dim],
                                "extras": {
                                    "std": dims.get(f"{dim}_std"),
                                    "samples": dims.get(f"{dim}_samples"),
                                },
                                "status": "complete",
                            }
                            items.append(item)
        except Exception as _e:
            print(f"\033[33mWarning: build_item_records failed for PromptAlign story {story_id}: {_e}\033[0m")
        return items
