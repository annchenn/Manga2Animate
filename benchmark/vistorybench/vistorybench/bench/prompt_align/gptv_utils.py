import time
import json
import requests
import io
import base64
from PIL import Image


# Params some OpenAI-compatible endpoints reject (e.g. Gemini rejects 'seed' with
# HTTP 400). Dropped for the rest of the process once an endpoint refuses them.
_UNSUPPORTED_PARAMS = set()


def gptv_query(transcript=None, top_p=0.2, temp=0., model_type="gpt-4.1", api_key='', base_url='', seed=123, max_tokens=512, wait_time=10, max_retries=5):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    base_url = base_url[:-1] if base_url.endswith('/') else base_url
    requests_url = f"{base_url}/chat/completions" if base_url.endswith('/v1') else f"{base_url}/v1/chat/completions"

    data = {
        'model': model_type,
        'max_tokens': max_tokens,
        'temperature': temp,
        'messages': transcript or [],
    }
    if seed is not None and 'seed' not in _UNSUPPORTED_PARAMS:
        data['seed'] = seed

    response_text, retry, response_json, last_error = '', 0, None, 'no attempt made'
    while len(response_text) < 2 and retry < max_retries:
        retry += 1
        try:
            response = requests.post(url=requests_url, headers=headers, data=json.dumps(data))
            response_json = response.json()
        except Exception as e:
            last_error = str(e)
            print(e)
            time.sleep(wait_time)
            continue
        if response.status_code != 200:
            body = response.text
            last_error = f"HTTP {response.status_code}: {body[:300]}"
            # An endpoint that rejects an optional param will keep rejecting it, so
            # drop it and retry immediately instead of burning the retry budget.
            dropped = False
            for param in ('seed', 'top_p'):
                if param in data and param in body and (
                    'Unknown name' in body or 'Unrecognized' in body or 'unsupported' in body.lower()
                ):
                    print(f"Endpoint rejected '{param}'; dropping it and retrying.")
                    _UNSUPPORTED_PARAMS.add(param)
                    del data[param]
                    dropped = True
            if dropped:
                retry -= 1
                continue
            print(response.headers, response.content)
            time.sleep(wait_time)
            data['temperature'] = min(data['temperature'] + 0.2, 1.0)
            continue
        if 'choices' not in response_json:
            last_error = f"no 'choices' in response: {str(response_json)[:300]}"
            time.sleep(wait_time)
            continue
        response_text = response_json["choices"][0]["message"]["content"]

    if len(response_text) < 2:
        raise RuntimeError(f"gptv_query failed after {max_retries} attempts. Last error: {last_error}")
    return response_text


def encode_image(image_input, image_mode='path', resize_to=(512, 512)):
    if image_mode == 'path':
        with Image.open(image_input) as img:
            img = img.resize(resize_to)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    elif image_mode == 'pil':
        img = image_input.resize(resize_to)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_img_content(image_input, image_mode='path', resize_to=(512, 512), pil_detail='low'):
    base64_image = encode_image(image_input, image_mode, resize_to=resize_to)
    image_meta = "data:image/jpeg;base64"
    content = {
        "type": "image_url",
        "image_url": {"url": f"{image_meta},{base64_image}"},
    }
    if image_mode == 'pil':
        content["image_url"]["detail"] = pil_detail  # PIL image is small, configurable detail
    return content