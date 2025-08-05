import openai
import base64
import mimetypes
from openai import OpenAI
from retrying import retry

@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def generate_gpt4o(prompt,client,model="Qwen3-8B",temperature=0.9,max_tokens=1024,image_paths=None,history=None,):
    # 设置 API 密钥
    # 创建消息历史记录，包含之前的对话
    messages = []
    if history:
        for h in history:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["assistant"]})
    if image_paths:  # image_paths 是图片路径组成的列表
        content = [{"type": "text", "text": prompt}]

        for image_path in image_paths:
            with open(image_path, "rb") as img_file:
                b64_image = base64.b64encode(img_file.read()).decode('utf-8')
            mime_type, _ = mimetypes.guess_type(image_path)
            mime_type = mime_type or "image/png"  # fallback
            data_url = f"data:{mime_type};base64,{b64_image}"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            })
        # 构造图文混合消息
        image_message = {
            "role": "user",
            "content": content
        }

        messages.append(image_message)
    else:
        # 纯文本输入
        messages.append({"role": "user", "content": prompt})
    # 调用模型
    response = client.chat.completions.create(
        model=model,##deepseek-r1
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


import requests
import base64
import mimetypes

def call_vllm_api(
    client,
    task: str,
    model: str,
    prompt: str = None,
    messages: list = None,
    inputs: list = None,
    query: str = None,
    documents: list = None,
    temperature: float = 0.9,
    max_tokens: int = None,
    image_paths: list = None,
    history: list = None,
    base_url : str = None,
    extra_body: dict = None
):
    def post_custom_api(path, body):
        try:
            # 优先尝试用 OpenAI 客户端内部 _client 发起请求
            return client._client.post(path, json=body).json()
        except Exception:
            base_url = getattr(client, "base_url", "http://localhost:8000/v1").rstrip("/")
            url = f"{base_url}/{path.lstrip('/')}"
            return requests.post(url, json=body).json()

    if task == "chat":
        chat_messages = messages or []
        if not chat_messages:
            if history:
                for h in history:
                    chat_messages.append({"role": "user", "content": h["user"]})
                    chat_messages.append({"role": "assistant", "content": h["assistant"]})
            if image_paths:
                content = [{"type": "text", "text": prompt}]
                for image_path in image_paths:
                    with open(image_path, "rb") as img_file:
                        b64_image = base64.b64encode(img_file.read()).decode("utf-8")
                    mime_type, _ = mimetypes.guess_type(image_path)
                    data_url = f"data:{mime_type or 'image/png'};base64,{b64_image}"
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
                chat_messages.append({"role": "user", "content": content})
            else:
                chat_messages.append({"role": "user", "content": prompt})

        return client.chat.completions.create(
            model=model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **(extra_body or {})
        )

    elif task == "completion":
        return client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **(extra_body or {})
        )

    elif task == "embedding":
        return client.embeddings.create(
            model=model,
            input=inputs,
            **(extra_body or {})
        )

    elif task == "score":
        body = {"model": model, "text_1": query, "text_2": documents}
        if extra_body: body.update(extra_body)
        return post_custom_api("/score", body)

    elif task == "rerank":
        body = {"model": model, "query": query, "documents": documents}
        if extra_body: body.update(extra_body)
        return post_custom_api("/rerank", body)

    elif task == "tokenize":
        body = {"model": model, "prompt": prompt}
        return post_custom_api("/tokenize", body)

    elif task == "detokenize":
        body = {"model": model, "tokens": inputs}
        return post_custom_api("/detokenize", body)

    else:
        raise ValueError(f"Unsupported task type: {task}")

    __all__ = ["call_vllm_api", "generate_gpt4o"]