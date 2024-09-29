import gradio as gr
from openai import OpenAI

model_name = "Translation-EnKo/gemma2-for-nlp-arxiv-translation"
model_port = 1785
base_url = f"http://0.0.0.0:{model_port}/v1"
api_key = "API_KEY_HERE"


def inference_api(message):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user","content": message}
        ],
        max_tokens=2048,
        stream=True,
    )
    return response


def stream_task_streaming(message, history):
    response = inference_api(message)
    collected_messages = []
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        collected_messages.append(chunk_message)
        collected_messages = [m for m in collected_messages if m is not None]
        full_reply_content = ''.join(collected_messages)
        yield full_reply_content


if __name__ == "__main__":
    client = OpenAI(base_url=base_url, api_key=api_key)

    demo = gr.ChatInterface(
        title=model_name,
        fn=stream_task_streaming,
        retry_btn=None,
        fill_height=True,
        description="영어 논문 단락을 입력해보세요. 읽기 편하게 변환해드려요!",
    )
    demo.launch(server_name="0.0.0.0")
