import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

def generate_title_openrouter(post_content, system_prompt=None):
    system_prompt = system_prompt or (
        "입력된 게시글 내용을 한글로 아주 간단하고 핵심만 담은 한 문장 제목으로 만들어주세요. "
        "제목은 15자 이내로, 자연스럽고 명확하게 작성하세요. "
        "영어, 특수문자, 오타 없이 한글로만 작성하세요."
    )
    prompt = (
        f"{system_prompt}\n"
        f"게시글 내용: {post_content}\n"
        f"생성된 제목:"
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4.1-2025-04-14",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32
    }
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"예외 발생: {e}"
    
if __name__ == "__main__":
    text = input("요약할 텍스트를 입력하세요:\n")
    print("\n요약 결과:", generate_title_openrouter(text))