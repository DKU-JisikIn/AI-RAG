##### 필요한 라이브러리 임포트
import json
import numpy as np
import joblib
from transformers import AutoTokenizer
import onnxruntime as ort
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
CHATBOT_DIR = Path(__file__).resolve().parent

##### .env 파일에서 환경변수 불러오기
load_dotenv(BASE_DIR / ".env")

QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")

API_URL = "https://api.openai.com/v1/chat/completions"

class ChatBot:
    ##### 모델 및 라벨 인코더 로딩
    session = ort.InferenceSession(CHATBOT_DIR / "subcat_model_quant.onnx")
    le = joblib.load(CHATBOT_DIR / "subcategory_label_encoder.pkl")
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

    ##### 임베딩 모델 로딩
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60.0
    )
    embed_model = SentenceTransformer("BM-K/KoSimCSE-roberta")

    ##### 서브카테고리 → 카테고리 매핑
    subcategory_to_category = {
        '학적': '학사', '성적': '학사', '수업': '학사', '교양': '학사', '교직': '학사',
        '복지시설': '일반', '예비군': '일반', '등록': '일반', 'IT서비스': '일반', '도서관': '일반',
        '기타': '일반', '기숙사': '일반', '제증명/학생증': '일반', '학생지원/교통': '일반',
        '장학': '장학',
        '국제(외국인유학생,한국어연수)': '국제', '국제교류(교환학생,어학연수)': '국제',
        '취업': '진로', '대학원': '진로', '자격증': '진로',
        '동아리': '동아리'
    }

    answer_system_prompt = (
    "당신은 단국대학생을 위한 대학생 전문 챗봇입니다. "
    "아래 참고자료를 바탕으로, 사용자의 질문에 대해 정확하고 친절하게 답변하세요. "
    "반드시 참고자료의 내용을 근거로 답변하고, 근거가 없으면 '자료에 해당 정보가 없습니다.'라고 안내하세요. "
    "답변은 3문장 이내로 간결하게 작성하세요. "
    "비슷한 내용의 문장은 나열하지 말아주세요. "
    "중요: 반드시 한국어로만 답변하세요. 영어, 태국어, 일본어 등 외국어, 특수문자, 오타를 절대 사용하지 마세요. "
    "F학점 등 학점 표기는 그대로 사용해도 되지만, 그 외에는 한글만 사용하세요. "
    "죽전캠퍼스 대상 챗봇이므로 천안 관련 답변은 하지 말아주세요. "
    )

    title_system_prompt = (
    "입력된 게시글 내용을 한글로 아주 간단하고 핵심만 담은 한 문장 제목으로 만들어주세요. "
    "제목은 15자 이내로, 자연스럽고 명확하게 작성하세요. "
    "영어, 특수문자, 오타 없이 한글로만 작성하세요."
    )

    ##### 함수 정의
    @staticmethod
    def predict_subcategory(question):
        inputs = ChatBot.tokenizer(
            question,
            return_tensors="np",
            truncation=True,
            padding='max_length',
            max_length=7
        )
        onnx_inputs = {k: v for k, v in inputs.items() if k in [x.name for x in ChatBot.session.get_inputs()]}
        outputs = ChatBot.session.run(None, onnx_inputs)
        logits = outputs[0]
        pred = int(np.argmax(logits, axis=1)[0])
        subcat = ChatBot.le.inverse_transform([pred])[0]
        return subcat

    @staticmethod
    def search_similar_questions(question, collection_name, top_k=5, threshold=0.7):
        query_vector = ChatBot.embed_model.encode(question).tolist()
        results = ChatBot.qdrant_client.search(
            collection_name=f"dku_{collection_name}",
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=True
        )
        filtered = [
            {
                "question": hit.payload["question"],
                "answer": hit.payload["answer"],
                "score": hit.score,
                "source": hit.payload.get("source", None)
            }
            for hit in results if hit.score >= threshold
        ]
        return filtered

    @staticmethod
    def generate_answer_openrouter(question, context_list):
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a, score in context_list])
        prompt = (
            f"{ChatBot.answer_system_prompt or ''}\n"
            f"질문: {question}\n"
            f"참고자료:\n{context}\n"
            f"답변:"
        )
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4.1-2025-04-14",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256
        }
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"예외 발생: {e}"

    @staticmethod
    def rag_pipeline(user_question):
        subcat = ChatBot.predict_subcategory(user_question)
        bigcat = ChatBot.subcategory_to_category.get(subcat)
        if not bigcat:
            return "해당 질문의 카테고리를 찾을 수 없습니다. 다시 질문해주세요."
        similar_qas = ChatBot.search_similar_questions(user_question, bigcat, top_k=5, threshold=0.7)
        if not similar_qas:
            return "해당하는 질문이 없습니다. 게시판에 글을 올려주세요."
        context_list = [(qa["question"], qa["answer"], qa["score"]) for qa in similar_qas]
        answer = ChatBot.generate_answer_openrouter(user_question, context_list)
        references = [
        {'title': qa['question'], 'source': qa['source']}
        for qa in similar_qas
        ]
        
        return {
        "answer": answer,
        "references": references
        }

    @staticmethod
    def generate_title_from_content(content: str) -> str:
        prompt = (
            f"{ChatBot.title_system_prompt}\n"
            f"게시글 내용: {content}\n"
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
                title = result["choices"][0]["message"]["content"].strip()
                return title
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"예외 발생: {e}"

if __name__ == "__main__":
    # 챗봇 답변
    user_question = input("질문을 입력하세요: ")
    answer = ChatBot.rag_pipeline(user_question)
    print(f"{answer}")

    # 제목 생성
    post_content = input("\n제목을 만들 게시글 내용을 입력하세요:\n")
    title = ChatBot.generate_title_from_content(post_content)
    print(f"{title}")