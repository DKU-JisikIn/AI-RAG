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
import warnings
warnings.filterwarnings("ignore")

##### .env 파일에서 환경변수 불러오기
load_dotenv()

QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")

##### 모델 및 라벨 인코더 로딩
session = ort.InferenceSession("subcat_model_quant.onnx")
le = joblib.load("subcategory_label_encoder.pkl")
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

##### 함수 정의
def predict_subcategory(question, session, tokenizer, le):
    inputs = tokenizer(
        question,
        return_tensors="np",
        truncation=True,
        padding='max_length',
        max_length=7
    )
    onnx_inputs = {k: v for k, v in inputs.items() if k in [x.name for x in session.get_inputs()]}
    outputs = session.run(None, onnx_inputs)
    logits = outputs[0]
    pred = int(np.argmax(logits, axis=1)[0])
    subcat = le.inverse_transform([pred])[0]
    return subcat

def search_similar_questions(question, collection_name, embed_model, client, top_k=5, threshold=0.7):
    query_vector = embed_model.encode(question).tolist()
    results = client.search(
        collection_name=f"dku_{collection_name}",
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=True
    )
    filtered = [
        (hit.payload["question"], hit.payload["answer"], hit.score)
        for hit in results if hit.score >= threshold
    ]
    return filtered

API_URL = "https://openrouter.ai/api/v1/chat/completions"

def generate_answer_openrouter(question, context_list, system_prompt=None):
    context = "\n".join([f"Q: {q}\nA: {a}" for q, a, score in context_list])
    prompt = (
        f"{system_prompt or ''}\n"
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

system_prompt = (
    "당신은 단국대학생을 위한 대학생 전문 챗봇입니다. "
    "아래 참고자료를 바탕으로, 사용자의 질문에 대해 정확하고 친절하게 답변하세요. "
    "반드시 참고자료의 내용을 근거로 답변하고, 근거가 없으면 '자료에 해당 정보가 없습니다.'라고 안내하세요. "
    "답변은 3문장 이내로 간결하게 작성하세요. "
    "비슷한 내용의 문장은 나열하지 말아주세요. "
    "중요: 반드시 한국어로만 답변하세요. 영어, 태국어, 일본어 등 외국어, 특수문자, 오타를 절대 사용하지 마세요. "
    "F학점 등 학점 표기는 그대로 사용해도 되지만, 그 외에는 한글만 사용하세요. "
        "죽전캠퍼스 대상 챗봇이므로 천안 관련 답변은 하지 말아주세요. "
)

def rag_pipeline(user_question):
    subcat = predict_subcategory(user_question, session, tokenizer, le)
    bigcat = subcategory_to_category.get(subcat)
    if not bigcat:
        return "해당 질문의 카테고리를 찾을 수 없습니다. 다시 질문해주세요."
    similar_qas = search_similar_questions(user_question, bigcat, embed_model, qdrant_client, top_k=5, threshold=0.7)
    if not similar_qas:
        return "해당하는 질문이 없습니다. 게시판에 글을 올려주세요."
    answer = generate_answer_openrouter(user_question, similar_qas, system_prompt=system_prompt)
    question_list = [q for q, a, score in similar_qas]
    references = f"참고한 Qusetion: {question_list}"
    return f"\n{answer}\n\n{references}"

user_question = input("")
answer = rag_pipeline(user_question)
print(f"{answer}")