##### 필요한 라이브러리 임포트
import json
import torch
import joblib
import numpy as np
from transformers import AutoModel
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file 
from qdrant_client import QdrantClient
from openai import OpenAI
import requests

# 경고 메세지 무시
import warnings
warnings.filterwarnings("ignore")

##### 모델 및 라벨 인코더 로딩
# ONNX 세션 생성
session = ort.InferenceSession("subcat_model_quant.onnx")

# 라벨 인코더 불러오기
le = joblib.load("subcategory_label_encoder.pkl")

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
text = "예시 문장입니다."
inputs = tokenizer(
    text,
    return_tensors="np",        # numpy 배열로 반환
    padding='max_length',
    max_length=7,               # 반드시 7로 맞춰야 함
    truncation=True
)

onnx_inputs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
}

outputs = session.run(None, onnx_inputs)

##### 임베딩 모델 로딩
# Qdrant 접속
qdrant_client = QdrantClient(
    url="https://2a09054d-de92-436e-bf8c-158f44d82df4.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.1VAec5LQRLXCskcPERZg3WgNpTpj00q4ZwVqVkCy0RA",
    timeout=60.0
)

# 임베딩 모델 로딩 (질문 검색용)
embed_model = SentenceTransformer("BM-K/KoSimCSE-roberta")

##### 질문 검색
# subcategory → category 매핑 딕셔너리
subcategory_to_category = {
    '학적': '학사', '성적': '학사', '수업': '학사', '교양': '학사', '교직': '학사',
    '복지시설': '일반', '예비군': '일반', '등록': '일반', 'IT서비스': '일반', '도서관': '일반',
    '기타': '일반', '기숙사': '일반', '제증명/학생증': '일반', '학생지원/교통': '일반',
    '장학': '장학',
    '국제(외국인유학생,한국어연수)': '국제', '국제교류(교환학생,어학연수)': '국제',
    '취업': '진로', '대학원': '진로', '자격증': '진로',
    '동아리': '동아리'
}

# 질문 → 서브카테고리 분류 함수
def predict_subcategory(question, session, tokenizer, le):
    inputs = tokenizer(question, return_tensors="np", truncation=True, padding=True, max_length=7)
    onnx_inputs = {k: v for k, v in inputs.items() if k in [x.name for x in session.get_inputs()]}
    outputs = session.run(None, onnx_inputs)
    logits = outputs[0]
    pred = int(np.argmax(logits, axis=1)[0])
    subcat = le.inverse_transform([pred])[0]
    return subcat

# Qdrant에서 유사 질문 5개(유사도 0.7 이상만) 검색
def search_similar_questions(question, collection_name, embed_model, client, top_k=5, threshold=0.7):
    query_vector = embed_model.encode(question).tolist()
    results = client.search(
        collection_name=f"dku_{collection_name}",
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=True  # 유사도 계산을 위해 벡터도 가져옴
    )
    # Qdrant는 score가 1에 가까울수록 유사 (cosine similarity)
    filtered = [
        (hit.payload["question"], hit.payload["answer"], hit.score)
        for hit in results if hit.score >= threshold
    ]
    return filtered

##### 답변 생성
API_URL = "https://openrouter.ai/api/v1/chat/completions"
# API_KEY = ""

def generate_answer_openrouter(question, context_list, system_prompt=None):
    context = "\n".join([f"Q: {q}\nA: {a}" for q, a, score in context_list])
    prompt = (
        f"{system_prompt or ''}\n"
        f"질문: {question}\n"
        f"참고자료:\n{context}\n"
        f"답변:"
    )
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def generate_answer_openrouter(question, context_list, system_prompt=None):

    context = "\n".join([f"Q: {q}\nA: {a}" for q, a, score in context_list])
    prompt = (
        f"{system_prompt or ''}\n"
        f"질문: {question}\n"
        f"참고자료:\n{context}\n"
        f"답변:"
    )
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256
    }
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        print("응답 코드:", response.status_code)  # 디버깅용
        print("응답 본문:", response.text)        # 디버깅용
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"예외 발생: {e}"

##### RAG
system_prompt = (
    "당신은 단국대학생을 위한 대학생 전문 챗봇입니다. "
    "아래 참고자료를 바탕으로, 사용자의 질문에 대해 정확하고 친절하게 답변하세요. "
    "반드시 참고자료의 내용을 근거로 답변하고, 근거가 없으면 '자료에 해당 정보가 없습니다.'라고 안내하세요. "
    "답변은 3문장 이내로 간결하게 작성하세요. "
    "비슷한 내용의 문장은 나열하지 말아주세요. "
    "중요: 반드시 한국어로만 답변하세요. 영어, 태국어, 일본어 등 외국어, 특수문자, 오타를 절대 사용하지 마세요. "
    "F학점 등 학점 표기는 그대로 사용해도 되지만, 그 외에는 한글만 사용하세요."
)

def rag_pipeline(user_question):
    # 1. 서브카테고리 분류
    subcat = predict_subcategory(user_question, session, tokenizer, le)
    # 2. 서브카테고리 → 카테고리 매핑
    bigcat = subcategory_to_category.get(subcat)
    if not bigcat:
        return "해당 질문의 카테고리를 찾을 수 없습니다. 다시 질문해주세요."
    # 3. Qdrant에서 유사 질문 5개(유사도 0.7 이상) 검색
    similar_qas = search_similar_questions(user_question, bigcat, embed_model, qdrant_client, top_k=5, threshold=0.7)
    if not similar_qas:
        return "해당하는 질문이 없습니다. 게시판에 글을 올려주세요."
    # 4. 답변 생성
    answer = generate_answer_openrouter(user_question, similar_qas, system_prompt=system_prompt)
    references = "\n".join(
        [f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a, score) in enumerate(similar_qas)])
    return f"AI 답변:\n{answer}\n\n[참고한 Q&A]\n{references}"