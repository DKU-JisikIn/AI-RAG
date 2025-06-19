import os
import RAG

##### 마지막 id 가져오기 및 저장
def load_last_id(file_path="last_id.txt"):
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("0")
        return 0
    try:
        with open(file_path, "r") as f:
            return int(f.read())
    except Exception:
        return 0

def save_last_id(last_id, file_path="last_id.txt"):
    with open(file_path, "w") as f:
        f.write(str(last_id))

last_id = load_last_id()

def get_next_id():
    global last_id
    last_id += 1
    save_last_id(last_id)
    return last_id

##### 컬렉션에 question/answer 인덱스 생성
def ensure_keyword_indexes(collection_name):
    # question 인덱스 생성
    try:
        RAG.qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="question",
            field_schema="keyword"
        )
    except Exception as e:
        # 이미 인덱스가 있으면 무시
        if "already exists" not in str(e):
            print(f"question 인덱스 생성 중 오류: {e}")
    # answer 인덱스 생성
    try:
        RAG.qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="answer",
            field_schema="keyword"
        )
    except Exception as e:
        if "already exists" not in str(e):
            print(f"answer 인덱스 생성 중 오류: {e}")

##### Qdrant에서 같은 question+answer가 있는지 확인
def is_duplicate(question, answer, collection_name):
    hits = RAG.qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {"key": "question", "match": {"value": question}},
                {"key": "answer", "match": {"value": answer}}
            ]
        },
        limit=1
    )
    return len(hits[0]) > 0

##### Qdrant에 질문/답변 쌍 저장
def save_qa_to_qdrant(question, answer):
    # 1. subcategory 예측
    predicted_subcategory = RAG.predict_subcategory(question, RAG.session, RAG.tokenizer, RAG.le)
    # 2. category 매핑
    predicted_category = RAG.subcategory_to_category.get(predicted_subcategory, "default")
    # 3. 컬렉션 이름 지정 (dku_ 접두어 추가)
    collection_name = f"dku_{predicted_category}"
    # 4. 인덱스 보장
    ensure_keyword_indexes(collection_name)
    # 5. 중복 체크
    formatted_question = f"[{predicted_subcategory}] {question}"
    if is_duplicate(formatted_question, answer, collection_name):
        print("이미 동일한 질문/답변 쌍이 존재합니다. 저장하지 않습니다.")
        return None
    # 6. 질문 벡터 임베딩
    vector = RAG.embed_model.encode(question).tolist()
    # 7. id 생성
    unique_id = get_next_id()
    # 8. Qdrant 저장
    RAG.qdrant_client.upsert(
        collection_name=collection_name,
        points=[{
            "id": unique_id,
            "vector": vector,
            "payload": {
                "id": unique_id,
                "campus": "죽전",
                "category": predicted_category,
                "subcategory": predicted_subcategory,
                "question": f"[{predicted_subcategory}] {question}",
                "answer": answer,
                "source": "DKU Jisikin",
                "predicted_category": predicted_category,
                "predicted_subcategory": predicted_subcategory
            }
        }]
    )
    return unique_id