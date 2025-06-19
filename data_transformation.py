import os
import re
import title_generator
from dotenv import load_dotenv
from qdrant_client import QdrantClient

##### .env 파일에서 환경변수 불러오기
load_dotenv()

QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")

##### 임베딩 모델 로딩
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0
)

collections = client.get_collections().collections
collection_name = [col.name for col in collections]

##### 데이터 답변 개수 삭제 코드
all_points = []
scroll_offset = None

while True:
    response = client.scroll(
        collection_name=collection_name,
        limit=1000,  # 한 번에 가져올 데이터 수 (적절히 조절)
        offset=scroll_offset
    )
    points, scroll_offset = response
    if not points:
        break
    all_points.extend(points)
    if scroll_offset is None:
        break

for point in all_points:
    if "question" in point.payload:
        point.payload["question"] = re.sub(r"\[\d+\]", "", point.payload["question"])

batch_size = 100
for i in range(0, len(all_points), batch_size):
    batch = all_points[i:i+batch_size]
    ids = [point.id for point in batch]
    questions = [point.payload["question"] for point in batch]
    payloads = [{"question": q} for q in questions]
    for pid, pld in zip(ids, payloads):
        client.set_payload(
            collection_name=collection_name,
            payload=pld,
            points=[pid]
        )

points, _ = client.scroll(
    collection_name=collection_name,
    limit=5  # 5개만 샘플로 확인
)
for point in points:
    print(point.id, point.payload.get("question"))

##### 소분류 대괄호를 중괄호로 변경
for collection_name in collection_name:
    print(f"Processing collection: {collection_name}")
    all_points = []
    scroll_offset = None

    # 모든 포인트 불러오기 (scroll)
    while True:
        points, scroll_offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=scroll_offset
        )
        if not points:
            break
        all_points.extend(points)
        if scroll_offset is None:
            break

    # 각 포인트의 question 필드 전처리 및 Qdrant에 반영
    for point in all_points:
        question = point.payload.get("question")
        if question:
            match = re.match(r'^\[([^\]]*)\](.*)', question)
            if match:
                inside_brackets = match.group(1)
                rest = match.group(2)
                new_question = f"{{{inside_brackets}}}{rest}"
                # Qdrant에 반영
                client.set_payload(
                    collection_name=collection_name,
                    payload={"question": new_question},
                    points=[point.id]
                )

##### 질문 형태에 안 맞는 질문 변경
def is_not_valid_question_format(question):
    return not (isinstance(question, str) and ' - ' in question)

for collection_name in collection_name:
    print(f"Processing collection: {collection_name}")
    scroll_offset = None

    while True:
        points, scroll_offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=scroll_offset,
            with_payload=True
        )
        if not points:
            break

        for point in points:
            if int(point.id) >= 450:
                continue
            payload = point.payload
            question = payload.get("question", "")
            predicted_subcategory = payload.get("predicted_subcategory", "")
            if is_not_valid_question_format(question):
                question_content = re.sub(r'^\{[^}]*\}', '', question).strip()
                # ChatGPT로 제목 생성
                new_title = title_generator.generate_title_openrouter(question_content)
                subcategory_bracketed = f"{{{predicted_subcategory}}}" if predicted_subcategory else ""
                new_question = f"{subcategory_bracketed}{new_title} - {question_content}"
                # Qdrant에 반영
                client.set_payload(
                    collection_name=collection_name,
                    payload={"question": new_question},
                    points=[point.id]
                )
                print(f"Updated point {point.id} in {collection_name}")