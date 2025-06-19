import os
import re
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