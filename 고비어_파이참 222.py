import os
import pandas as pd
from langchain.schema import Document
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import numpy as np

# Hugging Face API 토큰 설정
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Embedding 모델 로드
def load_embedding_model(model_name: str):
    """Loads a sentence-transformers model."""
    return SentenceTransformer(model_name)

# Embedding 모델 초기화
model_name = "sentence-transformers/all-mpnet-base-v2"
embedding_model = load_embedding_model(model_name)
print(f"Embedding model loaded: {model_name}")

# 데이터 처리
data_dir = "./data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Directory created at {data_dir}")

# amazon.csv 파일 로드 및 처리
csv_path = "amazon.csv"
try:
    data = pd.read_csv(csv_path, encoding="utf-8")  # 파일 인코딩을 UTF-8로 명시
except UnicodeDecodeError:
    raise RuntimeError(f"Failed to read {csv_path}. Check file encoding.")

df = data[:100].copy()
df.dropna(subset=['rating_count'], inplace=True)

# 카테고리 분리
df['sub_category'] = df['category'].astype(str).str.split('|').str[-1]
df['main_category'] = df['category'].astype(str).str.split('|').str[0]

# 중복 제거 및 필요한 컬럼 선택
df['product_name'] = df['product_name'].str.lower()
df = df.drop_duplicates(subset=['product_name'])
df2 = df[['product_id', 'product_name', 'about_product', 'main_category',
          'sub_category', 'actual_price', 'discount_percentage', 'rating', 'rating_count']]

# UTF-8로 저장
csv_output_path = os.path.join(data_dir, "amazon_rag.csv")
df2.to_csv(csv_output_path, index=False, encoding="utf-8")
print(f"Processed data saved to {csv_output_path}")

# 문서 리스트 생성
documents = [
    Document(page_content=row["about_product"], metadata={"product_name": row["product_name"]})
    for _, row in df2.iterrows()
]

print(f"Created {len(documents)} documents for processing.")

# Embedding 함수 정의
def embedding_function(texts):
    """Converts a list of texts into embeddings."""
    if isinstance(texts, str):  # 단일 문자열 처리
        texts = [texts]
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings)

# 문서 Embedding 및 벡터화
if documents:
    print("Embedding documents...")
    texts = [doc.page_content for doc in documents]  # 문서의 텍스트 추출

    # Chroma VectorStore 초기화
    persist_directory = "./chroma_db"  # 벡터 데이터 저장 디렉토리
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding_function=embedding_function,  # Embedding 함수 전달
        persist_directory=persist_directory
    )
    print(f"VectorStore created with {len(documents)} documents.")

    # 예시 쿼리로 검색
    query = "iPhone USB charger and adapter"

    # 검색 수행
    search_results = vectorstore.similarity_search(query, k=5)
    print(f"Search results for query: '{query}'")
    for i, result in enumerate(search_results, 1):
        print(f"{i}. {result.page_content[:100]}...")
else:
    print("No documents to process.")
