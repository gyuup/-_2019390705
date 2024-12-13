from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from typing import List
from langchain.schema import Document  # Import Document if needed
import pandas as pd
import os

from langchain_community.document_loaders.text import TextLoader

# 명시적으로 인코딩 지정
loader = TextLoader("Intro.txt", encoding="utf-8")
data = loader.load()
print(data)


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=150,
    chunk_overlap=50,
    encoding_name='cl100k_base'
)

texts = text_splitter.split_text(data[0].page_content)

texts

# Set the token
from langchain_huggingface import HuggingFaceEmbeddings
import os

# HuggingFace API 토큰 환경 변수 설정 (필요 시 토큰을 설정)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# HuggingFaceEmbeddings 객체 생성
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 모델 사용 예시
print(embeddings_model)



print(texts[0])
embeddings_model.embed_query(texts[0])[:3]

# Vectorized size
len(embeddings_model.embed_query(texts[0]))

db = Chroma.from_texts(
    texts,
    embeddings_model,
    collection_name = 'history',
    persist_directory = './db/chromadb',
    collection_metadata = {'hnsw:space': 'cosine'}, # l2 is the default
)

print(db)

query = '디지털경영전공은 무엇인가요?'
docs = db.similarity_search(query)
print(docs[0].page_content)

print(docs)

query = '데이터분석과 빅데이터는 무슨 상관인가요'
docs = db.similarity_search(query)
print(docs[0].page_content)