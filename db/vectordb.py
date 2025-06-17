from typing import List
from langchain_milvus import Milvus
from langchain_core.embeddings import Embeddings
from enum_constant import *
from langchain_community.vectorstores import VectorStore
import os


async def get_milvus_vectorstore(
    collection_name: str, _embedding: Embeddings
) -> VectorStore:
    """Milvus 벡터스토어 비동기-safe 생성 함수"""
    return Milvus(
        embedding_function=_embedding,
        collection_name=collection_name,
        connection_args={
            "url": os.getenv("MILVUS_URI", "http://localhost:19530"),
            "db_name": "edu",
        },
        index_params={"index_type": "FLAT", "metric_type": "COSINE"},
    )


def create_vectorstore(
    collection_name: str, docs: List[str], embedding_function: Embeddings
) -> VectorStore:
    """벡터스토 생성

    Args:
        docs (str): 텍스트 목록
        embedding_function (object): 임배딩 인스턴스

    Returns:
        _type_: Vectorstore
    """
    if len(docs) == 0:
        # return get_milvus_vectorstore(collection_name, embedding_function)
        return
    return Milvus.from_documents(
        documents=docs,
        embedding=embedding_function,
        collection_name=collection_name,
        connection_args={
            "url": os.getenv("MILVUS_URI", "http://localhost:19530"),
            "db_name": "edu",
        },
        index_params={"index_type": "FLAT", "metric_type": "COSINE"},
        drop_old=False,  # 기존 컬렉션 유지
    )


async def get_vectorstore(
    type: VectorType, collection_name: str, embedding: object
) -> VectorStore:
    if type == VectorType.Milvus:
        return await get_milvus_vectorstore(collection_name, embedding)
    return None
