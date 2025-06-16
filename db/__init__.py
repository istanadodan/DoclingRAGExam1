from typing import Dict, List

from tools.async_tools import run_async
from .vectordb import get_milvus_vectorstore, VectorStore, Embeddings


class MilvusVectorStorePool:
    embedding_function: Embeddings = None

    def __init__(
        self, url="http://localhost:19530", embedding_function=None, collection_names=[]
    ):
        self.store_map: Dict[VectorStore] = {}
        self.url: str = url
        self.embedding_function: Embeddings = embedding_function
        self.collection_names: List[str] = collection_names
        # 풀 초기화
        self._make_pool()

    def get_vectorstore(self, collection_name) -> VectorStore:
        return self.store_map.get(collection_name)

    def put_vectorstore(self, collection_name: str):
        if collection_name not in self.store_map:
            self.store_map[collection_name] = run_async(
                get_milvus_vectorstore(collection_name, self.embedding_function)
            )

    def _make_pool(self):
        for name in self.collection_names:
            self.put_vectorstore(name)
