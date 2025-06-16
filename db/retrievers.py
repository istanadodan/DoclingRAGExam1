import os
from langchain_community.vectorstores import VectorStore

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever

from langchain.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)


def get_retriever(vs: VectorStore):
    base_retriever = vs.as_retriever(search_kwargs={"k": 5})
    memory_retriever = get_memory_retriever(vs=vs)

    re_ranker = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(
            model_name=os.getenv("RERANKER_MODEL_ID"),
            model_kwargs=dict(trust_remote_code=True),
        ),
        top_n=3,
    )

    cross_encoder_reranker_retriever = ContextualCompressionRetriever(
        base_compressor=re_ranker, base_retriever=base_retriever
    )

    return cross_encoder_reranker_retriever


from langchain.memory import ConversationSummaryBufferMemory, VectorStoreRetrieverMemory


def get_memory_retriever(vs: VectorStore, llm: object = None):
    vector_memory = VectorStoreRetrieverMemory(
        retriever=vs.as_retriever(search_kwargs={"k": 3}),  # 관련 메모리 3개 검색
        memory_key="chat_history",
        input_key="query",
        return_docs=True,
        k=5,
    )

    # conversation_memory = ConversationSummaryBufferMemory(
    #     llm=llm,
    #     memory_key="conversation_summary",
    #     input_key="query",
    #     output_key="answer",
    #     max_token_limit=1000,
    #     return_docs=True,
    # )

    return vector_memory
