import os
from langchain_community.vectorstores import VectorStore

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever

from langchain.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)


def get_retriever(vs: VectorStore):
    base_retriever = vs.as_retriever(search_kwargs={"k": 5})

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
