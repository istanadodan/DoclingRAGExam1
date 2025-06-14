from langchain_core.prompts import PromptTemplate
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import VectorStore
from langchain_milvus import Milvus
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import vertexai
from vertexai.generative_models import GenerativeModel
from langchain_core.embeddings import Embeddings
from vertexai.generative_models._generative_models import GenerationResponse

vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))


async def query(input_str: str) -> str:
    """LLM 조회 메인함수

    Args:
        input_str (str): 질의내용

    Returns:
        str: 답변
    """

    RAG_PROMPT_TEMPLATE = """
  Context information is below
  ---------------------------
  {context}
  ---------------------------
  Given the context information and not prior knowledge, answer the query.
  Query: {input}
  Answer in Korean:
  """

    prompt = PromptTemplate.from_template(
        RAG_PROMPT_TEMPLATE, partial_variables=dict(context="", input=input_str)
    )
    # langfuse = Langfuse()
    # assert langfuse.auth_check()

    vs = getVectorstore(embedding())
    if not vs:
        return f"Milvus 초기화 실패"
    retriever = getRetriever(vs)
    llm = getLLM()
    rag_chain = (
        {
            "source": retriever,
            "query": RunnablePassthrough(),
        }
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["source"]),
            input=lambda x: x["query"],
        )
        # | RunnableLambda(lambda x: print("prompt: ", prompt.invoke(x)) or x)
        | prompt
        | RunnableLambda(
            lambda x: llm.generate_content(x.text if hasattr(x, "text") else x)
        )
        | RunnableLambda(lambda x: parse_protobuf_response(x))
    )
    try:
        return rag_chain.invoke(input_str)
    except TypeError as e:
        print(e)


def parse_protobuf_response(raw_response: GenerationResponse) -> str:
    return raw_response.text


def getLLM():
    model_kwargs = {
        # temperature (float): The sampling temperature controls the degree of
        # randomness in token selection.
        "temperature": 0.28,
        # max_output_tokens (int): The token limit determines the maximum amount of
        # text output from one prompt.
        "max_output_tokens": 1000,
        # top_p (float): Tokens are selected from most probable to least until
        # the sum of their probabilities equals the top-p value.
        "top_p": 0.95,
        # top_k (int): The next token is selected from among the top-k most
        # probable tokens. This is not supported by all model versions. See
        # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#valid_parameter_values
        # for details.
        "top_k": None,
        # safety_settings (Dict[HarmCategory, HarmBlockThreshold]): The safety
        # settings to use for generating content.
        # (you must create your safety settings using the previous step first).
        # "safety_settings": safety_settings,
    }
    return GenerativeModel(
        model_name=os.getenv("VERTEXAI_MODEL"),
        generation_config=model_kwargs,
        system_instruction="You are a helpful assistant.",
    )


def format_docs(docs: list[object]) -> str:
    result = ""
    for doc in docs:
        result += f"\\n\\n{doc.page_content}"
    return result


def embedding() -> Embeddings:
    # 임베딩 모델 초기화
    model_name = os.getenv("EMBED_MODEL_ID")
    embedding = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=dict(trust_remote_code=True)
    )
    return embedding


def milvusVectorstore(_embedding: Embeddings):
    """Milvus 벡터스토어 비동기-safe 생성 함수"""
    return Milvus(
        embedding_function=_embedding,
        collection_name="docling_transformer",
        connection_args={
            "url": os.getenv("MILVUS_URI", "http://localhost:19530"),
            "db_name": "edu",
        },
    )


def createVectorstore(url: str, embedding: object) -> VectorStore:
    """벡터스토 생성

    Args:
        url (str): 벡터DB url
        embedding (object): 임배딩 인스턴스

    Returns:
        _type_: Vectorstore
    """
    from tools.load_file import lanchainFileLoader

    return Milvus.from_documents(
        documents=lanchainFileLoader(file_path=url),
        embedding=embedding,
        collection_name="docling_transformer",
        connection_args={
            "url": os.getenv("MILVUS_URI"),
            "db_name": "edu",
        },
        index_params={"index_type": "FLAT", "metrics_type": "COSINE"},
        drop_old=True,  # 기존 컬렉션 삭제
    )


def getVectorstore(embedding: object) -> VectorStore:
    return milvusVectorstore(embedding)


def getRetriever(vs: VectorStore):
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
