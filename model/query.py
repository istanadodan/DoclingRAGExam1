import os
from pathlib import Path
from typing import Dict
import vertexai
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from vertexai.generative_models._generative_models import GenerationResponse
from db import MilvusVectorStorePool
from db.vectordb import get_vectorstore, create_vectorstore
from db.retrievers import get_retriever, get_memory_retriever
from model.llm_models import get_llm_model, get_embedding_model
from enum_constant import VectorType
from langchain_core.embeddings import Embeddings

vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))


async def create_url_db(collection_name: str, url: str, embedding_function):
    from data_loader.web_data_loader import langchainWebLoader

    docs = await langchainWebLoader(url)
    create_vectorstore(collection_name, docs, embedding_function)
    return f"{url} : upload 완료"


async def create_file_db(collection_name: str, file_path: Path, embedding_function):
    from data_loader.web_data_loader import langchainWebLoader

    from data_loader.load_file import lanchainFileLoader

    docs = await lanchainFileLoader(file_path)
    create_vectorstore(collection_name, docs, embedding_function)
    return f"{file_path.name} : upload 완료"


def query(
    input_str: str,
    collection_name,
    llm,
    embedding_function: Embeddings,
    vs_pool: MilvusVectorStorePool,
) -> str:
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
  chat history is bellow
  ---------------------------
  {chat_history}
  ---------------------------
  Given the context information and not prior knowledge, answer the query.
  Query: {query}
  Answer in Korean:
  """

    prompt = PromptTemplate.from_template(
        RAG_PROMPT_TEMPLATE,
        # partial_variables=dict(query=input_str),
    )

    # llm = get_llm_model(model_name, temperature)

    # lause = Langfuse()
    # assert langfuse.auth_check()

    # collection_name = "docling_transformer"
    vs = vs_pool.get_vectorstore(collection_name)
    # vs = get_vectorstore(VectorType.Milvus, collection_name, get_embedding_model())
    collection_name = "chat_history"
    vs2 = vs_pool.get_vectorstore(collection_name)
    # vs2 = get_vectorstore(VectorType.Milvus, collection_name, get_embedding_model())
    if not vs:
        return f"Milvus 초기화 실패"

    retriever = get_retriever(vs)
    memory = get_memory_retriever(vs2)

    def get_history(inputs):
        query_text = inputs if isinstance(inputs, str) else inputs.get("query")
        relevant_memories = memory.search_related_documents(query_text, k=5)
        return relevant_memories

    rag_chain = (
        {
            "source": retriever,
            "query": RunnablePassthrough(),
        }
        | RunnablePassthrough.assign(
            chat_history=lambda x: memory.load_memory_variables(
                dict(query=x["query"])
            ).get("chat_history")
        )
        # | RunnableLambda(lambda x: print("chat_history: ", x) or x)
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["source"]),
            query=lambda x: x["query"],
            additional_info={"source": lambda x: x["source"]},
        )
        # | RunnableLambda(lambda x: print("prompt: ", prompt.invoke(x)) or x)
        | RunnablePassthrough.assign(
            prompt_result=lambda x: prompt.invoke(x),
        )
        | RunnablePassthrough.assign(
            llm_result=lambda x: llm.generate_content(
                x["prompt_result"].text
                if hasattr(x["prompt_result"], "text")
                else x["prompt_result"]
            )
        )
        | RunnableLambda(lambda x: save_to_memory(x, memory))
        | RunnableLambda(lambda x: format_source_response(x))
    )
    try:
        return rag_chain.invoke(input_str)
    except TypeError as e:
        print(e)


def save_to_memory(context, memory):
    memory.save_context(
        inputs={"query": context["query"]},
        outputs={"output": context["llm_result"].text},
    )

    return context


def format_source_response(context: Dict) -> str:

    formatted_response = context["llm_result"].text
    source_docs = context["additional_info"].get("source", [])

    # 고유한 출처만 추출
    unique_sources = list(
        set(doc.metadata.get("source", "알 수 없음") for doc in source_docs)
    )

    if unique_sources:
        source_text = "\n\n**출처:**\n"
        for i, source in enumerate(unique_sources, 1):
            source_text += f"{i}. {source}\n"
        formatted_response += source_text

    return formatted_response


def format_docs(docs: list[object]) -> str:
    result = ""
    for doc in docs:
        result += f"\\n\\n{doc.page_content}\nsource:{doc.metadata['source']}"
    return result
