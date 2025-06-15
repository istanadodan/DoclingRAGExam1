import os
import vertexai
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from vertexai.generative_models._generative_models import GenerationResponse
from db.vectordb import get_vectorstore, create_vectorstore
from db.retrievers import get_retriever
from model.llm_models import get_llm_model, get_embedding_model
from enum_constant import VectorType

vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))


async def create_db(file_info: str):
    await create_vectorstore(file_info, get_embedding_model())
    print(file_info)
    return f"{file_info} : upload 완료"


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

    vs = get_vectorstore(VectorType.Milvus, get_embedding_model())
    if not vs:
        return f"Milvus 초기화 실패"
    retriever = get_retriever(vs)
    llm = get_llm_model()
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


def format_docs(docs: list[object]) -> str:
    result = ""
    for doc in docs:
        result += f"\\n\\n{doc.page_content}"
    return result
