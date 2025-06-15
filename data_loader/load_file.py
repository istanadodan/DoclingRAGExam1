from typing import List
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document
from pathlib import Path

""" 랭체인 파일 로더 """


def lanchainFileLoader(file_path: Path) -> List[Document]:
    try:
        return DoclingLoader(
            file_path=file_path,
            export_type=ExportType.DOC_CHUNKS,
            chunker=hybridChunker(),
        ).load()

    except Exception as e:
        return []


def docConverter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(
        do_table_structure=True
    )  # 테이블 구조 추출 활성화
    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE
    )  # 더 정확한 TableFormer 모델 사용

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return doc_converter


import os


def hybridChunker() -> HybridChunker:
    return HybridChunker(
        tokenzier=os.environ.get("EMBED_MODEL_ID"),  # 필요에 따라 토크나이저 설정
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        verbose=True,
    )


def extractDocs(url: str) -> List[DoclingDocument]:
    return docConverter().convert(url).document


def getChunkedDocs(docs: List[DoclingDocument]) -> List[str]:
    chunker = hybridChunker()
    return list(chunker.chunk(docs))


def recursiveSplitDocuments(docs: List[Document]) -> List[DoclingDocument]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    return splitter.split_documents(docs)


def loadDocsFromUrl(url: str = "https://arxiv.org/pdf/2408.09869"):
    # print(result.document.export_to_markdown())
    docs: List[DoclingDocument] = extractDocs(url)

    # 청크 목록 취득
    chunks: list = getChunkedDocs(docs)

    # 청크의 개수 확인
    print(f"Total chunks: {len(chunks)}")

    # 청크의 첫번째 항목 확인
    print(chunks[0])
    return chunks
