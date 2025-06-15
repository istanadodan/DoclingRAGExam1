import asyncio
from typing import List
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

page_url_default = "https://techcrunch.com/2025/03/14/ai-coding-assistant-cursor-reportedly-tells-a-vibe-coder-to-write-his-own-damn-code/"


async def langchainWebLoader(url: str) -> List[Document]:
    from data_loader.web_data_loader import web_loader
    from data_loader.load_file import recursiveSplitDocuments

    docs: List[Document] = await web_loader(url)
    return recursiveSplitDocuments(docs)


async def web_loader(page_url: str = page_url_default) -> List[Document]:
    loader = WebBaseLoader(
        web_path=page_url,
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                class_="entry-content wp-block-post-content is-layout-constrained wp-block-post-content-is-layout-constrained",
            ),
        },
        # 텍스트를 추출할 때 구분자를 설정하고 공백을 제거
        bs_get_text_kwargs={"separator": "\n", "strip": True},
    )
    docs = []

    async for doc in loader.alazy_load():
        docs.append(doc)

    return docs


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


if __name__ == "__main__":
    docs = run_async(web_loader())
    print(docs)
