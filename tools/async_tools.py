import nest_asyncio
import asyncio


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Streamlit 또는 Jupyter 환경
        nest_asyncio.apply()
        return asyncio.ensure_future(coro)
    else:
        return loop.run_until_complete(coro)
