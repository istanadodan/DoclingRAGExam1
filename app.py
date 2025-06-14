import asyncio
import streamlit as st
import datetime
import json
import orjson
import warnings
from tools.async_tools import run_async


def setup() -> None:
    from transformers import logging as logging_logger
    import logging

    # Milvus 로거만 대상으로 설정 (패키지 이름 확인 필요)
    milvus_logger = logging.getLogger(
        "milvus"
    )  # 또는 "pymilvus", "async_milvus_client"
    milvus_logger.setLevel(logging.INFO)

    logging_logger.set_verbosity_error()


def get_event_loop_safe():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        # Streamlit처럼 실행 중 루프가 없을 때는 기본 루프 가져오기
        return asyncio.get_event_loop()


def main():
    """_메인함수_

    Returns:
        _type_: _description_
    """
    if not "env_loaded" in st.session_state:
        from dotenv import load_dotenv

        load_dotenv()
        # setup()

    st.session_state.env_loaded = True
    # ScriptRunContext 경고 무시
    warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
    warnings.filterwarnings("ignore", category=UserWarning)

    # 페이지 설정
    st.set_page_config(page_title="AI 채팅 어시스턴트", page_icon="💬", layout="wide")

    # 세션 상태 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # 메인 제목
    st.title("💬 AI 채팅 어시스턴트")

    # 사이드바 - 옵션 설정
    with st.sidebar:
        st.header("⚙️ 설정")

        # 대화 이력 개수 설정
        max_history = st.slider(
            "대화 이력 표시 개수",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="최근 몇 개의 대화까지 표시할지 설정합니다.",
        )

        st.divider()

        # 모델 옵션 선택
        st.subheader("🤖 모델 설정")
        model_option = st.selectbox(
            "AI 모델 선택",
            ["GPT-4", "GPT-3.5-turbo", "Claude-3", "Gemini-Pro"],
            index=0,
        )

        temperature = st.slider(
            "창의성 수준 (Temperature)",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="높을수록 더 창의적이고 다양한 답변을 생성합니다.",
        )

        max_tokens = st.number_input(
            "최대 토큰 수", min_value=100, max_value=4000, value=2000, step=100
        )

        st.divider()

        # 파일 업로드 섹션
        st.subheader("📁 파일 업로드")
        uploaded_file = st.file_uploader(
            "파일을 선택하세요",
            type=["txt", "pdf", "docx", "csv", "json", "py", "md"],
            accept_multiple_files=True,
            help="텍스트, PDF, Word, CSV, JSON, Python, Markdown 파일을 업로드할 수 있습니다.",
        )

        if uploaded_file:
            for file in uploaded_file:
                if file not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(
                        {
                            "name": file.name,
                            "size": file.size,
                            "type": file.type,
                            "upload_time": datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }
                    )

            st.success(f"{len(uploaded_file)}개 파일이 업로드되었습니다!")

            # 업로드된 파일 목록 표시
            if st.session_state.uploaded_files:
                st.write("**업로드된 파일들:**")
                for i, file_info in enumerate(st.session_state.uploaded_files):
                    with st.expander(f"📄 {file_info['name']}"):
                        st.write(f"크기: {file_info['size']:,} bytes")
                        st.write(f"타입: {file_info['type']}")
                        st.write(f"업로드 시간: {file_info['upload_time']}")
                        if st.button(f"삭제", key=f"delete_{i}"):
                            st.session_state.uploaded_files.pop(i)
                            st.rerun()

        st.divider()

        # 대화 내역 관리
        st.subheader("📋 대화 관리")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ 전체 삭제", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        with col2:
            if st.button("💾 대화 저장", use_container_width=True):
                if st.session_state.chat_history:
                    chat_data = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "model": model_option,
                        "settings": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        "conversation": st.session_state.chat_history,
                    }
                    st.download_button(
                        label="📥 JSON 다운로드",
                        # data=json.dumps(chat_data, ensure_ascii=False, indent=2),
                        data=orjson.dumps(chat_data),
                        file_name=f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

    # 메인 채팅 영역
    col1, col2 = st.columns([3, 1])

    with col1:
        # 채팅 기록 표시 영역
        st.subheader("💭 대화창")

        chat_container = st.container()

        with chat_container:
            # 최근 대화 이력만 표시
            display_history = (
                st.session_state.chat_history[-max_history:]
                if len(st.session_state.chat_history) > max_history
                else st.session_state.chat_history
            )

            if not display_history:
                st.info("👋 안녕하세요! 무엇을 도와드릴까요?")
            else:
                for i, chat in enumerate(display_history):
                    timestamp = chat.get("timestamp", "")

                    # 사용자 메시지
                    with st.chat_message("user"):
                        st.write(f"**[{timestamp}]**")
                        st.write(chat["user"])

                    # AI 응답
                    with st.chat_message("assistant"):
                        st.write(f"**[{timestamp}] {model_option}**")
                        st.write(chat["assistant"])

        # 질의 입력창
        st.subheader("✍️ 질문 입력")

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "메시지를 입력하세요:",
                height=100,
                placeholder="여기에 질문을 입력하세요...",
                help="Ctrl+Enter로 빠르게 전송할 수 있습니다.",
            )

            col_submit1, col_submit2, col_submit3 = st.columns([1, 1, 2])

            with col_submit1:
                submitted = st.form_submit_button("🚀 전송", use_container_width=True)

            with col_submit2:
                clear_input = st.form_submit_button(
                    "🧹 지우기", use_container_width=True
                )

            with col_submit3:
                include_files = st.checkbox(
                    "업로드된 파일 포함", help="업로드된 파일 정보를 함께 전송합니다."
                )

    with col2:
        # 통계 및 정보 표시
        st.subheader("📊 대화 통계")

        total_chats = len(st.session_state.chat_history)

        # 메트릭 표시
        st.metric("총 대화 수", total_chats)
        st.metric("업로드된 파일", len(st.session_state.uploaded_files))
        st.metric("현재 모델", model_option)

        # 최근 대화 시간
        if st.session_state.chat_history:
            last_chat_time = st.session_state.chat_history[-1].get(
                "timestamp", "Unknown"
            )
            st.info(f"**마지막 대화:** {last_chat_time}")

        # 설정 요약
        with st.expander("🔧 현재 설정"):
            st.write(f"**모델:** {model_option}")
            st.write(f"**창의성:** {temperature}")
            st.write(f"**최대 토큰:** {max_tokens}")
            st.write(f"**표시 이력:** {max_history}개")

    # 폼 제출 처리
    if submitted and user_input.strip():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 파일 정보 포함 여부 확인
        file_context = ""
        if include_files and st.session_state.uploaded_files:
            file_context = "\n\n[업로드된 파일 정보]\n"
            for file_info in st.session_state.uploaded_files:
                file_context += f"- {file_info['name']} ({file_info['type']}, {file_info['size']:,} bytes)\n"

        # 실제 AI 모델 대신 간단한 응답 생성 (데모용)
        def generate_response(user_query: str, model: str, temp: float) -> str:
            responses = [
                f"안녕하세요! {model} 모델로 답변드립니다. 질문: '{user_query[:50]}...' 에 대해 생각해보고 있습니다.",
                f"흥미로운 질문이네요! {model}이 창의성 수준 {temp}로 답변을 생성하고 있습니다.",
                f"질문을 잘 이해했습니다. {model} 모델의 최대 {max_tokens} 토큰으로 답변해드리겠습니다.",
                f"좋은 질문입니다! {model}으로 분석한 결과를 말씀드리겠습니다.",
                f"네, 도움을 드릴 수 있습니다. {model} 모델로 최선의 답변을 준비했습니다.",
            ]
            import random

            return (
                random.choice(responses)
                + f"\n\n실제 구현에서는 여기에 {model} API 호출 결과가 표시됩니다."
                + file_context
            )

        # AI 응답 생성
        from model.query import query

        try:
            ai_response = run_async(query(user_input))
        except RuntimeError:
            ai_response = "ERROR"

        # ai_response = generate_response(
        #     user_input + file_context, model_option, temperature
        # )

        # 대화 히스토리에 추가
        chat_entry = {
            "timestamp": timestamp,
            "user": user_input,
            "assistant": ai_response,
            "model": model_option,
            "settings": {"temperature": temperature, "max_tokens": max_tokens},
        }

        st.session_state.chat_history.append(chat_entry)

        # 페이지 새로고침
        st.rerun()
    elif clear_input:
        st.rerun()

    # 하단 정보
    st.divider()
    st.caption(
        "💡 **사용 팁:** 사이드바에서 AI 모델과 설정을 변경할 수 있습니다. 파일을 업로드하여 문서 기반 질문도 가능합니다."
    )

    # 키보드 단축키 안내
    with st.expander("⌨️ 키보드 단축키"):
        st.write(
            """
        - **Ctrl + Enter**: 메시지 전송
        - **Tab**: 다음 입력창으로 이동
        - **Shift + Tab**: 이전 입력창으로 이동
        """
        )


if __name__ == "__main__":
    main()
