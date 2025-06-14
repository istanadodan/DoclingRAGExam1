def setup() -> None:
    from transformers import logging as logging_logger
    import logging

    # Milvus 로거만 대상으로 설정 (패키지 이름 확인 필요)
    milvus_logger = logging.getLogger(
        "milvus"
    )  # 또는 "pymilvus", "async_milvus_client"
    milvus_logger.setLevel(logging.INFO)

    logging_logger.set_verbosity_error()


if __name__ == "__main__":
    from model.query import query
    from dotenv import load_dotenv

    load_dotenv()
    setup()

    while True:
        input_path = input("질의내용: ")
        if "quit" in input_path.lower():
            print("대화를 종료합니다")
            break

        answer = query(input_str=input_path)
        print(f"답변은 {answer}")
