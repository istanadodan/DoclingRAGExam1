import time
from pathlib import Path
from .load_file import docConverter


def extract_and_export_tables(
    input_path: str,
    output_directory: Path,
) -> int:
    """
    PDF문서에서 표를 추출하고 csv 및 html 형식으로 내보내는 함수

    Returns:
    int : 추출된 표의 수
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    conv_res = docConverter().convert(input_path)
    doc_filename = conv_res.input.file.stem

    # 표 내보내기
    for table_ix, table in enumerate(conv_res.document.tables):
        table_df = table.export_to_dataframe()
        print(f"## table {table_ix}")
        print(table_df.to_markdown())

        # 표를 csv로 저장
        csv_filename = output_directory / f"{doc_filename}-table-{table_ix}.csv"
        print(f"## CSV save to {csv_filename}")
        table_df.to_csv(csv_filename, index=False)

        # 표를 html로 저장
        html_filename = output_directory / f"{doc_filename}-table-{table_ix}.html"
        print(f"## HTML save to {html_filename}")
        table_df.to_html(html_filename)
        # with html_filename.open("w") as f:
        #     f.write(table_df.to_html())

    elapsed_time = time.time() - start_time
    print(f"문서변환 완료: {elapsed_time}")

    # 추출된 표 수 확인
    table_count = len(conv_res.document.tables)
    print(f"추출된 표 수: {table_count}개")

    return table_count
