import json
from pathlib import Path
from typing import Generator

import pandas as pd


def json_lines_reader(file_path: Path | str) -> Generator[dict[str, str], None, None]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def filter_and_yield(data_generator: Generator[dict[str, str], None, None], min_q_len=10, min_a_len=5, max_len=256):
    for per in data_generator:
        q = per["instruction"] + (per.get("input", "") or "")
        a = per["output"]
        if min_q_len <= len(q) <= max_len and min_a_len <= len(a) <= max_len:
            yield {"prompt": q, "answer": a}


def batch_process_to_csv(
    input_files: list[Path | str] | Generator[Path | str, None, None], output_file: Path | str, batch_size=10000
):
    chunks = []
    total_written = 0

    for file_path in input_files:
        print(f"Processing {file_path}...")
        data_gen = json_lines_reader(file_path)
        filtered_gen = filter_and_yield(data_gen)

        for i, item in enumerate(filtered_gen):
            chunks.append(item)
            if len(chunks) >= batch_size:
                df_chunk = pd.DataFrame(chunks)
                df_chunk.to_csv(output_file, mode="a", header=not total_written, index=False)
                chunks.clear()
                total_written += len(df_chunk)
                print(f"{total_written} items processed.")

    if chunks:
        df_chunk = pd.DataFrame(chunks)
        df_chunk.to_csv(output_file, mode="a", header=False, index=False)
        total_written += len(df_chunk)
        print(f"All done. Total {total_written} items written to {output_file}.")


def main():
    data_dir = Path("./sft_data").resolve()
    data_dir.mkdir(exist_ok=True)
    raw_data_files = (data_dir / "raw").rglob("*.jsonl")
    output_csv = data_dir / "sft_data.csv"
    batch_process_to_csv(raw_data_files, output_csv)


if __name__ == "__main__":
    main()
