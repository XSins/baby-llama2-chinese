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


def process_and_write_to_parquet(input_files: list[Path | str] | Generator[Path | str, None, None], output_file: Path):
    all_data = []

    for file_path in input_files:
        print(f"Processing {file_path}...")
        data_gen = json_lines_reader(file_path)
        filtered_gen = filter_and_yield(data_gen)
        all_data.extend(filtered_gen)

    df = pd.DataFrame(all_data)
    df.to_parquet(output_file, engine="pyarrow", index=False)
    print(f"Total {len(df)} items written to {output_file} (in Parquet format).")


def main():
    data_dir = Path("./sft_data").resolve()
    raw_data_files = (data_dir / "raw").rglob("*.jsonl")
    output_parquet = data_dir / "sft_data.parquet"
    process_and_write_to_parquet(raw_data_files, output_parquet)


if __name__ == "__main__":
    main()
