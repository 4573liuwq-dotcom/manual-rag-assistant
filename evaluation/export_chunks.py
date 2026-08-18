"""将实验集合中的文本切片导出为便于人工标注的 JSONL。"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import MilvusClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="appliance_chunks_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / "knowledge" / ".env")
    client = MilvusClient(uri=os.environ["MILVUS_URL"])
    rows = client.query(
        collection_name=args.collection,
        filter="",
        output_fields=[
            "chunk_id", "title", "parent_title", "file_title", "item_name",
            "part", "content",
        ],
        limit=10_000,
    )
    rows.sort(key=lambda row: (row.get("file_title", ""), int(row["chunk_id"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已导出 {len(rows)} 个切片到 {args.output}")


if __name__ == "__main__":
    main()
