"""批量导入评测语料，并记录每份文档的真实处理规模。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / "knowledge" / ".env")

from knowledge.processor.import_process.main_graph import graph  # noqa: E402
from knowledge.processor.import_process.state import create_default_state  # noqa: E402


def import_document(pdf_path: Path, output_root: Path) -> Dict[str, Any]:
    output_dir = output_root / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    state = create_default_state(
        task_id=f"corpus_{pdf_path.stem}",
        import_file_path=str(pdf_path.resolve()),
        file_dir=str(output_dir.resolve()),
    )

    started_at = time.perf_counter()
    try:
        result = graph.invoke(state)
        chunks = result.get("chunks") or []
        entities = result.get("knowledge_entities") or result.get("entities") or []
        relations = result.get("knowledge_relations") or result.get("relations") or []
        return {
            "file": pdf_path.name,
            "status": "success",
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "item_name": result.get("item_name") or "",
            "chunk_count": len(chunks),
            "image_count": int(result.get("image_count") or 0),
            "entity_count": len(entities),
            "relation_count": len(relations),
            "kg_failed_chunk_count": len(result.get("kg_failed_chunks") or []),
            "md_path": str(result.get("md_path") or ""),
            "chunk_ids": [str(chunk.get("chunk_id")) for chunk in chunks if chunk.get("chunk_id")],
            "error": "",
        }
    except Exception as exc:
        return {
            "file": pdf_path.name,
            "status": "error",
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    pdfs: List[Path] = sorted(args.corpus.glob("*.pdf"))
    if args.start:
        pdfs = pdfs[args.start :]
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit(f"没有找到 PDF：{args.corpus}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if args.start and args.manifest.exists():
        rows = [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    for index, pdf_path in enumerate(pdfs, start=1):
        print(f"[{index}/{len(pdfs)}] 开始导入 {pdf_path.name}", flush=True)
        row = import_document(pdf_path, args.output_root)
        rows.append(row)
        args.manifest.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{index}/{len(pdfs)}] {row['status']} "
            f"chunks={row.get('chunk_count', 0)} images={row.get('image_count', 0)} "
            f"elapsed={row['elapsed_seconds']}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
