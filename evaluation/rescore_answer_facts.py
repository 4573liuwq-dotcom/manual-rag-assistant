"""在不重复调用 LLM 的情况下，用修订后的人工事实标注重新评分。"""

import argparse
import json
import re
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    labels = {
        row["case_id"]: row
        for row in (json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line)
    }
    rows = []
    for row in (json.loads(line) for line in args.answers.read_text(encoding="utf-8").splitlines() if line):
        label = labels[row["case_id"]]
        answer = row.get("answer", "")
        hits = [any(keyword in answer for keyword in group) for group in label["keyword_groups"]]
        citations = [int(value) for value in re.findall(r"\[资料(\d+)\]", answer)]
        valid = bool(citations) and all(1 <= value <= len(row.get("sources") or []) for value in citations)
        row.update(
            keyword_groups=label["keyword_groups"],
            keyword_group_hits=hits,
            key_fact_recall=round(sum(hits) / len(hits), 4),
            citation_valid=valid,
            answer_pass=all(hits) and valid,
        )
        rows.append(row)
    report = {
        "case_count": len(rows),
        "key_fact_recall": round(statistics.fmean(row["key_fact_recall"] for row in rows), 4),
        "valid_citation_rate": round(statistics.fmean(float(row["citation_valid"]) for row in rows), 4),
        "answer_pass_rate": round(statistics.fmean(float(row["answer_pass"]) for row in rows), 4),
        "avg_generation_latency_ms": round(statistics.fmean(row["latency_ms"] for row in rows), 2),
    }
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
