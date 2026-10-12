"""Grade generated recall answers with the frozen GPT-5-mini rubric. Strict accuracy counts CORRECT only.

    OPENAI_API_KEY=... python -m talkie_facts_replication.judge runs/main --output runs/main/labels.jsonl
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from talkie_facts_replication import data

MODEL = "gpt-5-mini-2025-08-07"
LABELS = ("CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "REVIEW")
RUBRIC = data.ROOT / "configs/judge_rubric.txt"


def request(packet):
    schema = dict(type="object", additionalProperties=False, required=["label", "reason"],
                  properties=dict(label=dict(type="string", enum=list(LABELS)), reason=dict(type="string")))
    return dict(model=MODEL, instructions=RUBRIC.read_text(), input=json.dumps(packet, ensure_ascii=False), store=False,
                reasoning={"effort": "low"}, max_output_tokens=4096,
                text={"format": dict(type="json_schema", name="factual_judgment", strict=True, schema=schema)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="directory of stage_XX.jsonl files written by write.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from openai import OpenAI

    client = OpenAI()
    facts = {f["id"]: f for f in data.facts()}
    items = []
    for path in sorted(args.run.glob("stage_*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            fact = facts[row["fact_id"]]
            items.append((dict(stage=int(path.stem.split("_")[1]), fact_id=row["fact_id"], question_index=row["question_index"]),
                          dict(question=row["question"], reference_answer=fact["answer"]["canonical"],
                               reference_fact=fact["statement"], model_response=row["model_response"])))

    def grade(item):
        source, packet = item
        verdict = json.loads(client.responses.create(**request(packet)).output_text)
        return dict(source, question=packet["question"], model_response=packet["model_response"], **verdict)

    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(grade, items))
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    main()
