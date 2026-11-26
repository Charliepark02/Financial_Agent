"""Score numerical rows from the renderer's known layout against archived tool output."""
import argparse
import json
from pathlib import Path

from src.grounding import PERCENT, RATIOS, check_claim


def evaluate(report):
    cases = []
    for row in report["results"]:
        outcomes = []
        reports = row["answer"].split("\n\n")
        observations = row.get("observations", [])
        for text, item in zip(reports, observations):
            if not item.get("success"):
                continue
            args = item["arguments"]
            for line in text.splitlines():
                field, sep, value = line.partition(": ")
                if not sep:
                    continue
                claim = {"symbol": text.split(" · ")[0], "date": text.split(" · ")[1],
                         "period":args.get("period"), "metric":field}
                if field in ("Adjusted daily close","Unadjusted daily close"):
                    words = value.split()
                    claim.update(metric="daily_close", basis=field.split()[0].lower(), unit=words[1])
                    value = words[0]
                elif field in PERCENT | RATIOS:
                    claim["unit"] = "%" if value.endswith("%") else "ratio"
                else:
                    continue
                if value == "unavailable":
                    outcomes.append("unavailable")
                    continue
                try:
                    claim["value"] = float(value.rstrip("%"))
                except ValueError:
                    outcomes.append("unsupported")
                    continue
                outcomes.append(check_claim(claim, [item]))
        cases.append({"id": row["id"], "verified":outcomes.count("verified"),
                      "unsupported":outcomes.count("unsupported"), "unavailable":outcomes.count("unavailable")})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.benchmark.read_text())
    cases = evaluate(report)
    counts = {key:sum(row[key] for row in cases) for key in ("verified","unsupported","unavailable")}
    total = sum(counts.values())
    checked = counts["verified"] + counts["unsupported"]
    tool_calls = [call for row in report["results"] for call in row["actual_calls"]]
    summary = dict(counts, structured_rows=total,
                   verified_fraction_of_all_rows=counts["verified"]/total if total else None,
                   validation_coverage=checked/total if total else None,
                   unsupported_fraction_of_checked=counts["unsupported"]/checked if checked else None,
                   tool_execution_success=sum(call["success"] for call in tool_calls),
                   tool_attempts=len(tool_calls))
    args.output.write_text(json.dumps({"source":args.benchmark.name, "cases":cases, "summary":summary},indent=2))
    print(summary)


if __name__ == "__main__":
    main()
