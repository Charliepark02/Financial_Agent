"""Compare two original-suite reports and one extended evaluation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    for key in ("baseline", "current", "extended", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    old, new, extra = [json.loads(path.read_text()) for path in (args.baseline,args.current,args.extended)]
    rows = new["results"]
    cases = [row["id"] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    ax = axes.flat
    ax[0].plot(cases,[r["latency"] for r in old["results"]], marker="o", label="Phase 3")
    ax[0].plot(cases,[r["latency"] for r in rows], marker="o", label="Phase 4")
    ax[0].set_title("End-to-end latency (seconds)")
    ax[0].legend()
    ax[1].bar(cases,[r["llm_latency"] for r in rows], label="LLM")
    ax[1].bar(cases,[r["tool_execution_time"] for r in rows], bottom=[r["llm_latency"] for r in rows],label="Tools")
    ax[1].set_title("Phase 4 execution time (seconds)")
    ax[1].legend()
    ax[2].bar(cases,[r["input_tokens"] for r in rows],label="Input")
    ax[2].bar(cases,[r["output_tokens"] for r in rows],bottom=[r["input_tokens"] for r in rows],label="Output")
    ax[2].set_title("Tokens per case")
    ax[2].legend()
    cache=rows[-1]["cache"]
    ax[3].bar(["Hit","Miss","Shared"],[cache["hits"],cache["misses"],cache["shared_calls"]])
    ax[3].set_title("Cache requests (original suite)")
    fixture=extra["fixtures"]
    ax[4].bar(["Submitted","With reference","Supported"],[fixture["claims_submitted"],fixture["claims_with_reference"],fixture["correct_supported_claims"]])
    ax[4].set_title("Numerical validation (offline fixtures)")
    ax[5].bar(["Queries","Correct results"],[fixture["retrieval_queries"],fixture["correct_retrievals"]])
    ax[5].set_title("Archive retrieval (offline fixtures)")
    fig.savefig(args.output, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
