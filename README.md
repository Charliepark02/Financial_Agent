Financial Agentic AI

A local LLM-powered agentic AI system for financial data analysis, quantitative research, and automated trading research.
## Running locally

Use Miniforge to create the environment with `conda env create -f environment.yml` on a new machine. Activate it with `conda activate financial-agent`. Start local Ollama, make qwen3:8b available with `ollama pull qwen3:8b`, then run `python main.py`. Default settings are in src/config.py (4,096-token window, thinking disabled, temperature zero).

The CLI supports ordinary questions, `plan REQUEST`, `tools`, `metrics`, `logs`, `clear` and `exit`. Enable the analysis archive with `memory on`. List records with `memory`, search with `memory search AAPL`, remove one with `memory delete ID`, or erase the archive with `memory clear CONFIRM`. Use `memory off` to close access for the session. The next session starts with storage disabled. Clearing conversation history affects only short-term memory.

See [architecture](ARCHITECTURE.md), [verification](PROGRESS.md), [roadmap](ROADMAP.md), and [handoff](HANDOFF.md). Run tests and benchmarks from the repository root as described in [AGENTS.md](AGENTS.md).
