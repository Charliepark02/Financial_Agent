# Handoff

Start with PROGRESS.md and the raw JSON reports. Run evaluation.phase4 --live when Ollama and Yahoo Finance are available. Database access is opt-in per CLI session. Current quotes always use the financial tool path (with the short in-process TTL). Historical commands read only the SQLite archive.

## Limits and follow-up
- Original ten prompts evaluate action selection, expected arguments and tool success. They do not establish general investment accuracy. Structured reports bypass model-generated numerical prose. Model-only explanations, unsupported-trade wording in arbitrary languages, and implicit references to older turns are not exhaustively verifiable.
- Deterministic reports emphasize close, returns, volatility, drawdown, Sharpe, VaR, ES and RSI. Technical indicators also present in raw reports may not be displayed. Reports use English metric names even for Korean questions. No downstream trading interface exists.
- Buffered streaming gives reliable validation and tool assembly; user-visible text can start much later than model text. No model cold-start is forced during a shared local session. The transport measures model-call first text; the agent adds preceding request time for request-level TTFT. First visible latency is measured separately.
- Archive search is lexical and explicit. It does not provide semantic recall, automatic preference inference, or cross-session conversational continuity. Per-record rendering can exclude a long record from a small display budget. Retrieval via Python supports session and date filters.
- Public FinancialAgent.chat and PlanningAgent.chat return strings. Deterministic formatting changes the text of data-backed answers. Planning is explicit in the CLI. History and latest-plan metrics belong to their respective agent instances.
- One retry policy wraps each external operation. Provider-internal retries and cross-request rate limiting are outside agent control. A transport timeout does not kill a running Python thread. Production-scale deployment would need process-level isolation and quotas.
- Keep old observations dated, compare models and caches separately, and rerun the suite over several market days before using latency or correctness as a release threshold. Extend structured units/provenance and specify Phase 5 input contracts before implementing the quantitative engine.
