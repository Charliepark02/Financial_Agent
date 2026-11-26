# Phase 4 verification
Baseline: 70 offline unittest cases passed on Python 3.12.16.
4-5: 72 passed after bounded I/O retry, transport timeouts and one planner regeneration.
4-6: 74 passed. Deterministic source/date/asset-bound metric reports; production quote validation at the provider boundary.
4-7: 76 passed. Buffered Ollama streaming, internal TTFT, visible latency and context estimate error. Pre-change LLM time 251.77/254.08 seconds.
4-8: 77 passed. Opt-in SQLite analysis records, separate historical retrieval budget, dated results, transactional deletion.

## Final gate

- 4-9: 94/94 offline unittest cases passed (zero skipped or failed), including nine multi-component agent flow cases. No live services or personal DB were used for unit tests. The extended runner reuses tests; its counts are not additional unique tests.
- Original live evaluation: 10/10 cases; tool selection 100.0%; expected argument accuracy 100.0%. Tool execution 8/8. Dataset phase3-original-10, qwen3:8b, 4096 context, thinking off, temperature zero, M3/16GB. Original suite is non-streaming. Model residency at original-suite start was not controlled.
- Final live mean/median/P95 21.90/25.41/33.26 s. Total 218.98 s, average LLM calls 1.7, input/output tokens 13781/3169. Phase 3: 63.91/73.79/96.07 s and 16959/2830 tokens. Pre-task Phase 4-4: 25.41/26.99/42.89 s and 16880/3278 tokens.
- Previous integration run before analysis currency propagation: 24.21/26.92/38.92 s and 13796/3365 tokens. With five missing currency references, 48 of 53 rows were verifiable. Currency metadata is now taken from the provider's history metadata. Live data, model cache and OS load vary. Each suite version had one repetition; these are observations, not a controlled speedup.
- Final financial report rows: 53 verified, 0 contradicted, 0 unavailable out of 53. Coverage 100.0%. These comparisons audit structured close/metric rows against the same saved provider output; they do not independently verify Yahoo's data or certify conceptual prose.
- LLM time 216.86 s; tool wall time 2.10 s. Original-suite cache hits/misses 0/8. Context estimate error (estimated minus Ollama prompt tokens) ranged from 363 to 423. Zero original-suite request overflows; a dedicated offline case forces overflow.
- Extended evaluation: 53/53 overlapping offline scenarios. Five fixture claim decisions, four with reference values; two supported, two contradicted and one unavailable. Two of two archive queries matched expected results (one positive, one negative). One intentionally transient request recovered after one retry.
- Controlled streaming pair: model resident at start [False, True]. First lookup cold, second cached. Request TTFT [10.422, 2.142] s, first visible [13.303, 4.796] s, TPOT [0.0638, 0.0623] s/token. Prior warm-model/cold-cache pair is in phase4_20261126_011644.json; its ttft field was per model call before the request-offset correction.
- CLI smoke: tools, metrics, logs, clear, disabled memory and exit exercised. SQLite and dotenv ignore rules checked. No trading integration. Review HANDOFF.md before Phase 5.

### Reproducible artifacts

- Live cases and profile: benchmark_20261126_012206.json and its CSV.
- Extended offline and streaming: phase4_20261126_012234.json and its CSV.
- Structured numerical checks: claims_20261126_012206.json. Recreate with python -m evaluation.phase4_metrics BENCHMARK_JSON OUTPUT_JSON.
- Chart: phase4_comparison_20261126_012206.png. Recreate with python -m evaluation.phase4_figures PHASE3_JSON FINAL_JSON EXTENDED_JSON OUTPUT_PNG.
- Cold-model probe: python -m evaluation.phase4 --live --cold-model. This unloads the selected Ollama model before the first request.

## 변경 파일과 역할

| 구분 | 파일 | 역할 |
|---|---|---|
| 신규 | src/recovery.py | 오류 분류, 재시도 상한, 백오프·지터·Retry-After |
| 신규 | src/grounding.py | 요청 검증, 수치·단위 대조, 금융 보고서 렌더링 |
| 신규 | src/memory/persistent.py | SQLite 트랜잭션과 명시적 조회·삭제 |
| 수정 | src/LLM/client.py | 타임아웃, 완성 스트림 수집, 모델 타이밍 |
| 수정 | src/agents/financial_agent.py | 검증·복구·스트리밍·영구 메모리 통합 |
| 수정 | src/agents/planner.py, src/agents/planning_agent.py | 계획 재생성과 계획 실행 결과 렌더링 |
| 수정 | src/tools/registry.py | 캐시 소유 호출의 재시도와 계측 |
| 수정 | src/tools/market_data.py, src/tools/preprocessing.py, src/tools/financial_analysis.py | 날짜·종가 검사와 분석 통화 전파 |
| 수정 | src/memory/context_manager.py, main.py | 과거 기록의 별도 예산, CLI 메모리·계획·출력 명령 |
| 수정 | evaluation/benchmark.py | 원본 평가 문항 유지, 실제 관측값과 호출 프로필 저장 |
| 신규 | evaluation/phase4.py, evaluation/phase4_metrics.py, evaluation/phase4_figures.py | 확장 평가, 구조화 수치 대조, 실측 시각화 |
| 신규 | tests/test_recovery.py, tests/test_grounding.py, tests/test_stream.py | 장애·수치·스트림 테스트 |
| 신규 | tests/test_archive.py, tests/test_phase4_flow.py, tests/test_cache_races.py | 저장소·종단 흐름·동시성 회귀 테스트 |
| 신규 | .gitignore, AGENTS.md, ROADMAP.md, ARCHITECTURE.md, HANDOFF.md, PROGRESS.md | 로컬 데이터 제외, 운영·설계·검증·인계 |
| 수정 | README.md | 설치·실행·CLI 사용법 |

## 학습 메모

- 지수 백오프는 일시적 장애 때 재요청 간격을 늘린다. 지터를 더하면 여러 요청이 동시에 재시도하는 현상을 줄인다. 같은 장애라도 입력 오류는 반복해서 해결되지 않으므로 오류 분류가 먼저다.
- 부분 실패 복구는 종목별 성공·실패 상태를 유지한다. 다른 종목이 실패해도 검증된 관측값을 전달할 수 있다.
- 수치 근거 검증은 값과 함께 종목·날짜·기간·조정 기준·단위를 확인한다. 수익률 0.12를 12%로 표시하는 변환은 Sharpe Ratio에 적용할 수 없다.
- 스트리밍 수신과 사용자 표시는 서로 다른 시점이다. 완성 전 도구 호출을 보류하고 검증 뒤 표시하면 TTFT와 첫 표시 시간이 벌어진다. TPOT는 생성 구간에 출력 토큰 하나가 차지한 평균 시간이다.
- 컨텍스트는 한정된 입력 공간이다. 현재 요청의 근거를 우선 보존하고 오래된 분석은 날짜와 함께 별도 예산 안에서만 조회한다.
- SQLite 트랜잭션은 한 요청의 일부 기록만 저장되는 상황을 방지한다. 명시적 종목·기간 검색은 재현 가능하며 의미 유사도 검색은 별도 과제다.
- 모의 테스트는 장애를 재현하고 실측은 실제 모델·공급자 연결을 확인한다. 겹치는 테스트 수를 더하거나 원본이 없는 주장을 정답으로 집계하면 평가가 왜곡된다.
