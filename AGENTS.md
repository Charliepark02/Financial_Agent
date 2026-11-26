# Working on financial-agent
Use the financial-agent Conda environment and Python 3.12. Run python -B -m unittest discover -s tests -v before reporting a change. Original live evaluation: python -B -m evaluation.benchmark. Offline extension: python -B -m evaluation.phase4. Add --live for the two-request streaming probe. Run from the repository root.

Preserve developer changes and the original benchmark dataset. Mock providers in offline tests, use temporary SQLite files, distinguish actual measurements from targets and keep financial formulas deterministic. The available financial tools are read-only.
