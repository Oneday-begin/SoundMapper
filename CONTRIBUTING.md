# Contributing

1. Create a branch describing your change.
2. Use Python 3.12 and run `setup.ps1` for the tested environment.
3. Keep changes focused; explain behavior and validation in your pull request.
4. Run `.\.venv\Scripts\python.exe -m pytest -q`.
5. Use synthetic or appropriately licensed fixtures. Never add real music, model weights, private configuration, or databases.
6. Inspect staged files before committing. Generated reports must retain their scope and provenance.

Do not relabel genre accuracy or machine-target error as human-similarity accuracy. Keep unsupported predictions and research limitations explicit. CPU-heavy validation with real models is separate from the isolated automated suite.
