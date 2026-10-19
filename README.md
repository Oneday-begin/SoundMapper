# SoundMap

**A local music-analysis and similarity explorer with evidence behind each comparison.**

Compare recordings using rhythm, harmonic patterns, timbre, dynamics, and frozen audio representations. Inspect charts, listen to candidate matching passages, and optionally arrange cited explanations with a local language model.

Python 3.12 | FastAPI | SQLite | JavaScript | librosa | PyTorch CPU | scikit-learn | Plotly

## Features

- Independent library/upload selection, file transfer progress, and analysis status.
- Whole-recording and local-passage comparisons with timestamped playback.
- Nine musical comparison dimensions, adjustable weights, evidence, and coverage.
- Frozen MERT embeddings; optional AST, Demucs/pYIN, and EnCodec measurements.
- Library discovery, visualization, and recurrence candidates.
- Background CPU processing with configuration-keyed, resumable caches.
- FMA genre classification and an experimental acoustic-rule regression model.
- TF-IDF reference retrieval with optional constrained local Ollama evidence selection.

## Quick start (Windows)

Install Python **3.12** with Python available from your terminal. Open PowerShell in this repository:

```powershell
.\setup.ps1
Copy-Item config.example.json config.json
.\setup-models.ps1
.\launch.ps1
```

Open **http://127.0.0.1:8765/** and upload your own licensed MP3/WAV recordings. The initial setup needs internet access and downloads dependencies and model assets. Cached inference runs locally. Audio and weights are not bundled here. Python 3.12 is the tested version.

If PowerShell blocks a local script, run that script for this invocation with:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

For a lighter acoustic-only run, set `mert_enabled` and `tagging_enabled` to `false`, and `ollama_model` to `null` in your ignored `config.json`; skip `setup-models.ps1`. Acoustic analysis remains usable, while model-specific features report unavailable.

### Optional features

```powershell
.\setup-separation.ps1   # sampled source/vocal-pitch support
.\setup-codec.ps1        # pinned EnCodec setup
```

Ollama is optional. Install it separately and provide the configured local model with its expected digest. Enable the checkbox under **Compare > Comparison priorities**. Missing or mismatched models fall back to deterministic evidence. The configured example is `qwen3:1.7b`; no LLM is required to compare audio.

Example data paths are repository-relative. Update ignored `config.json` if importing external samples or FMA. Use audio and datasets you have permission to use. FMA is optional for normal app use, and its full batch should not be started for a routine demo.

## Architecture

```text
Browser -> FastAPI -> SQLite / background worker
                       |
                 Audio preprocessing
                       |
          Acoustic features + frozen models
                       |
           Versioned caches / comparisons
                       |
       Charts, passages, evidence, optional Ollama
```

| Folder | Role |
| --- | --- |
| `soundmap/` | Backend, audio models, storage, jobs, experiments |
| `static/` | HTML/CSS/JavaScript; setup generates local Plotly |
| `tests/` | Isolated automated tests |
| `scripts/` | Validation and operational utilities |
| `reports/` | Selected machine-readable evaluation evidence |
| `data/` | Ignored local models, database, uploads, and caches; created at runtime |

The source repository starts without your processed library. Upload recordings or import your own data after setup. Saved evaluation reports describe the original experiment; they do not contain the full data required to reproduce it instantly.

## Recorded results

Principal validation was recorded on **10 October 2026**. These are saved experimental results, not a claim that CI reproduces the original dataset/model run.

| Experiment | Recorded outcome |
| --- | --- |
| Original FMA subset | 8,000 clips |
| Completed acoustic / frozen MERT representations | 7,994 / 7,993 |
| Genre classifier | 57.0% test accuracy on 800 clips |
| Acoustic-rule Ridge model | 5.08/100 combined-score MAE on 1,200 test pairs |
| Automated validation | 40 tests passed in the saved final run |
| Real-audio validation | Seven songs and 21 distinct pairs |

Genre accuracy is not song-similarity accuracy. The Ridge model learns machine-generated acoustic targets, not human listening judgments. Its candidate nDCG did not improve over frozen MERT. Display indices are uncalibrated and do not establish copying. A human-listener similarity model remains untrained for insufficient eligible labels.

See [current results](RESULTS.md), [completion and research limits](CHECKLIST.md), and [pair-model results](PSEUDO-MODEL-RESULTS.md).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use synthetic fixtures and an isolated temporary database. The GitHub Actions workflow runs the automated suite on Windows/Python 3.12 without real music or downloaded model checkpoints. A workflow badge should be added only after the repository exists and its first run succeeds.

## Technical references

- [Current measured results](RESULTS.md)
- [Implementation status and research limits](CHECKLIST.md)
- [Acoustic-rule model experiment](PSEUDO-MODEL-RESULTS.md)
- [FMA workflow](RUN-FMA.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Privacy and licensing

Audio processing and inference are local after provisioning. This repository excludes recordings, downloaded weights, databases, private settings, and large runtime caches. The app binds to localhost; it is not a production multi-user hosting configuration.

Original application code is under the [MIT license](LICENSE). Audio, datasets, model weights, Plotly, and the AudioSet-derived ontology retain separate terms. In particular, the project records MERT weights as CC BY-NC 4.0; no commercial weight rights are asserted. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md), keep tests isolated, and avoid committing audio, weights, or private data. Experimental scores must retain their target and validation limitations.
