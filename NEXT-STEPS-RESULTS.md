> Historical post-FMA report. Current implementation and blockers are consolidated in [RESULTS.md](RESULTS.md); EnCodec was added in the final pass.

# Post-FMA results — 10 October 2026

The local app is running at http://127.0.0.1:8765/. No new full FMA extraction batch is running.

## Dataset accounting

8,000 archive clips; 4 excluded during import; 7,996 registered. Retrying the two missing analyses confirmed decode failures for track IDs 098565 (Brandon Clark) and 098569 (Love and Fear). Original files remain untouched. Exclusions are recorded in reports/fma-recovery.json.

7,994 acoustic analyses exist for the active pipeline; 7,993 have frozen MERT representations. Naked Skin is silent and deliberately has no musical embedding. A damaged window checkpoint for one other recording was quarantined and successfully recomputed (22.29 seconds including CPU model loading). The encoder now recovers malformed, nonfinite or incorrectly shaped checkpoints and closes file handles before quarantine on Windows.

## What was trained

A regularized multiclass logistic genre classifier over frozen MERT embeddings, with artist separation and decoded-content duplicate audit. Supplied FMA splits are preserved. Layer 8 and C=0.01 were selected by the earlier validation experiment; this run did not retune on test data. Splits: {'training': 6393, 'validation': 800, 'test': 800}. Artist overlap: none. Cross-split decoded-content duplicates: none. Encoder weights were not fine-tuned.

Held-out genre accuracy: 57.0%; balanced accuracy: 57.0%; macro F1: 0.5677. Confusion matrix and all model settings are in reports/genre-model.json. These are genre metrics, not recommendation quality. Unknown pretraining overlap remains unaudited.

Optimized run: 60.62 seconds including data preparation; 30.20 seconds classifier fitting and test prediction; sampled whole-process peak RSS 1263.5 MiB. Only pooled vectors are fetched, rather than all window and waveform JSON. An earlier implementation used approximately 4,590 MiB; the optimized run is the delivered implementation. CPU BLAS concurrency is bounded at two threads.

## Comparing your songs

Compare and Analysis selectors show personal recordings, while Library defaults to My songs. FMA remains accessible through the FMA training dataset filter. Personal discovery and map requests filter before decoding analysis JSON. The selection endpoint returns lightweight summaries rather than entire analysis objects. Initial page load fetches only personal recordings; FMA summaries load on demand when its filter is selected.

A new **experimental learned genre-affinity** card compares classifier softmax profiles using 100 times the Bhattacharyya coefficient (sum of square roots of paired activations). It is symmetric, bounded, and returns 100 for identical profiles. Different recordings can have identical profiles. It does not measure instrumentation, melody, rhythm, or human similarity. Scores are not calibrated similarity probabilities. FMA's broad genres do not establish coverage for Hindi film music.

Actual Gehra Hua / Parvati: descriptive feature index 67.96, genre affinity 84.77, older acoustic/MERT aggregate 93.8. Cached comparison API took 1.56 seconds in the measured request. The nine-feature rows remain primary; unsupported/low-confidence channels are omitted with coverage disclosed. Full real result: reports/parvati-gehra-trained-proxy.json. All 21 sample pairs: reports/genre-affinity-samples.json. Those files are measurements, not perceptual-quality labels.

## Reproduce on Windows

From this project folder:

```powershell
.\train-genre.ps1
.\.venv\Scripts\python.exe -m pytest -q --junitxml=reports/tests-next-steps.xml
.\launch.ps1
```

Training requires the completed pooled MERT caches and reports/genre.json containing validation selection. Artifact: data/genre-model.json (ignored runtime data; plain JSON, no executable pickle). Changing the analysis key makes the artifact unavailable until retraining. No model download or complete extraction is necessary. Do not use start-background.ps1 for normal website-only launch: it also starts the FMA batch.

## Validation and demo

29 automated tests passed, including corrupt-window recovery, checkpoint reuse, symmetric and identity genre affinity, missing embeddings, stale model configuration, existing audio, API and listener leakage tests. Browser inspection confirmed the seven-song personal library, functioning charts, nine-feature comparison and genre card. Screenshot: reports/genre-affinity-ui.png.

Open Compare, choose Gehra Hua and Parvati, then Explore similarity. Read the individual evidence rows, the descriptive aggregate, and the separately labelled genre card. Change feature priorities and compare again. Use matching-passage controls to inspect the returned source timestamps. Evaluation shows the trained genre experiment while the listener model remains untrained.

## Completion checklist

- [x] Retry and document missing clips; preserve originals.
- [x] Repair damaged MERT checkpoint; checkpoint reuse remains supported.
- [x] Review genre experiment and split/duplicate audit.
- [x] Save tested genre classifier and expose an honest experimental affinity proxy.
- [x] Compare supplied songs and inspect updated local website.
- [x] Separate personal selections from training dataset by default.
- [x] Automated tests and reproducible Windows training command.
- [ ] Human-similarity ranking: untrained because sufficient listener labels are unavailable. Genre agreement is not substituted as listener ground truth.
- [x] Controlled five-second half-gain and reversal checks: genre affinity 100.0 and 80.82 respectively. Half-gain embedding max difference 0.00001854. Run took 13.16 seconds including model load; peak RSS 1,143.6 MiB. Reversal demonstrates that genre affinity is not a chronological-pattern metric. These are sanity checks, not recommendation-quality labels. Reproduce with `.\.venv\Scripts\python.exe -m soundmap.controlled_checks`. Results: reports/controlled-transforms.json.
- [ ] Validated melody/instrument/production/vocal recognition: existing channels remain explicitly labelled proxies or model hypotheses. EnCodec timbral-token claims are not implemented.

No automatic sleep was triggered. Monitoring remains ended; the former FMA batch is not restarted.
