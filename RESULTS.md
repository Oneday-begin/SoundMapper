# SoundMap — current results and completion status

Current final pass, 10 October 2026. This document supersedes the old running/pilot status. Earlier working logs are omitted from this source-only package.

## Delivered local application

The local CPU application implements Compare, Library, Analysis and Evaluation; real audio analysis, playback, charts, similarity search, model predictions, evidence references, optional Ollama, ratings and deletion. Launch with `./launch.ps1`; open http://127.0.0.1:8765/. No paid API or public deployment is required.

FMA is complete: 8,000 archive clips, four import exclusions, two further decode failures, **7,994 acoustic analyses**, **7,993 frozen MERT representations**. One silent recording has no musical embedding. Original files are preserved. No full FMA job remains to run.

## Models and what their scores mean

| Component | Status | Meaning |
|---|---|---|
| MERT | Frozen pretrained encoder | Audio representation; no local encoder fine-tuning |
| Genre classifier | Trained; 57.0% accuracy on 800 held-out clips | Auxiliary genre prediction; not human similarity |
| Acoustic-rule pair head | Trained on 6,000 pairs; 1,200 validation and 1,200 test pairs | Predicts calculated rhythm, harmony, timbre and dynamics targets |
| EnCodec | Installed, pinned, strict weights loaded; all seven songs analyzed | Gain-normalized token-distribution texture affinity, sampled excerpts |
| AST | Separate cached sampled predictions | Instrument/vocal hypotheses, often too weak to support a conclusion |
| Demucs + pYIN | Sampled source estimates and pitch measurements | Vocal pitch contour, voicing and stem-energy evidence |
| CQT templates | Chronological hypotheses and local key estimates | Major/minor triads and tonal profiles; uncertain intervals explicit |
| Human-listener ranker | Untrained; insufficient independent matching data | Never represented as trained by machine labels |

Pair-head combined-score MAE is **5.08 / 100 points against acoustic rules**; training-mean baseline 5.97, rescaled frozen-MERT cosine baseline 18.76. Candidate ranking nDCG@10 is 0.9635 versus frozen MERT 0.9658: ranking did not improve. These numbers are not human-listener accuracy. Detailed targets and provenance: [PSEUDO-MODEL-RESULTS.md](PSEUDO-MODEL-RESULTS.md).

## Final real-audio checks

All seven supplied songs and 21 pairs were checked. EnCodec uses five evenly spaced eight-second excerpts per song (40 seconds); original timelines and coverage are displayed. Checkpoint reuse, identity (100), pair-order symmetry, finite outputs and unchanged source hashes passed. Actual final run: **79.39 seconds**, sampled peak process RSS **1684.2 MiB**. Full report: [reports/final-validation.json](reports/final-validation.json).

Raw-amplitude codec histograms were sensitive to volume: halving gain gave only 54.5–60.8 affinity. The delivered descriptor normalizes input RMS before tokenization and preserves original-level measurements separately; the same seven real excerpts now return **100.0–100.0** under half gain. Reversal returns **58.4–63.6**. These are engineering observations, not judged positive/negative similarity labels. The raw-amplitude experiment is retained in reports/codec-raw-amplitude-validation.json.

Amplitude is measured from original audio. The EnCodec 24 kHz checkpoint has normalize=false and does not emit an amplitude-scale track; the app must not describe it as a learned dynamics detector. Reconstruction RMS error is a diagnostic, not dynamic-range similarity. Codec token IDs are categorical; Hellinger affinity compares per-codebook frequencies. Histogram pooling discards chronological order, so CQT and onset evidence remain separate. Original stereo width remains available because the codec descriptor is mono.

Chronological harmony now exposes conservative chord-template hypotheses and local major/minor estimates. Boundaries use original duration; a one-sample resampling-roundoff discrepancy found by real-audio validation was corrected and regression-tested. Estimated-source evidence requires at least two usable voiced excerpts in both recordings before reporting a vocal-movement index. This is not a singing-technique classifier. New exploratory indices are excluded from the nine-feature weighted aggregate.

## Coverage and failure examples

The fixed pair model was inspected across all eight FMA test genres. Per-dimension MAE and five largest errors are in reports/final-validation.json; groups overlap when pair endpoints have different genres. The largest observed pair's dynamics error was **48.61 points**, despite its smaller errors on other dimensions. This is why the combined score alone is insufficient. There was no test-driven retraining in this pass.

There are no verified language labels in this experiment. Seven related personal recordings do not establish Hindi-film generalization. AST instrument scores remain uncalibrated; low-confidence profiles are excluded. Major/minor templates do not cover ragas, all modes, modulations reliably, extended chords or full-song polyphonic melody. Public-model pretraining overlap is unknown.

## Independent human benchmark: concrete external blocker

The published [CCMSim annotations](https://github.com/pxaris/CCMSim) were fetched at revision `eff8ab428dd12f1973548d657928b5211dd290bb` with checksums and license copies. The actual mapping contains **486 recordings** and **1130 rated pairs**. **19 recordings** exist locally, but **0 rated pairs** have both recordings available. The repository redistributes annotations, not source audio; several source collections require registration or an access request. The public MagnaTagATune landing page also failed to load during this check.

External repeated-judgment dispersion was calculated: 78 repeatedly rated pairs, 178 within-pair rating comparisons, mean absolute disagreement 0.2093 on a normalized 0–1 scale. **This does not measure agreement with SoundMap.** No external or synthetic rating was inserted into the local listener table.

Human-similarity training/accuracy and validated semantic instrument/vocal/melody recognition therefore remain external data/evaluation requirements. Completing software cannot establish these scientific claims. Suitable licensed rated audio or task-specific labelled datasets are required; no new manual listening task was assigned to you. [reports/external-validation.json](reports/external-validation.json) records the audit.

## Reproduce and inspect

From this project directory in PowerShell:

```powershell
./launch.ps1
./setup-codec.ps1                 # one-time pinned download; already installed here
./analyze-codec.ps1               # personal songs; resumes cached excerpts
./train-pseudo.ps1                # reuses 8,400 machine-labelled pair checkpoints
./evaluate-external.ps1           # offline public-label/audio-availability audit
./.venv/Scripts/python.exe -m soundmap.final_validation
./.venv/Scripts/python.exe -m pytest -q --junitxml=reports/tests-final.xml
```

**40 tests passed**; pip check reports no broken requirements. Tests cover silence/short/malformed inputs, token categories, corrupted caches, identity/symmetry, missing evidence, tonal unknowns, source bounds, benchmark pair validation, original app regressions and split leakage. Test fixtures do not establish recommendation quality. Live API checks and browser inspection cover the real Gehra Hua/Parvati comparison, new evidence cards and Evaluation.

Demo: choose Gehra Hua and Parvati in Compare → Explore similarity → inspect the nine-feature rows → EnCodec texture → expand chronological harmony and play an interval → inspect estimated vocal/source evidence → open Evaluation for the separate trained-model and benchmark statuses. For a new uploaded song, queue EnCodec excerpts from its comparison card and compare again after the processing queue finishes.

## Attribution and licensing

- EnCodec: [Meta source](https://github.com/facebookresearch/encodec), [research paper](https://arxiv.org/abs/2210.13438), pinned `facebook/encodec_24khz` revision `c1dbe2ae3f1de713481a3b3e7c47f357092ee040`. Code is MIT; the downloaded checkpoint card has no explicit weight-license field. Commercial weight redistribution rights are not asserted. Weights stay in ignored local data.
- CCMSim: Charilaos Papaioannou / repository above. Annotation license CC BY 4.0; code MIT; source-audio licenses separate. Copies under data/benchmarks/ccmsim.
- CQT implementation reference: [librosa chroma_cqt](https://librosa.org/doc/0.10.2/generated/librosa.feature.chroma_cqt.html). References describe algorithms; they do not establish properties of a specific song.
- Existing FMA, MERT, AST, Demucs and Ollama attribution remains in README.md. FMA audio licenses are track-specific; the software repository license is not an audio redistribution license.

Live API verification is recorded in reports/final-api-check.json. Screenshots are omitted from this source-only package.
