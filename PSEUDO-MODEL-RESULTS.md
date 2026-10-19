# Automated acoustic-label model: actual results

Trained a standardized multi-output ridge pair head over frozen MERT layers 4, 8 and 12, with 32-component PCA fitted only on training tracks. No encoder fine-tuning and no human ratings generated.

Usable representations: {'training': 6393, 'validation': 800, 'test': 800}. Pairs: {'training': 6000, 'validation': 1200, 'test': 1200}. Artist and decoded-recording split audit: {'excluded': 0, 'artist_split_conflicts': 0, 'duplicate_split_conflicts': 0}.

Four machine targets: onset/BPM rhythm, chronological CQT harmony, MFCC/spectral timbre, and crest/RMS-spread dynamics. No instrument identities, isolated melody or vocal-delivery labels. Symmetric pair features use embeddings only; acoustic targets are never supplied as model inputs.

On 1,200 held-out pairs, combined-score MAE was 5.08 points / 100, compared with 5.97 for the training-mean baseline and 18.76 for uncalibrated frozen MERT cosine. Individual target errors are in reports/pseudo-model.json. Spearman correlation: 0.512. Candidate ranking nDCG@10: 0.9635 vs frozen MERT 0.9658; ranking did not improve. These are machine-rule metrics, never listener agreement or catalogue-wide quality.

Latest reproducible run: 54.91 seconds, peak measured process RSS 2492.0 MiB; reused 8400 cached pair labels. Initial run took 77.85 seconds with 2495.6 MiB peak RSS. Model data/pseudo-model.json; machine pair manifest data/pseudo-pairs.jsonl; resumable label cache data/pseudo-labels.sqlite; progress data/runtime/pseudo-training.json. Originals preserved. All 21 pairs of seven sample songs were exercised through the live comparison API and saved in reports/pseudo-sample-pairs.json.

Reproduce in PowerShell from this folder: `./train-pseudo.ps1`. It selects regularization on validation data, evaluates test data once for that run, and reuses labels keyed by dataset and pipeline configuration. Tests: 33 passed (reports/tests-pseudo-model.xml). A first integration test exposed a non-JSON NaN for a constant baseline correlation; fixed to null and rerun successfully.

Website: Compare shows a separate Machine-trained acoustic proxy card; Evaluation shows training counts and machine-target error. Existing calculated measurements remain primary. Demo: select Gehra Hua and Parvati, click Explore similarity, scroll to the experimental card.

Limitations: pseudo-labels reproduce the acoustic rules and can inherit their errors. The exact rule teacher necessarily has zero error against itself. Hindi full-song/domain coverage and perceptual similarity remain unvalidated. No manual listening is required to run this experiment; human accuracy cannot be established without independent human evidence.
