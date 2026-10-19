# Completion checklist — current final pass

## Implemented and verified locally

- [x] FastAPI / HTML / CSS / JavaScript local CPU app, four views, uploads, audio playback, charts, comparison, library discovery, ratings and grounded explanations.
- [x] Original audio preserved; uploads/cache deletion supported. Runtime audio, weights and caches excluded from version control.
- [x] FMA archive/import/extraction finished: 7,994 acoustic analyses, 7,993 MERT embeddings; six unavailable/invalid clips and one silent representation exclusion recorded. No pending full FMA batch.
- [x] Frozen MERT, separate AST instrument hypotheses, Demucs sampled stems, CQT harmony/key and pYIN vocal-pitch measurements.
- [x] Actual frozen EnCodec inference, pinned checkpoint and checksums, CPU limits, versioned resumable excerpt caches, gain-normalized token affinity and separate original-amplitude evidence.
- [x] Chronological chord-template hypotheses, local key estimates and source-timestamp playback; uncertain regions explicit.
- [x] Sampled voicing, pitch-spread and estimated stem-energy comparisons, with evidence gates and explicit unsupported semantic labels.
- [x] Trained FMA genre classifier and acoustic-rule distillation model; artist/recording split checks, validation-only model selection, held-out metrics and baseline comparisons.
- [x] Seven real sample songs, all 21 pairs, exact codec identity/symmetry, gain/reversal observations, cache recovery/reuse, source integrity and original timestamps checked.
- [x] Fixed-model held-out breakdown across eight FMA genres, error examples, language-coverage limits and explanation provenance documented.
- [x] Public human-annotation benchmark downloaded, pinned, licensed and audited; repeated-judgment dispersion measured independently of SoundMap.
- [x] 40 automated tests pass; live API and browser inspection; dependencies pass pip check. Reproduction commands and final demo in RESULTS.md.
- [x] Current results replace the stale running/pilot status; older local history is omitted from this package.

## External research requirements — not claimed complete

- Human-similarity model training, listener agreement with SoundMap and human-labelled retrieval quality: only two local judgments, and zero fully available rated pairs in the audited public benchmark. Requires licensed exact rated excerpts and sufficient independent labels. Machine-generated targets are not human judgments.
- Verified semantic instrumentation, production style, vocal technique and full-song melody/chord transcription: no matching labelled audio supplied for those tasks. Existing descriptors remain measurements/hypotheses and cannot establish general accuracy. EnCodec tokens do not supply those labels automatically.
- Cross-language and wider musical-system accuracy: no verified language labels or representative independent evaluation collection. Eight FMA genres and seven related personal recordings do not validate this claim.

These requirements are data/validation limits, not unattended processing jobs. No paid service, public deployment or new full FMA batch is needed to run the delivered app.
