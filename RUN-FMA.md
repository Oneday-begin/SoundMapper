# Run the FMA pilot yourself

## Website and FMA together (recommended for the already imported library)

Run `powershell -ExecutionPolicy Bypass -File .\start-background.ps1`. This starts separate hidden local processes, keeps the website available, and resumes all registered FMA tracks from checkpoints. Two CPU threads and below-normal batch priority leave resources for website work. Website analysis jobs run between FMA tracks using a shared process lock; they may wait for the current track to finish. Cached comparisons and playback stay available. Closing the launching terminal does not close these detached processes. Keep Windows awake; after a restart, rerun this command. Do not start the older exclusive launcher at the same time.

Live counts: `powershell -ExecutionPolicy Bypass -File .\scripts\status-fma.ps1`. Stage and logs: `data/runtime/fma-background.json`, `fma-background.stdout.log`, and `fma-background.stderr.log`. Slow instrument/pitch analysis remains on demand. The following sections document the original exclusive pilot/import workflow.

SoundMap's processing runs on your Windows computer. It does not use ChatGPT, an API key, a subscription allowance, or a cloud machine. You can close ChatGPT. The required pretrained MERT weights are already cached, and the launcher defaults to offline mode. Keep the processing window open, connect power, and keep the computer awake until it finishes.

## Start

In this project folder, double-click **run-fma-pilot.cmd**. Alternatively, open PowerShell here and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-fma-pilot.ps1
```

It checks the cached CPU model, pauses the managed SoundMap server if its queue is idle, verifies the archive/extracted files, imports the seeded 100-track pilot, computes acoustic features and frozen MERT embeddings, and fits/evaluates a small genre classifier. It restarts a managed server that it paused. Refresh <http://127.0.0.1:8765> afterwards and select **Library → FMA**. If the app was not running, use `launch.ps1` afterwards.

The archive is `C:\Users\HP\Downloads\fma_small.zip`. The audio is already extracted at **C:\Users\HP\Documents\SoundMap Data\fma_small\fma_small**. SoundMap's configuration now uses that actual nested layout. The original archive and audio are preserved. Verification passed for the official SHA-1, every extracted file's size/CRC, and 7-Zip's integrity test. See `reports/fma-extraction.json` and `reports/fma-7zip-test.log`.

## Interruptions and results

Closing ChatGPT or losing Wi-Fi does not stop this local process. Closing its processing window, restarting Windows, or pressing Ctrl+C interrupts it. Double-click the same launcher again to resume: completed analyses and individual five-second embeddings are reused. An interrupted window is recomputed. A second launcher is rejected while one is running. If an interactive SoundMap server cannot be stopped by the launcher, stop it with Ctrl+C first and rerun.

Read these files as they appear:

| File | Contents |
|---|---|
| `data/runtime/fma-run.log` | Run start times; new runs append |
| `data/runtime/m-soundmap-cli-*.stdout.log` | Per-stage CLI output and warnings, kept out of the console |
| `reports/fma-import.json` | Reproducible selected IDs and any unreadable files skipped |
| `reports/analysis-fma.json` | Actual per-track time, memory, windows, tempo, and model availability |
| `reports/dimensions-fma.json` | CQT/harmonic, key, dynamics, and tempo dimension measurements |
| `reports/genre.json` | Validation-selected layer/regularization and held-out test genre results |

Check the live import and analysis counts at any time with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\status-fma.ps1
```

If it fails, leave your audio and caches in place and rerun. If the cached model is missing, reconnect Wi-Fi and run `powershell -ExecutionPolicy Bypass -File .\run-fma-pilot.ps1 -Online`. Offline execution was tested successfully with the current cached model. `-CheckOnly` checks the model and metadata without importing or analyzing the pilot.

## What is trained

MERT is already pretrained; this run keeps it frozen and uses CPU inference. The trained model is regularized logistic regression for eight FMA genre labels, with FMA's supplied training/validation/test splits. Genre classification is an auxiliary representation experiment, not evidence that the app learned human musical similarity. The pilot tests feasibility before processing all 8,000 clips.

You do not need to rate pairs or train anything to use SoundMap. The optional listener model stays untrained, as requested. If you later choose to collect enough genuine judgments, the Evaluation view offers artist-separated training pairs and an enabled-when-ready training button. These optional commands also remain available:

```powershell
.\.venv\Scripts\python.exe -m soundmap.cli agreement
.\.venv\Scripts\python.exe -m soundmap.cli ranking
```

These commands do not fabricate ratings and cannot train with the current two judgments. Until sufficient eligible judgments exist it remains untrained. No ratings are invented. Large MERT fine-tuning is not part of this CPU pilot.

To expand to all 8,000 FMA Small clips after the pilot, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-fma-pilot.ps1 -Expand
```

The launcher imports the complete metadata/audio join once, resumes acoustic + frozen MERT analysis, computes the non-deep CQT/dynamics dimensions for every track, and runs the genre probe. It skips an already completed import and keeps verbose per-track CLI records in ignored `data/runtime` logs. It does not fine-tune MERT, perform listener training, or run slow vocal separation/AST tagging across every recording; the latter are computed on demand for comparisons. The measured 100-track pilot took 487.5 seconds of summed analysis time. Linear extrapolation for the remaining clips is about 10.7 hours for the same cached CPU pipeline, not a guarantee: disk, temperature, and Windows power settings affect elapsed time. Keep the laptop plugged in and awake; no Wi-Fi or active ChatGPT session is needed. Rerun the same command after interruption to resume checkpoints.

## Automatic sleep after this FMA run

An independent local watcher (`scripts/auto-sleep.ps1`) can keep Windows awake during the active FMA run and request ordinary sleep after all three stages (analysis, dimensions, genre) succeed and no website jobs remain. It checks the verified batch process, cancels on failures or an unexpected stop, and gives a two-minute countdown. It does not shut down Windows or force apps closed. Closing the lid or manually sleeping/shutting down can still interrupt processing; saved checkpoints are reusable.

To cancel automatic sleep while allowing FMA to continue, run from this project directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\auto-sleep.ps1 -Cancel
```

Watcher status and events are in `data/runtime/auto-sleep.log`; no power settings are permanently changed. Automatic sleep applies to the currently verified run and must be armed again after a reboot.

## Alternating CPU allocation

For the current FMA run, `scripts/cycle-fma-cpu.py` runs as an independent hidden controller. It alternates between 30 minutes at the original two-thread batch allocation with below-normal priority and 15 minutes on a single logical CPU at idle priority. It starts in the original allocation and ends when the verified batch ends. This is a timed load reduction, not temperature-controlled cooling. `data/runtime/cpu-cycle.json` records the current mode and next switch. Automatic sleep remains independent.

To cancel cycling and restore the original allocation, create `data/runtime/cpu-cycle.cancel` (for example, `Set-Content .\data\runtime\cpu-cycle.cancel cancelled`).

