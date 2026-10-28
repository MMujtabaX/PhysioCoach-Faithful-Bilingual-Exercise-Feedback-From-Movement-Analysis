# Build Guide: PhysioCoach-NLP, step by step

This guide takes you from an empty folder to a trained, evaluated, demo-ready
project. Every step says **where** it runs and **what you should see**.

## Where each part runs

| Where | What you do there | Why |
|---|---|---|
| **Your laptop** | Setup, dataset, demo, Urdu + speech, evaluation, report | Everything except training runs fine on a CPU |
| **Google Colab** (free T4 GPU) | Fine-tune Flan-T5 (step 6) | Training on a laptop CPU would take hours |
| **GitHub** | Host the code (`MMujtabaX/PhysioCoach-NLP`) | Colab clones it from there; it is also your portfolio link |
| **Ollama** (on your laptop) | The few-shot LLM comparison (step 8) | Free, local, no API key |

Disk space you will need on the laptop: about 6 GB (PyTorch, translation and
speech models, and an Ollama model).

## Overview

| Step | What | Where | Time |
|---|---|---|---|
| 1 | Set up the project | Laptop | 15 min |
| 2 | Run the demo (no ML needed) | Laptop | 5 min |
| 3 | Understand and adjust the config | Laptop | 1–2 h |
| 4 | Build the dataset | Laptop | 2 min |
| 5 | Push to GitHub | Laptop | 10 min |
| 6 | Fine-tune Flan-T5 | Colab | 30–45 min |
| 7 | Use the trained model locally | Laptop | 15 min |
| 8 | LLM comparison with Ollama | Laptop | 20 min |
| 9 | Urdu translation and speech | Laptop | 30 min |
| 10 | Hand-verified test set | Laptop + team | 1 day |
| 11 | Evaluation and ablation | Laptop | 2–3 h |
| 12 | Human evaluation | Team + volunteers | 2–3 days |
| 13 | Write the report and prepare the demo | Laptop | 1 week |

---

## Step 1: Set up the project (laptop)

**Python 3.11 or 3.12** is recommended. Make a *new* virtual environment for
this project; don't reuse your FYP environments (MediaPipe and TensorFlow have
their own protobuf pins).

Unzip the project, open a terminal **inside the `physiocoach-nlp` folder**, then:

Windows (PowerShell):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

macOS / Linux:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Check that everything works:
```bash
pytest
```
**You should see:** `62 passed`.

## Step 2: Run the demo (laptop, no ML needed)

```bash
physiocoach demo -n 5
physiocoach demo -n 5 --lang ur
```
**You should see:** random reps with their faults and an English (or Urdu)
coaching cue for each, then a session summary.

Start the web demo:
```bash
physiocoach serve
```
Open **http://127.0.0.1:8000**. Pick an exercise, click *Random record*, then
*Generate feedback*. Switch to اردو for Urdu. The interactive API docs are at
**http://127.0.0.1:8000/docs**. Stop the server with `Ctrl+C`.

At this point only the template generator (Method A) works. The other parts
light up as you complete the steps below.

## Step 3: Understand and adjust the config (laptop)

Open `src/physiocoach/config/exercises.yaml`. It is the heart of the project:

- **features:** how synthetic movement values are sampled, normal versus faulty.
- **faults:** for each fault, the English and Urdu cue templates, praise lines,
  session phrases, tips, and the **detection regexes** the faithfulness checker uses.

Things worth doing here (and writing about in the report):

1. **Read every Urdu line aloud** with your team and fix anything that sounds
   unnatural. Also check the number table in `src/physiocoach/numbers.py`.
2. **Match your FYP.** If PhysioLens uses different fault names or features,
   rename them here (see `docs/INTEGRATION.md`).
3. **Add a cue variant or two** per fault for more variety.

After *any* change, run `pytest`. The tests generate thousands of records and
fail if a template and the checker ever disagree.

## Step 4: Build the dataset (laptop)

```bash
python scripts/build_dataset.py --out data
```
**You should see:** about 4,800 records and about 14,300 input→feedback pairs,
split 80/10/10 *by record*, so no test record leaks into training. Files in `data/`:

| File | What it is |
|---|---|
| `train.jsonl`, `val.jsonl`, `test.jsonl` | Pairs: linearized input, target text, original record |
| `stats.json` | Counts per exercise and fault (use in the report's dataset section) |
| `test_to_verify.csv` | 250 test items for your team to verify by hand (step 10) |
| `translation_eval.csv` | 100 English cues for your team to translate (step 11) |

Optional, after step 8: add LLM paraphrases for more natural variety. Each
paraphrase is checked and dropped if it is not faithful.
```bash
python scripts/build_dataset.py --out data --paraphrase --paraphrase-model qwen2.5:3b
```
This takes roughly 20–40 minutes on a laptop. Report how many paraphrases were
kept versus dropped; it is a nice result in itself.

## Step 5: Push to GitHub (laptop)

Create an empty repository named `PhysioCoach-NLP` on GitHub (no README, since
the project has one), then from the project folder:
```bash
git init
git add .
git commit -m "PhysioCoach-NLP: initial version"
git branch -M main
git remote add origin https://github.com/MMujtabaX/PhysioCoach-NLP.git
git push -u origin main
```
`.gitignore` already keeps `data/`, `models/`, `cache/`, `results/` and `.venv/`
out of the repository.

## Step 6: Fine-tune Flan-T5 (Google Colab)

1. Go to **colab.research.google.com**, then *File → Upload notebook*, and
   choose `notebooks/train_on_colab.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*.
3. Run the cells top to bottom. The notebook clones your repository, installs
   the project, builds the dataset, trains `flan-t5-small`, evaluates it against
   the template baseline, and downloads `flan-t5-coach.zip` and `results.zip`.

**You should see** at the end of training something like:
```
{"n": 400, "pass_rate": 0.9..., "by_task": {...}, "eval_loss": 0.0..., ...}
```
`pass_rate` is the share of the model's outputs that pass the faithfulness
checker *before* any fallback. That is one of your headline numbers.

**Then train `google/flan-t5-base`** (change `--model` and `--out` in the
training cell). Comparing small and base model size against quality is an easy,
good extra result.

Colab sessions reset. Download the zip files before you close the tab, or mount
Google Drive (the notebook has a cell for it).

## Step 7: Use the trained model locally (laptop)

Install the ML extras (this downloads PyTorch and Transformers, about 2–3 GB):
```bash
pip install -e ".[ml,eval]"
```
Unzip `flan-t5-coach.zip` so that this folder exists:
`physiocoach-nlp/models/flan-t5-coach/` (containing `config.json`,
`model.safetensors`, the tokenizer files and `val_faithfulness.json`).

Try it:
```bash
physiocoach demo -n 5 --backend t5
physiocoach serve
```
In the web demo, choose **B · Fine-tuned Flan-T5**. The *Flan-T5* status chip
at the top turns green when the model folder is found.

**Checker-guided sampling (B+).** Plain Flan-T5 uses beam search, which always
picks the single most likely sentence, so it repeats its favourite phrasing.
The `t5-guided` backend samples 6 candidates, lets the faithfulness checker
remove every unfaithful one, and picks one of the rest at random. You get more
variety with the same faithfulness.
```bash
physiocoach demo -n 8 --backend t5-guided
```
In the web demo, choose **B+ · Flan-T5, checker-guided sampling**. The
*Guided decoding* row shows how many candidates were generated, how many were
distinct, and how many passed the checker. Change the number of candidates with
`PHYSIOCOACH_T5_SAMPLES` (more means more variety but slower).

### Make Flan-T5 fast enough for live cues (ONNX + int8)

On a laptop CPU, PyTorch with beam search takes about 0.8–1 s per cue, which is
too slow mid-exercise. Two changes fix it:

1. **Greedy decoding is now the default** (`PHYSIOCOACH_T5_BEAMS=1`). A fine-tuned
   model gives almost the same sentence as beam search at about 40% less time. Set
   `PHYSIOCOACH_T5_BEAMS=4` to go back to beam search.
2. **Export to ONNX with int8 quantization:**
   ```bash
   pip install -e ".[onnx]"
   python scripts/export_onnx.py --model models/flan-t5-coach-para-w5 --quantize
   ```
   This writes `models/flan-t5-coach-para-w5-onnx` and `...-onnx-int8` (about
   140 MB instead of 300 MB).

Measure it on your laptop, which gives the efficiency numbers for your report:
```bash
python scripts/benchmark_latency.py --models models/flan-t5-coach-para-w5 models/flan-t5-coach-para-w5-onnx models/flan-t5-coach-para-w5-onnx-int8
```
It times beam search, greedy and guided sampling for each version and also
reports the faithfulness pass rate, so you can see that the speed-up does not
cost correctness. On a model of this size, measured on a 2-core cloud CPU:
about 1,030 ms (PyTorch beam) → 500 ms (ONNX greedy) → **300 ms (int8 greedy)**.

If the int8 model stays faithful, make it the default:
- Windows: `[Environment]::SetEnvironmentVariable("PHYSIOCOACH_T5_DIR", "models/flan-t5-coach-para-w5-onnx-int8", "User")`
- macOS / Linux: `export PHYSIOCOACH_T5_DIR=models/flan-t5-coach-para-w5-onnx-int8`

Then re-run `scripts/evaluate.py` on it once, so the report's numbers are for the
model you actually demo.

## Step 8: LLM comparison with Ollama (laptop)

1. Install Ollama from **ollama.com** and start it. The desktop app runs it in
   the background; otherwise run `ollama serve`.
2. Pull a small model:
   ```bash
   ollama pull qwen2.5:3b
   ```
   If your laptop has 16 GB of RAM or more, also try `llama3.1:8b` and compare.
3. Try it:
   ```bash
   physiocoach demo -n 5 --backend llm
   ```
   Use a different model with `PHYSIOCOACH_OLLAMA_MODEL=llama3.1:8b`.

Expect the LLM to be fluent but slower (seconds per cue on a laptop) and to fail
the faithfulness check more often than Flan-T5. That contrast is the point of
the comparison.

## Step 9: Urdu translation and speech (laptop)

Needs the ML extras from step 7. The first Urdu request downloads:
- **NLLB-200 distilled 600M** (about 2.5 GB), for translation
- **MMS-TTS English and Urdu** (about 150 MB each), for speech

Run the server and try *اردو* with **Speak** ticked:
```bash
physiocoach serve
```
The first Urdu request takes a while because models load; later ones are fast,
and repeated cues come from the cache (`cache/translations.json` and
`cache/audio/`).

What happens behind the scenes:
1. The English cue is generated and checked.
2. NLLB translates it into Urdu script.
3. The **preservation check** confirms every number and the left/right side
   survived. If not, the hand-written Urdu template is used, and the web page
   shows *"Translation rejected → Urdu template"*.
4. Numbers are spelled out as Urdu words, and MMS speaks the text.

### Make live cues instant (do this once)

Out of the box, a new spoken Urdu cue can take several seconds: machine
translation plus speech synthesis on a laptop CPU. That is too slow mid-exercise,
so the project has a **live mode**, on by default for single reps:

- **Urdu text** for live reps comes from the hand-written Urdu templates, which
  are instant and say the same facts. Machine translation is used for session
  summaries, or when you untick *Live mode* in the web page.
- **Speech** is cached per sentence. Pre-record every cue sentence once:
  ```bash
  python scripts/precompute_audio.py
  ```
  This records about 860 sentences (every fault, side, and degree value from 1
  to 45, in English and Urdu) into `cache/audio/`. Expect roughly 15–45 minutes
  depending on your CPU. You can stop it and run it again later; finished
  sentences are skipped. Re-run it after changing any cue wording.

After that, the *Timing* row should show `speak` in single-digit milliseconds
with "(audio from cache)", and no `translate` step for live reps. The server
also loads all models in the background when it starts
(`PHYSIOCOACH_WARMUP=0` turns that off), so the first request isn't slow either.

For non-live Urdu (session summaries), `PHYSIOCOACH_NLLB_BEAMS=1` makes
translation roughly twice as fast, at a small quality cost. The server default
is 2; the evaluation script uses 4.

**Tune the glossary.** Run step 11's translation evaluation once, read the
machine translations, and add stiff or wrong terms you see repeatedly to
`glossary_ur.json` as `"machine wording": "preferred wording"`.

Settings (environment variables) if you need them:

| Variable | Use it to |
|---|---|
| `PHYSIOCOACH_TRANSLATOR=none` | Skip NLLB; Urdu comes from templates only (fast, offline) |
| `PHYSIOCOACH_TTS=none` | Turn speech off |
| `PHYSIOCOACH_TTS=piper` + `PHYSIOCOACH_PIPER_VOICE=path/voice.onnx` | Faster English voice (`pip install -e ".[piper]"`) |

Licence note for the report: NLLB and MMS weights are CC-BY-NC 4.0, so they are
fine for coursework but not for a commercial product.

## Step 10: Hand-verified test set (laptop + team)

The automatic test references come from templates, which flatters the template
baseline on BLEU and ROUGE. Your headline numbers should use references the team
wrote or corrected.

1. Open `data/test_to_verify.csv` in Excel or Google Sheets.
2. For each row, read `input` and rewrite `verified_reference` the way a good
   physiotherapist would say it, in your own words. Put `y` in `reviewed`.
   Split the 250 rows between teammates.
3. Save as CSV (UTF-8) and import:
   ```bash
   python scripts/import_verified.py --csv data/test_to_verify.csv
   ```
   It writes `data/test_verified.jsonl` and lists any verified reference the
   checker flags. Either fix the reference, or, if the wording is fine, add the
   phrase to that fault's `detect` regexes in the config. Report how often this
   happened; it measures the checker's vocabulary coverage.

## Step 11: Evaluation and ablation (laptop)

**Main comparison** (with Ollama running, Flan-T5 in `models/`):
```bash
python scripts/evaluate.py --test data/test_verified.jsonl --backends template,t5,t5-guided,llm --bertscore
```
Results go to `results/metrics.md` (paste it straight into the report),
`results/metrics.json`, and `results/predictions_<backend>.csv`.

| Column | Meaning |
|---|---|
| `bleu`, `chrf`, `rouge_l`, `bertscore_f1` | Similarity to the references |
| `faithful_raw_pct` | % of outputs passing the checker before fallback |
| `fault_precision` / `fault_recall` | Mentioned faults that are real / required faults that are mentioned |
| `hallucination_pct` | % of outputs that correct an absent fault or praise a present one |
| `number_error_pct` | % with a number not supported by the input |
| `distinct_2` | Variety (higher means less repetitive) |
| `latency_ms_mean` | Speed per output |
| `guided_rejected_pct` | t5-guided only: % of distinct sampled candidates the checker rejected during decoding |

**Ablation:** `faithful_raw_pct` is the system *without* the checker. With the
checker every output is faithful (`faithful_after_check_pct = 100`) at the cost
of `fallback_pct` falling back to templates. Report both side by side.

**Translation:** first fill `ur_reference` in `data/translation_eval.csv` (the
team translates the 100 cues by hand), then:
```bash
python scripts/evaluate_translation.py --csv data/translation_eval.csv
```
This reports chrF against your translations, the % of translations that keep
numbers and sides, the fallback rate and latency. It also writes
`results/translation_eval.csv` with columns for native-speaker adequacy and
fluency ratings.

**Speech:**
```bash
python scripts/speech_eval.py --n 20
```
This writes 40 clips and a shuffled rating sheet to `results/speech_eval/`, plus
cold versus cached latency in `latency.json`.

## Step 12: Human evaluation (team + volunteers)

```bash
python scripts/human_eval.py make --predictions "results/predictions_*.csv" --n 50
```
Give `results/human_eval_sheet.csv` to at least 3 raters (classmates, family,
ideally a physiotherapy student). The backend names are hidden and the order is
shuffled. Each rater fills clarity, actionability and naturalness (1–5) in their
own copy, saved as `results/rater_1.csv`, `results/rater_2.csv` and so on. Then:
```bash
python scripts/human_eval.py score results/rater_*.csv
```
Keep `results/human_eval_key.csv` away from raters until scoring.

Do the same for Urdu by giving Urdu speakers `results/translation_eval.csv` and
`results/speech_eval/sheet.csv`.

## Step 13: Report and demo

Where each result goes in the proposal's structure:

| Report section | Source |
|---|---|
| Dataset | `data/stats.json`, paraphrase kept/dropped counts, verification flag rate |
| Model comparison | `results/metrics.md` (verified test set) |
| Faithfulness and ablation | `faithful_raw_pct` vs `faithful_after_check_pct`, `fallback_pct`, examples from `predictions_*.csv` |
| Translation | `results/translation_metrics.json`, glossary entries added |
| Speech | `results/speech_eval/latency.json`, listener ratings |
| Human evaluation | `human_eval.py score` output |
| Efficiency | latency columns; Flan-T5 PyTorch vs ONNX; small vs base |

Good examples to show: a case where the LLM hallucinated a correction and the
checker caught it (search `predictions_llm.csv` for `faithful` = `False`), and a
translation the preservation check rejected.

**The two demo pages for the presentation**
- **http://127.0.0.1:8000/compare**: one record, all four generators side by
  side. Sentences that state something the analysis did not find are marked in
  red with the reason ("says 3 sets, but there were 2"), and the card shows what
  the patient hears instead. Set *Faults* to *Many* and pick *C* to show the LLM
  being caught.
- **http://127.0.0.1:8000/live**: a simulated workout. Each rep's cue appears and
  is spoken right after the rep, with a running fault tally and a spoken session
  summary at the end. Use *Fast* pace and 2 sets of 5 reps for a 1-minute demo.

**Demo-day checklist**
- [ ] Run `python scripts/precompute_audio.py` beforehand, so every live cue is pre-recorded.
- [ ] Start `physiocoach serve` a minute early; models load in the background.
- [ ] Have Ollama running if you show the LLM.
- [ ] Turn off *Faithfulness check* in the web page to show the ablation live:
      pick a record, generate with the LLM, and show the difference.
- [ ] Keep `PHYSIOCOACH_TRANSLATOR=none` as a backup if the internet or a model
      download fails; Urdu still works from templates.

## Troubleshooting

| Problem | Fix |
|---|---|
| `physiocoach: command not found` | Activate the virtual environment, then `pip install -e ".[dev]"` again |
| Flan-T5 chip stays grey | The folder must be exactly `models/flan-t5-coach/` with `config.json` inside, or set `PHYSIOCOACH_T5_DIR` |
| `Ollama is not running` | Start the Ollama app, or run `ollama serve`; check `ollama list` shows your model |
| Training loss becomes `nan` | Don't force fp16 with T5; the script uses bf16 only when the GPU supports it |
| Colab out of memory | `--batch 16` (or 8 for flan-t5-base) |
| Urdu shows as boxes | Your terminal font lacks Urdu; the web page loads Noto Nastaliq Urdu |
| MMS asks for `uroman` | `pip install uroman` |
| Spoken or Urdu cues are slow | Keep *Live mode* ticked and run `scripts/precompute_audio.py` once (step 9) |
| `WinError 126 ... torch\lib\shm.dll` | Install the Microsoft Visual C++ Redistributable (aka.ms/vs/17/release/vc_redist.x64.exe), or `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `[Errno 10048]` when starting the server | An old server is still running: close that terminal, or `netstat -ano \| findstr :8000` then `taskkill /PID <pid> /F` |
| Port 8000 in use | `physiocoach serve --port 8010` |

## After the course: integrate into PhysioLens

See `docs/INTEGRATION.md`. The short version: run this service next to
PhysioLens and send one HTTP request per finished rep.
