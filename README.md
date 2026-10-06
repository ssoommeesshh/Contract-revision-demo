# Today's mentor feasibility showcase

## Larger model on a T4

Use Qwen2.5-7B-Instruct with NF4 quantization to reduce GPU memory requirements.
Start with controls, then compare the same 60 cases and seed as the 3B run:

```python
!git pull origin main
%pip install -q -r requirements-4bit.txt
!python diagnose.py --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --controls-only
!python diagnose.py --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --limit 60 --seed 42
```

Use a GPU session without another loaded model. The initial download is larger
than the 3B download; quantization reduces GPU memory, not download size.
Both runs preserve the same prompts, sample and output-token limits. The manifest
records quantization, so report this as a 3B float16 versus 7B NF4 comparison.
For a comparison with the same quantization setting, optionally rerun the 3B model
with `--load-in-4bit`. Better accuracy on ACE is a hypothesis, not guaranteed.
Local syntax and selection checks do not verify GPU loading or performance.

## Investigate repeated Non-Compliant answers

```python
!git pull origin main
!python diagnose.py --limit 60
```

This runs six simple controls under the original and a grounded diagnostic prompt,
then runs both prompts on a seeded sample of 60 ACE examples (20 per label).
The original six inspected cases are excluded. It reports prediction counts,
per-class precision/recall/F1, macro-F1, errors and confusion matrices. Prompts,
raw responses and configuration are saved in separate timestamped folders under
`outputs/diagnostic_.../`, preserving the initial run. Generation scores are checked
for NaN or positive infinity; inputs are never silently truncated.

Start with controls only if short on time:

```python
!python diagnose.py --controls-only
```

If controls fail, compare an alternate attention backend:

```python
!python diagnose.py --controls-only --attention eager
```

For a numerical-precision check, with no other model using GPU memory:

```python
!python diagnose.py --controls-only --attention eager --dtype float32
```

Float32 requires substantially more GPU memory and may fail if the session has
another loaded model. These diagnostics do not reproduce COMPACT training.
The grounded prompt is a new development condition, not the original verification
condition. Further testing on a separate sample is needed after prompt development.
No GPU inference is performed by `--inspect`; use it for data/selection checks.

## Clone and run in Kaggle

Select a GPU accelerator and enable Internet in notebook settings. Run these
cells in a fresh GPU notebook:

```python
!git clone https://github.com/ssoommeesshh/Contract-revision-demo.git
%cd /kaggle/working/Contract-revision-demo
%pip install -q -r requirements.txt
!python run_t4.py --limit 6
```

In Colab, use `%cd /content/Contract-revision-demo` instead. Select a T4 GPU first.
For a shorter run: `!python run_t4.py --limit 3 --skip-revisions`.
The script saves responses under `outputs/` and creates `mentor_results.zip`.
Console results include comparison tables and wrapped explanations. To display
an existing run without loading the model or using a GPU:

```python
!python run_t4.py --show-saved
```

Raw JSON predictions remain in `outputs/` for evaluation.
Model inference needs a GPU runtime; the standard-library loader `run.py` can run locally.
The downloaded ACE test split and provenance are included in `data/`.
The runner uses this local copy when present. A standalone notebook uploaded without
the repository downloads the dataset automatically. Model weights and generated results are not committed.

Kaggle may report dependency conflicts with preinstalled Gradio or Diffusers after
installation. This runner does not import those packages. If installation completes,
continue with `!python run_t4.py --limit 6`, which starts a fresh Python process.
If that command raises a traceback, retain it for diagnosis.

## Notebook option

Upload `Mentor_Feasibility_T4.ipynb` to Colab or Kaggle. Select a T4 GPU;
enable Internet in Kaggle. Run cells in order. No API key or local ML installation is needed.
The notebook embeds the harness, so uploading that one file is sufficient.

The run downloads Qwen2.5-3B-Instruct, tests six ACE examples under two prompt
conditions, saves predictions and scores, and runs three synthetic SaaS revision
illustrations. Allow roughly 30–90 minutes including download/setup; this is an
estimate, not a measured GPU runtime. If time is short, change `load_cases(6)` to
`load_cases(3)` before the benchmark cell.

This runs a general pretrained model on the released ACE data. It does not
reproduce the COMPACT training procedure, graph construction or published results.
No GPU model execution has been performed in the local Windows environment.

## Show the mentor

1. One real ACE record: clauses, scenario, reference label.
2. Actual model prediction and supporting clause IDs.
3. Saved scores and at least one disagreement/error, if present.
4. Synthetic before/after edit output as the bridge to your research question.

ACE gives relevant clause sets, not full-contract parsing/retrieval annotations.
Not-Applicable means the clauses do not govern the scenario, not uncertainty.
The six cases are a convenience smoke-test sample. Synthetic cases are
author-constructed and have not received legal expert validation. Evidence-key
existence checks do not establish that the cited text supports a conclusion.

## Local dataset inspection

```powershell
python run.py --fetch --limit 6
python run.py --limit 6
```

The second command works offline after download. Inspect `outputs/selected_cases.txt`.
The harness also supports Ollama or a compatible chat endpoint; run `--help`.

Sources:
- ACE: https://github.com/FujitsuResearch/Fujitsu-Assessing-Compliance-in-Enterprise-Dataset
- COMPACT: https://aclanthology.org/2026.eacl-long.377/
- Model: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

ACE license is CC BY 4.0 per its README. Preserve attribution and provenance.
The model card identifies a Qwen research license.
