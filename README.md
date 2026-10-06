# Today's mentor feasibility showcase

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
Model inference needs a GPU runtime; the standard-library loader `run.py` can run locally.
Dataset files, downloaded model weights and generated results are not committed.

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
