"""Apply readable presentation to the generated mentor notebook."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
path = root / 'Mentor_Feasibility_T4.ipynb'
notebook = json.loads(path.read_text(encoding='utf-8'))


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


helpers = '''import json
import html
from pathlib import Path
from IPython.display import display, HTML, Markdown

def table(rows):
    if not rows:
        return
    columns = list(rows[0])
    header = ''.join('<th>' + html.escape(str(c)) + '</th>' for c in columns)
    body = ''.join('<tr>' + ''.join('<td>' + html.escape(str(row.get(c, ''))) + '</td>'
                                  for c in columns) + '</tr>' for row in rows)
    display(HTML('<style>th,td {padding:8px 12px;text-align:left;vertical-align:top;}'
                 'th {background:#edf2f7;} td {border-bottom:1px solid #ddd;}'
                 '</style><table><thead><tr>' + header + '</tr></thead><tbody>' + body + '</tbody></table>'))

def show_case(index, item):
    display(Markdown(f"### ACE case {index}"))
    display(Markdown('**Scenario**'))
    display(HTML('<p>' + html.escape(item['scenario_text']) + '</p>'))
    table([{'Evidence ID': key, 'Contract passage': text}
           for key, text in item['clauses'].items()])
    display(Markdown('**Reference label:** ' + item['gd_tr'] + ' — hidden from model input.'))

def show_ace_results(predictions):
    gold = dict(cases)
    rows = []
    for mode, values in predictions.items():
        for row in values:
            pred = row.get('prediction') or {}
            expected = gold[row['case_index']]['gd_tr']
            actual = pred.get('label', 'ERROR')
            rows.append({'Method': mode, 'Case': row['case_index'], 'Reference': expected,
                         'Model answer': actual, 'Match': 'Yes' if actual == expected else 'No',
                         'Seconds': round(row.get('timing', {}).get('seconds', 0), 1)})
    table(rows)
    table([{'Method': mode, 'Correct': str(benchmark.score(cases, values)['correct']) + '/' + str(len(cases)),
            'Accuracy': f"{benchmark.score(cases, values)['accuracy']:.1%}"}
           for mode, values in predictions.items()])

def show_answer(title, row):
    pred = row.get('prediction') or {}
    display(Markdown('### ' + title))
    table([{'Field': key.replace('_', ' ').title(), 'Value': pred.get(key, '—')}
           for key in ['label', 'before_status', 'after_status', 'introduced_violation', 'evidence_ids']
           if key in pred])
    display(HTML('<p><b>Explanation:</b> ' + html.escape(str(pred.get('explanation', row.get('error', 'No explanation')))) + '</p>'))
    issues = pred.get('unresolved_issues', [])
    if issues:
        table([{'Unresolved issue': issue} for issue in issues])

def show_revisions(values):
    table([{'Edit': row['id'], 'Expected before': row['reference_before'],
            'Model before': (row.get('prediction') or {}).get('before_status', 'ERROR'),
            'Expected after': row['reference_after'],
            'Model after': (row.get('prediction') or {}).get('after_status', 'ERROR')}
           for row in values])
    for row in values:
        show_answer(row['id'].replace('_', ' ').title(), row)
'''

out = [cell('markdown', '''# Contract revision · feasibility notebook

**Purpose:** run a small existing benchmark, inspect the model's evidence, then illustrate a SaaS edit check.

| Part | What you will see |
|---|---|
| 1. Setup | GPU check, repository and sample selection |
| 2. Existing benchmark | Six ACE examples, predicted labels and reference labels |
| 3. SaaS edits | Original wording, proposed edits and before/after findings |
| 4. Export | Saved predictions, scores and provenance |

Choose **T4 GPU** and enable **Internet**. Run cells in order for a fresh experiment.
If you already ran `run_t4.py`, use the **Read saved results** cell below to display those outputs without loading the model again.

This is a feasibility exercise, not reproduction of COMPACT or evidence of legal reliability.
''')]
for original in notebook['cells']:
    source = ''.join(original['source'])
    kind = original['cell_type']
    if kind == 'markdown' and source.startswith('# Mentor'):
        continue
    if kind == 'code' and source.startswith('%pip'):
        out.append(cell('markdown', '## 1 · Setup\nInstall the two libraries used for inference. The runner does not use Gradio or Diffusers.\n'))
    if "Path('run.py').write_text" in source:
        source = '''# Use readable source files from the repository, rather than embedding code here.
from pathlib import Path
import subprocess
import sys
import os

if not Path('run.py').exists():
    target = Path.cwd() / 'Contract-revision-demo'
    if not target.exists():
        subprocess.run(['git', 'clone', 'https://github.com/ssoommeesshh/Contract-revision-demo.git', str(target)], check=True)
    os.chdir(target)
sys.path.insert(0, str(Path.cwd()))
import run as benchmark
print('Working directory:', Path.cwd())
'''
        out.extend([cell('code', source), cell('code', helpers)])
        continue
    if 'cases = benchmark.load_cases(6)' in source:
        source = source[:source.index('print("Label distribution:')]+'''table([{'Reference label': label, 'Selected cases': count}
       for label, count in Counter(item['gd_tr'] for _, item in cases).items()])
show_case(*cases[0])
'''
        out.append(cell('markdown', '### Load the benchmark\nThe reference labels stay outside the model input. Relevant clauses are supplied by ACE; this does not test full-agreement retrieval.\n'))
        out.append(cell(kind, source))
        out.append(cell('markdown', '''### Read saved results — optional
After an earlier command-line run, execute this cell to inspect the saved answers.
You can skip the model-loading and inference cells if the files are present.
'''))
        out.append(cell('code', '''saved = {}
for mode in ['ordinary', 'verify']:
    path = Path('outputs') / f'predictions_{mode}.json'
    if path.exists():
        saved[mode] = json.loads(path.read_text(encoding='utf-8'))
if saved:
    show_ace_results(saved)
    for mode, rows in saved.items():
        show_answer(mode.title() + ' — first saved case', rows[0])
else:
    print('No saved predictions yet. Continue below for a fresh run.')
revision_path = Path('outputs/synthetic_revision_outputs.json')
if revision_path.exists():
    show_revisions(json.loads(revision_path.read_text(encoding='utf-8')))
'''))
        continue
    if 'from transformers import' in source:
        out.append(cell('markdown', '### Load the model\nQwen2.5-3B-Instruct runs on the T4. The first run downloads approximately 6 GB of weights.\n'))
    if kind == 'markdown' and source.startswith('## Existing'):
        source = '''## 2 · Existing benchmark: ACE
**Ordinary:** asks for an assessment. **Verify:** adds an explicit instruction to inspect clause interactions.
Both use the same examples, model and output limit. This is a preliminary prompting comparison.

Errors count as incorrect. A valid evidence ID does not mean the explanation is supported.
'''
    if 'for mode, summary in summaries.items():' in source:
        source = source[:source.index('for mode, summary in summaries.items():')] + '''show_ace_results(all_predictions)
display(Markdown('**Interpretation:** these six examples are a smoke test, not a reliable method comparison.'))
'''
    if source.startswith('# Inspect the actual explanation'):
        source = '''# Compare explanations for the first selected case.
show_case(*cases[0])
for mode, rows in all_predictions.items():
    show_answer(mode.title() + ' review', rows[0])
'''
        out.append(cell('markdown', '### Inspect the reasoning\nCompare each explanation against the displayed passage. Look for invented obligations, missed exceptions and unsupported citations.\n'))
    if kind == 'markdown' and source.startswith('## Bridge'):
        source = '''## 3 · SaaS revision illustrations
**Requirement:** retain termination after three consecutive months below the uptime target.

| Edit | Intended after-status |
|---|---|
| Explicitly overrides termination | Violated |
| Expressly preserves termination | Satisfied |
| Sole-remedy language with an unresolved conflict | Uncertain |

These are author-constructed snippets, not complete agreements or expert-validated benchmark cases.
'''
    if source.startswith('original ='):
        split = source.index('revision_outputs = []')
        definitions = source[:split] + '''display(Markdown('### Original agreement'))
table([{'Clause': key, 'Text': text} for key, text in original.items()])
display(Markdown('### Proposed replacements for clause 4.2'))
table([{'Edit': case['id'], 'Replacement text': case['new_text'],
        'Expected after': case['reference_after']} for case in revision_cases])
'''
        inference = source[split:].replace('    print(json.dumps(row, indent=2))', "    print('Completed:', case['id'])")
        out.extend([cell('code', definitions), cell('markdown', '### Run the edit checks\nAssess the original and revised wording independently. Results below show both statuses so a pre-existing issue cannot be mistaken for an introduced violation.\n'),
                    cell('code', inference), cell('code', 'show_revisions(revision_outputs)\n')])
        continue
    if kind == 'markdown' and source.startswith('## What'):
        source = source.replace('## What to tell the mentor', '## 4 · Interpretation and export')
    out.append(cell(kind, source))
for i, entry in enumerate(out):
    entry['id'] = f'cell-{i:02d}'
notebook['cells'] = out
path.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
print('Styled notebook with readable tables and saved-result viewer')
