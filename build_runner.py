"""Build a command-line equivalent of the notebook for Kaggle clone-and-run."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
notebook = json.loads((root / 'Mentor_Feasibility_T4.ipynb').read_text(encoding='utf-8'))
parts = ['''import argparse
from pathlib import Path
import run as benchmark

parser = argparse.ArgumentParser(description="T4 ACE feasibility run")
parser.add_argument('--limit', type=int, default=6)
parser.add_argument('--skip-revisions', action='store_true')
args = parser.parse_args()
if args.limit < 1:
    parser.error('--limit must be positive')
''']
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    if source.startswith('%') or "Path('run.py').write_text" in source:
        continue
    source = source.replace('benchmark.load_cases(6)', 'benchmark.load_cases(args.limit)')
    if source.startswith('original ='):
        source = 'if not args.skip_revisions:\n' + ''.join('    ' + line + '\n' for line in source.splitlines())
    if '# Save outputs' in source:
        source = source.split('try:\n')[0]
    parts.append(source)
(root / 'run_t4.py').write_text('\n\n'.join(parts), encoding='utf-8')
print('Built run_t4.py')
