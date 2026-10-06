"""Readable terminal reports; raw predictions stay in JSON files."""
import textwrap

def heading(title):
    print('\n' + '=' * 80 + '\n' + title + '\n' + '=' * 80)

def paragraph(label, value):
    print('\n' + label + ':\n' + textwrap.fill(str(value), width=96, initial_indent='  ', subsequent_indent='  '))

def table(headers, rows):
    rows = [[str(v) for v in row] for row in rows]
    widths = [max([len(h)] + [len(row[i]) for row in rows]) for i,h in enumerate(headers)]
    def line(row):
        return ' | '.join(v.ljust(widths[i]) for i,v in enumerate(row))
    print(line(headers))
    print('-+-'.join('-' * w for w in widths))
    for row in rows:
        print(line(row))

def case(index, item):
    heading(f'Example input: ACE case {index}')
    paragraph('Scenario', item['scenario_text'])
    for key, passage in item['clauses'].items():
        paragraph('Contract passage [' + key + ']', passage)
    print('\nReference:', item['gd_tr'], '(not sent to model)')

def answer(title, row):
    heading(title)
    pred = row.get('prediction') or {}
    for key in ['label','before_status','after_status','introduced_violation']:
        if key in pred:
            print(key.replace('_',' ').capitalize() + ':', pred[key])
    print('Evidence IDs:', ', '.join(map(str,pred.get('evidence_ids',[]))) or 'None')
    paragraph('Explanation', pred.get('explanation','No explanation returned.'))
    for issue in pred.get('unresolved_issues',[]):
        paragraph('Unresolved issue',issue)
    if row.get('error'):
        paragraph('Error',row['error'])

def ace_results(cases, predictions, summaries):
    gold = dict(cases)
    heading('ACE benchmark: answer comparison')
    table(['Method','Case','Reference','Model answer','Match'],[
        [mode,row['case_index'],gold[row['case_index']]['gd_tr'],
         (row.get('prediction') or {}).get('label','ERROR'),
         'Yes' if (row.get('prediction') or {}).get('label') == gold[row['case_index']]['gd_tr'] else 'No']
        for mode,values in predictions.items() for row in values])
    heading('ACE benchmark: scores')
    table(['Method','Correct','Accuracy','Invalid citation IDs'],[
        [mode,f"{s['correct']}/{s['n']}",f"{s['accuracy']:.1%}",len(s['invalid_evidence_ids'])]
        for mode,s in summaries.items()])
    for mode,s in summaries.items():
        print('\n' + mode.title() + ' confusion matrix (rows: reference; columns: prediction):')
        columns = list(next(iter(s['confusion_matrix'].values())))
        table(['Reference']+columns,[[label]+[counts[c] for c in columns] for label,counts in s['confusion_matrix'].items()])
    print('\nSmall-sample feasibility results. Citation existence does not establish evidence support.')

def revisions(values):
    heading('Synthetic SaaS edits: before/after comparison')
    table(['Edit','Expected before','Model before','Expected after','Model after'],[
        [r['id'],r['reference_before'],(r.get('prediction') or {}).get('before_status','ERROR'),
         r['reference_after'],(r.get('prediction') or {}).get('after_status','ERROR')] for r in values])
    for row in values:
        answer(row['id'].replace('_',' ').title(),row)
    print('\nSynthetic reference labels are illustrative, not expert-validated.')
