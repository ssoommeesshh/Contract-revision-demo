"""Larger ACE diagnostic, with independent simple controls and raw output logging."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import run as benchmark
import console_report as report


def select_cases(data, count, seed):
    if count < 3 or count % 3:
        raise ValueError('--limit must be a positive multiple of 3')
    rng = random.Random(seed)
    chosen = []
    # Keep the original six inspected examples out of this diagnostic sample.
    excluded = {0, 220, 440, 659, 879, 1099}
    for label in benchmark.LABELS:
        pool = [i for i, item in enumerate(data) if item['gd_tr'] == label and i not in excluded]
        chosen.extend(rng.sample(pool, count // 3))
    rng.shuffle(chosen)
    return [(i, data[i]) for i in chosen]


def metrics(cases, rows):
    summary = benchmark.score(cases, rows)
    matrix = summary['confusion_matrix']
    per_class = {}
    for label in benchmark.LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[gold][label] for gold in benchmark.LABELS)
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class[label] = {'support': support, 'precision': precision, 'recall': recall, 'f1': f1}
    summary.update(per_class=per_class,
                   macro_f1=sum(v['f1'] for v in per_class.values()) / 3,
                   prediction_counts=dict(Counter((r.get('prediction') or {}).get('label', 'ERROR') for r in rows)))
    return summary


def grounded_prompt(item):
    return (
        'Classify a scenario against ONLY the provided clauses. Do not infer obligations '
        'from complaints, opinions, or claims in the scenario. First decide whether the '
        'clauses govern the conduct in question. If they do not, label Not-Applicable. '
        'If applicable, label Non-Compliant only when an express obligation or restriction '
        'is breached; otherwise label Compliant. Inspect exceptions and permissions. '
        'Do not invent notice, fairness, promotion, or other obligations absent from the text. '
        'Return JSON only: label (Compliant, Non-Compliant, Not-Applicable), evidence_ids '
        '(list of supplied keys), explanation (short), unresolved_issues (list).\n'
        + json.dumps({'clauses': item['clauses'], 'scenario': item['scenario_text']}))


def controls():
    items = [
        ('Customer must pay invoices within 30 days.', 'Customer paid the invoice after 10 days.', 'Compliant'),
        ('Customer must pay invoices within 30 days.', 'Customer paid the invoice after 60 days.', 'Non-Compliant'),
        ('Customer must pay invoices within 30 days.', 'Provider painted its office blue. No invoice or payment is involved.', 'Not-Applicable'),
        ('Recipient must not disclose Confidential Information except to its employees.', 'Recipient disclosed Confidential Information only to its employees.', 'Compliant'),
        ('Recipient must not disclose Confidential Information except to its employees.', 'Recipient published Confidential Information on a public website.', 'Non-Compliant'),
        ('Recipient must not disclose Confidential Information except to its employees.', 'Customer changed the color of its logo. No information was disclosed.', 'Not-Applicable'),
    ]
    return [(i, {'clauses': {'evidence_1': clause}, 'scenario_text': scenario, 'gd_tr': label})
            for i, (clause, scenario, label) in enumerate(items)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--controls-only', action='store_true')
    parser.add_argument('--attention', choices=['sdpa', 'eager'], default='sdpa')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--load-in-4bit', action='store_true', help='NF4 quantization for larger models on T4')
    parser.add_argument('--inspect', action='store_true', help='Validate sample and metrics without GPU inference')
    args = parser.parse_args()
    if args.load_in_4bit and args.dtype != 'float16':
        parser.error('--load-in-4bit uses float16 compute; omit --dtype float32')
    path = benchmark.ROOT / 'data/ace_test.json'
    raw = path.read_bytes()
    data = benchmark.normalize_records(json.loads(raw))
    assert all(item['gd_tr'] in benchmark.LABELS for item in data)
    selected = select_cases(data, args.limit, args.seed)
    report.heading('ACE data and diagnostic selection')
    print('Dataset records:', len(data), 'Label counts:', dict(Counter(x['gd_tr'] for x in data)))
    print('Selected:', len(selected), 'Seed:', args.seed, 'Labels:', dict(Counter(x['gd_tr'] for _, x in selected)))
    print('Original six cases excluded. Reference labels are used only for sampling/scoring, not model inputs.')
    if args.inspect:
        print('Dataset and selection checks passed. No inference performed.')
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert torch.cuda.is_available(), 'Enable GPU before running.'
    output = Path('outputs') / ('diagnostic_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    print('Results:', output, 'GPU:', torch.cuda.get_device_name(0), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_options = {'torch_dtype': getattr(torch, args.dtype),
                    'device_map': {'': 0}, 'attn_implementation': args.attention}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_options['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    print('Model:', args.model, 'Quantization:', 'NF4 / float16 compute' if args.load_in_4bit else 'none', flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_options).eval()
    benchmark.write_json(output / 'manifest.json', {
        'args': vars(args), 'model_revision': getattr(model.config, '_commit_hash', None),
        'dataset_sha256': hashlib.sha256(raw).hexdigest(), 'case_indices': [i for i, _ in selected],
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'reference': 'https://github.com/FujitsuResearch/Fujitsu-Assessing-Compliance-in-Enterprise-Dataset',
        'note': 'Diagnostic prompting experiment, not reproduction of COMPACT.'})

    def evaluate(cases, mode, prompt_fn):
        rows = []
        for index, item in cases:
            prompt = prompt_fn(item)
            row = {'case_index': index, 'prompt': prompt, 'prediction': None}
            try:
                text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors='pt').to('cuda')
                n = inputs['input_ids'].shape[-1]
                if n > 6000:
                    raise ValueError('Input exceeds 6000-token budget; not truncated')
                started = time.perf_counter()
                with torch.inference_mode():
                    generation = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                        temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True, output_scores=True)
                generated = generation.sequences
                bad_scores = any(bool(torch.isnan(scores).any() or torch.isposinf(scores).any())
                                 for scores in generation.scores)
                row['non_finite_generation_scores'] = bad_scores
                text = tokenizer.decode(generated[0, n:], skip_special_tokens=True)
                row.update(raw=text, seconds=time.perf_counter() - started,
                           input_tokens=n, output_tokens=generated.shape[-1] - n,
                           reached_token_limit=generated.shape[-1] - n == 512)
                if bad_scores:
                    raise ValueError('NaN or positive infinity in generation scores; investigate dtype/attention')
                cleaned = text.strip()
                if cleaned.startswith('```'):
                    cleaned = '\n'.join(cleaned.splitlines()[1:-1])
                prediction = json.loads(cleaned)
                if not isinstance(prediction, dict) or prediction.get('label') not in benchmark.LABELS or not isinstance(prediction.get('evidence_ids'), list):
                    raise ValueError('Invalid output schema')
                row['prediction'] = prediction
            except Exception as exc:
                row['error'] = str(exc)
            rows.append(row)
            benchmark.write_json(output / f'{mode}_predictions.json', rows)
            print(f"{mode} {len(rows)}/{len(cases)} | case {index} | predicted {(row['prediction'] or {}).get('label', 'ERROR')} | reference {item['gd_tr']}", flush=True)
        summary = metrics(cases, rows)
        benchmark.write_json(output / f'{mode}_scores.json', summary)
        report.ace_results(cases, {mode: rows}, {mode: summary})
        print('Prediction counts:', summary['prediction_counts'], 'Macro-F1:', round(summary['macro_f1'], 3))
        report.table(['Label', 'Precision', 'Recall', 'F1'], [[label] + [f"{v[k]:.1%}" for k in ['precision', 'recall', 'f1']] for label, v in summary['per_class'].items()])
        return summary

    baseline = lambda item: benchmark.prompt(item, 'ordinary')
    control_summary = evaluate(controls(), 'controls_original', baseline)
    evaluate(controls(), 'controls_grounded', grounded_prompt)
    if control_summary['correct'] < 6:
        print('Original prompt failed simple controls. Treat ACE results as diagnostic, not reliable performance.')
    if not args.controls_only:
        evaluate(selected, 'ace_original', baseline)
        evaluate(selected, 'ace_grounded', grounded_prompt)
    print('\nComplete. Preserve the timestamped folder:', output)


if __name__ == '__main__':
    main()
