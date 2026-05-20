#!/usr/bin/env python3
"""
Recompute Maj@N from existing eval_results JSONs using FIXED clustering
(unformatted predictions cluster as INVALID and cannot win the vote-correct).
Reads per-problem 'generations' field from each JSON, recomputes majority,
and writes patched majority_vote_at_n / majority_vote_at_n_pct in place.

Usage:
    python recompute_majn.py [-h] [--apply] [PATTERN]

Examples:
    python recompute_majn.py                                # default pattern, no writeback
    python recompute_majn.py 'eval_results/qwen3_4b_base_*.json' --apply
    python recompute_majn.py --apply 'eval_results/*.json'

The repo root is auto-resolved from this file's location, so the script
can be invoked from any cwd and against any checkout that contains it.
"""
import argparse
import glob
import json
import os
from pathlib import Path

# Inline grade_answer (avoid importing evaluate_math.py which loads vLLM)
from math_verify import parse, verify

NO_BOXED = '[No boxed answer found]'

def grade_answer(predicted, ground_truth):
    """Verbatim copy of evaluate_math.py grade_answer (math_verify with
    normalized-string fallback on parser exception). Defensively coerces
    both arguments to str so non-string ground truths (e.g. numeric answers
    from some math datasets) cannot crash the fallback path."""
    if predicted is None:
        return False
    predicted = str(predicted)
    ground_truth = str(ground_truth)
    try:
        if '$' not in predicted:
            predicted = f'${predicted}$'
        if '$' not in ground_truth:
            ground_truth = f'${ground_truth}$'
        pred_parsed = parse(predicted, fallback_mode='no_fallback')
        gt_parsed = parse(ground_truth, fallback_mode='no_fallback')
        return verify(gt_parsed, pred_parsed, timeout_seconds=5)
    except Exception:
        # Match evaluate_math.py: fallback to normalized string equality
        pred_norm = predicted.replace('$', '').replace(' ', '').lower().strip()
        gt_norm = ground_truth.replace('$', '').replace(' ', '').lower().strip()
        return pred_norm == gt_norm

def is_formatted(pred, saved_flag=None):
    """Match evaluate_math.py: prefer the JSON's saved 'formatted' flag if
    available; otherwise infer from the predicted_answer string."""
    if saved_flag is not None:
        return bool(saved_flag)
    if pred is None:
        return False
    s = str(pred).strip()
    if s == '' or s == NO_BOXED:
        return False
    return True

def majority_vote_fixed(predictions, gt_answer, formatted_flags=None):
    if formatted_flags is None:
        formatted_flags = [None] * len(predictions)
    clusters = []
    for pred, fmt in zip(predictions, formatted_flags):
        if not is_formatted(pred, saved_flag=fmt):
            inv = next((c for c in clusters if c[0] is None), None)
            if inv is not None:
                inv[1] += 1
            else:
                clusters.append([None, 1])
            continue
        placed = False
        for c in clusters:
            if c[0] is None:
                continue
            try:
                if grade_answer(pred, c[0]) or grade_answer(c[0], pred):
                    c[1] += 1
                    placed = True
                    break
            except Exception:
                pass
        if not placed:
            clusters.append([pred, 1])
    if not clusters:
        return False
    clusters.sort(key=lambda x: -x[1])
    rep = clusters[0][0]
    if rep is None:
        return False
    return grade_answer(rep, gt_answer)

def recompute_one(path):
    with open(path) as f:
        d = json.load(f)
    results = d.get('results', [])
    if not results:
        return None
    n_total = d.get('num_problems', len(results))
    pct_old = float(d.get('majority_vote_at_n_pct', -1))
    n_old = int(d.get('majority_vote_at_n', -1))
    n_new = 0
    for r in results:
        # Each per-problem entry has 'generations' = list of dicts with predicted_answer
        gens = r.get('generations', [])
        preds = []
        fmts = []
        for g in gens:
            if isinstance(g, dict):
                preds.append(g.get('predicted_answer'))
                fmts.append(g.get('formatted'))
            else:
                preds.append(g)
                fmts.append(None)
        gt = r.get('ground_truth') or r.get('answer') or r.get('gt_answer')
        if gt is None or not preds:
            continue
        if majority_vote_fixed(preds, gt, formatted_flags=fmts):
            n_new += 1
    pct_new = 100.0 * n_new / max(n_total, 1)
    return {'old_n': n_old, 'new_n': n_new, 'old_pct': pct_old, 'new_pct': pct_new}

def main():
    # Script lives at eval/recompute_majn.py; repo root is the parent of eval/.
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Offline recompute of Maj@N over existing eval JSONs '
                    '(uses the same fixed clustering as evaluate_math.py).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'pattern',
        nargs='?',
        default='eval_results/qwen3_4b_base_*.json',
        help='Glob pattern for JSONs to recompute. Relative paths are resolved '
             'against the repo root (the directory containing this script).',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Write corrected majority_vote_at_n / majority_vote_at_n_pct back '
             'into each JSON in place. Without this flag, only a summary is printed.',
    )
    args = parser.parse_args()

    pattern = args.pattern
    if not os.path.isabs(pattern):
        pattern = str(repo / pattern)
    paths = sorted(glob.glob(pattern))
    print(f'Repo root: {repo}')
    print(f'Pattern: {pattern}')
    print(f'Found {len(paths)} JSONs')
    print(f'apply_changes={args.apply} (use --apply to write back to JSONs)')
    print()
    print(f'{"file":<70} {"N":>4} {"old%":>7} {"new%":>7} {"Δ":>7}')
    print('-' * 100)
    summary = []
    for p in paths:
        try:
            r = recompute_one(p)
        except Exception as e:
            print(f'  SKIP {os.path.basename(p)}: {type(e).__name__}: {e}')
            continue
        if r is None:
            continue
        sign = '+' if r['new_pct'] >= r['old_pct'] else ''
        delta = r['new_pct'] - r['old_pct']
        n_problems = r['old_n'] + r['new_n'] - r['old_n']  # placeholder shown as N
        print(f'{os.path.basename(p):<70} {n_problems:>4} {r["old_pct"]:>6.2f}% {r["new_pct"]:>6.2f}% {sign}{delta:>6.2f}')
        if args.apply:
            with open(p) as f:
                d = json.load(f)
            d['majority_vote_at_n_old_buggy'] = d.get('majority_vote_at_n', None)
            d['majority_vote_at_n_pct_old_buggy'] = d.get('majority_vote_at_n_pct', None)
            d['majority_vote_at_n'] = r['new_n']
            d['majority_vote_at_n_pct'] = r['new_pct']
            d['majority_vote_at_n_method'] = 'cluster-all-with-INVALID-marker (recomputed offline)'
            with open(p, 'w') as f:
                json.dump(d, f, indent=2)
        r['path'] = os.path.basename(p)
        summary.append(r)
    if args.apply:
        out_path = repo / 'eval_results_summary' / 'majn_recomputed_summary.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({'patches': summary, 'applied': args.apply}, f, indent=2)
        print(f'\nWrote {len(summary)} entries to {out_path}')
    else:
        print(f'\nDry run: {len(summary)} JSON(s) would be patched; pass --apply to write back.')

if __name__ == '__main__':
    main()
