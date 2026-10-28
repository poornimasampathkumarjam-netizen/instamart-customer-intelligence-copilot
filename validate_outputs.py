"""Validate the committed Part B–D result artifacts after run_all.py."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main() -> None:
    out = ROOT / 'outputs'
    segments = pd.read_csv(out / 'customer_segments.csv')
    retention = pd.read_csv(out / 'retention_list.csv')
    triage = pd.read_csv(out / 'triage_results.csv')
    budget = pd.read_csv(out / 'triage_budget_demo.csv')
    zero = pd.read_csv(out / 'triage_zero_shot.csv')
    diagnostics = json.loads((out / 'model_diagnostics.json').read_text())
    triage_summary = json.loads((out / 'triage_summary.json').read_text())
    rag = json.loads((out / 'rag_eval_summary.json').read_text())
    runs = [json.loads(line) for line in (out / 'agent_runs.jsonl').read_text().splitlines() if line.strip()]
    resume = json.loads((out / 'resume_demo.json').read_text())

    assert len(segments) == 1175 and segments.customer_id.is_unique
    premium_by_segment = segments.groupby('segment_id').is_premium.first()
    assert int(premium_by_segment.sum()) == 1
    assert segments.groupby('segment_id').is_premium.nunique().max() == 1
    assert len(retention) <= 10 and list(retention.columns) == ['customer_id', 'lapse_probability', 'actually_lapsed']
    assert diagnostics['cohort_size'] == 1156 and diagnostics['lapsed_count'] == 443
    assert diagnostics['train_size'] == 924 and diagnostics['test_size'] == 232
    assert diagnostics['logistic_regression']['test']['f1'] >= 0.55
    assert len(triage) == len(budget) == len(zero) == 164
    assert triage_summary['zero_shot']['classified'] == 147 and triage_summary['few_shot']['classified'] == 147
    assert triage_summary['zero_shot']['retry_errors_recovered'] == 19
    assert triage_summary['few_shot']['retry_errors_recovered'] == 19
    assert triage_summary['budget_demo']['budget_exhausted'] > 0
    assert (triage.status == 'classified_json').sum() == 135
    assert (triage.status == 'classified_keyword').sum() == 12
    assert (triage.status == 'flagged_invalid_output').sum() == 9
    assert (triage.status == 'skipped_empty').sum() == 4 and (triage.status == 'flagged_long').sum() == 4
    assert rag['overlap_10'] == {'expected_doc_hits': 12, 'expected_fact_matches': 12, 'fallbacks': 2}
    assert rag['overlap_0'] == {'expected_doc_hits': 12, 'expected_fact_matches': 10, 'fallbacks': 2}
    expected = {'S1':'REFUND','S2':'REFUND','S3':'ESCALATE','S4':'DENIED','S5':'DENIED','S6':'NOT_FOUND','S7':'FALLBACK_MAX_ITERATIONS'}
    assert len(runs) == 7 and {r['scenario_id']:r['status'] for r in runs} == expected
    assert resume['tools_executed'] == ['get_order','search_policy','calculate_refund','issue_refund']
    assert resume['status'] == 'REFUND' and 'INR 128.25' in resume['final_answer']
    assert (ROOT / 'state/S1_resume_demo.json').exists()
    print('ALL OUTPUT CHECKS PASSED')

if __name__ == '__main__':
    main()
