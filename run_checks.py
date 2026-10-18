"""Capture real unit/integration test results using a fresh writable pytest temp."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true', help='Read current measured results without running tests or models')
    parser.add_argument('--browser', action='store_true', help='Also run the existing end-to-end browser checks')
    parser.add_argument('--diagnose', action='store_true', help='Inspect five failed answer cases without model calls')
    parser.add_argument('--probe', action='store_true', help='Measure five representative answers only')
    args = parser.parse_args()
    if args.diagnose or args.probe:
        from rag_evaluation import load_cases, REPORT
        import core
        from chat_service import review_request
        from evaluation_metrics import answer_rubric, relevant_ids
        cases = [c for c in load_cases() if c['id'] in {'A01','A05','A08','A12','T04'}]
        old = json.loads(REPORT.read_text(encoding='utf-8'))
        measured = []
        for case in cases:
            chunks = json.loads(core._index_path('semantic',case['corpus']).read_text(encoding='utf-8'))['chunks']
            gold = relevant_ids(case,chunks)
            if args.diagnose:
                for mode in ('dense','hybrid_rerank'):
                    row = next(r for r in old['retrieval_results'] if r['id']==case['id'] and r['strategy']=='semantic' and r['mode']==mode)
                    print(json.dumps({'case':case['id'],'question':case['question'],'mode':mode,'chunks':[
                        {'id':i,'gold':i in gold,'text':next(c['text'] for c in chunks if c['id']==i)} for i in row['hit_ids']]},ensure_ascii=False),flush=True)
            else:
                result = review_request({'loan_id':case['loan_id'],'corpus':case['corpus'],'strategy':'semantic','rerank':False,'retrieval_mode':'dense','view':'RAG Evaluation'},case['question'],[])
                metrics = answer_rubric(case,result,chunks,core.get_loan(case['loan_id']))
                measured.append({'id':case['id'],'question':case['question'],'metrics':metrics,'answer':result['answer'],'selected_evidence':result['selected_evidence']})
                print(json.dumps({'id':case['id'],'metrics':metrics}),flush=True)
        if args.probe:
            (ROOT/'reports/answer_relevance_probe.json').write_text(json.dumps(measured,indent=2,ensure_ascii=False),encoding='utf-8')
        return
    if args.report:
        from rag_evaluation import REPORT, cached_report
        measured = json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else {}
        print(json.dumps({'current_code_matches_report': cached_report() is not None,
            'answers_complete': measured.get('answers_complete', False),
            'summary': measured.get('summary', []),
            'answers': [{'id': a['id'], 'metrics': a['metrics'], 'model_invoked': a.get('model_invoked')} for a in measured.get('answers', [])]},indent=2))
        return
    temporary = Path(tempfile.gettempdir()) / ('mortgage-uw-checks-' + uuid.uuid4().hex)
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp', str(temporary)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    output = result.stdout + result.stderr
    print(output, end='')
    counts = {label: int(number) for number, label in re.findall(r'(\d+) (passed|failed|skipped|errors?)', output)}
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
              'exit_code': result.returncode, 'counts': counts, 'command': command, 'output': output}
    destination = ROOT/'reports'
    destination.mkdir(exist_ok=True)
    (destination/'test_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (destination/'test_results.txt').write_text(output,encoding='utf-8')
    if args.browser and result.returncode == 0:
        browser = subprocess.run([sys.executable, '-X', 'utf8', 'verify_ui.py'], cwd=ROOT,
                                 capture_output=True, text=True, encoding='utf-8', errors='replace')
        print(browser.stdout + browser.stderr)
        (destination/'browser_test_results.txt').write_text(browser.stdout+browser.stderr,encoding='utf-8')
        if browser.returncode == 0:
            (destination/'browser_test_results.json').write_text((ROOT/'artifacts/ui-verification.json').read_text(encoding='utf-8'),encoding='utf-8')
        raise SystemExit(browser.returncode)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
