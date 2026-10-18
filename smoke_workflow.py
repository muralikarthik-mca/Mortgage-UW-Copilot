"""Live backend smoke check; use verify_ui.py for browser interaction checks."""
import json
from datetime import datetime, timezone
import sys

from streamlit.testing.v1 import AppTest
from core import ROOT, index_policies, ollama_status
from chat_service import context_id, review_request


def main():
    status = ollama_status()
    if status['missing']:
        raise RuntimeError(f"Missing models: {status['missing']}")
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    assert not app.exception, app.exception
    app.selectbox(key='selected_loan_id').select('LN-10046').run()
    assert not app.exception
    index_policies('semantic', 'fannie_mae')
    context = {'loan_id': 'LN-10046', 'corpus': 'fannie_mae', 'strategy': 'semantic', 'rerank': True, 'view': 'Dashboard'}
    result = review_request(context, 'Review the income documentation and identify outstanding items.', [])
    assert result['loan_id'] == 'LN-10046'
    assert result['facts']['record']['income_status'] == 'UNVERIFIED'
    assert not result['facts']['demo_recency_flags']
    assert all(p['corpus'] == 'fannie_mae' for p in result['policies'])
    assert result['citation_check_passed'], result['draft']
    output = ROOT / 'artifacts'
    output.mkdir(exist_ok=True)
    evidence = {'checked_at': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
                'ollama': status, 'context_id': context_id(context), 'result': result}
    (output / 'workflow-verification.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'loan_id':result['loan_id'],'citation_check_passed':True,'python':sys.executable},indent=2))


if __name__ == '__main__':
    main()
