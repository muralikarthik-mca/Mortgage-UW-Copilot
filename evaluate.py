"""Small labeled retrieval evaluation; fictional policies only, no eligibility scoring."""
import json
from core import index_policies, retrieve

CASES = [
    ('What is the fictional pay statement recency threshold?', 'TR-INC-02'),
    ('What evidence is needed for bonus income?', 'TR-INC-04'),
    ('How should income from multiple jobs be reviewed?', 'TR-INC-05'),
    ('How is self-employment tax information reviewed?', 'TR-INC-06'),
]


def run_evaluation():
    results = []
    for strategy in ('semantic', 'fixed'):
        count = index_policies(strategy, 'training')
        hits = [expected in {p['section_id'] for p in retrieve(question, strategy, True, 'training', k=3)} for question, expected in CASES]
        results.append({'strategy': strategy, 'chunks': count, 'recall_at_3': sum(hits)/len(hits), 'cases': len(hits)})
    return results


def main():
    print(json.dumps(run_evaluation(), indent=2))


if __name__ == '__main__':
    main()
