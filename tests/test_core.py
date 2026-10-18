from core import get_loan, evidence_review, split_policy

def test_mock_loan():
    assert get_loan('LN-10045')['employment_type']=='SALARIED'

def test_stale_demo_statement():
    assert evidence_review(get_loan('LN-10046'))['demo_recency_flags']

def test_policy_sections():
    assert len(split_policy('semantic'))>=7
    assert len(split_policy('fixed'))>=7
