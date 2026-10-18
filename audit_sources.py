"""Offline source audit over archived originals and verified official snapshots."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from pypdf import PdfReader
from core import AGENCY, ROOT


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def audit():
    manifest = json.loads((AGENCY/'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for filename, meta in manifest.items():
        for file_key, hash_key in [('active_file','active_sha256'), ('raw_html_file','raw_html_sha256'), ('article_html_file','article_html_sha256')]:
            if sha256((AGENCY/meta[file_key]).read_bytes()).hexdigest() != meta[hash_key]:
                raise ValueError(f'Provenance mismatch: {meta[file_key]}')
        original = AGENCY/filename
        if sha256(original.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Original PDF changed')
        original_text = '\n'.join(pg.extract_text() or '' for pg in PdfReader(original).pages)
        parsed = Text()
        parsed.feed((AGENCY/meta['article_html_file']).read_text(encoding='utf-8'))
        words = lambda s: re.findall(r'[a-z0-9]+', s.lower())
        expected = Counter(words(' '.join(parsed.parts)))
        missing = expected - Counter(words(original_text))
        original_coverage = 1 - sum(missing.values())/max(sum(expected.values()), 1)
        first_heading = re.search(r'(?m)^B\d+-[^\n]+', original_text)
        revision = re.search(r'\d{2}/\d{2}/\d{4}', first_heading.group() if first_heading else '')
        rows.append({'file': filename, 'original_sha256': meta['sha256'], 'original_quality': 'clipped/incomplete browser-print snapshot',
            'original_body_token_coverage': original_coverage,
            'readable_original_revision': revision.group() if revision else None,
            'verified_topic_revision': meta['topic_revision_date'], 'effective_date': None,
            'outdated_status': ('no revision change detected at verification' if revision and revision.group() == meta['topic_revision_date']
                                else 'original heading incomplete or different; rely on verified topic revision'),
            'active_file': meta['active_file'], 'active_sha256': meta['active_sha256'],
            'active_quality': meta['active_quality'], 'active_body_token_coverage': meta['body_token_coverage'],
            'source_url': meta['source_url'], 'verified_at': meta['verified_at'],
            'quality_warning': meta['quality_warning']})
    report = {'audited_at': datetime.now(timezone.utc).isoformat(), 'sources': rows,
        'method': 'Archived original PDF vs complete official #topic_content; original HTML and article HTML hashes retained. Width-safe active PDFs must cover 100% of body tokens. Token coverage does not establish table reading order, legal completeness of the entire guide, or effective-date applicability.',
        'coverage_gaps': ['Multiple-job agency topic', 'Self-employment calculation topics', 'Detailed referenced VOE and DU validation topics', 'FHA and other agencies']}
    output = ROOT/'reports'
    output.mkdir(exist_ok=True)
    (output/'source_quality.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    audit()
