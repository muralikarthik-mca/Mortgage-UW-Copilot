"""Fetch official topic bodies and produce width-safe PDFs; originals are retained.

Optional maintenance command (requires requirements-dev.txt and installed Edge).
No source downloads or browser execution occur during normal application startup.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import requests
from pypdf import PdfReader
from playwright.sync_api import sync_playwright

DIRECTORY = Path(__file__).parent / 'data/guidelines/agency/fannie_mae'


def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def refresh():
    manifest_path = DIRECTORY / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    output = DIRECTORY / 'verified'
    output.mkdir(exist_ok=True)
    # Validate every topic before replacing any active source file. A network or
    # completeness failure leaves the previous working snapshot untouched.
    staging_context = TemporaryDirectory(prefix='.refresh-', dir=DIRECTORY)
    staging = Path(staging_context.name)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page()
        page.route('**/*', lambda route: route.abort())
        for filename, entry in manifest.items():
            url = entry['source_url']
            response = requests.get(url, timeout=(10, 40))
            response.raise_for_status()
            if not response.url.startswith('https://selling-guide.fanniemae.com/sel/'):
                raise ValueError('Unexpected official-source redirect')
            raw = response.content
            safe = re.sub(r'<script\b[^>]*>.*?</script>', '', response.text, flags=re.S|re.I)
            page.set_content(safe, wait_until='domcontentloaded')
            topic = page.locator('#topic_content')
            if topic.count() != 1:
                raise ValueError(f'{filename}: missing unambiguous topic body')
            title = topic.locator('h1').inner_text().strip()
            if not title.startswith(entry['section_id']):
                raise ValueError('Source section changed; review manually')
            revision = re.search(r'\((\d{2}/\d{2}/\d{4})\)', title).group(1)
            duplicate_links = topic.locator('a.pdf-internal-link').count()
            topic.locator('a.pdf-internal-link').evaluate_all('(nodes)=>nodes.forEach(n=>n.remove())')
            topic.locator('script,style').evaluate_all('(nodes)=>nodes.forEach(n=>n.remove())')
            topic.locator('a[href]').evaluate_all('(nodes)=>nodes.forEach(n=>{const u=n.getAttribute("href");if(u.startsWith("/"))n.href="https://selling-guide.fanniemae.com"+u;})')
            body_text = topic.inner_text()
            body_html = topic.inner_html()
            stem = Path(filename).stem
            raw_path = staging / f'{stem}.source.html'
            raw_path.write_bytes(raw)
            html_path = staging / f'{stem}.article.html'
            html_path.write_text(body_html, encoding='utf-8')
            print_html = '''<!doctype html><meta charset="utf-8"><style>
                @page { size:A4; margin:14mm; }
                body {font:11px/1.5 Arial,sans-serif;color:#111;max-width:100%;}
                h1{font-size:18px;}h4{font-size:14px;break-after:avoid;}
                table{width:100%;table-layout:auto;border-collapse:collapse;margin:10px 0;}
                td,th{border:1px solid #aaa;padding:5px;vertical-align:top;overflow-wrap:normal;word-break:normal;hyphens:none;}
                td:first-child,th:first-child{min-width:110px;}
                tr{break-inside:avoid;}p,li{orphans:3;widows:3;}a{color:#173c68;}
                .source{font-size:8px;white-space:nowrap;}
                </style>'''
            page.set_content(print_html + '<p class="source">' + escape(url) + '</p>' + body_html,
                             wait_until='domcontentloaded')
            pdf_path = staging / filename
            page.pdf(path=str(pdf_path), format='A4', print_background=True, prefer_css_page_size=True)
            reader = PdfReader(pdf_path)
            pdf_text = '\n'.join(pg.extract_text() or '' for pg in reader.pages)
            expected, actual = Counter(words(body_text)), Counter(words(pdf_text))
            missing = expected - actual
            coverage = 1 - sum(missing.values()) / max(sum(expected.values()), 1)
            if coverage < 1.0:
                raise ValueError(f'{filename}: PDF body token coverage {coverage:.3f}; manifest not updated')
            entry['active_file'] = f'verified/{filename}'
            entry['active_sha256'] = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            entry['active_quality'] = 'complete_topic_body'
            entry['topic_revision_date'] = revision
            entry['effective_date'] = None
            entry['version'] = revision
            entry['version_basis'] = 'Official topic heading; not a policy effective date'
            entry['verified_at'] = datetime.now(timezone.utc).isoformat()
            entry['raw_html_file'] = f'verified/{stem}.source.html'
            entry['raw_html_sha256'] = hashlib.sha256(raw).hexdigest()
            entry['article_html_file'] = f'verified/{stem}.article.html'
            entry['article_html_sha256'] = hashlib.sha256(html_path.read_bytes()).hexdigest()
            entry['body_token_coverage'] = coverage
            entry['missing_body_tokens'] = dict(missing)
            entry['active_page_count'] = len(reader.pages)
            entry['transformations'] = [f'Removed {duplicate_links} duplicate PDF-only cross-reference links',
                'Kept complete #topic_content including notes, tables, exceptions, and announcements',
                'Rendered width-safe PDF from official body; no source sentences rewritten']
            entry['quality_warning'] = ('Original browser-print PDF clipped; retained unchanged for audit. '
                'Active PDF is a verified complete topic-body rendering, not the complete Selling Guide. '
                'Referenced topics and effective-date applicability require separate review.')
            print(json.dumps({'file': filename, 'revision': revision, 'pages': len(reader.pages),
                              'body_token_coverage': coverage, 'missing_tokens': dict(missing)}), flush=True)
        browser.close()
    for staged_file in staging.iterdir():
        staged_file.replace(output / staged_file.name)
    temporary_manifest = manifest_path.with_suffix('.tmp')
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    temporary_manifest.replace(manifest_path)
    staging_context.cleanup()


if __name__ == '__main__':
    refresh()
