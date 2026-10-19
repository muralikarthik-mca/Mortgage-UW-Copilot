"""Project-scoped Pinecone setup and live verification; never prints credentials."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess

from dotenv import dotenv_values, set_key

ROOT = Path(__file__).resolve().parent


def migrate():
    import core
    from vector_store import upsert_snapshot
    os.environ['VECTOR_BACKEND'] = 'pinecone'
    migrated = []
    for strategy in ('fixed', 'semantic'):
        for corpus in ('training', 'fannie_mae'):
            path = core._index_path(strategy, corpus)
            payload = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            reusable = (payload.get('schema') == 2 and payload.get('embedding_model') == core.EMBED_MODEL
                and payload.get('source_signature') == core._source_signature(corpus)
                and payload.get('chunking_version') == core.CHUNKING_VERSION
                and payload.get('fingerprint') == core._fingerprint(payload.get('chunks', [])))
            if not reusable:
                core.index_policies(strategy, corpus)
                payload = json.loads(path.read_text(encoding='utf-8'))
            else:
                payload['pinecone'] = upsert_snapshot(payload, corpus, strategy)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
                temporary.replace(path)
            row = {'corpus':corpus, 'strategy':strategy, 'reused_embeddings':reusable, **payload['pinecone']}
            migrated.append(row)
            print(json.dumps(row), flush=True)
    (ROOT/'reports/pinecone_migration.json').write_text(json.dumps(migrated,indent=2)+'\n', encoding='utf-8')


def verify():
    from datetime import datetime, timezone
    import core
    from rag_evaluation import load_cases
    rows = []
    cases = [c for c in load_cases() if c['id'] in {'A01','A05','A08','A12','T04'}]
    for case in cases:
        for mode in ('dense','hybrid','hybrid_rerank'):
            os.environ['VECTOR_BACKEND'] = 'local'
            local = core.retrieve_detailed(case['question'], 'semantic', case['corpus'], 5, mode)
            os.environ['VECTOR_BACKEND'] = 'pinecone'
            remote = core.retrieve_detailed(case['question'], 'semantic', case['corpus'], 5, mode)
            row = {'case':case['id'],'mode':mode,
                'local_ids':[h['id'] for h in local['hits']], 'pinecone_ids':[h['id'] for h in remote['hits']],
                'pinecone_latency_ms':remote['latency_ms'], 'local_latency_ms':local['latency_ms']}
            row['same_top_five'] = row['local_ids'] == row['pinecone_ids']
            rows.append(row)
            print(json.dumps({'case':case['id'],'mode':mode,'same_top_five':row['same_top_five']}), flush=True)
    (ROOT/'reports/pinecone_verification.json').write_text(json.dumps({
        'measured_at':datetime.now(timezone.utc).isoformat(),'results':rows},indent=2)+'\n',encoding='utf-8')
    if not all(r['same_top_five'] for r in rows):
        raise RuntimeError('Database rankings differ from the local baseline. Inspect the measured verification report before activating.')


def secure_configuration():
    """Recover a key accidentally placed in the public template, without logging it."""
    template = ROOT / '.env.example'
    private = ROOT / '.env'
    public_values = dotenv_values(template)
    private_values = dotenv_values(private)
    value = public_values.get('PINECONE_API_KEY', '') or ''
    is_key = value.startswith(('pcsk_', 'pck_'))
    if is_key and not private_values.get('PINECONE_API_KEY'):
        set_key(str(private), 'PINECONE_API_KEY', value)
    if is_key:
        text = template.read_text(encoding='utf-8')
        template.write_text(re.sub(r'(?m)^PINECONE_API_KEY=.*$', 'PINECONE_API_KEY=', text), encoding='utf-8')
    print(json.dumps({'private_key_configured': bool(dotenv_values(private).get('PINECONE_API_KEY')),
                      'template_key_removed': is_key}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--migrate', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--preserve-report', action='store_true')
    parser.add_argument('--check-security', action='store_true')
    args = parser.parse_args()
    if args.configure:
        secure_configuration()
    if args.check_security:
        value = dotenv_values(ROOT/'.env.example').get('PINECONE_API_KEY')
        tracked = subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',
                                 'ls-files','--cached','--others','--exclude-standard'],
                                cwd=ROOT,capture_output=True,text=True,check=True).stdout.splitlines()
        credential_pattern = re.compile(rb'pcsk_[A-Za-z0-9_-]{25,}')
        leaked = []
        for name in tracked:
            path = ROOT/name
            if path.is_file() and credential_pattern.search(path.read_bytes()):
                leaked.append(name)
        if value or '.env' in tracked or leaked:
            # Filenames only. Never print matching text or credential fragments.
            raise RuntimeError('Credential protection check failed. Files: ' + ', '.join(leaked))
        print('Credential check passed: template is blank, .env is ignored, no Pinecone key found in publishable files.')
    if args.preserve_report:
        import shutil
        for suffix in ('json','md'):
            source = ROOT/f'reports/rag_comparison.{suffix}'
            destination = ROOT/f'reports/pre_pinecone_rag_comparison.{suffix}'
            if source.exists() and not destination.exists():
                shutil.copyfile(source,destination)
        print('Preserved pre-migration measurements separately from Pinecone results.')
    if args.inspect:
        import inspect
        import core
        from pinecone import Pinecone
        print(json.dumps({'sdk_version': importlib.metadata.version('pinecone'),
            'api_key_configured': bool(__import__('os').getenv('PINECONE_API_KEY')),
            'client_methods': {n: str(inspect.signature(getattr(Pinecone, n))) for n in
                ('create_index','describe_index','Index') if hasattr(Pinecone,n)},
            'baseline_tests': json.loads((ROOT/'reports/test_results.json').read_text())['counts']}))
    if args.migrate:
        migrate()
    if args.verify:
        verify()
    if args.activate:
        results = json.loads((ROOT/'reports/pinecone_verification.json').read_text())['results']
        if not results or not all(r['same_top_five'] for r in results):
            raise RuntimeError('Run a successful --verify before activating Pinecone.')
        set_key(str(ROOT/'.env'), 'VECTOR_BACKEND', 'pinecone')
        print('Pinecone selected in local .env; restart Streamlit to apply configuration.')


if __name__ == '__main__':
    main()
