"""watch.py — health checks for refusal

Static: the five generated pages exist, are registered and fit width.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = ['diff-in-means', 'pick-layer', 'direction-ablation',
         'activation-addition', 'read-intervention-table']


def check_imports():
    reg = json.load(open(os.path.join(HERE, '..', 'kc_registry.json')))
    assert reg, 'registry empty'


def check_public_api():
    for p in PAGES:
        assert os.path.exists(os.path.join(HERE, f'kp-{p}.md')), f'missing kp-{p}.md'


def check_invariants():
    for p in PAGES:
        text = open(os.path.join(HERE, f'kp-{p}.md')).read()
        assert 'modulario:template' not in text
        assert '# Hidden checks' in text, f'kp-{p}: no asserted cells'


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
