"""Replay/check the three selective fallbacks discovered in weighted bank B.

Run from repository root with the project environment/oracle dependencies.
This is supplementary validation, not replacement of any generated draw.
"""
import hashlib
import time
from pathlib import Path
from scripts.followup_shifted_student_reference import global_mle, oracle_check, save, load
import numpy as np

out = Path('docs/research/evidence/shifted_student_followup_2026-10-10')
raw = Path('tmp/shifted_student_followup_20261010')
cfg = load(out/'config.json')
bank = np.load(raw/'weighted_b.npz')
base, saved_mle = bank['base'], bank['mle']
checks = []
start = time.monotonic()
for i, data in enumerate(base):
    value, diag = global_mle(data, 2, return_diagnostics=True)
    if value != saved_mle[i]:
        raise ArithmeticError('MLE replay not identical')
    if diag['fallback']:
        row = oracle_check(data, cfg['case'], 'weighted_b_exact_fallback', i)
        row['diagnostics'] = diag
        checks.append(row)
        save(out/'fallback_oracle_checks.json', checks)
        print('fallback', i, row.get('selected_location_error'), row.get('error'), flush=True)
save(out/'fallback_oracle_checks.json', checks)
save(out/'fallback_oracle_execution.json', {
    'checked_replay': len(saved_mle), 'fallback_checks': len(checks),
    'seconds': time.monotonic()-start,
    'bank_sha256': hashlib.sha256((raw/'weighted_b.npz').read_bytes()).hexdigest(),
    'reason': 'Unexpected selective fallbacks in weighted bank B; independently check every such selection.',
})
print('complete', len(checks), flush=True)
