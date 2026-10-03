"""Safe file IO and shared protocol definitions (no image loading)."""
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc')
SPLITS = {'strict_lesion_disjoint': 'data/splits/split_assignments.csv',
          'exploratory_image_level': 'data/splits/exploratory/image_level_dev_v1.csv'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def write_json(path, data):
    atomic_text(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows, fields=None):
    import io
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, out.getvalue())
