from pathlib import Path
import json
from src.data import ROOT, CLASSES, create_or_load_split

rows, paths = create_or_load_split()
print(f"metadata rows={len(rows)} resolved images={len(rows)} available JPGs={len(paths)} missing=0 duplicate mappings=0")
print(rows.split.value_counts().to_string())
(ROOT / "data/splits/label_mapping.json").write_text(json.dumps({name: i for i, name in enumerate(CLASSES)}, indent=2), encoding="utf-8")
