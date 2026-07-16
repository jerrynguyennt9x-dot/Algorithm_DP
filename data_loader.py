"""
================================================================================
MODULE: data_loader.py
MO TA: Ham doc va tien xu ly du lieu cho bai toan Knapsack 0/1
       Ho tro 3 bo du lieu: small (n=20, W=50), medium (n=50, W=200),
                              large  (n=100, W=500)
================================================================================
"""

import csv
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

BASE_DIR = Path(__file__).resolve().parent

_DEFAULT_FILES = {
    "small":  BASE_DIR / "dt02_knapsack_small.csv",
    "medium": BASE_DIR / "dt02_knapsack_medium.csv",
    "large":  BASE_DIR / "dt02_knapsack_large.csv",
}


@dataclass
class KnapsackItem:
    item_id: int
    weight:  int
    value:   int
    ratio:   float = 0.0

    def __post_init__(self):
        self.ratio = self.value / self.weight if self.weight > 0 else 0.0


@dataclass
class KnapsackDataset:
    scale:    str
    source:   str
    W:        int
    items:    List[KnapsackItem]
    stats:    Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def item_ids(self): return [it.item_id for it in self.items]
    @property
    def weights(self):  return [it.weight  for it in self.items]
    @property
    def values(self):   return [it.value   for it in self.items]
    @property
    def n(self):        return len(self.items)


def _parse_csv(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Khong tim thay file: {filepath}")
    items, W, data_lines = [], None, []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        raw_lines = f.readlines()
    for line in raw_lines:
        s = line.strip()
        if not s: continue
        m = re.search(r"#\s*W\s*=\s*(\d+)", s, re.IGNORECASE)
        if m:
            W = int(m.group(1)); continue
        if s.startswith("#"): continue
        data_lines.append(s)
    if W is None:
        raise ValueError(f"File '{filepath.name}' khong co dong '# W=<so>'.")
    reader = csv.DictReader(data_lines)
    for row in reader:
        items.append(KnapsackItem(int(row["item_id"]), int(row["weight"]), int(row["value"])))
    scale = "custom"
    for k, p in _DEFAULT_FILES.items():
        if filepath.resolve() == p.resolve():
            scale = k; break
    return KnapsackDataset(scale=scale, source=str(filepath.resolve()), W=W, items=items)


def validate_dataset(ds):
    warnings = []
    if ds.W <= 0:
        warnings.append(f"[LOI] Ngan sach W={ds.W} khong hop le.")
    seen_ids, infeasible = {}, []
    for i, it in enumerate(ds.items):
        if it.weight <= 0: warnings.append(f"[LOI] item_id={it.item_id} weight={it.weight}<=0.")
        if it.value  <= 0: warnings.append(f"[CANH BAO] item_id={it.item_id} value={it.value}<=0.")
        if it.item_id in seen_ids:
            warnings.append(f"[LOI] item_id={it.item_id} bi trung (dong {seen_ids[it.item_id]+1} va {i+1}).")
        seen_ids[it.item_id] = i
        if it.weight > ds.W: infeasible.append(it.item_id)
    if infeasible:
        warnings.append(f"[INFO] {len(infeasible)} item co weight > W (se bi loai): IDs={infeasible}")
    if not any(it.weight <= ds.W for it in ds.items):
        warnings.append("[LOI] Khong co item nao kha thi (tat ca weight > W).")
    ds.is_valid = not any(w.startswith("[LOI]") for w in warnings)
    ds.warnings = warnings
    return ds


def preprocess(ds):
    for it in ds.items:
        it.ratio = it.value / it.weight if it.weight > 0 else 0.0
    feasible = [it for it in ds.items if it.weight <= ds.W]
    def _s(lst):
        if not lst: return {"min":0,"max":0,"mean":0,"sum":0,"count":0}
        return {"min":min(lst),"max":max(lst),"mean":sum(lst)/len(lst),"sum":sum(lst),"count":len(lst)}
    ws = [it.weight for it in feasible]
    vs = [it.value  for it in feasible]
    rs = [it.ratio  for it in feasible]
    ds.stats = {
        "n_total":       len(ds.items),
        "n_feasible":    len(feasible),
        "n_infeasible":  len(ds.items)-len(feasible),
        "W":             ds.W,
        "weight": _s(ws), "value": _s(vs), "ratio": _s(rs),
        "total_weight_all":      sum(it.weight for it in ds.items),
        "total_weight_feasible": sum(ws),
        "best_ratio_item":   max(feasible, key=lambda x: x.ratio)  if feasible else None,
        "highest_value_item":max(feasible, key=lambda x: x.value)  if feasible else None,
    }
    return ds


def summary(ds):
    sep = "=" * 65
    print(sep)
    print(f"  BO DU LIEU: {ds.scale.upper()}  |  Nguon: {Path(ds.source).name}")
    print(sep)
    print(f"  So du an (n)   : {ds.n}")
    print(f"  Ngan sach (W)  : {ds.W}")
    print(f"  Hop le         : {'CO' if ds.is_valid else 'KHONG'}")
    if ds.warnings:
        print("\n  -- Canh bao / Ghi chu --")
        for w in ds.warnings: print(f"    {w}")
    if ds.stats:
        st = ds.stats
        print("\n  -- Thong ke (item kha thi) --")
        print(f"    Kha thi / Tong : {st['n_feasible']} / {st['n_total']}")
        print(f"    Weight -> min={st['weight']['min']}, max={st['weight']['max']}, mean={st['weight']['mean']:.1f}, sum={st['weight']['sum']}")
        print(f"    Value  -> min={st['value']['min']},  max={st['value']['max']},  mean={st['value']['mean']:.1f}, sum={st['value']['sum']}")
        print(f"    Ratio  -> min={st['ratio']['min']:.2f}, max={st['ratio']['max']:.2f}, mean={st['ratio']['mean']:.2f}")
        if st.get("best_ratio_item"):
            br = st["best_ratio_item"]
            print(f"    Item ratio cao nhat: ID={br.item_id}, w={br.weight}, v={br.value}, ratio={br.ratio:.2f}")
        if st.get("highest_value_item"):
            hv = st["highest_value_item"]
            print(f"    Item value cao nhat: ID={hv.item_id}, w={hv.weight}, v={hv.value}, ratio={hv.ratio:.2f}")
    print(f"\n  -- 5 item dau tien --")
    print(f"  {'ID':>4}  {'Weight':>7}  {'Value':>6}  {'Ratio':>7}")
    print(f"  {'-'*33}")
    for it in ds.items[:5]:
        print(f"  {it.item_id:>4}  {it.weight:>7}  {it.value:>6}  {it.ratio:>7.2f}")
    if ds.n > 5: print(f"  ... ({ds.n-5} dong con lai)")
    print(sep)


def load_from_csv(filepath, W=None, run_validate=True, run_preprocess=True):
    ds = _parse_csv(filepath)
    if W is not None: ds.W = W
    if run_validate:   validate_dataset(ds)
    if run_preprocess: preprocess(ds)
    return ds


def load_dataset(scale="small", run_validate=True, run_preprocess=True):
    scale = scale.strip().lower()
    if scale not in _DEFAULT_FILES:
        raise ValueError(f"scale='{scale}' khong hop le. Chon: {list(_DEFAULT_FILES.keys())}")
    return load_from_csv(_DEFAULT_FILES[scale], run_validate=run_validate, run_preprocess=run_preprocess)


def load_all_datasets(run_validate=True, run_preprocess=True):
    return {s: load_dataset(s, run_validate, run_preprocess) for s in _DEFAULT_FILES}


if __name__ == "__main__":
    print("\n" + "#" * 65)
    print("  KIEM TRA TIEN XU LY DU LIEU – CA 3 BO KNAPSACK")
    print("#" * 65 + "\n")
    scales = list(_DEFAULT_FILES.keys())
    if len(sys.argv) > 1 and sys.argv[1] in scales:
        scales = [sys.argv[1]]
    for scale in scales:
        try:
            ds = load_dataset(scale)
            summary(ds)
            print()
        except FileNotFoundError as e:
            print(f"[LOI] {e}\n")
