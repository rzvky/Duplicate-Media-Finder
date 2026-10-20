"""Compare scan results with the answer key from make_dataset.py.

Usage: python tests/evaluate.py DATASET_DIR [--verbose]
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path


def norm(p: str) -> str:
    return p.replace("\\", "/")


def evaluate(dataset: Path, verbose: bool = False) -> dict:
    dataset = Path(dataset)
    truth = json.loads((dataset / "truth.json").read_text(encoding="utf-8"))
    res = json.loads((dataset / "photos" / "_DupeFinder" / "results.json").read_text(encoding="utf-8"))
    group_of, role_of = {}, {}
    for g in res["groups"]:
        for m in g["files"]:
            group_of[norm(m["rel"])] = g["id"]
            role_of[norm(m["rel"])] = m["role"]

    # pairs that should be together
    should = set()
    for files in truth["families"].values():
        rels = sorted(f["rel"] for f in files)
        should.update(itertools.combinations(rels, 2))
    found = set()
    for g in res["groups"]:
        rels = sorted(norm(m["rel"]) for m in g["files"])
        found.update(itertools.combinations(rels, 2))
    tp = len(should & found)
    precision = tp / len(found) if found else 1.0
    recall = tp / len(should) if should else 1.0

    problems = []
    for a, b in sorted(should - found):
        problems.append(f"MISSED pair: {a} <-> {b}")
    for a, b in sorted(found - should):
        problems.append(f"WRONG pair: {a} <-> {b}")

    # role checks
    expect_role = {
        "exact": {"exact"},
        "lower": {"lower"},
        "edit": {"similar"},
        "rawpair": {"rawpair", "keep"},
        "burst_blurry": {"similar"},
        "original": {"keep", "similar", "rawpair"},
        "burst_sharp": {"keep", "similar"},
    }
    role_ok = role_total = 0
    for files in truth["families"].values():
        for f in files:
            r = role_of.get(f["rel"])
            if r is None:
                continue
            role_total += 1
            if r in expect_role.get(f["kind"], {r}):
                role_ok += 1
            else:
                problems.append(f"ROLE {f['rel']}: got {r}, expected {sorted(expect_role[f['kind']])}")
        kinds = {f["kind"] for f in files}
        if "original" in kinds and not kinds & {"burst_sharp", "rawpair", "edit"}:
            orig = [f["rel"] for f in files if f["kind"] == "original"][0]
            if role_of.get(orig) not in (None, "keep"):
                problems.append(f"KEEPER: original {orig} not chosen as keeper ({role_of.get(orig)})")
        if "burst_blurry" in kinds:
            blurry = [f["rel"] for f in files if f["kind"] == "burst_blurry"][0]
            if role_of.get(blurry) == "keep":
                problems.append(f"KEEPER: blurry burst shot chosen as keeper {blurry}")

    errs = {norm(e["rel"]) for e in res["errors"]}
    for e in truth["errors"]:
        if e not in errs:
            problems.append(f"ERROR not reported: {e}")
    scanned = set(group_of) | errs
    for s in truth["skipped"]:
        if s in scanned:
            problems.append(f"SHOULD BE SKIPPED: {s}")
    for d in truth.get("damaged_ok", []):
        if d in errs:
            problems.append(f"DAMAGED file not readable: {d}")
    out = {
        "pairs_expected": len(should),
        "pairs_found": len(found),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "roles_ok": f"{role_ok}/{role_total}",
        "problems": problems,
        "groups": len(res["groups"]),
        "errors": len(res["errors"]),
        "ai_used": res["ai_used"],
    }
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit("Usage: python tests/evaluate.py DATASET_DIR [--verbose]")
    r = evaluate(Path(sys.argv[1]), "--verbose" in sys.argv)
    probs = r.pop("problems")
    print(json.dumps(r, indent=1))
    for p in probs[:60]:
        print(" ", p)
    sys.exit(0 if not probs else 1)
