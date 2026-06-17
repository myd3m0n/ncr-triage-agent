import json
from collections import Counter
from pathlib import Path
from classifier import classify
ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "eval" / "eval_set.json"
def main():
    cases = json.loads(EVAL_FILE.read_text())
    total = len(cases); correct = 0; misses = []
    print(f"Running eval on {total} cases...\n")
    for i, case in enumerate(cases, 1):
        predicted = classify(case["input"])["defect_type"]
        expected = case["expected_defect_type"]
        ok = predicted == expected; correct += ok
        print(f"{'OK ' if ok else 'XX '}[{i:>2}] {case['id']}  pred={predicted!r}  exp={expected!r}")
        if not ok: misses.append((expected,))
    acc = correct/total if total else 0
    print(f"\n=== Accuracy: {correct}/{total} = {acc:.0%} ===")
    if misses:
        print("\nMisses by expected type:")
        for t, n in Counter(m[0] for m in misses).most_common():
            print(f"  {n}  {t}")
if __name__ == "__main__":
    main()
