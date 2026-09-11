from collections import defaultdict

_RETR_KEYS = ["precision", "recall", "f1", "bundle_size", "context_tokens"]
_ANS_KEYS = ["em", "answer_f1", "completion_tokens"]

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def aggregate(records: list[dict]) -> dict[str, dict]:
    by_method = defaultdict(list)
    for r in records:
        if "error" not in r:
            by_method[r["method"]].append(r)
    out = {}
    for method, recs in by_method.items():
        feasible = [r for r in recs if r.get("feasible")]
        row = {k: _mean([r[k] for r in recs]) for k in _RETR_KEYS}
        row.update({k: _mean([r[k] for r in feasible]) for k in _ANS_KEYS})
        row["feasible_n"] = len(feasible)
        row["total_n"] = len(recs)
        out[method] = row
    return out

def render_table(agg: dict) -> str:
    cols = ["recall", "precision", "f1", "em", "answer_f1", "bundle_size", "context_tokens", "feasible_n", "total_n"]
    lines = ["| method | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for method, row in sorted(agg.items()):
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append("n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

def pr_curve(sweeps: dict, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for method, points in sweeps.items():
        points = sorted(points, key=lambda t: t[0])  # by recall
        ax.plot([r for r, _ in points], [p for _, p in points], marker="o", label=method)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend(); ax.set_title("Precision-Recall")
    fig.savefig(out_path, bbox_inches="tight")
