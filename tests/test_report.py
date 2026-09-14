from contexeval.report import aggregate, render_table

RECORDS = [
    {"method": "contex", "precision": 1.0, "recall": 1.0, "f1": 1.0, "em": 1.0,
     "answer_f1": 1.0, "bundle_size": 2, "context_tokens": 200, "completion_tokens": 3, "feasible": True},
    {"method": "contex", "precision": 0.5, "recall": 1.0, "f1": 0.667, "em": 0.0,
     "answer_f1": 0.5, "bundle_size": 4, "context_tokens": 400, "completion_tokens": 5, "feasible": True},
    {"method": "dump-all", "precision": 0.01, "recall": 1.0, "f1": 0.02, "em": None,
     "answer_f1": None, "bundle_size": 500, "context_tokens": 999999, "completion_tokens": None, "feasible": False},
]

def test_aggregate_means_and_feasibility():
    agg = aggregate(RECORDS)
    assert abs(agg["contex"]["recall"] - 1.0) < 1e-9
    assert abs(agg["contex"]["precision"] - 0.75) < 1e-9
    assert agg["contex"]["em"] == 0.5                  # (1.0 + 0.0)/2
    assert agg["dump-all"]["feasible_n"] == 0
    assert agg["dump-all"]["total_n"] == 1
    assert agg["dump-all"]["em"] is None                # no feasible answers to average

def test_render_table_has_rows():
    table = render_table(aggregate(RECORDS))
    assert "contex" in table and "dump-all" in table and "|" in table
    assert "completion_tokens" in table  # cost column must appear (generation tokens)
