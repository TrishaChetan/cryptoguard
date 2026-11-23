from cryptoguard.explainer import Explainer


def test_explanation_for_noisy_miner(model, data):
    _, _, test, _ = data
    row = test[test["scenario"] == "noisy"].iloc[[10]]
    exp = Explainer(model)
    out = exp.explain(row, {"top_process_name": row["top_proc_name"].iloc[0]})
    assert out["method"] == "TreeSHAP"
    assert out["reasons"], "expected at least one reason"
    assert any("CPU" in r["text"] or "process" in r["text"] for r in out["reasons"])
    assert 0.75 <= out["compute_share"] <= 1.0
    assert out["summary"].startswith("Looks like cryptojacking")


def test_surge_is_explained_by_traffic(model, data):
    _, _, test, _ = data
    rows = test[test["scenario"] == "traffic_surge"].iloc[5:15]
    share = Explainer(model).compute_share(rows)
    assert share.mean() < 0.75
