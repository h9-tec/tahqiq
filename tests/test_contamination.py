import math

from tahqiq.contamination import anomalies, load_declared, median_polish


def _table(n_models=8, datasets=("a", "b", "c", "d", "e", "f")):
    # additive truth: model levels × dataset difficulty
    levels = [0.2 + 0.05 * i for i in range(n_models)]
    diff = {d: 1.0 + 0.3 * k for k, d in enumerate(datasets)}
    return {f"m{i}": {d: levels[i] * diff[d] for d in datasets} for i in range(n_models)}


def test_median_polish_residuals_near_zero_on_additive_data():
    t = {m: {d: math.log(v) for d, v in row.items()} for m, row in _table().items()}
    res = median_polish(t)
    assert max(abs(v) for row in res.values() for v in row.values()) < 1e-6


def test_anomaly_flags_only_the_leaked_cell():
    t = _table()
    t["m3"]["d"] = t["m3"]["d"] * 0.4   # 60% better than it should be on dataset d
    out = anomalies(t)
    assert set(out) == {"m3"} and set(out["m3"]) == {"d"}
    assert out["m3"]["d"]["residual"] < -0.3 and out["m3"]["d"]["expected_wer"] > t["m3"]["d"]


def test_no_flags_without_enough_models():
    assert anomalies(_table(n_models=4)) == {}


def test_declared_loads():
    d = load_declared()
    assert "mgb2" in d["artst_v3"] and d["artst_v3"]["mgb2"]["source"].startswith("https://")
    assert set(d["nvidia_fastconformer_ar_pc"]) == {"masc_clean", "masc_noisy", "commonvoice"}


def _score(mid, test, dev=None, pert=None, lik=None):
    d = {"model_id": mid, "scores": {"api": {"datasets": {ds: {"elm": {"wer": {"point": w}}} for ds, w in test.items()}}}, "extras": {}}
    if dev: d["extras"]["dev"] = {ds: {"wer": w} for ds, w in dev.items()}
    if pert: d["extras"]["pert"] = {k: {"wer": w} for k, w in pert.items()}
    if lik: d["extras"]["likelihood"] = {ds: {"ref_preferred_rate": r} for ds, r in lik.items()}
    return d


def test_dev_gap_flags_only_leaked_corpus():
    from tahqiq.contamination import dev_gap_flags
    sets = ["a", "b", "c", "d"]
    clean = [_score(f"m{i}", {s: 0.3 + 0.02 * i for s in sets}, dev={s: 0.31 + 0.02 * i for s in sets}) for i in range(5)]
    leaked = _score("leak", {"a": 0.12, "b": 0.3, "c": 0.3, "d": 0.3}, dev={"a": 0.31, "b": 0.31, "c": 0.31, "d": 0.31})
    out = dev_gap_flags(clean + [leaked])
    assert set(out) == {"leak"} and set(out["leak"]) == {"a"} and out["leak"]["a"]["wer_dev"] == 0.31


def test_pert_gap_flags_corpus_where_advantage_collapses():
    from tahqiq.contamination import pert_gap_flags
    sets = ["a", "b", "c", "d"]; variants = ["speed094", "noise20"]
    def mk(mid, base, bump):
        test = {s: base for s in sets}
        pert = {f"{v}__{s}": base * (1.3 + (bump if s == "a" else 0)) for v in variants for s in sets}
        return _score(mid, test, pert=pert)
    scores = [mk(f"m{i}", 0.3, 0.0) for i in range(5)] + [mk("leak", 0.3, 1.2)]
    out = pert_gap_flags(scores)
    assert set(out) == {"leak"} and set(out["leak"]) == {"a"}


def test_likelihood_flags():
    from tahqiq.contamination import likelihood_flags
    s = _score("m", {"a": .3, "b": .3, "c": .3, "d": .3}, lik={"a": 0.45, "b": 0.05, "c": 0.06, "d": 0.04})
    assert set(likelihood_flags([s])["m"]) == {"a"}
