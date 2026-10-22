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
