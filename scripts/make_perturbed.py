"""Perturbed copies of the common slice for the perturbation-gap contamination test.

    uv run python scripts/make_perturbed.py            # all variants, all sets
    uv run python scripts/make_perturbed.py --limit 20 # smoke

Variants (word-preserving, published, seeded):
  speed094   resample so the clip plays 6% slower (pitch moves with it, as Kaldi speed-perturb does)
  speed106   6% faster
  noise20    additive pink noise at 20 dB SNR
  tel8k      telephone band: resample 16k -> 8k -> 16k (band-limited to 4 kHz)

Writes audio/pert/<variant>/<dataset>/<id>.wav and data/pert_manifests/<variant>__<dataset>.jsonl
(same ids and references as the common slice; audio_rel points at the perturbed file).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

from tahqiq.io import SAMPLE_RATE, load_audio_16k_mono  # noqa: E402
from tahqiq.manifest import Utterance, load_manifest, manifest_paths, sha256_file, write_manifest, write_manifest_hash  # noqa: E402

AUDIO = Path(os.environ.get("TAHQIQ_AUDIO", ROOT / "audio"))
OUT_MANIFESTS = ROOT / "data" / "pert_manifests"
VARIANTS = ("speed094", "speed106", "noise20", "tel8k")


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(white)
    f = np.arange(len(spec), dtype=np.float32)
    f[0] = 1.0
    spec = spec / np.sqrt(f)
    pink = np.fft.irfft(spec, n).astype(np.float32)
    return pink / (np.abs(pink).max() + 1e-9)


def perturb(x: np.ndarray, variant: str, rng: np.random.Generator) -> np.ndarray:
    if variant == "speed094":   # play slower: more samples
        return _resample(x, 16000, 17021)[: int(len(x) / 0.94) + 1]
    if variant == "speed106":
        return _resample(x, 16000, 15094)
    if variant == "noise20":
        noise = pink_noise(len(x), rng)
        sig_p = float(np.mean(x ** 2)) + 1e-12
        noise_p = float(np.mean(noise ** 2)) + 1e-12
        scale = np.sqrt(sig_p / (noise_p * 10 ** (20 / 10)))
        return np.clip(x + scale * noise, -1, 1).astype(np.float32)
    if variant == "tel8k":
        return _resample(_resample(x, 16000, 8000), 8000, 16000)[: len(x)]
    raise ValueError(variant)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    OUT_MANIFESTS.mkdir(parents=True, exist_ok=True)
    for p in manifest_paths(ROOT / "data" / "manifests"):
        ds = p.stem
        utts = [u for u in load_manifest(p) if u.in_api_slice]
        if args.limit:
            utts = utts[: args.limit]
        for variant in args.variants.split(","):
            rng = np.random.default_rng(abs(hash((variant, ds))) % (2**32))
            out_utts: list[Utterance] = []
            for u in tqdm(utts, desc=f"{variant}/{ds}", unit="utt"):
                rel = f"pert/{variant}/{ds}/{u.id}.wav"
                dst = AUDIO / rel
                if not dst.exists():
                    x, _ = load_audio_16k_mono(AUDIO / u.audio_rel)
                    y = perturb(x, variant, rng)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(dst), y, SAMPLE_RATE, subtype="PCM_16")
                v = Utterance(**{**u.to_dict(), "audio_rel": rel, "audio_sha256": sha256_file(dst),
                                 "duration_s": round(len(sf.SoundFile(str(dst))) / SAMPLE_RATE, 3), "in_api_slice": True})
                out_utts.append(v)
            write_manifest(OUT_MANIFESTS / f"{variant}__{ds}.jsonl", out_utts)
    print("PERT_MANIFEST_SHA256", write_manifest_hash(OUT_MANIFESTS))


if __name__ == "__main__":
    main()
