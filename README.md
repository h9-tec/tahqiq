# Tahqiq (تحقيق)

**A reproducible Arabic ASR leaderboard.** *Tahqiq* (تحقيق) is the slowest, most exact style of recitation, where every letter is given its full right; in everyday Arabic it means verification. Every model's Arabic gets the tahqiq treatment. Open models on the same public test sets as Elm's Open Universal Arabic ASR Leaderboard, re-scored with: an Elm-identical normalizer for comparability, a variant-tolerant normalizer, numeral and named-entity probes, a 95% confidence interval on every number, per-dataset and per-dialect columns, and every per-utterance output published.

**Live board and submissions:** https://huggingface.co/spaces/HeshamHaroon/tahqiq · Results: [tahqiq-results](https://huggingface.co/datasets/HeshamHaroon/tahqiq-results) · Queue: [tahqiq-requests](https://huggingface.co/datasets/HeshamHaroon/tahqiq-requests) · Method: [docs/methodology.md](docs/methodology.md)

## Submit your model

1. Open the [Space](https://huggingface.co/spaces/HeshamHaroon/tahqiq), tab **Submit a model**, sign in with Hugging Face (it only asks for permission to open a discussion).
2. Paste the model id (`owner/name`). Transformers-native ASR architectures (Whisper, wav2vec2/HuBERT CTC, Qwen3-ASR, Cohere Transcribe, Voxtral) and NeMo `.nemo` checkpoints up to 24 GB run automatically; repositories that need `trust_remote_code` wait for a maintainer.
3. A worker on our GPUs runs it on all test sets, scores it, and answers in the request thread with the numbers and a link to every per-utterance output. Typical turnaround: 1 to 6 hours.

## Leaderboard snapshot

<!-- tahqiq:table:start -->
| System | Avg WER | Casablanca | CV18 | MASC-c | MASC-n | SADA | Numerals | Entities | Δ vs Whisper v3 |
|---|---|---|---|---|---|---|---|---|---|
| NVIDIA FastConformer Arabic (pc) | 30.9 | 60.0 | 6.8 | 15.8 | 28.3 | 43.7 | 71% | 69% | -18.1 |
| NVIDIA FastConformer Arabic (pc + diacritics) | 31.5 | 60.8 | 7.0 | 16.1 | 28.6 | 45.0 | 71% | 69% | -16.3 |

WER % on the full test sets, `elm` normalizer, corpus-level, sorted by unweighted average over 5 sets. Confidence intervals, CER, the tolerant normalizer and per-dialect numbers are on the [live board](https://huggingface.co/spaces/HeshamHaroon/tahqiq). Snapshot of 2 systems, manifest `20052b67ac3c`, 2026-10-22.
<!-- tahqiq:table:end -->

## What is different from Elm's board

| | Elm | Tahqiq |
|---|---|---|
| Who runs the evaluation | model authors, self-reported | this repo, same code for every system |
| Normalizer | one, partial | two, both published and tested (`elm` is a verified 1:1 port) |
| Uncertainty | none | utterance-level bootstrap CI on every number, paired Δ + McNemar vs Whisper large-v3 |
| Beyond WER | none | numeral probe (digits and spoken numbers), entity probe (PERS/LOC/ORG containment) |
| Dialects | in the paper only | columns on the board where the test set carries labels |
| Replication | manifests in repo | manifests + hash + every hypothesis + run.json per system |

## Reproduce one row

```bash
git clone https://github.com/h9-tec/tahqiq && cd tahqiq
uv sync --extra data --extra gpu
# obtain the test sets under their own terms, then:
uv run python scripts/prepare_data.py sada --src /path/to/sada2022/raw
uv run python -m tahqiq.runner --model whisper_large_v3_turbo --dataset sada
uv run python scripts/build.py build && uv run python scripts/build.py table
```

Per-utterance outputs, `run.json` and scores for every system live in the [results dataset](https://huggingface.co/datasets/HeshamHaroon/tahqiq-results). `scripts/build.py check` refuses any run whose manifest hash is not the current one.

## Scope and policy

Open-weight models and global vendors only. Vendors domiciled in the GCC appear only after written opt-in. Cohere's hosted API is excluded by its terms; the open-weight checkpoint is included. All text is numeric; the board makes no claims beyond the measured values.

## License

Code Apache-2.0. Results, manifests and probe files CC-BY-4.0. Audio is never redistributed.
