# Tahqiq (تحقيق)

**A reproducible Arabic ASR leaderboard.** *Tahqiq* (تحقيق) is the slowest, most exact style of recitation, where every letter is given its full right; in everyday Arabic it means verification. Every model's Arabic gets the tahqiq treatment: Open models on the same public test sets as Elm's Open Universal Arabic ASR Leaderboard, re-scored with: an Elm-identical normalizer for comparability, a variant-tolerant normalizer, numeral and named-entity probes, a 95% confidence interval on every number, per-dataset and per-dialect columns, and every per-utterance output published.

Live board: https://h9-tec.github.io/tahqiq · Method: [docs/methodology.md](docs/methodology.md)

## Leaderboard

<!-- tahqiq:table:start -->
| System | Avg WER | Casablanca | CV18 | MASC-c | MASC-n | SADA | Numerals | Entities | Δ vs Whisper v3 |
|---|---|---|---|---|---|---|---|---|---|

WER % on the full test sets, `elm` normalizer, corpus-level, sorted by unweighted average over 5 sets. Confidence intervals, CER, the tolerant normalizer and per-dialect numbers are on the site. Manifest `20052b67ac3c`, built 2026-10-22.
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

Results in `results/<model>/` are the published artifact. `scripts/build.py check` refuses any run whose manifest hash is not the current one.

## Scope and policy

Open-weight models and global vendors only. Vendors domiciled in the GCC appear only after written opt-in. Cohere's hosted API is excluded by its terms; the open-weight checkpoint is included. All text is numeric; the board makes no claims beyond the measured values.

## License

Code Apache-2.0. Results, manifests and probe files CC-BY-4.0. Audio is never redistributed.
