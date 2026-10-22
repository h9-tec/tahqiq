---
title: Tahqiq · Arabic ASR Leaderboard
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
hf_oauth: true
hf_oauth_scopes:
  - write-discussions
short_description: Arabic ASR, verified. CIs, dialects, submit any model.
---

# Tahqiq (تحقيق)

An independent, reproducible leaderboard for Arabic speech recognition. Open-weight models are run by one harness on six public test sets (SADA, Casablanca, Common Voice 18, MASC clean and noisy, MGB-2), scored with two published normalizers, and reported with a 95% confidence interval on every number, per-dataset and per-dialect columns, numeral and entity probes, and a production profile (RTFx, latency, VRAM, streaming, timestamps, license).

**Submit any model** from the Hub in the *Submit a model* tab. It is evaluated on Hugging Face compute and appears here with a results thread.

- Results: [HeshamHaroon/tahqiq-results](https://huggingface.co/datasets/HeshamHaroon/tahqiq-results)
- Queue: [HeshamHaroon/tahqiq-requests](https://huggingface.co/datasets/HeshamHaroon/tahqiq-requests)
- Code and methodology: [github.com/h9-tec/tahqiq](https://github.com/h9-tec/tahqiq)
