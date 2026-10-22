---
title: Tahqiq worker
emoji: ⚙️
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Evaluates Tahqiq submissions on ZeroGPU (common slice).
---

# Tahqiq worker

Background evaluator for https://huggingface.co/spaces/HeshamHaroon/tahqiq. Polls the request queue, runs each submitted model on the common slice (about 1,000 utterances per test set) on ZeroGPU, scores it, and publishes to `HeshamHaroon/tahqiq-results`.
