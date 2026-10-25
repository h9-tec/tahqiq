"""SpeechBrain CTC checkpoints (hyperparams.yaml + *.ckpt) through ``EncoderASR.from_hparams``, greedy CTC as
declared in each repo's hyperparams. Batched with relative lengths, as SpeechBrain expects.

Needs the ``speechbrain`` package (not in the main env): run with
``uv run --no-sync --with speechbrain --with 'transformers==4.57.6' ...`` (see box_queue ``pip``).
"""

from __future__ import annotations

import os

import numpy as np

from ..base import ASRModel


class SpeechBrainASR(ASRModel):
    kind = "open"
    decode_params = {"decoder": "ctc_greedy (repo hyperparams)", "runtime": "speechbrain EncoderASR"}
    batch_size = 16

    def __init__(self, model_id: str, hf_id: str, display_name: str):
        self.model_id = model_id
        self.hf_id = hf_id
        self.display_name = display_name
        self.asr = None

    def _marker(self) -> None:
        pass

    def load(self) -> None:
        try:
            from speechbrain.inference.ASR import EncoderASR   # speechbrain >= 1.0
        except ImportError:
            from speechbrain.pretrained import EncoderASR      # 0.5.x, for repos whose hyperparams use the old class paths

        save = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "speechbrain", self.model_id)
        self.asr = EncoderASR.from_hparams(source=self.hf_id, savedir=save, run_opts={"device": "cuda"})
        self.asr.mods.eval()

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        import torch

        n = max(len(a) for a in audios)
        wavs = torch.zeros(len(audios), n)
        for i, a in enumerate(audios):
            wavs[i, : len(a)] = torch.from_numpy(a.astype(np.float32))
        lens = torch.tensor([len(a) / n for a in audios])
        with torch.no_grad():
            words, _ = self.asr.transcribe_batch(wavs, lens)
        return [(" ".join(w) if isinstance(w, list) else str(w)).strip() for w in words]

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_batch([audio])[0]

    def env_info(self) -> dict:
        info = super().env_info()
        try:
            import speechbrain

            info["speechbrain"] = speechbrain.__version__
        except Exception:
            pass
        return info
