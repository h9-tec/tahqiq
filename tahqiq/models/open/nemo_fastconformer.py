"""NVIDIA FastConformer Hybrid (CTC/RNNT) Arabic checkpoints via NeMo. Batched.

NeMo pins its own torch; run these in the separate env created by
``scripts/setup_nemo_env.sh`` (``.venv-nemo``).

Decoding: RNNT (transducer) greedy, the checkpoint default. Both checkpoints
emit punctuation/casing ("pc"); the "pcd" one also emits diacritics, which
both normalizers strip.
"""

from __future__ import annotations

import numpy as np

from ..base import ASRModel


class NemoFastConformer(ASRModel):
    kind = "open"
    decode_params = {"decoder": "rnnt", "strategy": "greedy_batch"}
    batch_size = 32

    def __init__(self, model_id: str, hf_id: str, display_name: str):
        self.model_id = model_id
        self.hf_id = hf_id
        self.display_name = display_name
        self.asr_model = None

    def _marker(self) -> None:
        pass

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr

        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.hf_id, map_location="cuda")
        self.asr_model.eval()
        try:
            self.asr_model.change_decoding_strategy(decoder_type="rnnt")
        except Exception:
            pass

    @staticmethod
    def _text(hyp) -> str:
        if isinstance(hyp, list):
            hyp = hyp[0]
        return str(getattr(hyp, "text", hyp)).strip()

    def transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        out = self.asr_model.transcribe([a.astype(np.float32) for a in audios], batch_size=len(audios), verbose=False)
        if isinstance(out, tuple):  # (best, all) in some NeMo versions
            out = out[0]
        return [self._text(h) for h in out]

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_batch([audio])[0]

    def env_info(self) -> dict:
        info = super().env_info()
        try:
            import nemo

            info["nemo"] = nemo.__version__
        except Exception:
            pass
        return info


def nvidia_fastconformer_ar_pc() -> NemoFastConformer:
    return NemoFastConformer(
        "nvidia_fastconformer_ar_pc", "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0", "NVIDIA FastConformer Arabic (pc)"
    )


def nvidia_fastconformer_ar_pcd() -> NemoFastConformer:
    return NemoFastConformer(
        "nvidia_fastconformer_ar_pcd",
        "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0",
        "NVIDIA FastConformer Arabic (pc + diacritics)",
    )
