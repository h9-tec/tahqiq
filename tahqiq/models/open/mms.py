"""Meta MMS-1b-all with the Arabic adapter (CTC, greedy, batched)."""

from __future__ import annotations

from .ctc import CTCModel


class MMS(CTCModel):
    decode_params = {"target_lang": "ara", "decoding": "greedy"}

    def __init__(self) -> None:
        super().__init__("mms_1b_all", "facebook/mms-1b-all", "MMS-1b-all (ara)", target_lang="ara")


def mms_1b_all() -> MMS:
    return MMS()
