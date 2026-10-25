from tahqiq.normalize.buckwalter import looks_like_buckwalter_vocab, to_arabic


def test_untransliterate_with_slash_delimiter():
    assert to_arabic("nAk/klb/ynbH") == "ناك كلب ينبح"
    # diacritics keep Buckwalter order: kin~a = kaf, kasra, nun, shadda, fatha
    assert to_arabic("walakin~a/nu") == "\u0648\u064e\u0644\u064e\u0643\u0650\u0646\u0651\u064e \u0646\u064f"
    assert to_arabic("<lY/>n") == "إلى أن"


def test_vocab_detection():
    bw = {c: i for i, c in enumerate(list("AbtvjHxd*rzs$SDTZEgfqklmnhwYy'|><&}~aiuo/"))}
    ar = {c: i for i, c in enumerate(list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي|"))}
    latin = {c: i for i, c in enumerate(list("abcdefghijklmnopqrstuvwxyzḥ'|"))}
    assert looks_like_buckwalter_vocab(bw)
    assert not looks_like_buckwalter_vocab(ar)
    assert not looks_like_buckwalter_vocab(latin)
