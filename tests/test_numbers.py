from tahqiq.normalize.numbers import digit_string, has_written_number


def test_has_written_number():
    assert has_written_number("رقم ٠٥٠")
    assert has_written_number("room 12")
    assert not has_written_number("خمسة")


def test_digit_string_digits_only():
    assert digit_string("رقمي ٠٥٠ ١٢٣ ٤٥") == "05012345"
    assert digit_string("no digits") == ""


def test_digit_string_number_words():
    assert digit_string("خمسة وعشرين", map_number_words=True) == "5"
    assert digit_string("ثلاثة أربعة", map_number_words=True) == "34"
    assert digit_string("تلاته", map_number_words=True) == "3"


def test_has_number_words_but_not_ambiguous():
    from tahqiq.normalize.numbers import has_number
    assert has_number("عندي خمسة أولاد")
    assert has_number("الساعة ٣")
    assert not has_number("يا ست الكل")          # ست = lady
    assert not has_number("واحد من الناس")        # واحد = someone
    assert not has_number("كيف الحال")
