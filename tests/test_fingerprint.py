from tahqiq.fingerprint import features, fingerprint_report, mean_vector


def test_features_capture_conventions():
    masc_like = features("كان الامام الطبري مع علمه الواسع ضحوك السن محبا لاهل الفضل")
    cv_like = features("هو يحبّها و هي تحبّه أيضا، والإمام في المدرسة.")
    assert masc_like["hamza_forms"] == 0.0 and cv_like["hamza_forms"] > 0.0
    assert cv_like["punctuation"] > 0 and masc_like["punctuation"] == 0
    assert cv_like["ta_marbuta"] == 0.5  # المدرسة (ة) vs تحبّه (ه)
    assert features("") is None


def test_flag_only_matching_corpus():
    ref = {"masc": mean_vector(["كان الامام في المدرسه", "محبا لاهل الفضل", "ذهب الولد الى البيت"] * 4),
           "cv": mean_vector(["هو يحبّها، والإمام في المدرسة.", "أيضاً، ذهبت إلى الجامعة.", "أين الكتاب؟"] * 4),
           "sada": mean_vector(["وش رايك يا ابو بدر", "الدنيا حلوه والله", "خلاص يا شيخ"] * 4),
           "mgb2": mean_vector(["أهلا بكم مشاهدينا الكرام", "في حلقة جديدة من البرنامج", "موضوع حلقتنا اليوم"] * 4)}
    cv_style = ["ذهب أحمد إلى السوق، واشترى خبزاً.", "هل رأيت الإشارة؟", "نعم، في المساء."] * 4
    masc_style = ["ذهب احمد الى السوق واشترى خبزا", "هل رايت الاشاره", "نعم في المساء"] * 4
    hyp_a = {d: mean_vector(cv_style) for d in ref}          # one house style everywhere
    hyp_b = dict(hyp_a); hyp_b["masc"] = mean_vector(masc_style)  # mirrors MASC's captions on MASC only
    rep = fingerprint_report(ref, {"A": hyp_a, "B": hyp_b})
    assert "masc" not in rep["flags"].get("A", {})
    assert "masc" in rep["flags"].get("B", {})
