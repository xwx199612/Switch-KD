from vlm_distill.stage_evaluation import exact_match, token_f1


def test_exact_match_normalizes_case_and_spaces():
    assert exact_match("  A Cup ", "a cup") == 1.0


def test_token_f1_partial_overlap():
    assert token_f1("red cup on table", "red cup") == 2 * 0.5 * 1.0 / (0.5 + 1.0)
