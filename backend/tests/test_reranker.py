from app.reranker import sigmoid


def test_sigmoid_relevance_score_is_between_zero_and_one() -> None:
    assert 0 < sigmoid(-7.701) < 1
    assert 0 < sigmoid(0) < 1
    assert 0 < sigmoid(7.701) < 1


def test_sigmoid_preserves_score_order() -> None:
    assert sigmoid(-2.0) < sigmoid(0.0) < sigmoid(2.0)
