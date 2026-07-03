from app.metrics import time_block


def test_time_block_records_elapsed_ms() -> None:
    timings: dict[str, float] = {}

    with time_block("step_ms", timings):
        value = 1 + 1

    assert value == 2
    assert "step_ms" in timings
    assert timings["step_ms"] >= 0
