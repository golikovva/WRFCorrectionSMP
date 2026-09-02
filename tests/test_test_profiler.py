import json
from types import SimpleNamespace

from lib.pipeline.test_profiler import TestLoopProfiler, count_temporal_entries


def test_profiler_writes_window_report_without_console_output(tmp_path, capsys):
    cfg = SimpleNamespace(
        device="cpu",
        test_config=SimpleNamespace(
            profiling=SimpleNamespace(
                enabled=True,
                log_every=1,
                warmup_batches=0,
                top_k=2,
                sync_cuda=False,
                console_output=False,
                output_file="profile.jsonl",
            )
        ),
    )
    profiler = TestLoopProfiler(cfg, tmp_path)

    profiler.record("metric/example", 0.25)
    profiler.finish_batch(regional_date_entries=6)
    profiler.close()

    assert capsys.readouterr().out == ""
    records = [
        json.loads(line)
        for line in (tmp_path / "profile.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["batch"] == 1
    assert records[0]["regional_date_entries"] == 6
    assert records[0]["timings_ms"]["metric/example"] == 250.0


def test_count_temporal_entries_ignores_metadata_and_other_aggregators():
    results = {
        "metric_a": {
            "RegionalTemporalAggregator": {
                "region_a": {"date_1": {}, "date_2": {}},
                "region_b": {"date_1": {}},
                "__meta__": {"channel_count": 2},
            },
            "SpatialAggregator": {"sum": 1},
        },
        "metric_b": {
            "RegionalTemporalAggregator": {
                "region_a": {"date_1": {}, "date_2": {}, "date_3": {}},
                "__meta__": {"channel_count": 1},
            }
        },
    }

    assert count_temporal_entries(results, "RegionalTemporalAggregator") == 6
