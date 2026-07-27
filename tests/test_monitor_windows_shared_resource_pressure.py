import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/monitor_windows_shared_resource_pressure.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("resource_monitor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module()


def _sample(*, passed=True, windows_free=3.0, gpu_free=4_000):
    return {
        "schema": "cinebotrl_windows_shared_resource_admission_v2",
        "phase": "runtime",
        "observed": {
            "windows_free_memory_gib": windows_free,
            "gpu_free_memory_mib": gpu_free,
        },
        "runtime_started": True,
        "authorization_consumed": True,
        "passed": passed,
    }


def test_monitor_report_accepts_observed_clean_exit() -> None:
    report = MODULE.build_report(
        monitored_pid=123,
        samples=[
            _sample(windows_free=3.2, gpu_free=4_200),
            _sample(windows_free=2.8, gpu_free=3_900),
        ],
        termination_requested=False,
        process_exit_observed=True,
    )

    assert report["passed"] is True
    assert report["sample_count"] == 2
    assert report["minimum_observed_windows_free_memory_gib"] == 2.8
    assert report["minimum_observed_gpu_free_memory_mib"] == 3_900


def test_monitor_report_rejects_pressure_termination() -> None:
    report = MODULE.build_report(
        monitored_pid=123,
        samples=[_sample(passed=False, windows_free=1.2)],
        termination_requested=True,
        process_exit_observed=False,
    )

    assert report["passed"] is False
    assert report["termination_requested"] is True


def test_monitor_report_rejects_missing_samples_or_unobserved_exit() -> None:
    assert MODULE.build_report(
        monitored_pid=123,
        samples=[],
        termination_requested=False,
        process_exit_observed=True,
    )["passed"] is False
    assert MODULE.build_report(
        monitored_pid=123,
        samples=[_sample()],
        termination_requested=False,
        process_exit_observed=False,
    )["passed"] is False
