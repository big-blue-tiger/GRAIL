"""Low-overhead diagnostics for online DAgger CUDA/Kit interaction."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import re
import time
from typing import Any, Iterator

import torch


def _call(value: Any, name: str) -> Any:
    attribute = getattr(value, name, None)
    return attribute() if callable(attribute) else attribute


def _timeline_status(timeline: Any) -> str:
    if timeline is None:
        return "timeline=unavailable"
    values = {
        "time": _call(timeline, "get_current_time"),
        "end": _call(timeline, "get_end_time"),
        "playing": _call(timeline, "is_playing"),
        "stopped": _call(timeline, "is_stopped"),
    }
    return "timeline=" + ",".join(
        f"{key}={value!r}" for key, value in values.items() if value is not None
    )


def _nvml_process_memory(device_index: int) -> int | None:
    """Return this process's device memory, when optional NVML bindings exist."""
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        pid = os.getpid()
        total = 0
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if process.pid == pid:
                total += int(process.usedGpuMemory or 0)
        return total
    except Exception:  # pragma: no cover - depends on the host driver/runtime
        return None


def cuda_memory_status(*, nvml: bool = False, timeline: Any = None) -> str:
    """Return non-synchronizing CUDA, optional NVML, and timeline telemetry."""
    if not torch.cuda.is_available():
        return "cuda=unavailable " + _timeline_status(timeline)

    try:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        free, total = torch.cuda.mem_get_info()
        stream = getattr(torch.cuda.current_stream(), "cuda_stream", "unknown")
        fields = [
            f"device={device}",
            f"allocated={allocated / 2**30:.2f}GiB",
            f"reserved={reserved / 2**30:.2f}GiB",
            f"max_allocated={peak / 2**30:.2f}GiB",
            f"free={free / 2**30:.2f}GiB",
            f"total={total / 2**30:.2f}GiB",
            f"stream={stream}",
        ]
        if nvml:
            process_memory = _nvml_process_memory(device)
            if process_memory is not None:
                fields.append(f"nvml_process={process_memory / 2**30:.2f}GiB")
        return ", ".join(fields) + " " + _timeline_status(timeline)
    except Exception as exc:  # pragma: no cover - CUDA runtime dependent
        return f"cuda_status_error={type(exc).__name__}: {exc} " + _timeline_status(timeline)


def _tensor_census(limit: int = 8) -> str:
    """Summarize live CUDA tensors without retaining any of the objects inspected."""
    counts: Counter[str] = Counter()
    bytes_by_device: Counter[str] = Counter()
    largest: list[tuple[int, str]] = []
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor):
                continue
            device = str(obj.device)
            counts[device] += 1
            nbytes = int(obj.numel() * obj.element_size())
            bytes_by_device[device] += nbytes
            if device.startswith("cuda") and nbytes:
                largest.append((nbytes, f"{tuple(obj.shape)}/{obj.dtype}"))
        except Exception:
            # Objects can disappear while the garbage collector is traversed.
            continue
    largest.sort(reverse=True)
    largest_text = ";".join(
        f"{size / 2**20:.1f}MiB:{description}" for size, description in largest[:limit]
    )
    totals = ",".join(
        f"{device}={count}/{bytes_by_device[device] / 2**30:.2f}GiB"
        for device, count in sorted(counts.items())
    )
    return f"tensors={totals or 'none'}, largest={largest_text or 'none'}"


@contextmanager
def phase_probe(
    name: str,
    *,
    enabled: bool = True,
    synchronize: bool = False,
    nvml: bool = False,
    nvtx: bool = False,
    timeline: Any = None,
    tensor_census: bool = False,
) -> Iterator[None]:
    """Print a START/DONE pair for a phase and optionally force CUDA ordering."""
    if not enabled:
        yield
        return

    if synchronize and torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.monotonic()
    if nvtx and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    print(
        f"[dagger][phase] START {name} ({cuda_memory_status(nvml=nvml, timeline=timeline)})",
        flush=True,
    )
    try:
        yield
    except BaseException:
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f"[dagger][phase] FAILED {name} after {time.monotonic() - started:.3f}s "
            f"({cuda_memory_status(nvml=nvml, timeline=timeline)})",
            flush=True,
        )
        raise
    else:
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        status = cuda_memory_status(nvml=nvml, timeline=timeline)
        census = f", {_tensor_census()}" if tensor_census else ""
        print(
            f"[dagger][phase] DONE {name} after {time.monotonic() - started:.3f}s "
            f"({status}{census})",
            flush=True,
        )
    finally:
        if nvtx and torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


class CudaDiagnostics:
    """Configuration-bound phase probes and PyTorch memory snapshots."""

    def __init__(self, config: Any, output_dir: Path, timeline: Any = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.phase_logging = bool(config.get("phase_logging", True))
        self.synchronize = bool(config.get("synchronize", False))
        self.nvml = bool(config.get("nvml", False))
        self.nvtx = bool(config.get("nvtx", False))
        self.tensor_census = bool(config.get("tensor_census", False))
        self.safe_teacher_cleanup = bool(config.get("safe_teacher_cleanup", False))
        self.collection_mode = str(config.get("collection_mode", "full"))
        self.collect_only = bool(config.get("collect_only", False))
        self.forward_only = bool(config.get("forward_only", False))
        self.disable_optimizer = bool(config.get("disable_optimizer", False))
        self.skip_checkpoint = bool(config.get("skip_checkpoint", False))
        valid_collection_modes = {"full", "environment_only", "teacher_only", "atm_only"}
        if self.collection_mode not in valid_collection_modes:
            raise ValueError(
                "diagnostics.collection_mode must be one of "
                + ", ".join(sorted(valid_collection_modes))
            )
        if self.enabled is False and (
            self.collection_mode != "full"
            or self.collect_only
            or self.forward_only
            or self.disable_optimizer
            or self.skip_checkpoint
        ):
            raise ValueError(
                "diagnostic isolation modes require diagnostics.enabled=true"
            )
        self.memory_history = bool(config.get("memory_history", False))
        self.memory_snapshot = bool(config.get("memory_snapshot", False))
        self.max_history_entries = int(config.get("max_history_entries", 100000))
        configured_dir = config.get("snapshot_dir", "diagnostics/memory")
        snapshot_dir = Path(str(configured_dir))
        self.snapshot_dir = snapshot_dir if snapshot_dir.is_absolute() else output_dir / snapshot_dir
        self.timeline = timeline
        self._history_started = False
        self._snapshot_index = 0

    def start_memory_history(self) -> None:
        if not self.enabled or not self.memory_history or not torch.cuda.is_available():
            return
        recorder = getattr(torch.cuda.memory, "_record_memory_history", None)
        if recorder is None:
            print("[dagger][diagnostics] CUDA memory history is unavailable", flush=True)
            return
        try:
            recorder(max_entries=self.max_history_entries)
            self._history_started = True
            print(
                f"[dagger][diagnostics] CUDA memory history enabled: "
                f"max_entries={self.max_history_entries}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - runtime/version dependent
            print(
                f"[dagger][diagnostics] CUDA memory history failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def phase(self, name: str, *, synchronize: bool | None = None):
        return phase_probe(
            name,
            enabled=self.enabled and self.phase_logging,
            synchronize=self.synchronize if synchronize is None else synchronize,
            nvml=self.nvml,
            nvtx=self.nvtx,
            timeline=self.timeline,
            tensor_census=self.tensor_census,
        )

    def snapshot(self, label: str) -> Path | None:
        if not self.enabled or not self.memory_snapshot or not torch.cuda.is_available():
            return None
        dumper = getattr(torch.cuda.memory, "_dump_snapshot", None)
        if dumper is None:
            print("[dagger][diagnostics] CUDA memory snapshots are unavailable", flush=True)
            return None
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "snapshot"
        path = self.snapshot_dir / f"{self._snapshot_index:03d}_{safe_label}.pickle"
        self._snapshot_index += 1
        try:
            dumper(str(path))
        except Exception as exc:  # pragma: no cover - runtime/version dependent
            print(
                f"[dagger][diagnostics] CUDA memory snapshot failed at {path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None
        print(f"[dagger][diagnostics] CUDA memory snapshot: {path}", flush=True)
        return path
