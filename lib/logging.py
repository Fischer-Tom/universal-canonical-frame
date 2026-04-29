import json
import logging
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch

try:
    from tools.ddp_tools import is_main_process
except ImportError:
    def is_main_process() -> bool:
        return True


class LogLevel:
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class MetricsTracker:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.global_metrics: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def update(self, metrics: Dict[str, Union[float, torch.Tensor]]) -> None:
        with self._lock:
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.metrics[key].append(value)
                self.global_metrics[key].append(value)

    def get_current_avg(self, key: str) -> float:
        with self._lock:
            values = self.metrics.get(key, [])
            return float(np.mean(values)) if values else 0.0

    def get_state(self) -> Dict[str, List[float]]:
        with self._lock:
            return {k: list(v) for k, v in self.global_metrics.items()}

    def restore_state(self, state: Dict[str, List[float]]) -> None:
        with self._lock:
            for key, values in state.items():
                self.global_metrics[key] = list(values)
                self.metrics[key] = deque(values[-self.window_size:], maxlen=self.window_size)


class AdvancedLogger:
    """File + console logger with windowed metric averages and JSONL output."""

    def __init__(
        self,
        total_iters: int,
        run_dir: str,
        experiment_name: str = "experiment",
        log_level: int = LogLevel.INFO,
        save_frequency: int = 100,
        metrics_window_size: int = 100,
        log_subdir: str = "logs",
        vis_subdir: str = "vis",
        checkpoints_subdir: str = "checkpoints",
    ):
        self.total_iters = total_iters
        self.run_dir = Path(run_dir)
        self.log_dir = self.run_dir / log_subdir
        self.vis_dir = self.run_dir / vis_subdir
        self.checkpoints_dir = self.run_dir / checkpoints_subdir
        self.experiment_name = experiment_name
        self.log_level = log_level
        self.save_frequency = save_frequency
        self.metrics_tracker = MetricsTracker(metrics_window_size)
        self.start_time = time.time()
        self.current_iter = 0

        if is_main_process():
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.vis_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.main_log_file = self.log_dir / "train.log"
        self.metrics_log_file = self.log_dir / "metrics.jsonl"
        self.jsonl_log_file = self.log_dir / "train.jsonl"

        self.logger = logging.getLogger(f"ucf.{experiment_name}")
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()
        self.logger.propagate = False

        if is_main_process():
            fh = logging.FileHandler(self.main_log_file)
            fh.setFormatter(logging.Formatter(
                f"%(asctime)s | %(levelname)8s | {experiment_name} | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)8s | %(message)s"))
        self.logger.addHandler(ch)

        self.info(f"Logger initialized for experiment: {experiment_name}")
        self.info(f"Run directory: {self.run_dir}")
        self.info(f"Log directory: {self.log_dir}")
        self.info(f"Checkpoint directory: {self.checkpoints_dir}")
        self.info(f"Visualization directory: {self.vis_dir}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log(self, level: int, message: str, **kwargs) -> None:
        if not is_main_process():
            return
        self.logger.log(level, message, **kwargs)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "iteration": self.current_iter,
            "experiment": self.experiment_name,
            **kwargs,
        }
        try:
            with open(self.jsonl_log_file, "a") as f:
                f.write(json.dumps(entry, default=self._json_serializer) + "\n")
        except Exception as e:
            print(f"Failed to write structured log: {e}", file=sys.stderr)

    def info(self, message: str, **kwargs) -> None:
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self.log(LogLevel.ERROR, message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        self.log(LogLevel.DEBUG, message, **kwargs)

    @staticmethod
    def _json_serializer(obj):
        if hasattr(obj, "_metadata"):
            from omegaconf import OmegaConf
            return OmegaConf.to_container(obj, resolve=True)
        if hasattr(obj, "item") and hasattr(obj, "numel"):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "__fspath__"):
            return str(obj)
        return str(obj)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def step(self, iteration: int, metrics: Dict[str, Union[float, torch.Tensor]]) -> None:
        self.current_iter = iteration
        processed = {
            k: (v.item() if isinstance(v, torch.Tensor) else float(v))
            for k, v in metrics.items()
        }
        self.metrics_tracker.update(processed)

        if iteration % self.save_frequency == 0:
            self._save_metrics_checkpoint(iteration, processed)
            self._log_training_summary(iteration)

    def _save_metrics_checkpoint(self, iteration: int, metrics: Dict[str, float]) -> None:
        if not is_main_process():
            return
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time,
            "metrics": metrics,
            "averages": {k: self.metrics_tracker.get_current_avg(k) for k in self.metrics_tracker.metrics},
            "experiment": self.experiment_name,
        }
        try:
            with open(self.metrics_log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.error(f"Failed to save metrics checkpoint: {e}")

    def _log_training_summary(self, iteration: int) -> None:
        elapsed = time.time() - self.start_time
        speed = iteration / elapsed if elapsed > 0 else 0.0
        lines = [
            f"Training Summary - Iteration {iteration}/{self.total_iters}",
            f"Elapsed: {elapsed / 3600:.2f}h | Speed: {speed:.2f} iter/s",
        ]
        for name in self.metrics_tracker.metrics:
            avg = self.metrics_tracker.get_current_avg(name)
            lines.append(f"{name}: {avg:.4g}")
        self.info("\n" + "\n".join(lines))

    # ------------------------------------------------------------------
    # Checkpoint state
    # ------------------------------------------------------------------
    def get_checkpoint_state(self) -> Dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "log_dir": str(self.log_dir),
            "metrics_history": self.metrics_tracker.get_state(),
            "current_iter": self.current_iter,
            "elapsed_before_resume": time.time() - self.start_time,
        }

    def restore_from_checkpoint(self, state: Dict[str, Any]) -> None:
        if "metrics_history" in state:
            self.metrics_tracker.restore_state(state["metrics_history"])
            self.info(f"Restored {len(state['metrics_history'])} metric histories")
        if "current_iter" in state:
            self.current_iter = state["current_iter"]
        if "elapsed_before_resume" in state:
            self.start_time = time.time() - state["elapsed_before_resume"]
            self.info(f"Adjusted start time for {state['elapsed_before_resume']:.1f}s of previous training")
