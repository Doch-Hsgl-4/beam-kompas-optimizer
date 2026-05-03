from __future__ import annotations

import json
import logging
import os
import shutil
import stat
from pathlib import Path

from beam_optimizer.models import BeamRequest, CandidateResult, OptimizationSummary


def ensure_output_structure(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "models": output_dir / "models",
        "reports": output_dir / "reports",
        "logs": output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def clear_output_structure(output_dir: Path) -> None:
    root_path = output_dir.resolve()
    _detach_output_file_handlers(root_path)
    logging.shutdown()
    if not root_path.exists():
        return

    locked_paths: list[str] = []

    def _onerror(func, path, _exc_info):
        target = Path(path)
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            func(path)
            return
        except Exception:
            locked_paths.append(str(target))

    shutil.rmtree(root_path, onerror=_onerror)
    if root_path.exists():
        locked_paths.append(str(root_path))

    if locked_paths:
        unique_locked = sorted(set(locked_paths))
        preview = "\n".join(unique_locked[:5])
        if len(unique_locked) > 5:
            preview += f"\n... и ещё {len(unique_locked) - 5}"
        raise PermissionError(
            "Не удалось удалить часть файлов в output. Скорее всего, они заняты другим процессом "
            "(обычно КОМПАС или OneDrive):\n"
            f"{preview}"
        )


def configure_logging(log_dir: Path) -> Path:
    log_path = log_dir / "beam_optimizer.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
        for handler in root_logger.handlers
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root_logger.addHandler(file_handler)
    return log_path


def save_run_report(
    request: BeamRequest,
    summary: OptimizationSummary,
    report_dir: Path,
) -> Path:
    payload = {
        "length_m": request.length_m,
        "section_type": request.section_type.value,
        "material": request.material.title,
        "support_type": request.support_type.value,
        "load_type": request.load_case.load_type.value,
        "load_magnitude": request.load_case.magnitude,
        "load_position_m": request.load_case.position_m,
        "goal": request.optimization_goal.value,
        "total_generated": summary.total_generated,
        "feasible_count": summary.feasible_count,
        "selected": [_candidate_to_dict(candidate) for candidate in summary.selected],
        "alternatives": [_candidate_to_dict(candidate) for candidate in summary.alternatives],
    }
    report_path = report_dir / "run_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _candidate_to_dict(candidate: CandidateResult) -> dict:
    return {
        "section": candidate.section.title,
        "dimensions_mm": candidate.section.dimensions_mm,
        "mass_kg": candidate.analysis.mass_kg,
        "max_stress_pa": candidate.analysis.max_stress_pa,
        "max_deflection_m": candidate.analysis.max_deflection_m,
        "safety_factor": candidate.analysis.safety_factor,
        "is_feasible": candidate.is_feasible,
        "violations": candidate.violations,
        "model_path": str(candidate.model_path) if candidate.model_path else None,
    }


def _detach_output_file_handlers(output_dir: Path) -> None:
    root_logger = logging.getLogger()
    handlers_to_remove: list[logging.Handler] = []
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                handler_path = Path(handler.baseFilename).resolve()
            except Exception:
                continue
            if output_dir in handler_path.parents:
                handler.close()
                handlers_to_remove.append(handler)
    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)
