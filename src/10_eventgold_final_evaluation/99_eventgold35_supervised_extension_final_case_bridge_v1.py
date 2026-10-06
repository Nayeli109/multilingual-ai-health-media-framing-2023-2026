#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path.cwd().resolve()
FINAL = ROOT / "01_event_aware_v3" / "10_eventgold_final_evaluation"
EVALUATOR = FINAL / "97_evaluate_eventgold_supervised_extension_final_v1.py"
EXPECTED_EVALUATOR_SHA256 = "a4a0a7d6c94f9d84f765568011a6cacd5cb0fa6ba14439b391faf03c610bb920"

STANCE_CASE_MAP = {
    "cautious or mixed about AI in health": "cautious or mixed about ai in health",
    "critical of AI risks in health": "critical of ai risks in health",
    "optimistic about AI in health": "optimistic about ai in health",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_frozen_evaluator():
    if not EVALUATOR.is_file():
        raise RuntimeError(f"Frozen evaluator not found: {EVALUATOR}")
    observed = sha256(EVALUATOR)
    if observed != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError(
            "Frozen evaluator SHA256 mismatch.\n"
            f"OBSERVED={observed}\n"
            f"EXPECTED={EXPECTED_EVALUATOR_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "eventgold35_frozen_evaluator_v97",
        EVALUATOR,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen evaluator module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def install_stance_case_bridge(module) -> None:
    original_validate = module.validate_eventgold

    def validate_eventgold_case_bridge(df, frozen_maps):
        stance_targets = {str(v) for v in frozen_maps["stance"].values()}

        for src, dst in STANCE_CASE_MAP.items():
            if src.lower() != dst:
                raise RuntimeError(
                    f"Case bridge is not orthographic-only: {src!r} -> {dst!r}"
                )
            if dst not in stance_targets:
                raise RuntimeError(
                    f"Mapped target absent from frozen stance map: {dst!r}"
                )

        col = module.LABEL_COLUMNS["stance"]
        if col not in df.columns:
            raise RuntimeError(f"Missing stance label column: {col}")

        out = df.copy()
        before = out[col].astype(str).copy()
        out[col] = out[col].astype(str).replace(STANCE_CASE_MAP)
        after = out[col].astype(str)

        changed = before != after
        changed_sources = set(before[changed].tolist())
        unexpected = changed_sources - set(STANCE_CASE_MAP)

        if unexpected:
            raise RuntimeError(
                f"Unexpected stance labels changed: {sorted(unexpected)}"
            )

        print("STANCE_CASE_BRIDGE=ACTIVE")
        print("STANCE_CASE_BRIDGE_CHANGED_ROWS=" + str(int(changed.sum())))
        print("STANCE_CASE_BRIDGE_MAPPING_COUNT=3")

        return original_validate(out, frozen_maps)

    module.validate_eventgold = validate_eventgold_case_bridge

def main() -> None:
    module = load_frozen_evaluator()
    install_stance_case_bridge(module)
    module.main()

if __name__ == "__main__":
    main()
