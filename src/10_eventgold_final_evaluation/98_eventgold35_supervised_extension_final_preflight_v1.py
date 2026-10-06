#!/usr/bin/env python3
"""
EVE-FRAME — EventGold35 supervised-extension final non-performance preflight v1.

Purpose
-------
Verify the frozen final-evaluation package before any final-gold model inference.

This script:
- verifies frozen evaluator/package hashes;
- runs the evaluator's --self-check-only mode;
- verifies the 15 frozen checkpoints byte-for-byte;
- verifies the 3 x 5 task/seed grid;
- verifies final-gold structure without model inference;
- verifies the previously frozen bootstrap RNG;
- verifies result namespaces are absent;
- writes and freezes a PASS report only after all checks succeed.

This script DOES NOT call forward_document, does not compute probabilities,
does not compute predictions, and does not compute performance metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple


# =============================================================================
# Frozen project constants
# =============================================================================

ROOT = Path.cwd().resolve()

FINAL = (
    ROOT
    / "01_event_aware_v3"
    / "10_eventgold_final_evaluation"
)

EVAL = (
    FINAL
    / "97_evaluate_eventgold_supervised_extension_final_v1.py"
)

EVAL_LOCK = (
    FINAL
    / "EVENTGOLD35_SUPERVISED_EXTENSION_FINAL_EVALUATOR_LOCK_v1.txt"
)

EVAL_FREEZE = (
    FINAL
    / "FROZEN_EVENTGOLD35_SUPERVISED_EXTENSION_FINAL_EVALUATOR_SHA256SUMS_v1.txt"
)

EVAL_PROTOCOL = (
    FINAL
    / "EVENTGOLD35_SUPERVISED_EXTENSION_FINAL_EVALUATION_PROTOCOL_v1.txt"
)

EVAL_PROTOCOL_FREEZE = (
    FINAL
    / "FROZEN_EVENTGOLD35_SUPERVISED_EXTENSION_FINAL_EVALUATION_PROTOCOL_SHA256SUMS_v1.txt"
)

GOLD = (
    FINAL
    / "eventgold35_supervised_extension_v1"
    / "EventGold35_SUPERVISED_EXTENSION_FINAL_GOLD_v1.csv"
)

GOLD_FREEZE = (
    FINAL
    / "FROZEN_EVENTGOLD35_SUPERVISED_EXTENSION_FINAL_GOLD_SHA256SUMS_v1.txt"
)

EXTENSION_PROTOCOL = (
    FINAL
    / "EventGold35_SUPERVISED_EXTENSION_PROTOCOL_v1.txt"
)

TRAINER = (
    ROOT
    / "01_event_aware_v3"
    / "07_eve_frame_stage_a_no_event"
    / "37_train_eve_frame_stage_a_v3.py"
)

MANIFEST = (
    FINAL
    / "FINAL_REFIT_CHECKPOINT_MANIFEST_v1.csv"
)

CHECKPOINT_FREEZE = (
    FINAL
    / "FROZEN_FINAL_REFIT_CHECKPOINTS_SHA256SUMS_v1.txt"
)

RNG = (
    FINAL
    / "EVENTGOLD_BOOTSTRAP_RNG_v1.json"
)

OUTDIR = (
    FINAL
    / "eventgold35_supervised_extension_final_results_v1"
)

TMPDIR = (
    FINAL
    / "eventgold35_supervised_extension_final_results_v1_tmp"
)

REPORT = (
    FINAL
    / "EVENTGOLD35_SUPERVISED_EXTENSION_FINAL_PREFLIGHT_v1.txt"
)

REPORT_FREEZE = (
    FINAL
    / "FROZEN_EVENTGOLD35_SUPERVISED_EXTENSION_FINAL_PREFLIGHT_SHA256SUMS_v1.txt"
)


EXPECTED_SHA256: Dict[Path, str] = {
    EVAL:
        "a4a0a7d6c94f9d84f765568011a6cacd5cb0fa6ba14439b391faf03c610bb920",

    EVAL_LOCK:
        "1d4be4231fd73835b8653c65d919bb4006de883e5dd6ec3b5f46928eaf63c6de",

    EVAL_FREEZE:
        "6a6ef1cf495820cbb54dbb0a744ee0f5e9c6e1f0e72a2fc302417c61eb5e6d85",

    EVAL_PROTOCOL:
        "7dd54e5fe8ac6ccadeb21f882c446f0fd7e4860a38e06ecd34e57c5c6ca3c405",

    EVAL_PROTOCOL_FREEZE:
        "c4dba0d887b0e04cf8c8e233d5be35eeab134ad71de19803900a2d16294dcf09",

    GOLD:
        "4b2f271386a1b092e7d0b31ed382a234252ae649ff2580c75774a7276708fd6d",

    GOLD_FREEZE:
        "9b067f2a5567fbcdbe4659a0922725793de2746cd3f9ea9698c6d50eb26231cd",

    EXTENSION_PROTOCOL:
        "ff3d4f2361aa4fe6e435348a7b69b46bc9bca79245ffdc5d9e8da1c27ff46ff4",

    TRAINER:
        "62f5f55fcdeaf4e5c947313bcc326daa8ceb225e2ab954ffa7b24430e1e760b9",

    MANIFEST:
        "05d8cbfb7aba539d149af8925dda435de58cdfa4f5cd28e6d8b6344671da860a",

    CHECKPOINT_FREEZE:
        "946413f086bdb70aca8e905d5e95e646f54fd91e28929a6fbe2e4dfd2fb17cb2",

    RNG:
        "4c15cd3357629e5a0e9fc48c08e80ec7dcbf2a81e4800ad5308b64ea1b82405d",
}


TASKS = [
    "primary_frame",
    "stance",
    "misinformation_relation",
]

SEEDS = [
    11,
    29,
    47,
    83,
    131,
]

EXPECTED_MANIFEST_COLUMNS = {
    "run_name",
    "task",
    "seed",
    "final_refit_epoch",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "summary_sha256",
    "label_maps_sha256",
    "document_chunking_sha256",
}

REQUIRED_GOLD_COLUMNS = {
    "doc_id",
    "dapt_text",
    "final_primary_frame_validated",
    "final_stance_validated",
    "final_misinformation_relation_validated",
}

EXPECTED_RNG = {
    "primary_frame": 2026092701,
    "stance": 2026092702,
    "misinformation_relation": 2026092703,
}


# =============================================================================
# Helpers
# =============================================================================

def fail(message: str) -> "NoReturn":
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def assert_file(path: Path) -> None:
    if not path.is_file():
        fail(f"Missing required file: {path}")


def assert_hash(path: Path, expected: str) -> None:
    assert_file(path)

    observed = sha256(path)

    if observed != expected:
        fail(
            "SHA256 mismatch\n"
            f"FILE={path}\n"
            f"OBSERVED={observed}\n"
            f"EXPECTED={expected}"
        )

    print(f"HASH_OK={path}")


def resolve_checkpoint_path(raw_path: str) -> Path:
    p = Path(raw_path)

    if not p.is_absolute():
        p = ROOT / p

    return p.resolve()


def freeze_read_only(path: Path) -> None:
    os.chmod(path, 0o444)


# =============================================================================
# Checks
# =============================================================================

def check_project_root() -> None:
    if not FINAL.is_dir():
        fail(
            "Project root mismatch. Run this script from:\n"
            "~/Q1_MediaDiscourse_2026"
        )

    print(f"PROJECT_ROOT={ROOT}")


def check_new_artifact_namespace() -> None:
    for p in (REPORT, REPORT_FREEZE):
        if p.exists():
            fail(
                "Unexpected pre-existing final preflight artifact: "
                f"{p}"
            )

    for p in (OUTDIR, TMPDIR):
        if p.exists():
            fail(
                "Final result namespace already exists before "
                f"preflight: {p}"
            )

    print("FINAL_RESULT_DIRECTORY=ABSENT")
    print("TEMP_RESULT_DIRECTORY=ABSENT")


def check_frozen_package_hashes() -> None:
    print("\n=== FROZEN PACKAGE HASHES ===")

    for path, expected in EXPECTED_SHA256.items():
        assert_hash(
            path,
            expected,
        )


def check_freeze_manifest_contents() -> None:
    print("\n=== EVALUATOR FREEZE MANIFEST ===")

    expected_entries = {
        str(EVAL.relative_to(ROOT)):
            EXPECTED_SHA256[EVAL],
        str(EVAL_LOCK.relative_to(ROOT)):
            EXPECTED_SHA256[EVAL_LOCK],
        str(EVAL_PROTOCOL.relative_to(ROOT)):
            EXPECTED_SHA256[EVAL_PROTOCOL],
        str(EVAL_PROTOCOL_FREEZE.relative_to(ROOT)):
            EXPECTED_SHA256[EVAL_PROTOCOL_FREEZE],
        str(GOLD.relative_to(ROOT)):
            EXPECTED_SHA256[GOLD],
        str(GOLD_FREEZE.relative_to(ROOT)):
            EXPECTED_SHA256[GOLD_FREEZE],
        str(MANIFEST.relative_to(ROOT)):
            EXPECTED_SHA256[MANIFEST],
        str(CHECKPOINT_FREEZE.relative_to(ROOT)):
            EXPECTED_SHA256[CHECKPOINT_FREEZE],
        str(RNG.relative_to(ROOT)):
            EXPECTED_SHA256[RNG],
    }

    observed_entries: Dict[str, str] = {}

    for raw in EVAL_FREEZE.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw.strip()

        if not line:
            continue

        parts = line.split(
            maxsplit=1
        )

        if len(parts) != 2:
            fail(
                "Malformed evaluator freeze manifest line: "
                + raw
            )

        digest = parts[0].strip()

        name = (
            parts[1]
            .strip()
            .lstrip("*")
        )

        observed_entries[name] = digest

    if observed_entries != expected_entries:
        fail(
            "Evaluator freeze manifest contents do not match "
            "the frozen expected package."
        )

    print("EVALUATOR_FREEZE_MANIFEST_CONTENT=PASS")


def run_evaluator_self_check() -> None:
    print("\n=== EVALUATOR NON-PERFORMANCE SELF-CHECK ===")

    proc = subprocess.run(
        [
            sys.executable,
            str(EVAL),
            "--self-check-only",
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    print(proc.stdout, end="")

    if proc.returncode != 0:
        fail(
            "Frozen evaluator self-check returned "
            f"exit code {proc.returncode}"
        )

    required_markers = [
        "EVENTGOLD SUPERVISED-EXTENSION EVALUATOR SELF-CHECK OK",
        "Frozen checkpoint manifest : 15",
        "Frozen tasks               : 3",
        "Frozen seeds per task      : 5",
        "Bootstrap replicates       : 10000",
        "EventGold CSV parsed       : NO",
        "EventGold performance      : NO",
        "EVENTGOLD_STATUS=SEALED_NOT_ACCESSED",
    ]

    missing = [
        marker
        for marker in required_markers
        if marker not in proc.stdout
    ]

    if missing:
        fail(
            "Evaluator self-check output missing markers: "
            + repr(missing)
        )

    print("EVALUATOR_SELF_CHECK=PASS")


def load_checkpoint_manifest() -> List[dict]:
    print("\n=== CHECKPOINT MANIFEST ===")

    with MANIFEST.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        fields = set(
            reader.fieldnames or []
        )

        if fields != EXPECTED_MANIFEST_COLUMNS:
            fail(
                "Checkpoint manifest schema mismatch.\n"
                f"OBSERVED={sorted(fields)}\n"
                f"EXPECTED={sorted(EXPECTED_MANIFEST_COLUMNS)}"
            )

        rows = list(reader)

    if len(rows) != 15:
        fail(
            f"Expected 15 checkpoints; got {len(rows)}"
        )

    pairs = [
        (
            row["task"],
            int(row["seed"]),
        )
        for row in rows
    ]

    if len(set(pairs)) != 15:
        fail(
            "Duplicate task/seed pair in checkpoint manifest."
        )

    if set(
        row["task"]
        for row in rows
    ) != set(TASKS):
        fail(
            "Checkpoint task set mismatch."
        )

    for task in TASKS:
        observed = sorted(
            int(row["seed"])
            for row in rows
            if row["task"] == task
        )

        if observed != SEEDS:
            fail(
                f"Seed grid mismatch for {task}: {observed}"
            )

    print("CHECKPOINT_MANIFEST_ROWS=15")
    print("TASK_SEED_GRID=3x5_PASS")

    return rows


def check_checkpoints(rows: List[dict]) -> None:
    print("\n=== CHECKPOINT BYTE INTEGRITY ===")

    seen_paths = set()

    for i, row in enumerate(
        rows,
        start=1,
    ):

        p = resolve_checkpoint_path(
            row["checkpoint_path"]
        )

        if p in seen_paths:
            fail(
                f"Duplicate checkpoint path: {p}"
            )

        seen_paths.add(p)

        if not p.is_file():
            fail(
                f"Missing checkpoint: {p}"
            )

        observed_bytes = (
            p.stat().st_size
        )

        expected_bytes = int(
            row["checkpoint_bytes"]
        )

        if observed_bytes != expected_bytes:
            fail(
                "Checkpoint byte mismatch\n"
                f"FILE={p}\n"
                f"OBSERVED={observed_bytes}\n"
                f"EXPECTED={expected_bytes}"
            )

        observed_sha = sha256(p)

        expected_sha = (
            row["checkpoint_sha256"]
            .strip()
        )

        if observed_sha != expected_sha:
            fail(
                "Checkpoint SHA mismatch\n"
                f"FILE={p}\n"
                f"OBSERVED={observed_sha}\n"
                f"EXPECTED={expected_sha}"
            )

        print(
            "CHECKPOINT_OK="
            f"{i:02d}/15 "
            f"{row['task']} "
            f"seed={row['seed']} "
            f"epoch={row['final_refit_epoch']}"
        )

    if len(seen_paths) != 15:
        fail(
            "Expected 15 unique checkpoint paths."
        )

    print("CHECKPOINTS_VERIFIED=15")


def check_manifest_sidecar_hash_consistency(
    rows: List[dict],
) -> None:
    print("\n=== CHECKPOINT SIDECAR HASH CONSISTENCY ===")

    for column, label in [
        (
            "label_maps_sha256",
            "LABEL_MAPS",
        ),
        (
            "document_chunking_sha256",
            "DOCUMENT_CHUNKING",
        ),
    ]:
        vals = {
            str(row[column]).strip()
            for row in rows
        }

        if len(vals) != 1:
            fail(
                f"{label} SHA differs across checkpoints: "
                f"{sorted(vals)}"
            )

        value = next(iter(vals))

        if len(value) != 64:
            fail(
                f"{label} SHA is not a 64-character digest."
            )

        print(
            f"CHECKPOINT_{label}_SHA_CONSISTENT=YES"
        )


def check_final_gold_structure() -> None:
    print("\n=== FINAL GOLD STRUCTURE ===")

    with GOLD.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        fields = set(
            reader.fieldnames or []
        )

        rows = list(reader)

    missing = sorted(
        REQUIRED_GOLD_COLUMNS
        - fields
    )

    if missing:
        fail(
            "Final gold missing required fields: "
            + repr(missing)
        )

    if len(rows) != 35:
        fail(
            f"Expected 35 final-gold rows; got {len(rows)}"
        )

    doc_ids = [
        str(
            row["doc_id"]
        )
        for row in rows
    ]

    if any(
        not x.strip()
        for x in doc_ids
    ):
        fail(
            "Missing/empty final-gold doc_id."
        )

    if len(set(doc_ids)) != 35:
        fail(
            "Final gold doc_id values are not unique."
        )

    for row in rows:

        if not str(
            row["dapt_text"]
        ).strip():

            fail(
                "Empty final-gold dapt_text."
            )

        for col in [
            "final_primary_frame_validated",
            "final_stance_validated",
            "final_misinformation_relation_validated",
        ]:

            if not str(
                row[col]
            ).strip():

                fail(
                    f"Empty final-gold label: {col}"
                )

    print("FINAL_GOLD_ROWS=35")
    print("FINAL_GOLD_UNIQUE_DOC_IDS=35")
    print("FINAL_GOLD_STRUCTURE=PASS")


def check_rng() -> None:
    print("\n=== FROZEN BOOTSTRAP RNG ===")

    x = json.loads(
        RNG.read_text(
            encoding="utf-8"
        )
    )

    if x.get(
        "method"
    ) != "numpy.random.default_rng":

        fail(
            "Frozen RNG method mismatch."
        )

    if int(
        x.get(
            "bootstrap_replicates",
            -1,
        )
    ) != 10000:

        fail(
            "Frozen bootstrap replicate count mismatch."
        )

    if x.get(
        "ci_percentiles"
    ) != [0.025, 0.975]:

        fail(
            "Frozen CI percentiles mismatch."
        )

    if x.get(
        "stratification"
    ) != "gold_label_within_task":

        fail(
            "Frozen bootstrap stratification mismatch."
        )

    if x.get(
        "seeds"
    ) != EXPECTED_RNG:

        fail(
            "Frozen bootstrap RNG seed mapping mismatch."
        )

    print("BOOTSTRAP_REPLICATES=10000")
    print("BOOTSTRAP_RNG=PASS")


def check_no_forward_execution() -> None:
    """
    Static guard only. It proves this preflight script itself contains
    no call to forward_document or model(...). The evaluator self-check
    is separately required to report that EventGold was not parsed and
    no performance was computed.
    """
    source = Path(__file__).read_text(
        encoding="utf-8"
    )

    forbidden_runtime_calls = [
        ".forward_document(",
        "torch.softmax(",
        ".argmax(",
        "metric_dict(",
        "bootstrap_macro_f1(",
    ]

    # The strings above occur in this guard itself, so inspect executable
    # AST call targets instead of doing raw substring matching.
    import ast

    tree = ast.parse(source)

    calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func

        if isinstance(func, ast.Attribute):
            calls.append(func.attr)

        elif isinstance(func, ast.Name):
            calls.append(func.id)

    forbidden_names = {
        "forward_document",
        "softmax",
        "argmax",
        "metric_dict",
        "bootstrap_macro_f1",
    }

    hits = sorted(
        forbidden_names
        & set(calls)
    )

    if hits:
        fail(
            "Preflight script contains forbidden model/performance calls: "
            + repr(hits)
        )

    print("MODEL_FORWARD_CALLS=0")
    print("PERFORMANCE_METRICS_COMPUTED=0")
    print("NON_PERFORMANCE_PREFLIGHT_CORE=PASS")


def write_final_report() -> None:
    print("\n=== WRITE FINAL PREFLIGHT REPORT ===")

    if REPORT.exists() or REPORT_FREEZE.exists():
        fail(
            "Preflight output unexpectedly exists before write."
        )

    text = """EVE-FRAME — EVENTGOLD35 SUPERVISED EXTENSION FINAL PREFLIGHT v1
=================================================================

STATUS=PASS

FINAL_ARCHITECTURE=STAGE_A_NO_EVENT
FINAL_EVALUATOR_FROZEN=YES
FINAL_EVALUATION_PROTOCOL_FROZEN=YES
FINAL_HUMAN_GOLD_FROZEN=YES
FINAL_STAGE_A_CHECKPOINTS_FROZEN=YES

CHECKPOINTS_VERIFIED=15
TASKS=3
SEEDS_PER_TASK=5
TASK_SEED_GRID=3x5_PASS
CHECKPOINT_LABEL_MAP_SHA_CONSISTENT=YES
CHECKPOINT_DOCUMENT_CHUNKING_SHA_CONSISTENT=YES

FINAL_GOLD_ROWS=35
FINAL_GOLD_UNIQUE_DOC_IDS=35
FINAL_GOLD_STRUCTURE=PASS

BOOTSTRAP_REPLICATES=10000
BOOTSTRAP_RNG=PASS

MODEL_FORWARD_CALLS=0
MODEL_PREDICTIONS_COMPUTED=NO
PERFORMANCE_AVAILABLE=NO

FINAL_RESULT_DIRECTORY=ABSENT
TEMP_RESULT_DIRECTORY=ABSENT

METHODOLOGY_FROZEN=YES
FINAL_PREFLIGHT=PASS
NEXT_ALLOWED_PHASE=FINAL_GPU_EVALUATION
"""

    tmp_report = REPORT.with_name(
        REPORT.name + ".tmp"
    )

    if tmp_report.exists():
        fail(
            f"Unexpected temporary report exists: {tmp_report}"
        )

    tmp_report.write_text(
        text,
        encoding="utf-8",
    )

    os.replace(
        tmp_report,
        REPORT,
    )

    manifest_targets = [
        REPORT,
        EVAL,
        EVAL_LOCK,
        EVAL_FREEZE,
        EVAL_PROTOCOL,
        EVAL_PROTOCOL_FREEZE,
        GOLD,
        GOLD_FREEZE,
        MANIFEST,
        CHECKPOINT_FREEZE,
        RNG,
    ]

    lines = []

    for path in manifest_targets:
        lines.append(
            f"{sha256(path)}  "
            f"{path.relative_to(ROOT)}"
        )

    tmp_freeze = REPORT_FREEZE.with_name(
        REPORT_FREEZE.name + ".tmp"
    )

    if tmp_freeze.exists():
        fail(
            f"Unexpected temporary freeze exists: {tmp_freeze}"
        )

    tmp_freeze.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp_freeze,
        REPORT_FREEZE,
    )

    freeze_read_only(
        REPORT
    )

    freeze_read_only(
        REPORT_FREEZE
    )

    print(
        f"PREFLIGHT_REPORT_SHA256={sha256(REPORT)}"
    )

    print(
        "PREFLIGHT_FREEZE_SHA256="
        f"{sha256(REPORT_FREEZE)}"
    )

    print(
        f"PREFLIGHT_REPORT_MODE={oct(REPORT.stat().st_mode & 0o777)}"
    )

    print(
        f"PREFLIGHT_FREEZE_MODE={oct(REPORT_FREEZE.stat().st_mode & 0o777)}"
    )


def main() -> None:
    print(
        "=" * 79
    )
    print(
        "EVE-FRAME FINAL NON-PERFORMANCE PREFLIGHT v1"
    )
    print(
        "=" * 79
    )

    check_project_root()

    check_new_artifact_namespace()

    check_frozen_package_hashes()

    check_freeze_manifest_contents()

    run_evaluator_self_check()

    rows = load_checkpoint_manifest()

    check_checkpoints(
        rows
    )

    check_manifest_sidecar_hash_consistency(
        rows
    )

    check_final_gold_structure()

    check_rng()

    check_no_forward_execution()

    # Re-check namespaces immediately before writing PASS report.
    if OUTDIR.exists() or TMPDIR.exists():
        fail(
            "Result namespace appeared during preflight."
        )

    write_final_report()

    print()
    print(
        "=" * 79
    )
    print(
        "FINAL PREFLIGHT PASS"
    )
    print(
        "=" * 79
    )
    print(
        "METHODOLOGY_FROZEN=YES"
    )
    print(
        "FINAL_EVALUATOR_FROZEN=YES"
    )
    print(
        "FINAL_PREFLIGHT=PASS"
    )
    print(
        "MODEL_PREDICTIONS_COMPUTED=NO"
    )
    print(
        "PERFORMANCE_AVAILABLE=NO"
    )
    print(
        "NEXT_PHASE=FINAL_GPU_EVALUATION"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            "\nFINAL_PREFLIGHT=FAIL",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
