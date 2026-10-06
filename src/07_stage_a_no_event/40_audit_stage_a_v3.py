from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path(
    "01_event_aware_v3/07_eve_frame_stage_a_no_event"
)

MANIFEST = ROOT / "stage_a_manifest_v3.csv"
RUNROOT = ROOT / "runs_v3"
OUTDIR = ROOT / "audit_v3"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

manifest = pd.read_csv(
    MANIFEST
)

rows = []

for r in manifest.itertuples(
    index=False
):

    run_name = (
        f"{r.target_task}"
        f"__fold{int(r.outer_fold)}"
        f"__seed{int(r.model_seed)}"
    )

    run_dir = RUNROOT / run_name

    summary_path = (
        run_dir
        / "run_summary_v3.json"
    )

    pred_path = (
        run_dir
        / "outer_test_predictions_v3.csv"
    )

    valid = False
    reason = ""
    elapsed = None
    peak = None

    if not summary_path.exists():
        reason = "missing_summary"

    elif not pred_path.exists():
        reason = "missing_predictions"

    else:
        try:
            s = json.loads(
                summary_path.read_text()
            )

            checks = [
                s.get("status")
                    == "COMPLETE",

                s.get("architecture")
                    == "EVE_FRAME_STAGE_A_NO_EVENT",

                s.get("event_context")
                    == "DISABLED",

                s.get("target_task")
                    == r.target_task,

                int(
                    s.get(
                        "outer_fold",
                        -1,
                    )
                )
                    == int(
                        r.outer_fold
                    ),

                int(
                    s.get(
                        "seed",
                        -1,
                    )
                )
                    == int(
                        r.model_seed
                    ),

                s.get("eventgold_status")
                    == "SEALED_NOT_ACCESSED",

                s.get(
                    "strict_deterministic_algorithms"
                )
                    is True,

                s.get(
                    "attention_implementation"
                )
                    == "eager",

                int(
                    s.get(
                        "refit_scheduler_horizon_epochs",
                        -1,
                    )
                )
                    == 8,
            ]

            valid = all(
                checks
            )

            if not valid:
                reason = (
                    "summary_validation_failed"
                )

            elapsed = s.get(
                "elapsed_seconds"
            )

            peak = s.get(
                "peak_gpu_memory_gb"
            )

        except Exception as e:
            reason = (
                f"exception:{type(e).__name__}"
            )

    rows.append({
        "run_id":
            int(r.run_id),

        "run_name":
            run_name,

        "valid":
            bool(valid),

        "reason":
            reason,

        "elapsed_seconds":
            elapsed,

        "peak_gpu_memory_gb":
            peak,
    })


audit = pd.DataFrame(
    rows
)

audit.to_csv(
    OUTDIR
    / "stage_a_run_audit_v3.csv",
    index=False,
)


valid_n = int(
    audit["valid"].sum()
)

summary = {
    "expected_runs":
        60,

    "summaries_found":
        int(
            sum(
                (
                    RUNROOT
                    / (
                        f"{r.target_task}"
                        f"__fold{int(r.outer_fold)}"
                        f"__seed{int(r.model_seed)}"
                    )
                    / "run_summary_v3.json"
                ).exists()
                for r
                in manifest.itertuples(
                    index=False
                )
            )
        ),

    "valid_runs":
        valid_n,

    "missing_or_invalid_runs":
        int(
            60
            - valid_n
        ),

    "all_60_valid":
        bool(
            valid_n == 60
        ),

    "total_recorded_gpu_hours":
        float(
            pd.to_numeric(
                audit[
                    "elapsed_seconds"
                ],
                errors="coerce",
            ).fillna(
                0
            ).sum()
            / 3600
        ),

    "max_peak_gpu_memory_gb":
        (
            float(
                pd.to_numeric(
                    audit[
                        "peak_gpu_memory_gb"
                    ],
                    errors="coerce",
                ).max()
            )
            if audit[
                "peak_gpu_memory_gb"
            ].notna().any()
            else None
        ),
}

summary_path = (
    OUTDIR
    / "stage_a_audit_summary_v3.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)

print(
    json.dumps(
        summary,
        indent=2,
    )
)

if not summary[
    "all_60_valid"
]:
    print()
    print(
        audit[
            ~audit["valid"]
        ].to_string(
            index=False
        )
    )
