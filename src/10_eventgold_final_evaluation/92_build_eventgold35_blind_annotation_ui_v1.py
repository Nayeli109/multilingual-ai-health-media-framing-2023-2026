from pathlib import Path
import csv
import hashlib
import html
import json


ROOT = Path.cwd()

FINAL = (
    ROOT
    / "01_event_aware_v3"
    / "10_eventgold_final_evaluation"
)

EXT = FINAL / "eventgold35_supervised_extension_v1"

PROTOCOL = (
    FINAL
    / "EventGold35_SUPERVISED_EXTENSION_PROTOCOL_v1.txt"
)

CSV_A = EXT / "EventGold35_ANNOTATOR_A_BLIND_v1.csv"
CSV_B = EXT / "EventGold35_ANNOTATOR_B_BLIND_v1.csv"

UI_A = EXT / "EventGold35_ANNOTATOR_A_OFFLINE_v1.html"
UI_B = EXT / "EventGold35_ANNOTATOR_B_OFFLINE_v1.html"

MANIFEST = EXT / "EventGold35_ANNOTATION_UI_MANIFEST_v1.json"


FRAME_OPTIONS = [
    "clinical innovation and diagnostic benefit",
    "health safety risk and medical harm",
    "misinformation, deepfakes, and fraud",
    "regulation, governance, and policy",
    "public trust and accountability",
    "access, inequality, and digital divide",
    "labor automation and professional displacement",
    "mental health and therapy risk",
    "other / unclear",
]

STANCE_OPTIONS = [
    "optimistic about AI in health",
    "cautious or mixed about AI in health",
    "critical of AI risks in health",
    "focused on regulation and governance",
    "unclear / not applicable",
]

MISINFO_OPTIONS = [
    "direct misinformation focus",
    "indirect misinformation concern",
    "no clear misinformation relation",
    "unclear / not applicable",
]


FRAME_DEFS = {
    "clinical innovation and diagnostic benefit":
        "AI improves diagnosis, imaging, clinical workflows, screening, treatment support, or healthcare efficiency.",

    "health safety risk and medical harm":
        "Patient risk, unsafe deployment, incorrect medical advice, bias, hallucinations, harm, or clinical unreliability.",

    "misinformation, deepfakes, and fraud":
        "False information, synthetic media, scams, fake ads, deepfakes, manipulated content, or misleading health claims.",

    "regulation, governance, and policy":
        "Laws, regulation, institutional governance, standards, enforcement, or policy debates.",

    "public trust and accountability":
        "Trust, transparency, responsibility, accountability, institutional credibility, or public confidence.",

    "access, inequality, and digital divide":
        "Unequal access, geographic disparities, cost/language barriers, or technological dependence.",

    "labor automation and professional displacement":
        "Job replacement, professional roles, productivity, workers, clinicians, students, or labor automation.",

    "mental health and therapy risk":
        "AI therapy, mental-health chatbots, suicide, psychosis, emotional dependence, psychological or psychiatric risk.",

    "other / unclear":
        "Use only when none of the other primary-frame labels clearly applies.",
}


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_packet(path, suffix):

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    expected = [
        "annotation_id",
        "title",
        "annotation_excerpt",
        f"human_primary_frame_{suffix}",
        f"human_stance_{suffix}",
        f"human_misinformation_relation_{suffix}",
        f"human_notes_{suffix}",
    ]

    if not rows:
        raise RuntimeError(
            f"Empty packet: {path}"
        )

    if list(rows[0].keys()) != expected:
        raise RuntimeError(
            f"Unexpected packet schema for {suffix}: "
            f"{list(rows[0].keys())}"
        )

    if len(rows) != 35:
        raise RuntimeError(
            f"Expected 35 rows for {suffix}; "
            f"observed {len(rows)}."
        )

    ids = [
        r["annotation_id"].strip()
        for r in rows
    ]

    if len(set(ids)) != 35:
        raise RuntimeError(
            f"annotation_id uniqueness failure for {suffix}"
        )

    for r in rows:

        if not r["annotation_id"].strip():
            raise RuntimeError(
                f"Missing annotation_id in {suffix}"
            )

        if not r["title"].strip():
            raise RuntimeError(
                f"Missing title in {suffix}"
            )

        if not r["annotation_excerpt"].strip():
            raise RuntimeError(
                f"Missing excerpt in {suffix}"
            )

        if len(r["annotation_excerpt"]) > 1400:
            raise RuntimeError(
                f"Excerpt >1400 characters in {suffix}"
            )

        for c in expected[3:]:
            if r[c].strip():
                raise RuntimeError(
                    f"Nonblank annotation field before annotation: {c}"
                )

    return rows, expected


def js_json(x):
    return json.dumps(
        x,
        ensure_ascii=False,
    ).replace("</", "<\\/")


def make_html(rows, suffix, columns):

    role = f"Annotator {suffix}"

    storage_key = (
        f"eventgold35_supervised_extension_v1_annotator_{suffix}"
    )

    export_name = (
        f"EventGold35_ANNOTATOR_{suffix}_COMPLETED_v1.csv"
    )

    frame_field = f"human_primary_frame_{suffix}"
    stance_field = f"human_stance_{suffix}"
    misinfo_field = f"human_misinformation_relation_{suffix}"
    notes_field = f"human_notes_{suffix}"

    payload = []

    for r in rows:
        payload.append({
            "annotation_id": r["annotation_id"],
            "title": r["title"],
            "annotation_excerpt": r["annotation_excerpt"],
        })

    frame_options_html = "\n".join(
        f'<option value="{html.escape(x)}">{html.escape(x)}</option>'
        for x in FRAME_OPTIONS
    )

    stance_options_html = "\n".join(
        f'<option value="{html.escape(x)}">{html.escape(x)}</option>'
        for x in STANCE_OPTIONS
    )

    misinfo_options_html = "\n".join(
        f'<option value="{html.escape(x)}">{html.escape(x)}</option>'
        for x in MISINFO_OPTIONS
    )

    frame_defs_html = "\n".join(
        (
            '<div class="definition">'
            f'<b>{html.escape(label)}</b><br>'
            f'{html.escape(FRAME_DEFS[label])}'
            '</div>'
        )
        for label in FRAME_OPTIONS
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>EventGold35 Blind Annotation — {role}</title>

<style>
body {{
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #f5f7f9;
    color: #17212b;
}}

header {{
    position: sticky;
    top: 0;
    z-index: 20;
    background: white;
    border-bottom: 1px solid #d8dee5;
    padding: 14px 22px;
}}

header h1 {{
    margin: 0 0 8px 0;
    font-size: 21px;
}}

.status {{
    font-size: 14px;
    font-weight: 700;
}}

.controls {{
    margin-top: 10px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}}

button {{
    border: 0;
    border-radius: 7px;
    padding: 9px 13px;
    font-weight: 700;
    cursor: pointer;
}}

.primary {{
    background: #173f67;
    color: white;
}}

.secondary {{
    background: #e6eaf0;
    color: #17212b;
}}

main {{
    max-width: 1050px;
    margin: 22px auto 80px auto;
    padding: 0 16px;
}}

.notice {{
    background: #fff8e6;
    border: 1px solid #f1ce73;
    border-radius: 8px;
    padding: 13px;
    margin-bottom: 18px;
    line-height: 1.45;
}}

.codebook {{
    background: white;
    border: 1px solid #d8dee5;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 18px;
}}

.definition {{
    margin: 9px 0;
    font-size: 13px;
    line-height: 1.4;
}}

.card {{
    background: white;
    border: 1px solid #d8dee5;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 18px;
}}

.annotation-id {{
    font-size: 12px;
    color: #687785;
    margin-bottom: 8px;
}}

.title {{
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 13px;
}}

.excerpt {{
    white-space: pre-wrap;
    line-height: 1.6;
    padding: 14px;
    background: #f8fafb;
    border: 1px solid #e1e6eb;
    border-radius: 8px;
    margin-bottom: 18px;
}}

.field {{
    margin: 15px 0;
}}

label {{
    display: block;
    font-weight: 700;
    margin-bottom: 6px;
}}

select, textarea {{
    width: 100%;
    box-sizing: border-box;
    padding: 10px;
    border: 1px solid #bcc6cf;
    border-radius: 7px;
    font-size: 14px;
    background: white;
}}

textarea {{
    min-height: 80px;
    resize: vertical;
}}

.done {{
    border-left: 6px solid #5f7d65;
}}

.incomplete {{
    border-left: 6px solid #ad7d35;
}}
</style>
</head>

<body>

<header>
<h1>EventGold35 — Blind Human Annotation ({role})</h1>

<div class="status" id="status">
0 / 35 documents complete
</div>

<div class="controls">
<button class="primary" onclick="exportCSV()">
Export completed CSV
</button>

<button class="secondary" onclick="jumpIncomplete()">
First incomplete
</button>

<button class="secondary" onclick="saveNow()">
Save locally
</button>
</div>
</header>

<main>

<div class="notice">
<b>Blinded annotation.</b><br>
Annotate each document independently using only the title,
the frozen excerpt, and the codebook below.

Do not consult model predictions, probabilities, zero-shot outputs,
the other annotator's decisions, or original EventGold semantic fields.
</div>

<div class="codebook">

<h2>Frozen codebook</h2>

<h3>Primary frame</h3>
{frame_defs_html}

<h3>Stance toward AI in health</h3>

<div class="definition">
<b>optimistic about AI in health</b><br>
Benefits, promise, innovation, progress, or positive transformation.
</div>

<div class="definition">
<b>cautious or mixed about AI in health</b><br>
Both benefits and risks, or balanced/uncertain tone.
</div>

<div class="definition">
<b>critical of AI risks in health</b><br>
Danger, harm, misinformation, fraud, bias, or negative consequences.
</div>

<div class="definition">
<b>focused on regulation and governance</b><br>
Rules, policy, governance, regulation, or institutional oversight
rather than a clearly positive or negative evaluation.
</div>

<div class="definition">
<b>unclear / not applicable</b><br>
The stance cannot be inferred.
</div>

<h3>Relation to misinformation</h3>

<div class="definition">
<b>direct misinformation focus</b><br>
The article is mainly about misinformation, deepfakes, fake ads,
scams, false health claims, or propaganda.
</div>

<div class="definition">
<b>indirect misinformation concern</b><br>
Misinformation is a risk/background concern but not the main focus.
</div>

<div class="definition">
<b>no clear misinformation relation</b><br>
The article does not meaningfully discuss misinformation.
</div>

<div class="definition">
<b>unclear / not applicable</b><br>
The title/excerpt does not support a reliable classification.
</div>

</div>

<div id="cards"></div>

</main>

<script>

const DOCUMENTS = {js_json(payload)};

const FRAME_OPTIONS = {js_json(FRAME_OPTIONS)};
const STANCE_OPTIONS = {js_json(STANCE_OPTIONS)};
const MISINFO_OPTIONS = {js_json(MISINFO_OPTIONS)};

const STORAGE_KEY = {js_json(storage_key)};

const FRAME_FIELD = {js_json(frame_field)};
const STANCE_FIELD = {js_json(stance_field)};
const MISINFO_FIELD = {js_json(misinfo_field)};
const NOTES_FIELD = {js_json(notes_field)};

const EXPORT_NAME = {js_json(export_name)};

let annotations = {{}};


function esc(s) {{
    return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
}}


function optionHTML(options) {{

    let x = '<option value="">-- select --</option>';

    for (const v of options) {{
        x += `<option value="${{esc(v)}}">${{esc(v)}}</option>`;
    }}

    return x;
}}


function ensureRecord(id) {{

    if (!annotations[id]) {{
        annotations[id] = {{
            frame: "",
            stance: "",
            misinfo: "",
            notes: ""
        }};
    }}

    return annotations[id];
}}


function render() {{

    const container = document.getElementById("cards");

    let out = "";

    DOCUMENTS.forEach((d, i) => {{

        const a = ensureRecord(d.annotation_id);

        out += `
        <div class="card incomplete"
             id="card_${{i}}">

            <div class="annotation-id">
                ${{esc(d.annotation_id)}} · Document ${{i + 1}} of ${{DOCUMENTS.length}}
            </div>

            <div class="title">
                ${{esc(d.title)}}
            </div>

            <div class="excerpt">
                ${{esc(d.annotation_excerpt)}}
            </div>

            <div class="field">
                <label>Primary frame</label>
                <select
                    id="frame_${{i}}"
                    onchange="update(${{i}})"
                >
                    ${{optionHTML(FRAME_OPTIONS)}}
                </select>
            </div>

            <div class="field">
                <label>Stance toward AI in health</label>
                <select
                    id="stance_${{i}}"
                    onchange="update(${{i}})"
                >
                    ${{optionHTML(STANCE_OPTIONS)}}
                </select>
            </div>

            <div class="field">
                <label>Relation to misinformation</label>
                <select
                    id="misinfo_${{i}}"
                    onchange="update(${{i}})"
                >
                    ${{optionHTML(MISINFO_OPTIONS)}}
                </select>
            </div>

            <div class="field">
                <label>Optional notes</label>
                <textarea
                    id="notes_${{i}}"
                    oninput="update(${{i}})"
                    placeholder="Ambiguity, insufficient context, mixed framing, translation uncertainty..."
                ></textarea>
            </div>

        </div>
        `;
    }});

    container.innerHTML = out;

    DOCUMENTS.forEach((d, i) => {{

        const a = ensureRecord(d.annotation_id);

        document.getElementById(`frame_${{i}}`).value =
            a.frame || "";

        document.getElementById(`stance_${{i}}`).value =
            a.stance || "";

        document.getElementById(`misinfo_${{i}}`).value =
            a.misinfo || "";

        document.getElementById(`notes_${{i}}`).value =
            a.notes || "";
    }});

    refreshStatus();
}}


function update(i) {{

    const d = DOCUMENTS[i];
    const a = ensureRecord(d.annotation_id);

    a.frame =
        document.getElementById(`frame_${{i}}`).value;

    a.stance =
        document.getElementById(`stance_${{i}}`).value;

    a.misinfo =
        document.getElementById(`misinfo_${{i}}`).value;

    a.notes =
        document.getElementById(`notes_${{i}}`).value;

    saveNow();
    refreshStatus();
}}


function isComplete(a) {{
    return Boolean(
        a.frame &&
        a.stance &&
        a.misinfo
    );
}}


function refreshStatus() {{

    let complete = 0;

    DOCUMENTS.forEach((d, i) => {{

        const a = ensureRecord(d.annotation_id);
        const card = document.getElementById(`card_${{i}}`);

        if (isComplete(a)) {{
            complete += 1;
            card.classList.remove("incomplete");
            card.classList.add("done");
        }}
        else {{
            card.classList.remove("done");
            card.classList.add("incomplete");
        }}
    }});

    document.getElementById("status").textContent =
        `${{complete}} / ${{DOCUMENTS.length}} documents complete`;
}}


function saveNow() {{

    localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(annotations)
    );
}}


function loadSaved() {{

    const raw = localStorage.getItem(STORAGE_KEY);

    if (!raw) {{
        return;
    }}

    try {{
        const obj = JSON.parse(raw);

        if (
            obj &&
            typeof obj === "object"
        ) {{
            annotations = obj;
        }}
    }}
    catch (e) {{
        alert(
            "Could not load local annotation backup."
        );
    }}
}}


function jumpIncomplete() {{

    for (
        let i = 0;
        i < DOCUMENTS.length;
        i++
    ) {{

        const a = ensureRecord(
            DOCUMENTS[i].annotation_id
        );

        if (!isComplete(a)) {{

            document
                .getElementById(`card_${{i}}`)
                .scrollIntoView({{
                    behavior: "smooth",
                    block: "start"
                }});

            return;
        }}
    }}

    alert("All 35 documents are complete.");
}}


function csvEscape(x) {{

    const s = String(
        x === undefined || x === null
            ? ""
            : x
    );

    return '"' + s.replaceAll('"', '""') + '"';
}}


function exportCSV() {{

    const incomplete = [];

    DOCUMENTS.forEach((d, i) => {{

        const a = ensureRecord(d.annotation_id);

        if (!isComplete(a)) {{
            incomplete.push(i + 1);
        }}
    }});

    if (incomplete.length) {{

        alert(
            "Cannot export final CSV. "
            + incomplete.length
            + " documents are incomplete."
        );

        return;
    }}

    const header = [
        "annotation_id",
        "title",
        "annotation_excerpt",
        FRAME_FIELD,
        STANCE_FIELD,
        MISINFO_FIELD,
        NOTES_FIELD
    ];

    const lines = [
        header.map(csvEscape).join(",")
    ];

    DOCUMENTS.forEach((d) => {{

        const a = ensureRecord(d.annotation_id);

        const row = [
            d.annotation_id,
            d.title,
            d.annotation_excerpt,
            a.frame,
            a.stance,
            a.misinfo,
            a.notes
        ];

        lines.push(
            row.map(csvEscape).join(",")
        );
    }});

    const blob = new Blob(
        [lines.join("\\n") + "\\n"],
        {{
            type: "text/csv;charset=utf-8"
        }}
    );

    const url =
        URL.createObjectURL(blob);

    const link =
        document.createElement("a");

    link.href = url;
    link.download = EXPORT_NAME;

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);

    URL.revokeObjectURL(url);
}}


loadSaved();
render();

</script>

</body>
</html>
"""


rows_a, cols_a = read_packet(
    CSV_A,
    "A",
)

rows_b, cols_b = read_packet(
    CSV_B,
    "B",
)


# Both annotators must receive the exact same
# documents in the exact same frozen order.
ids_a = [
    r["annotation_id"]
    for r in rows_a
]

ids_b = [
    r["annotation_id"]
    for r in rows_b
]

if ids_a != ids_b:
    raise RuntimeError(
        "A/B annotation order mismatch."
    )


content_a = make_html(
    rows_a,
    "A",
    cols_a,
)

content_b = make_html(
    rows_b,
    "B",
    cols_b,
)


# Fail-closed checks against private/model information.
for name, content in [
    ("A", content_a),
    ("B", content_b),
]:

    prohibited_literals = [
        "doc_id",
        "problem_definition",
        "evaluation_tone",
        "coder_confidence",
        "machine_primary_frame",
        "machine_stance",
        "primary_frame_zs",
        "stance_zs",
        "probability",
        "logit",
    ]

    for token in prohibited_literals:
        if token in content:
            raise RuntimeError(
                f"Prohibited literal in UI {name}: {token}"
            )


UI_A.write_text(
    content_a,
    encoding="utf-8",
)

UI_B.write_text(
    content_b,
    encoding="utf-8",
)


manifest = {
    "version": "v1",
    "purpose": (
        "offline blinded independent annotation "
        "of prospective EventGold35 supervised extension"
    ),
    "rows_per_annotator": 35,
    "same_order_for_A_and_B": True,
    "annotation_id_visible": True,
    "doc_id_visible": False,
    "original_eventgold_semantic_fields_visible": False,
    "model_predictions_visible": False,
    "model_probabilities_visible": False,
    "external_network_required": False,
    "browser_local_storage_enabled": True,
    "final_export_requires_all_three_tasks": True,
    "annotator_a_source_sha256": sha256(CSV_A),
    "annotator_b_source_sha256": sha256(CSV_B),
    "protocol_sha256": sha256(PROTOCOL),
    "ui_a_sha256": sha256(UI_A),
    "ui_b_sha256": sha256(UI_B),
}

MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


print("ANNOTATOR_A_ROWS=35")
print("ANNOTATOR_B_ROWS=35")
print("A_B_ORDER_IDENTICAL=YES")
print("DOC_ID_VISIBLE=NO")
print("ORIGINAL_EVENTGOLD_SEMANTIC_FIELDS_VISIBLE=NO")
print("MODEL_PREDICTIONS_VISIBLE=NO")
print("EXTERNAL_NETWORK_REQUIRED=NO")
print("FINAL_EXPORT_REQUIRES_COMPLETE_LABELS=YES")
print("BLIND_ANNOTATION_UI_BUILD=PASS")
