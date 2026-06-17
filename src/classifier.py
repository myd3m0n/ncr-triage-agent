import json, re
from llm import complete
DEFECT_TYPES = ["Surface Scratch/Dig","Fiber Alignment / Insertion Loss","Dimensional Out-of-Tolerance","Solder / Bond Defect","Contamination / Coating Defect"]
SEVERITIES = ["Minor","Major","Critical"]
SYSTEM = """You are a manufacturing quality inspector that classifies nonconforming material reports (NCRs). You are precise and never invent categories.

Use these keyword hints to pick the defect_type:
- Surface Scratch/Dig: scratch, dig, scratch-dig spec (e.g. 10-5, 40-20), clear aperture, optical surface blemish.
- Fiber Alignment / Insertion Loss: insertion loss, IL, dB, core offset, re-termination, splice, connector, alignment. Optical loss measured in dB is ALWAYS this type, even when written as a measured-vs-target value.
- Dimensional Out-of-Tolerance: a physical/geometric dimension out of spec - mm, GD&T, datum, position, diameter, bore, slot/step/hole, length/width/height. Use only for geometric size or position, never for optical loss.
- Solder / Bond Defect: solder, cold solder joint, reflow, voiding, bridging, wire bond, pull test, IPC.
- Contamination / Coating Defect: contamination, particulate, particles, coating."""
PROMPT_TEMPLATE = """Classify the following NCR description.

Choose the defect_type from EXACTLY this list (copy the label verbatim):
{types}

Choose severity from: {sevs}
Give a confidence between 0.0 and 1.0.

Respond with ONLY a JSON object, no preamble, no markdown, in this exact shape:
{{"defect_type": "...", "severity": "...", "confidence": 0.0}}

NCR description:
\"\"\"{desc}\"\"\""""
def _extract_json(text):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group(0))
def _validate(obj):
    dtype = obj.get("defect_type", "")
    if dtype not in DEFECT_TYPES:
        dtype = next((t for t in DEFECT_TYPES if t.lower()==str(dtype).lower()), "Unknown")
    sev = obj.get("severity", "")
    if sev not in SEVERITIES: sev = "Unknown"
    try: conf = round(float(obj.get("confidence", 0.0)), 2)
    except (TypeError, ValueError): conf = 0.0
    return {"defect_type": dtype, "severity": sev, "confidence": conf}
def classify(description):
    prompt = PROMPT_TEMPLATE.format(types="\n".join(f"- {t}" for t in DEFECT_TYPES), sevs=", ".join(SEVERITIES), desc=description)
    raw = complete(prompt, system=SYSTEM, temperature=0.0)
    try:
        return _validate(_extract_json(raw))
    except Exception:
        return {"defect_type":"Unknown","severity":"Unknown","confidence":0.0}
if __name__ == "__main__":
    print(classify("Insertion loss measured 1.1 dB, exceeds 0.5 dB max after active alignment."))
