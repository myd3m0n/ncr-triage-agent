import csv, json, random, hashlib
from datetime import date, timedelta
from pathlib import Path
random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; EVAL = ROOT / "eval"
ARCHETYPES = [
 {"process":"Optical Assembly","part_prefix":"OPT","part_names":["Lens Cell","Beam Splitter Mount","Mirror Substrate","Window Assembly"],"defect_type":"Surface Scratch/Dig","templates":["Inspection found a {size} um scratch on the {surface} optical surface, exceeds scratch-dig spec {spec}.","Visible dig defect near clear aperture on {surface} side; measured {size} um, spec is {spec}.","Surface scratch on coated face observed under {light}; length approx {size} um, outside {spec}."],"fills":{"size":[40,60,80,120,150],"surface":["front","rear","S1","S2"],"spec":["20-10","40-20","10-5"],"light":["collimated inspection lamp","fiber illuminator","dark-field"]}},
 {"process":"Fiber Assembly","part_prefix":"FBR","part_names":["Fiber Pigtail","Collimator Assembly","Splice Module","Connector Term"],"defect_type":"Fiber Alignment / Insertion Loss","templates":["Insertion loss measured {loss} dB, exceeds {spec} dB max after active alignment.","Return loss {rl} dB on {conn} connector, below {spec_rl} dB requirement.","Core offset suspected; IL {loss} dB vs {spec} dB target, re-termination needed."],"fills":{"loss":[0.6,0.8,1.1,1.5,2.0],"spec":[0.5,0.3],"rl":[-35,-40,-42],"spec_rl":[-45,-50],"conn":["LC","FC/APC","SC"]}},
 {"process":"Precision Machining","part_prefix":"MCH","part_names":["Housing","Flexure Mount","Baseplate","Spacer Ring"],"defect_type":"Dimensional Out-of-Tolerance","templates":["{feature} measured {actual} mm, drawing calls {nominal} +/- {tol} mm. Out of tolerance.","Bore diameter {actual} mm vs {nominal} mm nominal ({tol} mm tol); oversize.","{feature} out by {dev} mm relative to datum, exceeds {tol} mm GD&T callout."],"fills":{"feature":["Bore dia","Slot width","Hole position","Step height"],"actual":[10.06,24.98,5.13,12.07],"nominal":[10.00,25.00,5.00,12.00],"tol":[0.02,0.05,0.01],"dev":[0.04,0.08,0.12]}},
 {"process":"Electronics / Bonding","part_prefix":"ELx","part_names":["Driver PCBA","Wire-bond Module","Solder Assembly","Flex Cable"],"defect_type":"Solder / Bond Defect","templates":["Cold solder joint on {ref}; reflow voiding {void}% exceeds {spec}% IPC limit.","Wire bond lifted on pad {ref}; pull test {pull} g below {spec_pull} g minimum.","Bridging observed between {ref} and adjacent pad after reflow."],"fills":{"ref":["U3","R12","J1 pin 4","Q7"],"void":[28,35,40],"spec":[25],"pull":[2.1,3.0,1.8],"spec_pull":[4.0]}},
 {"process":"Coating / Cleanliness","part_prefix":"CTG","part_names":["AR-Coated Optic","Filter","Coated Window","Reflector"],"defect_type":"Contamination / Coating Defect","templates":["Particulate contamination on coated surface, {count} particles > {size} um in clear aperture.","Coating pinhole defect, {count} sites observed under {light}.","Residue / haze on {surface} surface after coating, fails cleanliness {spec}."],"fills":{"count":[3,5,8,12],"size":[25,50,100],"light":["dark-field","UV lamp"],"surface":["front","rear"],"spec":["MIL-C-48497","Level 50"]}},
]
# --- MRB decision history (disposition + root_cause) ----------------------------
# Derived deterministically from sha256(ncr_id) — entirely OUTSIDE the random.* stream
# above, so adding this cannot perturb any previously generated field. Disposition is a
# probabilistic function of (defect_type, process, per-record severity factor): each
# defect_type has a dominant disposition with a realistic minority spilling into more
# severe dispositions for high-severity / critical-part cases.

# Parts whose nonconformance is more consequential (clear-aperture optics, structural
# flexures, active drivers) -> their severity factor skews high, pulling the tail toward
# Scrap / Return to Vendor.
CRITICAL_PARTS = {
    "Mirror Substrate", "Beam Splitter Mount", "Collimator Assembly",
    "Flexure Mount", "Driver PCBA", "Wire-bond Module", "AR-Coated Optic",
}

# Per defect_type: (disposition, base_weight, severity_gain).
# effective_weight = max(eps, base_weight + severity_gain * s), s in [0,1].
# gain > 0  -> grows for severe/critical cases (Scrap, Return to Vendor, Repair)
# gain < 0  -> the mild tail (Use-As-Is) that shrinks as severity rises.
DISPOSITION_PROFILES = {
    "Surface Scratch/Dig": [
        ("Rework", 0.66, -0.08),               # dominant: re-polish / re-finish
        ("Use-As-Is", 0.18, -0.10),            # minor cosmetic, outside aperture
        ("Use Under Deviation", 0.10, 0.04),
        ("Scrap", 0.06, 0.22),                 # dig in clear aperture on critical optic
    ],
    "Fiber Alignment / Insertion Loss": [
        ("Rework", 0.68, -0.10),               # dominant: re-terminate / re-align
        ("Repair", 0.18, 0.03),
        ("Scrap", 0.08, 0.18),                 # damaged pigtail/core
        ("Return to Vendor", 0.06, 0.10),      # vendor connector defect
    ],
    "Dimensional Out-of-Tolerance": [
        ("Rework", 0.60, -0.10),               # dominant: re-machine oversize feature
        ("Use Under Deviation", 0.22, -0.03),  # within functional limits
        ("Scrap", 0.10, 0.18),                 # undersize / unrecoverable
        ("Return to Vendor", 0.08, 0.10),
    ],
    "Solder / Bond Defect": [
        ("Repair", 0.64, -0.08),               # dominant: rework the joint/bond
        ("Rework", 0.22, -0.04),
        ("Scrap", 0.14, 0.20),                 # lifted pad / board damage
    ],
    "Contamination / Coating Defect": [
        ("Rework", 0.62, -0.10),               # dominant: re-clean / strip & recoat
        ("Use-As-Is", 0.18, -0.08),            # contamination outside clear aperture
        ("Return to Vendor", 0.12, 0.14),      # vendor coating adhesion failure
        ("Scrap", 0.08, 0.18),
    ],
}

# Per defect_type root-cause mix, correlated with the process: machining leans
# Machine/Equipment, coating leans Process Drift / Material, hand assembly leans
# Operator Error / Tooling. Static weights, drawn from an independent hash channel.
ROOT_CAUSE_PROFILES = {
    "Surface Scratch/Dig": [("Operator Error", 0.40), ("Tooling/Fixture", 0.25), ("Process Drift", 0.20), ("Machine/Equipment", 0.10), ("Material/Supplier", 0.05)],
    "Fiber Alignment / Insertion Loss": [("Operator Error", 0.35), ("Tooling/Fixture", 0.25), ("Process Drift", 0.18), ("Material/Supplier", 0.15), ("Machine/Equipment", 0.07)],
    "Dimensional Out-of-Tolerance": [("Machine/Equipment", 0.34), ("Tooling/Fixture", 0.26), ("Process Drift", 0.18), ("Operator Error", 0.12), ("Design/Spec", 0.10)],
    "Solder / Bond Defect": [("Machine/Equipment", 0.32), ("Process Drift", 0.24), ("Operator Error", 0.20), ("Material/Supplier", 0.16), ("Tooling/Fixture", 0.08)],
    "Contamination / Coating Defect": [("Process Drift", 0.34), ("Material/Supplier", 0.26), ("Machine/Equipment", 0.18), ("Operator Error", 0.14), ("Design/Spec", 0.08)],
}


def _hash_unit(ncr_id, salt):
    """Deterministic float in [0,1) from sha256(ncr_id|salt). Independent per salt so
    severity / disposition / root_cause draws don't correlate. No random.* involved."""
    h = hashlib.sha256(f"{ncr_id}|{salt}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def _weighted_pick(weighted, u):
    """Pick a label from [(label, weight), ...] using uniform u in [0,1)."""
    total = sum(w for _, w in weighted)
    threshold = u * total
    cum = 0.0
    for label, w in weighted:
        cum += w
        if threshold < cum:
            return label
    return weighted[-1][0]


def _severity_factor(rec):
    """Per-record severity in [0,1] from the hash; critical parts skew high."""
    s = _hash_unit(rec["ncr_id"], "severity")
    if rec["part_name"] in CRITICAL_PARTS:
        s = 0.35 + 0.45 * s   # critical parts skew high but don't saturate
    return s


def disposition_for(rec):
    s = _severity_factor(rec)
    profile = DISPOSITION_PROFILES[rec["defect_type"]]
    weighted = [(label, max(0.001, base + gain * s)) for label, base, gain in profile]
    return _weighted_pick(weighted, _hash_unit(rec["ncr_id"], "disposition"))


def root_cause_for(rec):
    return _weighted_pick(ROOT_CAUSE_PROFILES[rec["defect_type"]], _hash_unit(rec["ncr_id"], "root_cause"))


def fill_template(t, fills):
    for key, choices in fills.items():
        tok = "{"+key+"}"
        if tok in t: t = t.replace(tok, str(random.choice(choices)))
    return t
def make_ncr(i):
    arc = random.choice(ARCHETYPES)
    d = date(2025,1,1) + timedelta(days=random.randint(0,500))
    rec = {"ncr_id":f"NCR-2025-{i:04d}","date":d.isoformat(),"part_id":f"{arc['part_prefix']}-{random.randint(1000,9999)}","part_name":random.choice(arc["part_names"]),"process":arc["process"],"defect_type":arc["defect_type"],"description":fill_template(random.choice(arc["templates"]), arc["fills"])}
    # MRB decision history, derived from sha256(ncr_id) — see DISPOSITION_PROFILES above.
    # Computed after rec is fully built; uses no random.*, so every field above is unchanged.
    rec["disposition"] = disposition_for(rec)
    rec["root_cause"] = root_cause_for(rec)
    return rec
def main(n=200, eval_n=20):
    DATA.mkdir(exist_ok=True); EVAL.mkdir(exist_ok=True)
    records = [make_ncr(i) for i in range(1,n+1)]
    with open(DATA/"ncrs.jsonl","w") as f:
        for r in records: f.write(json.dumps(r)+"\n")
    with open(DATA/"ncrs.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
    eval_cases = [{"id":r["ncr_id"],"input":r["description"],"expected_defect_type":r["defect_type"]} for r in random.sample(records, eval_n)]
    with open(EVAL/"eval_set.json","w") as f: json.dump(eval_cases, f, indent=2)
    from collections import Counter
    print(f"Wrote {len(records)} NCRs and {len(eval_cases)} eval cases")
    for k,v in Counter(r["defect_type"] for r in records).most_common(): print(f"  {v:>3}  {k}")
    print("\nDisposition distribution per defect_type (dominant + tail):")
    for dt,_ in Counter(r["defect_type"] for r in records).most_common():
        dist = Counter(r["disposition"] for r in records if r["defect_type"]==dt)
        print(f"  {dt}")
        for disp,c in dist.most_common(): print(f"      {c:>3}  {disp}")
if __name__ == "__main__":
    main()
