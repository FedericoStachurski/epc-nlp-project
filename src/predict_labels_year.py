"""
run_year_inference.py
Per-year EPC inference pipeline:
- Load all quarters for YEAR
- Clean + parse improvements, aggregate per RRN
- Predict labels (Transformer multilabel, top-k)
- Geocode postcodes to lat/lon (postcodes.io)
- Save CSV for downstream maps/analytics
"""

import os, re, glob, sys, gc, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import http.client
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Project imports ---
sys.path.insert(0, os.path.abspath(".."))
from src.preprocess import split_improvements, parse_improvement, clean_text
from src.labels import labels_measure  # not used for inference, but handy for QA

# -------------------------
# Config
# -------------------------
DATA_GLOB   = "data/raw/D_EPC_data_*/*.csv"
OUT_DIR     = "data/predictions"
MODEL_DIR   = "models/epc_distilbert_multilabel/best"
MLB_PATH    = "models/mlb.joblib"

ID_COL, TEXT_COL, POST_COL = "REPORT_REFERENCE_NUMBER", "IMPROVEMENTS", "POSTCODE"
CHUNKSIZE   = 50_000
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Inference selection: at least 1, at most 3 labels, with a floor
MIN_K, MAX_K, MIN_PROB = 1, 3, 0.05

# Geocode batching
POSTCODES_BATCH = 100
GEOCODE_SLEEP   = 0.05   # seconds between batches; bump if you hit rate limits

# -------------------------
# Helpers
# -------------------------
period_re = re.compile(r"(?<!\d)(\d{4})Q([1-4])(?!\d)")
def infer_period_from_path(path: str) -> str:
    fn = os.path.basename(path)
    m  = period_re.search(fn)
    if m: return m.group(0)
    matches = list(period_re.finditer(path))
    return matches[-1].group(0) if matches else "UNKNOWN"

RRN_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{4}-\d{4}$")
BAD_TEXT = {"improvement", "improvements", "measure", "measures", "text"}

def select_labels(probs, labels, min_k=1, max_k=3, min_prob=0.05):
    ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    filtered = [(lbl, p) for lbl, p in ranked if p >= min_prob]
    top = filtered[:max_k] if filtered else ranked[:min_k]
    return [lbl for lbl, _ in top]

@torch.no_grad()
def predict_probs(texts, tokenizer, model, batch_size=64):
    # normalize input
    if isinstance(texts, pd.Series): texts = texts.astype(str).tolist()
    else: texts = [str(t) for t in texts]

    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        probs = torch.sigmoid(model(**enc).logits).cpu().numpy()
        out.append(probs)
    return np.vstack(out) if out else np.empty((0,))

def fetch_postcode_locations(postcodes, batch_size=100, sleep_s=0.05):
    """
    Bulk lookup UK postcodes via postcodes.io
    Returns dict: {postcode: (lat, lon)} ; missing -> (None, None)
    """
    # sanitize postcodes (trim spaces, upper)
    clean = []
    seen = set()
    for p in postcodes:
        if not isinstance(p, str): continue
        q = p.strip().upper()
        if q and q not in seen:
            seen.add(q); clean.append(q)

    results = {p: (None, None) for p in clean}
    if not clean:
        return results

    conn = http.client.HTTPSConnection("api.postcodes.io")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    for i in tqdm(range(0, len(clean), batch_size), desc="Geocoding postcodes"):
        batch = clean[i:i+batch_size]
        payload = json.dumps({"postcodes": batch})
        conn.request("POST", "/postcodes", body=payload, headers=headers)
        res = conn.getresponse()
        data = res.read()
        out = json.loads(data.decode("utf-8"))
        for r in out.get("result", []):
            q = r["query"]
            rr = r["result"]
            if rr is not None:
                results[q] = (rr["latitude"], rr["longitude"])
        time.sleep(sleep_s)

    return results

# -------------------------
# Main per-year runner
# -------------------------
def run_year(year: int, save_csv=True):
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) find all quarters for year
    all_csvs = sorted(glob.glob(DATA_GLOB))
    targets = []
    for f in all_csvs:
        per = infer_period_from_path(f)
        if per != "UNKNOWN" and int(per[:4]) == year:
            targets.append((f, per))
    if not targets:
        raise RuntimeError(f"No CSVs found for year {year} with pattern {DATA_GLOB}")

    print(f"Year {year}: {len(targets)} quarterly files → {[p for _, p in targets]}")

    # 2) aggregate per RRN across all quarters of the year
    prop_texts, prop_post, prop_periods = defaultdict(list), {}, defaultdict(set)

    for fpath, period in targets:
        print(f"Reading {fpath} ({period})")
        reader = pd.read_csv(
            fpath, chunksize=CHUNKSIZE, low_memory=False,
            usecols=[ID_COL, TEXT_COL, POST_COL]
        )
        for chunk in reader:
            for rrn, txt, post in chunk[[ID_COL, TEXT_COL, POST_COL]].itertuples(index=False, name=None):
                # skip junk rows
                if not isinstance(rrn, str) or not RRN_RE.match(rrn): 
                    continue
                if not isinstance(txt, str):
                    continue
                if txt.strip().lower() in BAD_TEXT or len(txt.strip()) < 8:
                    continue
                # split → parse → keep measure text
                for seg in split_improvements(txt):
                    rec = parse_improvement(seg) or {}
                    m = rec.get("measure") or seg
                    prop_texts[rrn].append(m)
                # store one postcode per RRN if seen
                if rrn not in prop_post and isinstance(post, str) and len(post.strip()) > 0:
                    prop_post[rrn] = post.strip().upper()
                prop_periods[rrn].add(period)
            del chunk; gc.collect()

    # 3) build per-RRN rows (one row per property in the year)
    rows = []
    for rrn, measures in prop_texts.items():
        txt_raw = " . ".join(measures)
        txt_clean = clean_text(txt_raw)
        periods_sorted = sorted(prop_periods[rrn])
        rows.append({
            "rrn": rrn,
            "periods": periods_sorted,
            "period": periods_sorted[0] if periods_sorted else None,  # keep first period in year for convenience
            "text": txt_clean,
            "postcode": prop_post.get(rrn, None),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No valid rows produced for year {year}.")
    print(f"Built {len(df):,} rows for year {year}")

    # 4) load model assets
    tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    mlb   = joblib.load(MLB_PATH)
    classes = mlb.classes_.tolist()
    label_cols = [f"p_{c}" for c in classes]

    # 5) model inference (batched)
    probs = predict_probs(df["text"], tok, model, batch_size=64)
    # attach probabilities + top-k predictions
    for j, cls in enumerate(classes):
        df[f"p_{cls}"] = probs[:, j].astype(float)
    df["prediction"] = [
        select_labels(probs[i], classes, min_k=MIN_K, max_k=MAX_K, min_prob=MIN_PROB)
        for i in range(len(df))
    ]

    # 6) geocode postcodes → lat/lon (batched); cache optional
    uniq_pc = df["postcode"].dropna().unique().tolist()
    pc2ll = fetch_postcode_locations(uniq_pc, batch_size=POSTCODES_BATCH, sleep_s=GEOCODE_SLEEP)
    df["lat"] = df["postcode"].map(lambda p: pc2ll.get(p, (None, None))[0] if isinstance(p, str) else None)
    df["lon"] = df["postcode"].map(lambda p: pc2ll.get(p, (None, None))[1] if isinstance(p, str) else None)

    # 7) save
    out_csv = os.path.join(OUT_DIR, f"epc_preds_{year}.csv")
    if save_csv:
        df.to_csv(out_csv, index=False)
        print(f"Saved → {out_csv}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EPC NLP inference for a given year")
    parser.add_argument("year", type=int, help="Year to process (e.g. 2023)")
    parser.add_argument("--no-save", action="store_true", help="Do not save CSV output")
    args = parser.parse_args()

    YEAR = args.year
    df = run_year(YEAR, save_csv=not args.no_save)

    print(df.head())
    print(f"Done running inference for year {YEAR}")
