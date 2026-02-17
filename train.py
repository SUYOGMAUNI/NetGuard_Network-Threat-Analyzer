"""
train.py – Train the NetGuard threat classifier on CIC-IDS-2017.

Usage
-----
  # Train on real dataset (single combined CSV or folder of day-files):
  python train.py --data-dir data/cicids2017/

  # If no dataset is present, trains on rich synthetic data instead:
  python train.py --synthetic

  # Evaluate a saved model without retraining:
  python train.py --eval-only --data-dir data/cicids2017/

  # Show what labels are in your CSV without training:
  python train.py --inspect --data-dir data/cicids2017/

Dataset
-------
  Kaggle (single combined CSV):
    https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
  UNB (original day-files):
    https://www.unb.ca/cic/datasets/ids-2017.html

  Both formats are supported automatically. Place any/all CSV files in
  --data-dir and the script handles the rest.

Notes
-----
  - CICIDS2017 column headers may have LEADING/TRAILING SPACES — stripped auto.
  - Flow Duration and IAT columns are in microseconds in the raw CSV → converted
    to seconds to match the feature space used by analyzer.py at runtime.
  - The dataset is highly imbalanced (BENIGN >> attacks). Classes with more than
    max_per_class samples are undersampled; absent classes are padded with
    synthetic samples from ml_model._build_synthetic_dataset() so the model
    always knows all 7 attack types.

Fix log vs original:
  - "infiltration" → SQL_INJECT removed; infiltration is now mapped to BOTNET
    (network-layer exfiltration, not application-layer SQL injection).
  - Unknown labels now fall through to NORMAL (safe default) instead of
    WEB_ATTACK (which had no real dataset support in CICIDS2017).
  - _synth_for_class() now delegates to ml_model._build_synthetic_dataset()
    so training and the auto-fallback in load_model() are always identical.
  - trained_on_real_data=True is passed to classifier.train() when called
    from here, so the accuracy disclaimer is suppressed in the UI.
  - Accuracy metrics are printed with an explicit disclaimer when absent
    classes were synthesised, so 99.99% results are not misleading.
"""

import argparse
import glob
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_model import ThreatClassifier, THREAT_LABELS, _build_synthetic_dataset

# ---------------------------------------------------------------------------
# CIC-IDS-2017 → NetGuard label mapping
#
# IMPORTANT: order matters — more specific strings must come FIRST.
# FIX: "infiltration" moved from SQL_INJECT to BOTNET (it describes
#      network-layer exfiltration via a C2 channel, not SQL injection).
# FIX: unknown labels fall through to NORMAL instead of WEB_ATTACK.
# ---------------------------------------------------------------------------
_RAW_LABEL_MAP = {
    # ── NORMAL ───────────────────────────────────────────────────────────────
    "benign":                                       "NORMAL",

    # ── DDoS / DoS ────────────────────────────────────────────────────────────
    "ddos":                                         "DDOS",
    "dos hulk":                                     "DDOS",
    "dos goldeneye":                                "DDOS",
    "dos slowloris":                                "DDOS",
    "dos slowhttptest":                             "DDOS",
    "heartbleed":                                   "DDOS",

    # ── Port Scan ─────────────────────────────────────────────────────────────
    "portscan":                                     "PORT_SCAN",

    # ── Brute Force ───────────────────────────────────────────────────────────
    "ftp-patator":                                  "BRUTE_FORCE",
    "ssh-patator":                                  "BRUTE_FORCE",
    "web attack \u2013 brute force":               "BRUTE_FORCE",
    "web attack \u2014 brute force":               "BRUTE_FORCE",
    "web attack \x96 brute force":                 "BRUTE_FORCE",
    "web attack - brute force":                     "BRUTE_FORCE",
    "web attack  brute force":                      "BRUTE_FORCE",

    # ── Web Attacks (XSS and SQL-over-HTTP) ───────────────────────────────────
    "web attack \u2013 xss":                       "WEB_ATTACK",
    "web attack \u2014 xss":                       "WEB_ATTACK",
    "web attack \x96 xss":                         "WEB_ATTACK",
    "web attack - xss":                             "WEB_ATTACK",
    "web attack  xss":                              "WEB_ATTACK",
    "web attack \u2013 sql injection":             "WEB_ATTACK",
    "web attack \u2014 sql injection":             "WEB_ATTACK",
    "web attack \x96 sql injection":               "WEB_ATTACK",
    "web attack - sql injection":                   "WEB_ATTACK",
    "web attack  sql injection":                    "WEB_ATTACK",
    "web attack":                                   "WEB_ATTACK",

    # ── Botnet / C2 ────────────────────────────────────────────────────────────
    "bot":                                          "BOTNET",
    # FIX: infiltration is network-layer exfil via C2, not SQL injection
    "infiltration":                                 "BOTNET",
}

# Canonical feature column names as they appear in CICIDS2017 CSVs.
CICIDS_FEATURE_COLS = [
    "Flow Duration",               # 1  → flow_duration_s    (µs → s)
    "Total Fwd Packets",           # 2  → fwd_pkts
    "Total Backward Packets",      # 3  → bwd_pkts
    "Total Length of Fwd Packets", # 4  → fwd_bytes
    "Total Length of Bwd Packets", # 5  → bwd_bytes
    "Fwd Packet Length Mean",      # 6  → fwd_pkt_len_mean
    "Bwd Packet Length Mean",      # 7  → bwd_pkt_len_mean
    "Fwd Packet Length Max",       # 8  → fwd_pkt_len_max
    "Bwd Packet Length Max",       # 9  → bwd_pkt_len_max
    "Fwd Packet Length Std",       # 10 → fwd_pkt_len_std
    "Bwd Packet Length Std",       # 11 → bwd_pkt_len_std
    "Flow Bytes/s",                # 12 → flow_bytes_per_s
    "Flow Packets/s",              # 13 → flow_pkts_per_s
    "Fwd IAT Mean",                # 14 → fwd_iat_mean       (µs → s)
    "Bwd IAT Mean",                # 15 → bwd_iat_mean       (µs → s)
    "Fwd IAT Std",                 # 16 → fwd_iat_std        (µs → s)
    "Bwd IAT Std",                 # 17 → bwd_iat_std        (µs → s)
    "SYN Flag Count",              # 18 → syn_flag
    "FIN Flag Count",              # 19 → fin_flag
    "RST Flag Count",              # 20 → rst_flag
    "PSH Flag Count",              # 21 → psh_flag
    "ACK Flag Count",              # 22 → ack_flag
    "Destination Port",            # 23 → dst_port_norm      (/65535)
    "Average Packet Size",         # 24 → bytes_ratio proxy
]

_LABEL_COL_CANDIDATES = ["Label", " Label", "label", "CLASS", "Class"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_col_resolver(df: pd.DataFrame):
    strip_map = {c.strip(): c for c in df.columns}

    def resolve(name: str):
        canonical = name.strip()
        if name in df.columns:
            return name
        if canonical in df.columns:
            return canonical
        if canonical in strip_map:
            return strip_map[canonical]
        return None

    return resolve


def _map_label(raw: str) -> str:
    """
    Map a raw CICIDS2017 label string to a NetGuard THREAT_LABELS entry.

    Strategy:
      1. Exact match after strip + lower
      2. Prefix match (longest key wins)
      3. FIX: unknown labels → NORMAL (safe default, logged once per value)
    """
    clean = str(raw).strip().lower()

    if clean in _RAW_LABEL_MAP:
        return _RAW_LABEL_MAP[clean]

    # Prefix match (longest key wins)
    best_key, best_val = "", None
    for k, v in _RAW_LABEL_MAP.items():
        if clean.startswith(k) and len(k) > len(best_key):
            best_key, best_val = k, v
    if best_val:
        return best_val

    # FIX: fall through to NORMAL (was WEB_ATTACK)
    if not hasattr(_map_label, "_warned"):
        _map_label._warned = set()
    if clean not in _map_label._warned:
        print(f"  [!] Unknown label '{raw}' → NORMAL (safe default)")
        _map_label._warned.add(clean)
    return "NORMAL"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cicids2017(data_dir: str) -> tuple:
    """
    Load all CICIDS2017 CSV files from data_dir.

    Returns
    -------
    X          : np.ndarray  shape (n, 24)  float32
    y          : np.ndarray  shape (n,)     int32
    class_names: list[str]   = THREAT_LABELS
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            "Download from:\n"
            "  https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset"
        )

    print(f"[*] Found {len(csv_files)} CSV file(s) in {data_dir}:")
    for f in csv_files:
        print(f"     • {os.path.basename(f)}")

    frames = []
    for path in csv_files:
        fname = os.path.basename(path)
        print(f"[*] Loading {fname} …", end="", flush=True)
        try:
            df_raw = pd.read_csv(path, low_memory=False, encoding="latin-1")
        except Exception as e:
            print(f" SKIP ({e})")
            continue
        print(f" {len(df_raw):,} rows  ×  {len(df_raw.columns)} cols")
        frames.append(df_raw)

    if not frames:
        raise RuntimeError("All CSV files failed to load. Check file encoding.")

    df = pd.concat(frames, ignore_index=True)
    print(f"[+] Combined: {len(df):,} rows  ×  {len(df.columns)} columns")

    resolve = _build_col_resolver(df)

    # Locate label column
    label_col = None
    for candidate in _LABEL_COL_CANDIDATES:
        r = resolve(candidate)
        if r is not None:
            label_col = r
            break
    if label_col is None:
        raise RuntimeError(
            f"Label column not found. Tried: {_LABEL_COL_CANDIDATES}\n"
            f"Available: {list(df.columns)}"
        )

    raw_labels = df[label_col].value_counts()
    print(f"\n[*] Raw label column = '{label_col}'  ({len(raw_labels)} unique values):")
    for lbl, cnt in raw_labels.items():
        mapped = _map_label(lbl)
        print(f"     {str(lbl):<45}  {cnt:>10,}  →  {mapped}")

    df["_label"] = df[label_col].apply(_map_label)

    print("\n[*] Mapped label distribution:")
    for lbl, cnt in df["_label"].value_counts().items():
        print(f"     {lbl:<15} {cnt:>10,}")

    # Resolve feature columns
    resolved_features, missing_cols = [], []
    for col in CICIDS_FEATURE_COLS:
        r = resolve(col)
        if r:
            resolved_features.append(r)
        else:
            missing_cols.append(col)

    if missing_cols:
        print(f"\n[!] {len(missing_cols)} feature column(s) not found in CSV:")
        for mc in missing_cols:
            print(f"     • {mc}")
        if len(missing_cols) > 6:
            raise RuntimeError(
                "Too many feature columns missing. Verify this is a CIC-IDS-2017 CSV."
            )

    df_feat = df[resolved_features].copy()
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat.fillna(df_feat.median(numeric_only=True), inplace=True)

    # Clip extreme outliers
    for col in df_feat.columns:
        q1, q3 = df_feat[col].quantile([0.25, 0.75])
        df_feat[col] = df_feat[col].clip(upper=q3 + 3 * (q3 - q1))

    # Unit conversions: µs → s
    for col_name in ["Flow Duration", "Fwd IAT Mean", "Bwd IAT Mean",
                     "Fwd IAT Std", "Bwd IAT Std"]:
        r = resolve(col_name)
        if r and r in df_feat.columns:
            df_feat[r] = df_feat[r] / 1_000_000.0

    # Normalize destination port
    dp_r = resolve("Destination Port")
    if dp_r and dp_r in df_feat.columns:
        df_feat[dp_r] = df_feat[dp_r] / 65535.0

    # Pad missing columns with zeros
    if len(resolved_features) < len(CICIDS_FEATURE_COLS):
        n_missing = len(CICIDS_FEATURE_COLS) - len(resolved_features)
        pad = pd.DataFrame(
            np.zeros((len(df_feat), n_missing), dtype=np.float32),
            index=df_feat.index,
        )
        df_feat = pd.concat([df_feat, pad], axis=1)

    X = df_feat.values.astype(np.float32)
    assert X.shape[1] == len(CICIDS_FEATURE_COLS), (
        f"Feature count mismatch: got {X.shape[1]}, expected {len(CICIDS_FEATURE_COLS)}"
    )

    # Map labels to indices using THREAT_LABELS order explicitly.
    # Do NOT use LabelEncoder — it sorts alphabetically and breaks
    # the index↔label correspondence assumed by balance_dataset().
    label_to_idx = {lbl: i for i, lbl in enumerate(THREAT_LABELS)}
    y = np.array(
        [label_to_idx[lbl] for lbl in df["_label"].values],
        dtype=np.int32,
    )

    classes_present = sorted(df["_label"].unique())
    print(f"\n[+] Feature matrix: {X.shape}  |  Classes present: {classes_present}")
    return X, y, THREAT_LABELS


# ---------------------------------------------------------------------------
# Dataset inspection (no training)
# ---------------------------------------------------------------------------

def inspect_dataset(data_dir: str):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"[!] No CSV files in '{data_dir}'")
        return
    for path in csv_files:
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(path)}")
        df_full = pd.read_csv(path, low_memory=False, encoding="latin-1")
        print(f"Columns ({len(df_full.columns)}): {list(df_full.columns)}")
        resolve   = _build_col_resolver(df_full)
        label_col = next(
            (resolve(c) for c in _LABEL_COL_CANDIDATES if resolve(c)), None
        )
        if label_col:
            print(f"\nLabel column: '{label_col}'")
            for v, cnt in df_full[label_col].value_counts().items():
                print(f"  {str(v):<45}  {cnt:>8,}  →  {_map_label(v)}")
        else:
            print("[!] Label column not found")


# ---------------------------------------------------------------------------
# Dataset balancing
# FIX: delegates to _build_synthetic_dataset() from ml_model so both
# training paths (train.py and load_model() fallback) are identical.
# ---------------------------------------------------------------------------

def _synth_for_class(cls_idx: int, n: int, seed: int = 42) -> np.ndarray:
    """
    Generate n synthetic rows for cls_idx using the shared generator.
    We generate a larger batch and slice the needed rows so per-class
    distributions match exactly what _build_synthetic_dataset produces.
    """
    # Build enough rows for this class
    n_batch   = max(n, 500)
    X_all, y_all = _build_synthetic_dataset(n_per_class=n_batch, seed=seed)
    mask      = y_all == cls_idx
    X_cls     = X_all[mask]
    if len(X_cls) >= n:
        return X_cls[:n]
    # Should never happen but guard against it
    reps = (n // len(X_cls)) + 1
    return np.tile(X_cls, (reps, 1))[:n]


def balance_dataset(X, y,
                    max_per_class: int = 30_000,
                    min_per_class: int = 3_000,
                    seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    parts_X, parts_y = [], []

    real_counts    = {i: int(np.sum(y == i)) for i in range(len(THREAT_LABELS))}
    present_counts = [c for c in real_counts.values() if c > 0]
    target         = min(max_per_class, max(present_counts)) if present_counts else min_per_class
    target         = max(target, min_per_class)

    # Track whether any class was synthesised (affects accuracy disclaimer)
    any_synthesised = False
    print(f"  [i] Per-class target: {target:,} samples")

    for cls_idx in range(len(THREAT_LABELS)):
        n   = real_counts[cls_idx]
        lbl = THREAT_LABELS[cls_idx]

        if n == 0:
            synth = _synth_for_class(cls_idx, target, seed)
            parts_X.append(synth)
            parts_y.append(np.full(target, cls_idx, dtype=np.int32))
            any_synthesised = True
            print(f"  [~] {lbl:<15} absent   → synthesised {target:,}")

        elif n < target:
            n_synth  = target - n
            real_idx = np.where(y == cls_idx)[0]
            synth    = _synth_for_class(cls_idx, n_synth, seed)
            parts_X.append(np.vstack([X[real_idx], synth]))
            parts_y.append(np.full(target, cls_idx, dtype=np.int32))
            any_synthesised = True
            print(f"  [+] {lbl:<15} {n:>8,} real + {n_synth:>6,} synth = {target:,}")

        else:
            idx = rng.choice(np.where(y == cls_idx)[0], size=target, replace=False)
            parts_X.append(X[idx])
            parts_y.append(np.full(target, cls_idx, dtype=np.int32))
            print(f"  [-] {lbl:<15} {n:>8,} → undersampled to {target:,}")

    if any_synthesised:
        print(
            "\n  [!] WARNING: One or more classes were absent from the real dataset "
            "and were padded with synthetic samples.\n"
            "      Evaluation metrics on the test split will be inflated for those "
            "classes and do NOT reflect real-world detection capability.\n"
            "      Use the full 8-day CICIDS2017 dataset for valid benchmarks.\n"
        )

    X_bal = np.vstack(parts_X)
    y_bal = np.concatenate(parts_y).astype(np.int32)
    perm  = rng.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(X, y, class_names, cv_folds: int = 0):
    print("\n[*] Balancing dataset …")
    X_bal, y_bal = balance_dataset(X, y)
    print(f"[+] Balanced dataset: {len(X_bal):,} samples  ({len(np.unique(y_bal))} classes)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.20, stratify=y_bal, random_state=42
    )
    print(f"[*] Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    clf = ThreatClassifier()

    print("\n[*] Training ensemble (RF + HGB) …")
    t0 = time.time()
    clf.train(X_tr, y_tr, eval_X=X_te, eval_y=y_te, real_data=True)
    print(f"[+] Training complete in {time.time() - t0:.1f}s")

    # Full evaluation — batch predict (no per-sample loop)
    print("\n[*] Full evaluation on test set …")
    preds = clf.pipeline.predict(X_te).astype(np.int32)

    acc  = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, average="macro", zero_division=0)
    rec  = recall_score(y_te, preds,    average="macro", zero_division=0)
    f1   = f1_score(y_te, preds,        average="macro", zero_division=0)

    print(f"\n{'='*55}")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%  (macro)")
    print(f"  Recall   : {rec * 100:.2f}%  (macro)")
    print(f"  F1       : {f1:.4f}  (macro)")
    print(f"{'='*55}\n")

    all_cls      = sorted(set(y_te.tolist()) | set(preds.tolist()))
    target_names = [THREAT_LABELS[i] for i in all_cls]
    print(classification_report(
        y_te, preds, labels=all_cls,
        target_names=target_names, zero_division=0,
    ))

    if cv_folds > 1:
        print(f"\n[*] {cv_folds}-fold stratified CV (lightweight RF) …")
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline as SKPipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        cv_pipe = SKPipeline([
            ("sc", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=50, max_depth=12,
                n_jobs=-1, random_state=42, class_weight="balanced",
            )),
        ])
        skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(cv_pipe, X_bal, y_bal,
                                 cv=skf, scoring="f1_macro", n_jobs=-1)
        print(f"[+] CV F1 (macro): {scores.mean():.4f} ± {scores.std():.4f}")

    print("\n[+] Model saved → models/threat_classifier.joblib")
    return clf


# ---------------------------------------------------------------------------
# Eval-only
# ---------------------------------------------------------------------------

def eval_saved_model(data_dir: str):
    clf = ThreatClassifier()
    clf.load_model()
    if not clf.trained:
        print("[!] No saved model found. Run without --eval-only first.")
        sys.exit(1)

    X, y, _ = load_cicids2017(data_dir)
    preds = clf.pipeline.predict(X).astype(np.int32)
    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, average="macro", zero_division=0)
    print(f"\nAccuracy: {acc*100:.2f}%   F1 (macro): {f1:.4f}")

    all_cls      = sorted(set(y.tolist()) | set(preds.tolist()))
    target_names = [THREAT_LABELS[i] for i in all_cls]
    print(classification_report(
        y, preds, labels=all_cls, target_names=target_names, zero_division=0
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train / evaluate NetGuard threat classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data/cicids2017/",
                   help="Directory containing CIC-IDS-2017 CSV file(s)")
    p.add_argument("--synthetic", action="store_true",
                   help="Train on synthetic data (no dataset required)")
    p.add_argument("--eval-only", action="store_true",
                   help="Evaluate saved model on --data-dir, skip training")
    p.add_argument("--inspect", action="store_true",
                   help="Print label values and columns, then exit")
    p.add_argument("--cv", type=int, default=0, metavar="K",
                   help="Run K-fold cross-validation after training (0=skip)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.inspect:
        inspect_dataset(args.data_dir)
        return

    if args.eval_only:
        eval_saved_model(args.data_dir)
        return

    if args.synthetic:
        print("[*] Training on synthetic data …")
        ThreatClassifier()._train_synthetic()
        print("[+] Done.")
        return

    if not os.path.isdir(args.data_dir):
        print(f"[!] Data directory not found: '{args.data_dir}'")
        print("[*] Falling back to synthetic training …\n")
        ThreatClassifier()._train_synthetic()
        print("[+] Done.")
        return

    X, y, class_names = load_cicids2017(args.data_dir)
    train_and_evaluate(X, y, class_names, cv_folds=args.cv)


if __name__ == "__main__":
    main()