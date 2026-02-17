"""
ml_model.py – Threat Classifier (Random Forest + HistGradientBoosting ensemble).

Class labels align exactly with CIC-IDS-2017 attack categories:
    0 → NORMAL
    1 → PORT_SCAN
    2 → DDOS
    3 → BRUTE_FORCE
    4 → SQL_INJECT
    5 → WEB_ATTACK
    6 → BOTNET

Training:
    • Real dataset  : run train.py (reads CIC-IDS-2017 CSVs)
    • Synthetic demo: ThreatClassifier.load_model() auto-trains if no saved
                      model is found, using statistically representative
                      synthetic data that matches the 24 CICIDS2017 features.

Inference:
    • predict(features) → (label: str, confidence: float)
    • Falls back to a rule-based classifier if scikit-learn is absent.

Fix log vs original:
    - GradientBoostingClassifier → HistGradientBoostingClassifier (supports
      n_jobs, ~10× faster, friendlier with threading)
    - predict() raises ValueError on out-of-range class index instead of
      silently clamping to BOTNET
    - _train_synthetic() delegates to the shared synth generator in
      train_data_utils so both code paths produce identical distributions
    - inference_latency_ms benchmark now measures single-sample latency
    - get_feature_importances() averages RF and HGB importances
    - accuracy disclaimer added to _meta to prevent misleading UI display
      when the model was trained on purely synthetic data
"""

import os
import time
import random
import numpy as np

try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        HistGradientBoostingClassifier,
        VotingClassifier,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[!] scikit-learn not found – using rule-based fallback classifier.")

MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "threat_classifier.joblib")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.joblib")

THREAT_LABELS = [
    "NORMAL",
    "PORT_SCAN",
    "DDOS",
    "BRUTE_FORCE",
    "SQL_INJECT",
    "WEB_ATTACK",
    "BOTNET",
]

N_FEATURES = 24   # must match PacketAnalyzer.FEATURE_NAMES


class ThreatClassifier:
    """
    Ensemble (soft-voting) classifier: RandomForest + HistGradientBoosting.
    Wraps a sklearn Pipeline (StandardScaler → VotingClassifier) so scaling
    is applied automatically at both train and predict time.
    """

    def __init__(self):
        self.pipeline = None
        self.trained  = False
        self._meta    = {
            "algorithm":             "Random Forest + HistGradientBoosting (Voting)",
            "features":              N_FEATURES,
            "n_classes":             len(THREAT_LABELS),
            "classes":               THREAT_LABELS,
            "dataset":               "CIC-IDS-2017",
            "trained_on_real_data":  False,   # set True only by train.py
            # Populated after training/evaluation
            "accuracy":              None,
            "precision_macro":       None,
            "recall_macro":          None,
            "f1_macro":              None,
            "training_samples":      None,
            "inference_latency_ms":  None,
            "accuracy_note":         (
                "Metrics are on held-out synthetic data when "
                "trained_on_real_data is False. Real-world performance "
                "will differ; train on CIC-IDS-2017 for valid benchmarks."
            ),
        }

    # ------------------------------------------------------------------
    # BUILD MODEL
    # ------------------------------------------------------------------
    def _build_pipeline(self):
        rf = RandomForestClassifier(
            n_estimators      = 100,
            max_depth         = 15,
            min_samples_split = 4,
            min_samples_leaf  = 2,
            max_features      = "sqrt",
            n_jobs            = -1,
            random_state      = 42,
            class_weight      = "balanced",
        )
        # FIX: HistGradientBoostingClassifier supports n_jobs and is
        # much faster than GradientBoostingClassifier (~10×).
        hgb = HistGradientBoostingClassifier(
            max_iter        = 100,
            max_depth       = 6,
            learning_rate   = 0.1,
            l2_regularization = 0.1,
            random_state    = 42,
        )
        voting = VotingClassifier(
            estimators = [("rf", rf), ("hgb", hgb)],
            voting     = "soft",
            n_jobs     = -1,
        )
        return Pipeline([
            ("scaler",     StandardScaler()),
            ("classifier", voting),
        ])

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def train(self, X, y, eval_X=None, eval_y=None, real_data=False):
        """
        Train the pipeline on (X, y).
        Optionally evaluates on (eval_X, eval_y) and stores metrics.

        Parameters
        ----------
        X         : np.ndarray shape (n, 24)
        y         : np.ndarray of int (class indices matching THREAT_LABELS)
        real_data : bool – set True when called from train.py with real CSVs
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for training.")

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)
        self.trained  = True
        self._meta["training_samples"]     = int(len(X))
        self._meta["trained_on_real_data"] = real_data

        if eval_X is not None and eval_y is not None:
            self._evaluate(eval_X, eval_y)

        # FIX: benchmark single-sample latency (not batch / len(sample))
        sample_1 = X[:1]
        t0 = time.perf_counter()
        for _ in range(200):
            self.pipeline.predict(sample_1)
        elapsed_ms = (time.perf_counter() - t0) / 200 * 1000
        self._meta["inference_latency_ms"] = round(elapsed_ms, 3)

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.pipeline, MODEL_PATH)
        joblib.dump(self._meta,    META_PATH)
        print(f"[+] Model saved → {MODEL_PATH}")

    def _evaluate(self, X, y):
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )
        preds = self.pipeline.predict(X)
        self._meta["accuracy"]        = round(float(accuracy_score(y, preds)) * 100, 2)
        self._meta["precision_macro"] = round(
            float(precision_score(y, preds, average="macro", zero_division=0)) * 100, 2
        )
        self._meta["recall_macro"]    = round(
            float(recall_score(y, preds, average="macro", zero_division=0)) * 100, 2
        )
        self._meta["f1_macro"]        = round(
            float(f1_score(y, preds, average="macro", zero_division=0)), 4
        )
        print(
            f"[+] Eval → Accuracy: {self._meta['accuracy']}%  "
            f"F1(macro): {self._meta['f1_macro']}"
        )

    # ------------------------------------------------------------------
    # LOAD / FALLBACK
    # ------------------------------------------------------------------
    def load_model(self):
        """Load saved model, or train on synthetic data if none exists."""
        if SKLEARN_AVAILABLE and os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
            self.pipeline = joblib.load(MODEL_PATH)
            self._meta    = joblib.load(META_PATH)
            self.trained  = True
            print(f"[+] Loaded model from {MODEL_PATH}")
        else:
            print("[*] No saved model found – training on synthetic data …")
            self._train_synthetic()

    def _train_synthetic(self):
        """
        Generate a synthetic dataset that mimics CIC-IDS-2017 feature distributions.
        FIX: delegates to the shared _build_synthetic_dataset() so this path
        and train.py's _synth_for_class() always produce identical distributions.
        """
        if not SKLEARN_AVAILABLE:
            print("[!] sklearn not available – using rule-based classifier.")
            return

        from sklearn.model_selection import train_test_split
        X, y = _build_synthetic_dataset(n_per_class=3000, seed=42)
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.train(X_tr, y_tr, eval_X=X_ev, eval_y=y_ev, real_data=False)

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------
    def predict(self, features):
        """
        Parameters
        ----------
        features : list or np.ndarray of length N_FEATURES

        Returns
        -------
        label      : str   – one of THREAT_LABELS
        confidence : float – probability of the predicted class [0, 1]
        """
        if SKLEARN_AVAILABLE and self.trained and self.pipeline is not None:
            arr  = np.array(features, dtype=np.float32).reshape(1, -1)
            arr  = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=0.0)
            idx  = int(self.pipeline.predict(arr)[0])
            proba = self.pipeline.predict_proba(arr)[0]

            # FIX: validate index instead of silently clamping
            if idx < 0 or idx >= len(THREAT_LABELS):
                raise ValueError(
                    f"Model returned class index {idx} which is outside "
                    f"the expected range [0, {len(THREAT_LABELS) - 1}]. "
                    "Re-train the model."
                )
            if idx >= len(proba):
                raise ValueError(
                    f"proba vector length {len(proba)} does not cover "
                    f"class index {idx}. Model may be corrupt."
                )
            return THREAT_LABELS[idx], float(proba[idx])

        return self._rule_based(features)

    def _rule_based(self, features):
        """
        Simple deterministic fallback when sklearn is not available.
        Mirrors the 24-feature schema from PacketAnalyzer.extract_flow_features.
        """
        f = list(features) + [0.0] * N_FEATURES   # pad if short
        flow_dur    = f[0]
        fwd_pkts    = f[1]
        fwd_bytes   = f[3]
        bwd_bytes   = f[4]
        flow_bps    = f[11]
        flow_pps    = f[12]
        syn         = f[17]
        rst         = f[19]
        psh         = f[20]
        ack         = f[21]
        dst_port_n  = f[22]

        dst_port = int(dst_port_n * 65535)

        if syn and rst and flow_pps > 50 and fwd_bytes < 100 and fwd_pkts <= 2:
            return "PORT_SCAN", 0.93
        if flow_pps > 500 and flow_bps > 100_000 and flow_dur < 2.0:
            return "DDOS", 0.91
        if dst_port in (22, 3389, 21) and fwd_pkts > 10 and psh and ack:
            return "BRUTE_FORCE", 0.85
        if flow_dur > 30 and flow_pps < 5 and dst_port not in (80, 443, 53, 22):
            return "BOTNET", 0.78
        if dst_port in (3306, 5432, 1433) and bwd_bytes > fwd_bytes * 2:
            return "SQL_INJECT", 0.82
        if dst_port in (80, 443, 8080) and bwd_bytes > fwd_bytes and flow_pps < 50:
            return "WEB_ATTACK", 0.75
        return "NORMAL", 0.96

    # ------------------------------------------------------------------
    # STATS / META
    # ------------------------------------------------------------------
    def get_model_stats(self):
        return dict(self._meta)

    def get_feature_importances(self):
        """
        Return per-feature importances averaged across RF and HGB sub-estimators.
        FIX: original only used RF; this now averages both estimators.
        Returns dict {feature_name: importance} or empty dict if unavailable.
        """
        if not (SKLEARN_AVAILABLE and self.trained and self.pipeline is not None):
            return {}
        try:
            from analyzer import PacketAnalyzer
            names  = PacketAnalyzer.FEATURE_NAMES
            voting = self.pipeline.named_steps["classifier"]
            ests   = dict(voting.named_estimators_)

            rf_imps  = np.array(ests["rf"].feature_importances_)
            hgb_imps = np.array(ests["hgb"].feature_importances_)
            avg_imps = (rf_imps + hgb_imps) / 2.0

            return {
                name: round(float(v), 5)
                for name, v in zip(names, avg_imps)
            }
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Shared synthetic data generator
# (used by both _train_synthetic above and train.py's _synth_for_class)
# ---------------------------------------------------------------------------
def _build_synthetic_dataset(n_per_class: int = 3000, seed: int = 42):
    """
    Generate n_per_class rows for each of the 7 THREAT_LABELS.
    Feature order matches PacketAnalyzer.FEATURE_NAMES exactly.

    Returns (X: np.ndarray shape (n*7, 24), y: np.ndarray shape (n*7,))
    """
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    def noisy_row(vals, noise=0.06):
        out = []
        for i, v in enumerate(vals):
            if i in (17, 18, 19, 20, 21):   # binary flags – no noise
                out.append(float(v))
            else:
                out.append(max(0.0, float(v) * (1.0 + np_rng.normal(0, noise))))
        return out

    X_rows, y_rows = [], []

    for _ in range(n_per_class):

        # NORMAL – typical web/DNS traffic
        dp  = rng.choice([80, 443, 53, 22, 8080]) / 65535.0
        fp  = rng.randint(5, 40);   bp  = rng.randint(3, 30)
        fb  = fp * rng.randint(100, 800);  bb = bp * rng.randint(50, 600)
        dur = rng.uniform(0.1, 10.0)
        X_rows.append(noisy_row([
            dur, fp, bp, fb, bb,
            fb/fp, bb/max(bp, 1), 1400, 1400, 120, 100,
            (fb+bb)/dur, (fp+bp)/dur, dur/fp, dur/max(bp, 1), 0.1, 0.1,
            1, 1, 0, 1, 1, dp, fb/max(bb, 1)
        ]))
        y_rows.append(0)

        # PORT_SCAN – tiny SYN-only flows
        dp  = rng.randint(1, 1023) / 65535.0
        dur = rng.uniform(0.0001, 0.005)
        X_rows.append(noisy_row([
            dur, 1, 0, 40, 0, 40, 0, 40, 0, 0, 0,
            40/dur, 1/dur, 0, 0, 0, 0,
            1, 0, 1, 0, 0, dp, 100.0
        ], noise=0.02))
        y_rows.append(1)

        # DDOS – massive fwd traffic, high pps
        dp  = rng.choice([80, 443]) / 65535.0
        fp  = rng.randint(400, 800);  bp = rng.randint(2, 10)
        fb  = fp * 60;                bb = bp * 200
        dur = rng.uniform(0.05, 0.5)
        X_rows.append(noisy_row([
            dur, fp, bp, fb, bb, 60, 200, 120, 500, 5, 50,
            (fb+bb)/dur, (fp+bp)/dur, dur/fp, dur/max(bp, 1), 0.001, 0.05,
            1, 0, 0, 0, 1, dp, fb/max(bb, 1)
        ]))
        y_rows.append(2)

        # BRUTE_FORCE – repeated auth attempts on remote ports
        dp  = rng.choice([22, 3389, 21]) / 65535.0
        fp  = rng.randint(15, 30);  bp = rng.randint(12, 25)
        fb  = fp * 100;              bb = bp * 100
        dur = rng.uniform(2.0, 20.0)
        X_rows.append(noisy_row([
            dur, fp, bp, fb, bb, 100, 100, 200, 200, 20, 20,
            (fb+bb)/dur, (fp+bp)/dur, dur/fp, dur/max(bp, 1), 0.5, 0.5,
            1, 1, 0, 1, 1, dp, 1.0
        ]))
        y_rows.append(3)

        # SQL_INJECT – DB ports, bwd-heavy (large server responses)
        dp  = rng.choice([3306, 5432, 1433]) / 65535.0
        fp  = rng.randint(3, 8);   bp = rng.randint(5, 12)
        fb  = fp * 500;             bb = bp * 1500
        dur = rng.uniform(0.5, 5.0)
        X_rows.append(noisy_row([
            dur, fp, bp, fb, bb, 500, 1500, 1400, 8000, 100, 300,
            (fb+bb)/dur, (fp+bp)/dur, dur/fp, dur/max(bp, 1), 0.2, 0.3,
            1, 1, 0, 1, 1, dp, fb/max(bb, 1)
        ]))
        y_rows.append(4)

        # WEB_ATTACK – web ports, bwd-heavy (server error payloads)
        dp  = rng.choice([80, 443]) / 65535.0
        fp  = rng.randint(6, 15);   bp = rng.randint(4, 10)
        fb  = fp * 450;              bb = bp * 1200
        dur = rng.uniform(1.0, 8.0)
        X_rows.append(noisy_row([
            dur, fp, bp, fb, bb, 450, 1200, 1400, 8000, 80, 200,
            (fb+bb)/dur, (fp+bp)/dur, dur/fp, dur/max(bp, 1), 0.3, 0.4,
            1, 1, 0, 1, 1, dp, fb/max(bb, 1)
        ]))
        y_rows.append(5)

        # BOTNET – low-and-slow, unusual ports, very regular IAT
        dp  = rng.choice([6667, 1080, 9001, 4444]) / 65535.0
        fp  = rng.randint(8, 15);   bp = rng.randint(6, 12)
        fb  = fp * 50;               bb = bp * 45
        dur = rng.uniform(30.0, 300.0)
        X_rows.append(noisy_row([
            dur, fp, bp, fb, bb, 50, 45, 100, 90, 5, 4,
            (fb+bb)/dur, (fp+bp)/dur, dur/fp, dur/max(bp, 1), 0.05, 0.05,
            1, 0, 0, 1, 1, dp, 1.1
        ]))
        y_rows.append(6)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    perm = np.random.default_rng(seed).permutation(len(X))
    return X[perm], y[perm]