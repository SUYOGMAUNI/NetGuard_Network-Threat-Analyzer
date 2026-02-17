"""
app.py – NetGuard: Network Packet Analyzer & Threat Detector

Routes
------
  GET  /                  → Dashboard (index.html)
  POST /api/start         → Start packet capture loop
  POST /api/stop          → Stop packet capture loop
  GET  /api/stats         → Analyzer summary stats
  GET  /api/model-info    → Model metadata (accuracy, classes, etc.)
  GET  /api/features      → Feature names used by the model
  POST /api/predict       → Manual prediction from JSON flow dict
  POST /api/block-ip      → Add IP to blocked set
  POST /api/unblock-ip    → Remove IP from blocked set
  GET  /api/blocked-ips   → List all blocked IPs
  GET  /api/feature-importances → Per-feature importance scores

WebSocket events (server → client)
-----------------------------------
  new_packet     → {src_ip, dst_ip, src_port, dst_port, protocol,
                    fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes,
                    threat, confidence, timestamp}
  new_alert      → {type, title, src_ip, dst_ip, port, confidence, time}
  capture_status → {active: bool}
  stats_snapshot → summary stats dict
  error          → {message: str}

Fix log vs original:
  - SECRET_KEY read from environment variable; startup aborts if missing/default
  - _start_lock serialises concurrent POST /api/start calls (eliminates race)
  - All analyzer state mutations go through analyzer.increment_packet() /
    analyzer.add_blocked_ip() / analyzer.remove_blocked_ip() which hold the
    single internal lock — no second lock in app.py
  - Duplicate `from analyzer import PacketAnalyzer` removed
  - socketio.emit() calls include namespace="/" for thread-safety
  - /api/predict returns 400 on missing fields (no silent fallback)
  - /api/feature-importances returns 503 with explanation if unavailable
"""

import os
import threading
import time
import logging
from datetime import datetime

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from analyzer import PacketAnalyzer
from ml_model import ThreatClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("netguard")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# FIX: read secret key from environment; refuse to start with the placeholder
_SECRET_KEY = os.environ.get("NETGUARD_SECRET_KEY", "")
if not _SECRET_KEY or _SECRET_KEY == "netguard-secret-2025":
    raise RuntimeError(
        "NETGUARD_SECRET_KEY environment variable is not set or is still "
        "the default placeholder. Set a strong random value before starting."
    )
app.config["SECRET_KEY"] = _SECRET_KEY

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)

# ---------------------------------------------------------------------------
# Shared state
# FIX: analyzer._lock is the single source of truth for all mutable state.
#      _start_lock is a separate, narrow lock that only serialises the
#      start/stop of the capture thread – it is never held while emitting
#      to the socket or doing any I/O.
# ---------------------------------------------------------------------------
analyzer = PacketAnalyzer()
classifier = ThreatClassifier()

_capture_active = False
_capture_thread = None
_start_lock     = threading.Lock()   # guards thread start/stop only

# ---------------------------------------------------------------------------
# Startup – load / train model
# ---------------------------------------------------------------------------
log.info("Loading ML model …")
classifier.load_model()
log.info("Model ready.")

# ---------------------------------------------------------------------------
# Packet capture background thread
# ---------------------------------------------------------------------------
_CAPTURE_INTERVAL = 0.25   # seconds between simulated packets


def _packet_capture_loop():
    global _capture_active
    log.info("Capture loop started.")

    while True:
        # Read flag without holding any lock that could block the socket layer
        with _start_lock:
            active = _capture_active
        if not active:
            break

        try:
            flow               = analyzer.get_next_packet()
            features           = analyzer.extract_features(flow)
            label, confidence  = classifier.predict(features)

            # FIX: single atomic call handles all counter updates via analyzer._lock
            protocol = flow.get("protocol", "OTHER")
            analyzer.increment_packet(label, protocol)

            record = {
                "src_ip":     flow.get("src_ip", "0.0.0.0"),
                "dst_ip":     flow.get("dst_ip", "0.0.0.0"),
                "src_port":   int(flow.get("src_port", 0)),
                "dst_port":   int(flow.get("dst_port", 0)),
                "protocol":   str(protocol),
                "fwd_pkts":   int(flow.get("fwd_pkts", 0)),
                "bwd_pkts":   int(flow.get("bwd_pkts", 0)),
                "fwd_bytes":  int(flow.get("fwd_bytes", 0)),
                "bwd_bytes":  int(flow.get("bwd_bytes", 0)),
                "threat":     label,
                "confidence": round(confidence * 100, 1),
                "timestamp":  datetime.now().strftime("%H:%M:%S"),
            }

            # FIX: explicit namespace="/" for background-thread emission
            socketio.emit("new_packet", record, namespace="/")

            if label != "NORMAL":
                alert = {
                    "type":       "critical" if confidence >= 0.85 else "warning",
                    "title":      f"{label} Detected",
                    "src_ip":     record["src_ip"],
                    "dst_ip":     record["dst_ip"],
                    "port":       record["dst_port"],
                    "confidence": record["confidence"],
                    "time":       record["timestamp"],
                }
                socketio.emit("new_alert", alert, namespace="/")
                log.warning(
                    "[THREAT] %s | %s → %s:%d | conf=%.1f%%",
                    label, record["src_ip"], record["dst_ip"],
                    record["dst_port"], record["confidence"],
                )

        except Exception as exc:
            log.exception("Error in capture loop: %s", exc)
            socketio.emit("error", {"message": str(exc)}, namespace="/")

        time.sleep(_CAPTURE_INTERVAL)

    log.info("Capture loop stopped.")


# ---------------------------------------------------------------------------
# HTTP Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    stats = classifier.get_model_stats()
    return render_template("index.html", stats=stats)


@app.route("/api/start", methods=["POST"])
def start_capture():
    global _capture_active, _capture_thread

    # FIX: entire check-and-start is inside _start_lock to prevent concurrent
    # POST /api/start calls from spawning duplicate threads.
    with _start_lock:
        if _capture_active:
            return jsonify({"status": "already_running"}), 200
        _capture_active = True
        _capture_thread = threading.Thread(
            target=_packet_capture_loop, name="PacketCapture", daemon=True
        )
        _capture_thread.start()

    socketio.emit("capture_status", {"active": True}, namespace="/")
    log.info("Capture started.")
    return jsonify({"status": "started"}), 200


@app.route("/api/stop", methods=["POST"])
def stop_capture():
    global _capture_active

    with _start_lock:
        if not _capture_active:
            return jsonify({"status": "not_running"}), 200
        _capture_active = False

    socketio.emit("capture_status", {"active": False}, namespace="/")
    log.info("Capture stopped.")
    return jsonify({"status": "stopped"}), 200


@app.route("/api/stats")
def get_stats():
    return jsonify(analyzer.get_summary_stats()), 200


@app.route("/api/model-info")
def model_info():
    return jsonify(classifier.get_model_stats()), 200


@app.route("/api/features")
def feature_names():
    return jsonify({
        "feature_names": PacketAnalyzer.FEATURE_NAMES,
        "n_features":    len(PacketAnalyzer.FEATURE_NAMES),
    }), 200


@app.route("/api/predict", methods=["POST"])
def manual_predict():
    """
    Accepts a JSON body describing a network flow.
    Required fields: fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes.

    Minimal example body:
    {
        "dst_port": 22,
        "fwd_pkts": 20,
        "bwd_pkts": 18,
        "fwd_bytes": 2000,
        "bwd_bytes": 1800,
        "flow_duration_s": 5.0,
        "syn_flag": 1,
        "fin_flag": 1,
        "psh_flag": 1,
        "ack_flag": 1
    }
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    # FIX: validate required fields explicitly; dict_to_features now raises
    # ValueError on missing fields rather than silently building a synthetic flow.
    required = {"fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes"}
    missing  = required - data.keys()
    if missing:
        return jsonify({
            "error":   "Missing required fields",
            "missing": sorted(missing),
        }), 400

    try:
        features           = analyzer.dict_to_features(data)
        label, confidence  = classifier.predict(features)
        return jsonify({
            "threat":     label,
            "confidence": round(confidence * 100, 1),
            "features":   dict(zip(PacketAnalyzer.FEATURE_NAMES, features)),
        }), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        log.exception("Prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/block-ip", methods=["POST"])
def block_ip():
    data = request.get_json(silent=True)
    if not data or "ip" not in data:
        return jsonify({"error": "Missing 'ip' field"}), 400
    ip = str(data["ip"]).strip()
    analyzer.add_blocked_ip(ip)   # FIX: uses analyzer's own lock
    log.info("Blocked IP: %s", ip)
    return jsonify({"status": "blocked", "ip": ip}), 200


@app.route("/api/unblock-ip", methods=["POST"])
def unblock_ip():
    data = request.get_json(silent=True)
    if not data or "ip" not in data:
        return jsonify({"error": "Missing 'ip' field"}), 400
    ip = str(data["ip"]).strip()
    analyzer.remove_blocked_ip(ip)   # FIX: uses analyzer's own lock
    log.info("Unblocked IP: %s", ip)
    return jsonify({"status": "unblocked", "ip": ip}), 200


@app.route("/api/blocked-ips")
def list_blocked_ips():
    ips = analyzer.get_blocked_ips()   # already sorted, lock-safe
    return jsonify({"blocked_ips": ips, "count": len(ips)}), 200


@app.route("/api/feature-importances")
def feature_importances():
    imps = classifier.get_feature_importances()
    if not imps:
        return jsonify({
            "error":  "Feature importances not available",
            "reason": "Model must be trained with scikit-learn and loaded "
                      "before importances can be computed.",
        }), 503
    sorted_imps = dict(sorted(imps.items(), key=lambda x: x[1], reverse=True))
    return jsonify(sorted_imps), 200


# ---------------------------------------------------------------------------
# WebSocket handlers
# ---------------------------------------------------------------------------

@socketio.on("connect")
def handle_connect():
    with _start_lock:
        active = _capture_active
    emit("status", {
        "message": "Connected to NetGuard",
        "active":  active,
        "version": "2.1",
    })
    log.info("Client connected.")


@socketio.on("disconnect")
def handle_disconnect():
    log.info("Client disconnected.")


@socketio.on("request_stats")
def handle_stats_request():
    emit("stats_snapshot", analyzer.get_summary_stats())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,        # reloader conflicts with background threads
        allow_unsafe_werkzeug=True # required for dev; use gunicorn in production
    )