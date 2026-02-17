# NetGuard — Network Threat Analyzer

A real-time network threat detection system powered by an ensemble ML model trained on the CICIDS2017 dataset. Features a live dashboard with WebSocket-based flow streaming, automated threat classification, and IP blocking.


Features

- ML Threat Classification — Random Forest + Histogram Gradient Boosting ensemble classifies flows into 7 categories in real time
- Live Dashboard — WebSocket-powered UI with traffic charts, threat gauge, protocol breakdown, and alert feed
- IP Blocking. — Manual and automatic IP blocking with persistent session state
- Feature Importance Visualization — Top contributing features displayed per model
- REST API — Full API for stats, predictions, blocked IPs, and model metadata

---

ML Model Performance

Trained on the full .CICIDS2017. dataset (2,830,743 flows across 8 capture days).

Dataset

| File | Rows |
|------|------|
| Friday-DDos | 225,745 |
| Friday-PortScan | 286,467 |
| Friday-Morning | 191,033 |
| Monday | 529,918 |
| Thursday-Infiltration | 288,602 |
| Thursday-WebAttacks | 170,366 |
| Tuesday | 445,909 |
| Wednesday | 692,703 |
| Total | 2,830,743 |

Label Mapping

| Raw Label | Mapped Class | Count |
|-----------|-------------|-------|
| BENIGN | NORMAL | 2,273,097 |
| DoS Hulk / DDoS / GoldenEye / Slowloris / Slowhttptest / Heartbleed | DDOS | 380,699 |
| PortScan | PORT_SCAN | 158,930 |
| FTP-Patator / SSH-Patator | BRUTE_FORCE | 13,835 |
| Bot / Infiltration | BOTNET | 2,002 |
| Web Attack (Brute Force / XSS / SQLi) | WEB_ATTACK | 2,180 |

Training

Balanced to 30,000 samples per class. 80/20 train-test split.

| Split | Samples |
|-------|---------|
| Train | 168,000 |
| Test | 42,000 |

 Results
=======================================================
  Accuracy : 99.47%
  Precision: 99.47%  (macro)
  Recall   : 99.47%  (macro)
  F1       : 0.9947  (macro)
=======================================================

              precision    recall  f1-score   support

      NORMAL       0.99      0.98      0.98      6000
   PORT_SCAN       1.00      1.00      1.00      6000
        DDOS       0.99      1.00      1.00      6000
 BRUTE_FORCE       1.00      1.00      1.00      6000
  SQL_INJECT       1.00      1.00      1.00      6000
  WEB_ATTACK       0.99      1.00      1.00      6000
      BOTNET       0.99      0.99      0.99      6000

    accuracy                           0.99     42000
   macro avg       0.99      0.99      0.99     42000
weighted avg       0.99      0.99      0.99     42000


> Note: SQL\_INJECT class was absent from the raw dataset and padded with synthetic samples. Metrics for that class do not reflect real-world detection capability.

 Tech Stack

- Backend:. Python, Flask, Flask-SocketIO
- ML: Scikit-learn (RandomForestClassifier + HistGradientBoostingClassifier)
- Packet Capture: Scapy (requires Npcap on Windows)
- Frontend: Vanilla JS, Chart.js, Socket.IO client
- Data: CICIDS2017 (Canadian Institute for Cybersecurity)


 Project Structure

NetGuard/
├── app.py
├── analyzer.py
├── ml_model.py
├── train.py
├── models/
│   └── threat_classifier.joblib
├── data/
│   └── cicids2017/
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/dashboard.js
├── .env.example
├── requirements.txt
└── README.md

Setup

1. Clone & install

git clone https://github.com/yourusername/netguard.git
cd netguard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

2. Set environment variable

set NETGUARD_SECRET_KEY=your-strong-random-key-here

 3. Train the model (optional — pre-trained model included)

Download CICIDS2017 CSVs from the Canadian Institute for Cybersecurity and place them in data/cicids2017/

python train.py --data-dir data/cicids2017/

4. Run

python app.py

Open http://127.0.0.1:5000 in your browser.

> Windows note:. For real packet capture, install Npcap and run as Administrator. Without it, the app runs in simulation mode using synthetic flows.

 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/stats | Analyzer summary stats |
| GET | /api/model-info | Model metadata and accuracy |
| GET | /api/features | Feature names used by the model |
| GET | /api/feature-importances | Per-feature importance scores |
| POST | /api/predict | Classify a flow from JSON body |
| POST | /api/start | Start packet capture |
| POST | /api/stop | Stop packet capture |
| POST | /api/block-ip | Block an IP address |
| POST | /api/unblock-ip | Unblock an IP address |
| GET | /api/blocked-ips | List all blocked IPs |

WebSocket Events

| Event | Direction | Payload |
|-------|-----------|---------|
| new_packet | Server → Client | Flow record with classification |
| new_alert | Server → Client | Threat alert details |
| capture_status | Server → Client | { active: bool } |
| stats_snapshot | Server → Client | Summary stats |
| request_stats | Client → Server | Triggers stats emission |


Built by [Suyog Mauni](https://suyogmauni.com.np) · 2025
