"""
analyzer.py – Packet capture, flow aggregation, and feature extraction.

Architecture:
  - Live mode  : Scapy captures raw packets → FlowTracker aggregates into flows
                 → extract_flow_features() produces the 24 CICIDS2017-aligned features.
  - Simulation : get_next_packet() produces pre-aggregated flow dicts matching
                 those same 24 features, so training and inference are always consistent.

All 24 features mirror the most discriminative columns in CIC-IDS-2017:
  Flow Duration, packet counts, byte counts, IAT statistics, flag counts,
  packet-length statistics, and derived ratios.

Fix log vs original:
  - fwd/bwd direction now based on TCP SYN initiator, not IP-string sort
  - FlowTracker uses a heap-based expiry queue (O(log n)) instead of O(n) scan
  - All state protected by a single _lock (eliminated split-lock race)
  - Simulation generates bidirectional traffic (internal→external and reverse)
  - Protocol field restricted to real IP protocols (TCP/UDP/ICMP/OTHER)
"""

import random
import math
import time
import heapq
import threading
from collections import deque, defaultdict

# ---------------------------------------------------------------------------
# Optional Scapy import
# ---------------------------------------------------------------------------
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[!] Scapy not available – running in simulation mode.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INTERNAL_IPS = [f"192.168.1.{i}" for i in range(1, 20)]
EXTERNAL_IPS = [
    "185.220.101.42", "103.21.58.11", "45.33.32.156",
    "8.8.8.8",        "1.1.1.1",      "104.21.33.212",
    "172.67.68.91",   "91.108.4.1",   "52.3.8.24",
    "20.189.173.2",   "34.64.4.112",  "66.22.196.5",
]

# Attack simulation templates (flow-level, matching CICIDS2017 statistical profile)
_ATTACK_TEMPLATES = {
    "PORT_SCAN": {
        "dst_ports":        list(range(1, 1025)),
        "fwd_pkts":         1,
        "bwd_pkts":         0,
        "fwd_bytes":        40,
        "bwd_bytes":        0,
        "flow_duration_us": 500,
        "syn_flag":         1,
        "fin_flag":         0,
        "rst_flag":         1,
        "psh_flag":         0,
        "ack_flag":         0,
        "iat_mean_us":      500,
    },
    "DDOS": {
        "dst_ports":        [80, 443],
        "fwd_pkts":         500,
        "bwd_pkts":         5,
        "fwd_bytes":        30000,
        "bwd_bytes":        1000,
        "flow_duration_us": 100_000,
        "syn_flag":         1,
        "fin_flag":         0,
        "rst_flag":         0,
        "psh_flag":         0,
        "ack_flag":         1,
        "iat_mean_us":      200,
    },
    "BRUTE_FORCE": {
        "dst_ports":        [22, 3389, 21],
        "fwd_pkts":         20,
        "bwd_pkts":         15,
        "fwd_bytes":        2000,
        "bwd_bytes":        1500,
        "flow_duration_us": 5_000_000,
        "syn_flag":         1,
        "fin_flag":         1,
        "rst_flag":         0,
        "psh_flag":         1,
        "ack_flag":         1,
        "iat_mean_us":      250_000,
    },
    "SQL_INJECT": {
        "dst_ports":        [3306, 5432, 1433],
        "fwd_pkts":         5,
        "bwd_pkts":         8,
        "fwd_bytes":        800,
        "bwd_bytes":        4000,
        "flow_duration_us": 2_000_000,
        "syn_flag":         1,
        "fin_flag":         1,
        "rst_flag":         0,
        "psh_flag":         1,
        "ack_flag":         1,
        "iat_mean_us":      400_000,
    },
    "WEB_ATTACK": {
        "dst_ports":        [80, 443],
        "fwd_pkts":         8,
        "bwd_pkts":         6,
        "fwd_bytes":        3500,
        "bwd_bytes":        8000,
        "flow_duration_us": 3_000_000,
        "syn_flag":         1,
        "fin_flag":         1,
        "rst_flag":         0,
        "psh_flag":         1,
        "ack_flag":         1,
        "iat_mean_us":      375_000,
    },
    "BOTNET": {
        "dst_ports":        [6667, 1080, 9001],
        "fwd_pkts":         12,
        "bwd_pkts":         10,
        "fwd_bytes":        600,
        "bwd_bytes":        500,
        "flow_duration_us": 60_000_000,
        "syn_flag":         1,
        "fin_flag":         0,
        "rst_flag":         0,
        "psh_flag":         1,
        "ack_flag":         1,
        "iat_mean_us":      5_000_000,
    },
}


# ---------------------------------------------------------------------------
# Flow tracker (used in live-capture mode to aggregate packets into flows)
# ---------------------------------------------------------------------------
class FlowTracker:
    """
    Groups raw packets by 5-tuple into flows.
    A flow is exported after FLOW_TIMEOUT seconds of inactivity.

    FIX: expiry now uses a min-heap for O(log n) per-packet cost instead
    of scanning all flows on every packet.

    FIX: fwd/bwd direction is now determined by who sent the first SYN
    (TCP initiator), falling back to IP-string comparison for non-TCP.
    """
    FLOW_TIMEOUT = 5.0  # seconds

    def __init__(self, on_flow_ready):
        self._flows    = {}          # key → flow state dict
        self._cb       = on_flow_ready
        self._lock     = threading.Lock()
        # min-heap of (expiry_time, key) for efficient expiry
        self._heap: list = []

    @staticmethod
    def _key(pkt):
        """Canonical 5-tuple key — src/dst ordered so both directions share it."""
        proto = "TCP" if TCP in pkt else ("UDP" if UDP in pkt else "OTHER")
        sp = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dp = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
        src, dst = pkt[IP].src, pkt[IP].dst
        # Canonical ordering so (A→B) and (B→A) map to the same key
        if (src, sp) < (dst, dp):
            return (src, dst, sp, dp, proto)
        return (dst, src, dp, sp, proto)

    @staticmethod
    def _is_initiator(pkt, key):
        """
        True when this packet is in the 'forward' (initiator) direction.
        For TCP: the SYN-sender is always forward.  Falls back to the
        canonical key's first IP for non-TCP or mid-stream packets.
        """
        if TCP in pkt:
            flags = int(pkt[TCP].flags)
            syn   = bool(flags & 0x02)
            ack   = bool(flags & 0x10)
            if syn and not ack:
                # Pure SYN → this is definitively the initiator
                return True
            if syn and ack:
                # SYN-ACK → this is the responder
                return False
        # Non-TCP or mid-stream: use canonical key's "first" src
        return pkt[IP].src == key[0]

    def add_packet(self, pkt):
        if IP not in pkt:
            return
        now    = time.time()
        key    = self._key(pkt)
        size   = len(pkt)
        is_fwd = self._is_initiator(pkt, key)
        flags  = int(pkt[TCP].flags) if TCP in pkt else 0

        with self._lock:
            self._expire(now)
            if key not in self._flows:
                self._flows[key] = self._new_flow(pkt, key, now, is_fwd, size, flags)
                heapq.heappush(self._heap, (now + self.FLOW_TIMEOUT, key))
            else:
                self._update_flow(self._flows[key], now, is_fwd, size, flags)

    def _new_flow(self, pkt, key, now, is_fwd, size, flags):
        sp = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dp = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
        return {
            "src_ip":        pkt[IP].src,
            "dst_ip":        pkt[IP].dst,
            "src_port":      sp,
            "dst_port":      dp,
            "protocol":      "TCP" if TCP in pkt else ("UDP" if UDP in pkt else "ICMP"),
            "start_ts":      now,
            "last_ts":       now,
            "fwd_pkts":      1 if is_fwd else 0,
            "bwd_pkts":      0 if is_fwd else 1,
            "fwd_bytes":     size if is_fwd else 0,
            "bwd_bytes":     0 if is_fwd else size,
            "fwd_pkt_sizes": [size] if is_fwd else [],
            "bwd_pkt_sizes": [] if is_fwd else [size],
            "fwd_iats":      [],
            "bwd_iats":      [],
            "prev_fwd_ts":   now if is_fwd else None,
            "prev_bwd_ts":   None if is_fwd else now,
            "syn_flag":      1 if (flags & 0x02) else 0,
            "fin_flag":      1 if (flags & 0x01) else 0,
            "rst_flag":      1 if (flags & 0x04) else 0,
            "psh_flag":      1 if (flags & 0x08) else 0,
            "ack_flag":      1 if (flags & 0x10) else 0,
            "urg_flag":      1 if (flags & 0x20) else 0,
        }

    def _update_flow(self, flow, now, is_fwd, size, flags):
        flow["last_ts"] = now
        if is_fwd:
            flow["fwd_pkts"]  += 1
            flow["fwd_bytes"] += size
            flow["fwd_pkt_sizes"].append(size)
            if flow["prev_fwd_ts"] is not None:
                flow["fwd_iats"].append(now - flow["prev_fwd_ts"])
            flow["prev_fwd_ts"] = now
        else:
            flow["bwd_pkts"]  += 1
            flow["bwd_bytes"] += size
            flow["bwd_pkt_sizes"].append(size)
            if flow["prev_bwd_ts"] is not None:
                flow["bwd_iats"].append(now - flow["prev_bwd_ts"])
            flow["prev_bwd_ts"] = now
        # OR-accumulate flags (any packet in flow sets the bit)
        flow["syn_flag"] |= 1 if (flags & 0x02) else 0
        flow["fin_flag"] |= 1 if (flags & 0x01) else 0
        flow["rst_flag"] |= 1 if (flags & 0x04) else 0
        flow["psh_flag"] |= 1 if (flags & 0x08) else 0
        flow["ack_flag"] |= 1 if (flags & 0x10) else 0
        flow["urg_flag"] |= 1 if (flags & 0x20) else 0

    def _expire(self, now):
        """Pop heap entries whose expiry time has passed; export completed flows."""
        while self._heap and self._heap[0][0] <= now:
            _, key = heapq.heappop(self._heap)
            flow   = self._flows.pop(key, None)
            if flow is not None:
                self._cb(flow)

    def flush_all(self):
        """Force-export all tracked flows (called on capture stop)."""
        with self._lock:
            for flow in self._flows.values():
                self._cb(flow)
            self._flows.clear()
            self._heap.clear()


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------
class PacketAnalyzer:
    """
    Unified interface for both live capture and simulation.
    Produces flow-level feature vectors (24 features) consistent with the
    CIC-IDS-2017 feature space used during training.

    FIX: single _lock guards all mutable state (total_packets, threats_found,
    blocked_ips, _stats).  app.py must not use a separate lock on these fields.
    """

    # Ordered list of 24 feature names (must match extract_flow_features output)
    FEATURE_NAMES = [
        "flow_duration_s",      # 1
        "fwd_pkts",             # 2
        "bwd_pkts",             # 3
        "fwd_bytes",            # 4
        "bwd_bytes",            # 5
        "fwd_pkt_len_mean",     # 6
        "bwd_pkt_len_mean",     # 7
        "fwd_pkt_len_max",      # 8
        "bwd_pkt_len_max",      # 9
        "fwd_pkt_len_std",      # 10
        "bwd_pkt_len_std",      # 11
        "flow_bytes_per_s",     # 12
        "flow_pkts_per_s",      # 13
        "fwd_iat_mean",         # 14
        "bwd_iat_mean",         # 15
        "fwd_iat_std",          # 16
        "bwd_iat_std",          # 17
        "syn_flag",             # 18
        "fin_flag",             # 19
        "rst_flag",             # 20
        "psh_flag",             # 21
        "ack_flag",             # 22
        "dst_port_norm",        # 23
        "bytes_ratio",          # 24
    ]

    def __init__(self):
        self.packet_queue  = deque(maxlen=1000)
        self._stats        = defaultdict(int)
        self.total_packets = 0
        self.threats_found = 0
        self.blocked_ips   = set()
        self._flow_tracker = FlowTracker(on_flow_ready=self._on_flow_complete)
        self._lock         = threading.Lock()   # single lock for all state

    # ------------------------------------------------------------------
    # LIVE CAPTURE
    # ------------------------------------------------------------------
    def start_live_capture(self, iface="eth0", packet_count=0):
        """Blocking – run inside a daemon thread."""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is not installed; cannot start live capture.")
        sniff(iface=iface, prn=self._on_raw_packet,
              store=False, count=packet_count)
        self._flow_tracker.flush_all()

    def _on_raw_packet(self, pkt):
        if IP not in pkt:
            return
        self._flow_tracker.add_packet(pkt)
        with self._lock:
            self.total_packets += 1

    def _on_flow_complete(self, flow):
        """Called by FlowTracker when a flow expires."""
        proto = flow.get("protocol", "OTHER")
        with self._lock:
            self.packet_queue.append(flow)
            self._stats["protocol_" + proto] += 1

    # ------------------------------------------------------------------
    # SIMULATION
    # ------------------------------------------------------------------
    def get_next_packet(self):
        """
        Returns a flow-level dict for demo/testing.
        ~12 % of packets are attacks; the rest are normal.
        FIX: normal flows are bidirectional (internal↔external) to match
        realistic traffic and improve botnet/lateral-movement simulation.
        """
        if random.random() < 0.12:
            return self._simulate_attack_flow()
        return self._simulate_normal_flow()

    def _simulate_normal_flow(self):
        fwd_pkts  = random.randint(2, 30)
        bwd_pkts  = random.randint(1, 20)
        fwd_bytes = fwd_pkts * random.randint(40, 1460)
        bwd_bytes = bwd_pkts * random.randint(40, 1460)
        duration  = random.uniform(0.01, 10.0)

        fwd_sizes = [max(40, fwd_bytes // fwd_pkts + random.randint(-10, 10))
                     for _ in range(fwd_pkts)]
        bwd_sizes = [max(40, bwd_bytes // bwd_pkts + random.randint(-10, 10))
                     for _ in range(bwd_pkts)]
        fwd_iats  = [random.uniform(0.001, duration / max(fwd_pkts, 1))
                     for _ in range(max(fwd_pkts - 1, 0))]
        bwd_iats  = [random.uniform(0.001, duration / max(bwd_pkts, 1))
                     for _ in range(max(bwd_pkts - 1, 0))]

        # FIX: mix inbound and outbound normal traffic
        if random.random() < 0.5:
            src_ip, dst_ip = random.choice(INTERNAL_IPS), random.choice(EXTERNAL_IPS)
        else:
            src_ip, dst_ip = random.choice(EXTERNAL_IPS), random.choice(INTERNAL_IPS)

        return {
            "src_ip":        src_ip,
            "dst_ip":        dst_ip,
            "src_port":      random.randint(1024, 65535),
            "dst_port":      random.choice([80, 443, 53, 22, 8080, 8443]),
            "protocol":      random.choice(["TCP", "UDP", "ICMP"]),
            "start_ts":      time.time() - duration,
            "last_ts":       time.time(),
            "fwd_pkts":      fwd_pkts,
            "bwd_pkts":      bwd_pkts,
            "fwd_bytes":     fwd_bytes,
            "bwd_bytes":     bwd_bytes,
            "fwd_pkt_sizes": fwd_sizes,
            "bwd_pkt_sizes": bwd_sizes,
            "fwd_iats":      fwd_iats,
            "bwd_iats":      bwd_iats,
            "syn_flag":      1,
            "fin_flag":      1,
            "rst_flag":      0,
            "psh_flag":      random.randint(0, 1),
            "ack_flag":      1,
            "urg_flag":      0,
        }

    def _simulate_attack_flow(self):
        attack_type = random.choice(list(_ATTACK_TEMPLATES.keys()))
        t           = _ATTACK_TEMPLATES[attack_type]
        dst_port    = random.choice(t["dst_ports"])
        duration    = t["flow_duration_us"] / 1_000_000

        fwd_pkts = max(t["fwd_pkts"] + random.randint(-2, 2), 1)
        bwd_pkts = max(t["bwd_pkts"] + random.randint(-1, 1), 0)
        fwd_bytes = max(t["fwd_bytes"] + random.randint(-20, 20), fwd_pkts * 40)
        bwd_bytes = max(t["bwd_bytes"] + random.randint(-20, 20), 0)

        fwd_sizes = [fwd_bytes // fwd_pkts] * fwd_pkts
        bwd_sizes = ([bwd_bytes // bwd_pkts] * bwd_pkts) if bwd_pkts > 0 else []

        iat_s    = t["iat_mean_us"] / 1_000_000
        fwd_iats = [iat_s + random.uniform(-iat_s * 0.1, iat_s * 0.1)
                    for _ in range(max(fwd_pkts - 1, 0))]
        bwd_iats = [iat_s + random.uniform(-iat_s * 0.1, iat_s * 0.1)
                    for _ in range(max(bwd_pkts - 1, 0))]

        return {
            "src_ip":        random.choice(EXTERNAL_IPS),
            "dst_ip":        random.choice(INTERNAL_IPS),
            "src_port":      random.randint(1024, 65535),
            "dst_port":      dst_port,
            "protocol":      "TCP",
            "start_ts":      time.time() - duration,
            "last_ts":       time.time(),
            "fwd_pkts":      fwd_pkts,
            "bwd_pkts":      bwd_pkts,
            "fwd_bytes":     fwd_bytes,
            "bwd_bytes":     bwd_bytes,
            "fwd_pkt_sizes": fwd_sizes,
            "bwd_pkt_sizes": bwd_sizes,
            "fwd_iats":      fwd_iats,
            "bwd_iats":      bwd_iats,
            "syn_flag":      t["syn_flag"],
            "fin_flag":      t["fin_flag"],
            "rst_flag":      t["rst_flag"],
            "psh_flag":      t["psh_flag"],
            "ack_flag":      t["ack_flag"],
            "urg_flag":      0,
            "_label":        attack_type,   # simulation hint only; not used by model
        }

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION – 24 CICIDS2017-aligned flow-level features
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    @staticmethod
    def _safe_std(lst):
        if len(lst) < 2:
            return 0.0
        m = sum(lst) / len(lst)
        return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))

    @staticmethod
    def _safe_max(lst):
        return max(lst) if lst else 0.0

    def extract_flow_features(self, flow):
        """
        Convert a flow dict into a 24-element feature vector mirroring
        CIC-IDS-2017 statistical features used during training.
        Returns list[float].
        """
        duration    = max(flow["last_ts"] - flow["start_ts"], 1e-9)
        fwd_pkts    = max(int(flow.get("fwd_pkts", 0)), 0)
        bwd_pkts    = max(int(flow.get("bwd_pkts", 0)), 0)
        fwd_bytes   = max(int(flow.get("fwd_bytes", 0)), 0)
        bwd_bytes   = max(int(flow.get("bwd_bytes", 0)), 0)
        total_bytes = fwd_bytes + bwd_bytes
        total_pkts  = fwd_pkts + bwd_pkts

        fwd_sizes   = flow.get("fwd_pkt_sizes", [])
        bwd_sizes   = flow.get("bwd_pkt_sizes", [])
        fwd_iats    = flow.get("fwd_iats", [])
        bwd_iats    = flow.get("bwd_iats", [])
        dst_port    = int(flow.get("dst_port", 0))

        bytes_ratio = min(fwd_bytes / max(bwd_bytes, 1), 1000.0)

        raw = [
            min(duration, 3600.0),                          # 1  flow_duration_s
            float(fwd_pkts),                                # 2  fwd_pkts
            float(bwd_pkts),                                # 3  bwd_pkts
            float(fwd_bytes),                               # 4  fwd_bytes
            float(bwd_bytes),                               # 5  bwd_bytes
            self._safe_mean(fwd_sizes),                     # 6  fwd_pkt_len_mean
            self._safe_mean(bwd_sizes),                     # 7  bwd_pkt_len_mean
            float(self._safe_max(fwd_sizes)),               # 8  fwd_pkt_len_max
            float(self._safe_max(bwd_sizes)),               # 9  bwd_pkt_len_max
            self._safe_std(fwd_sizes),                      # 10 fwd_pkt_len_std
            self._safe_std(bwd_sizes),                      # 11 bwd_pkt_len_std
            min(total_bytes / duration, 1e9),               # 12 flow_bytes_per_s
            min(total_pkts  / duration, 1e6),               # 13 flow_pkts_per_s
            self._safe_mean(fwd_iats),                      # 14 fwd_iat_mean
            self._safe_mean(bwd_iats),                      # 15 bwd_iat_mean
            self._safe_std(fwd_iats),                       # 16 fwd_iat_std
            self._safe_std(bwd_iats),                       # 17 bwd_iat_std
            float(int(flow.get("syn_flag", 0))),            # 18 syn_flag
            float(int(flow.get("fin_flag", 0))),            # 19 fin_flag
            float(int(flow.get("rst_flag", 0))),            # 20 rst_flag
            float(int(flow.get("psh_flag", 0))),            # 21 psh_flag
            float(int(flow.get("ack_flag", 0))),            # 22 ack_flag
            dst_port / 65535.0,                             # 23 dst_port_norm
            bytes_ratio,                                    # 24 bytes_ratio
        ]

        return [float(0.0 if not math.isfinite(v) else v) for v in raw]

    # Alias used by app.py and the API predict endpoint
    def extract_features(self, flow):
        return self.extract_flow_features(flow)

    def dict_to_features(self, d):
        """
        Convert a plain dict (e.g. from the /api/predict JSON body) into a
        feature vector. Raises ValueError if required fields are missing.
        """
        required = {"fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes"}
        missing  = required - d.keys()
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        d = dict(d)
        if "start_ts" not in d:
            d["start_ts"] = time.time() - float(d.get("flow_duration_s", 1.0))
            d["last_ts"]  = time.time()
        for key in ("fwd_pkt_sizes", "bwd_pkt_sizes", "fwd_iats", "bwd_iats"):
            d.setdefault(key, [])
        return self.extract_flow_features(d)

    # ------------------------------------------------------------------
    # STATS  (all reads guarded by self._lock)
    # ------------------------------------------------------------------
    def increment_packet(self, label, protocol):
        """Called by app.py capture loop to update counters atomically."""
        with self._lock:
            self.total_packets += 1
            self._stats["protocol_" + protocol] += 1
            if label != "NORMAL":
                self.threats_found += 1

    def get_summary_stats(self):
        with self._lock:
            return {
                "total_packets": self.total_packets,
                "threats_found": self.threats_found,
                "blocked_ips":   len(self.blocked_ips),
                "protocols": {
                    k.replace("protocol_", ""): v
                    for k, v in self._stats.items()
                    if k.startswith("protocol_")
                },
            }

    def add_blocked_ip(self, ip: str):
        with self._lock:
            self.blocked_ips.add(ip)

    def remove_blocked_ip(self, ip: str):
        with self._lock:
            self.blocked_ips.discard(ip)

    def get_blocked_ips(self):
        with self._lock:
            return sorted(self.blocked_ips)