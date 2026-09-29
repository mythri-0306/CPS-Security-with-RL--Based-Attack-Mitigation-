# MiniCPS Power Grid Frequency Control Testbed (Phases 1–3)

This project implements a Cyber-Physical System (CPS) security testbed simulating a power grid primary frequency control loop using **MiniCPS** on top of **Mininet**.

---

## 1. Architecture Overview

### Physical Process (Swing Equation)
The power grid rotational dynamics are governed by the swing equation:
$$\frac{df}{dt} = \frac{P_{\text{gen}} - P_{\text{load}}}{2H}$$

Discretized using forward Euler integration with fixed timestep $\Delta t = 0.1\text{ s}$:
$$f(t + \Delta t) = f(t) + \left(\frac{P_{\text{gen}} - P_{\text{load}}}{2H}\right) \Delta t$$

- Nominal frequency $f_0 = 50.00\text{ Hz}$
- Inertia constant $H = 5.0\text{ s}$
- Nominal base load $P_{\text{load}} = 100.00\text{ MW}$
- Generator setpoint $P_{\text{gen}}$ initialized at $100.00\text{ MW}$

### Network & Devices
A star topology connects 3 Mininet hosts to OpenFlow/OVS switch `s1`:
1. **`sensor` (`192.168.1.10`)**: Runs a Modbus/TCP server on port `502`. Exposes:
   - **Holding Register 0 (`HR 0`)**: Current grid frequency $\times 100$ ($50.00\text{ Hz} \to 5000$).
   - **Holding Register 1 (`HR 1`)**: Generator setpoint $\times 100$ ($100.00\text{ MW} \to 10000$).
2. **`controller` (`192.168.1.20`)**: Runs a Modbus/TCP client polling `sensor` at $T = 0.2\text{ s}$.
   - **Threshold Control Rule**:
     - If $f < 49.80\text{ Hz}$: Ramp generation UP ($P_{\text{gen}} \leftarrow P_{\text{gen}} + 0.50\text{ MW}$).
     - If $f > 50.20\text{ Hz}$: Ramp generation DOWN ($P_{\text{gen}} \leftarrow P_{\text{gen}} - 0.50\text{ MW}$).
     - Else: HOLD setpoint.
   - Writes updated $P_{\text{gen}} \times 100$ back to `sensor` (`HR 1`).
3. **`attacker` (`192.168.1.77`)**: Unauthenticated third-party host on the control network executing attack scenarios.

---

## 2. File Structure

```
examples/powergrid/
├── utils.py              # Constants, network addresses, Modbus tags, SQLite schema
├── init.py               # Initializes powergrid_db.sqlite state table
├── physics.py            # GridPhysics class + PowerGridProcess simulation loop
├── sensor_device.py      # MiniCPS PLC for sensor host (Modbus/TCP server)
├── controller_device.py  # MiniCPS PLC for controller host (Modbus/TCP client)
├── logger.py             # Standardized CSV and DataFrame logger
├── plots.py              # Time-series plotting with dual panels, safe band, attack windows
├── topo.py               # Mininet Topo definition
├── run_baseline.py       # Baseline orchestrator
├── run_baseline.sh       # Baseline shell launcher
├── attacks/              # Phase 3 Attack Modules
│   ├── __init__.py
│   ├── spoof.py          # Modbus FC 6 unauthenticated register overwrite
│   ├── replay.py         # Modbus FC 3 record -> FC 6 replay
│   ├── delay.py          # Linux tc/netem latency injection
│   └── dos.py            # Linux tc/netem packet loss DoS
├── run_spoof.py / .sh    # Spoof attack orchestrator & launcher
├── run_replay.py / .sh   # Replay attack orchestrator & launcher
├── run_delay.py / .sh    # Delay attack orchestrator & launcher
├── run_dos.py / .sh      # DoS attack orchestrator & launcher
├── tests.py              # Unit & integration tests
└── README.md             # Documentation
```

---

## 3. Phase 3 Attack Scenarios

All attacks use standard protocol commands (Modbus/TCP) or Linux kernel traffic shaping (`tc`/`netem`) without packet forgery or ARP spoofing.

| Scenario | Vulnerability / Mechanism | Protocol / Tool | Physical Consequence |
|---|---|---|---|
| **1. Spoof (FDI)** | Absent Modbus/TCP authentication | Modbus FC 6 (`write_register`) | Injects fake low frequency ($48.5\text{ Hz}$), causing controller to over-generate until severe over-frequency trip. |
| **2. Replay** | Absent timestamping / freshness | Modbus FC 3 $\to$ FC 6 | Freezes controller perception using past nominal state, blinding it during actual load disturbances. |
| **3. Delay** | Unbounded network jitter / latency | Linux `tc qdisc netem delay 500ms` | Out-of-phase control actions destabilizing grid frequency oscillation. |
| **4. DoS** | Lack of transport availability | Linux `tc qdisc netem loss 100%` | Complete loss of sensor observability; controller freezes last command. |

---

## 4. How to Run

### Quick Start (300s Simulations + Auto-Plotting)

```bash
# Baseline
./run_baseline.sh 300 baseline_run1

# Scenario 1: Value Spoofing Attack
./run_spoof.sh 300 spoof_run1

# Scenario 2: Stale-Value Replay Attack
./run_replay.sh 300 replay_run1

# Scenario 3: Network Latency Injection Attack
./run_delay.sh 300 delay_run1

# Scenario 4: Packet Loss DoS Attack
./run_dos.sh 300 dos_run1
```

### Standalone Plotting
```bash
python plots.py logs/spoof_run1.csv --run-label "Spoof Attack Run 1"
```

---

## 5. Output Artifacts

- **`logs/<run_label>.csv`**: Time-series log ($0.1\text{ s}$ sampling).
- **`logs/<run_label>.png`**: Dual-panel graph with ground truth frequency, received readings, generation setpoints, and red-hatched attack intervals.
- **`logs/*.log`**: Component output streams.
