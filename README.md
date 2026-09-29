# CPS Security with RL-Based Attack Mitigation (Power Grid Testbed)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: MiniCPS](https://img.shields.io/badge/framework-MiniCPS-orange.svg)](https://github.com/scy-phy/minicps)
[![Network: Mininet](https://img.shields.io/badge/network-Mininet-green.svg)](http://mininet.org/)
[![Protocol: Modbus/TCP](https://img.shields.io/badge/protocol-Modbus%2FTCP-red.svg)](http://www.modbus.org/)
[![Status: Deliverable 1 Complete](https://img.shields.io/badge/status-Deliverable%201%20Complete-brightgreen.svg)]()

A Cyber-Physical System (CPS) security research testbed simulating a **power grid primary frequency control loop**. This repository demonstrates how cyberattacks targeting unauthenticated industrial communication protocols (Modbus/TCP) cause physical-layer disruptions (frequency excursions, blackout drift, and oscillatory hunting) and establishes the testbed for autonomous **Reinforcement Learning (RL) based attack detection and mitigation**.

---

## 📌 Project Overview

In power grid operation, real-time frequency stability ($50.00\text{ Hz}$ or $60.00\text{ Hz}$) requires a strict balance between active generation ($P_{\text{gen}}$) and load demand ($P_{\text{load}}$). Phasor Measurement Units (PMUs) and SCADA sensors report frequency telemetry across industrial networks to controllers (PLCs/RTUs) that adjust generator output.

Legacy industrial protocols such as **Modbus/TCP** lack native authentication, message freshness checks, and encryption. This allows adversaries with control-network access to inject false data, replay stale measurements, delay telemetry, or cause denial of service.

```
                  ┌───────────────────────────────────────────────┐
                  │                 POWER GRID                    │
                  │   df/dt = (f0 / 2H·S_base)*ΔP - D*(f - f0)    │
                  └───────▲───────────────────────────────┬───────┘
                          │ P_gen Setpoint                │ True Grid Frequency
                          │                               ▼
                  ┌───────┴──────────────┐   ┌────────────────────────────┐
                  │    CONTROLLER PLC    │   │         SENSOR PLC         │
                  │  (Modbus/TCP Client) │   │    (Modbus/TCP Server)     │
                  │  • Threshold Control │   │  • HR 0: Scaled Freq (×100)│
                  │  • Droop+PI Control  │   │  • HR 1: Scaled Gen (×100) │
                  └───────▲──────────────┘   └────────────┬───────────────┘
                          │                               │
                          └────────[ VIRTUAL NETWORK ]────┘
                                    (Mininet OVS)
                                          ▲
                                          │ Cyberattacks
                                   ┌──────┴──────┐
                                   │  ATTACKER   │
                                   └─────────────┘
```

---

## 🎯 Deliverables & Roadmap

* **Deliverable 1 (Completed ✅)**:
  * High-fidelity physical swing equation simulation with per-unit scaling and system load damping.
  * Modbus/TCP industrial communication layer over Mininet virtual network topology.
  * Baseline frequency controllers (Naive Threshold and Primary Droop + Secondary PI).
  * 4 cyberattack scenarios proving **Cyberattack $\to$ Physical Consequence**.
  * Real-time $0.1\text{ s}$ logging pipeline and dual-panel time-series visualization.
* **Deliverable 2 (In Progress 🚀)**:
  * Physics-based innovation anomaly detector (Kalman Filter state estimator).
  * Reinforcement Learning (DQN via Stable-Baselines3) response agent.
  * Multi-method defense evaluation: *No Defense* vs. *Rule-Based Defense* vs. *RL Defender*.

---

## ⚙️ Physical Process Modeling

The rotational frequency dynamics are governed by the power system **swing equation**, scaled in physical units ($\text{Hz}$) with self-regulating load damping:

$$\frac{df}{dt} = \frac{f_0}{2 H S_{\text{base}}} (P_{\text{gen}} - P_{\text{load}}) - D (f - f_0)$$

Discretized via forward Euler integration ($\Delta t = 0.1\text{ s}$):

$$f(t + \Delta t) = f(t) + \left[ \frac{f_0}{2 H S_{\text{base}}} (P_{\text{gen}} - P_{\text{load}}) - D (f - f_0) \right] \Delta t$$

| Parameter | Symbol | Default Value | Description |
|---|---|---|---|
| Nominal Frequency | $f_0$ | $50.00\text{ Hz}$ | Target grid synchronous frequency |
| Inertia Constant | $H$ | $5.0\text{ s}$ | Generator rotational inertia |
| Base Power Rating | $S_{\text{base}}$ | $100.00\text{ MVA}$ | Normalization system base capacity |
| Load Damping Factor | $D$ | $1.0\text{ p.u./Hz}$ | Self-regulating load-frequency sensitivity |
| Base Electrical Load | $P_{\text{load}}$ | $100.00\text{ MW}$ | Initial power demand (steps to $103\text{ MW}$ at $t=10\text{s}$) |
| Initial Generation | $P_{\text{gen}}$ | $100.00\text{ MW}$ | Initial generator output |

---

## 🌐 Network Architecture & Modbus Mapping

A star topology connects three virtual hosts to an Open vSwitch (`s1`):

1. **`sensor` (`192.168.1.10`)**: Modbus/TCP Server (Port 502)
   * **Holding Register 0 (`HR 0`)**: Frequency scaled by $100$ ($50.00\text{ Hz} \to 5000$).
   * **Holding Register 1 (`HR 1`)**: Generation setpoint scaled by $100$ ($100.00\text{ MW} \to 10000$).
2. **`controller` (`192.168.1.20`)**: Modbus/TCP Client ($T = 0.2\text{ s}$ polling)
   * **Threshold Mode (`--mode threshold`)**:
     * If $f < 49.80\text{ Hz}$: Ramp generation UP ($P_{\text{gen}} \leftarrow P_{\text{gen}} + 0.50\text{ MW}$).
     * If $f > 50.20\text{ Hz}$: Ramp generation DOWN ($P_{\text{gen}} \leftarrow P_{\text{gen}} - 0.50\text{ MW}$).
     * Else: HOLD setpoint.
   * **Droop + Secondary PI Mode (`--mode pi`)**: Proportional droop ($K_p = 15.0\text{ MW/Hz}$) + Integral error elimination ($K_i = 2.5\text{ MW/(Hz}\cdot\text{s})$).
   * **Liveness / Heartbeat Tracking**: Flags communication loss if $\ge 3$ consecutive polls fail ($0.6\text{ s}$).
3. **`attacker` (`192.168.1.77`)**: Unauthenticated host executing attack vectors.

---

## ⚔️ Implemented Attack Scenarios (Deliverable 1)

All attacks use standard protocol operations or Linux kernel traffic shaping (`tc`/`netem`) without requiring external packet crafting or ARP poisoning:

| Scenario | Vulnerability / Mechanism | Protocol / Tool | Attack Window | Physical Impact |
|---|---|---|---|---|
| **1. Value Spoofing (FDI)** | Absent Modbus/TCP authentication | Modbus FC 6 (`write_register`) | $t=60\text{s} \to 180\text{s}$ | Injects fake $48.50\text{ Hz}$; controller ramps generation to maximum ($160+\text{ MW}$), causing a **severe over-frequency excursion ($>52.5\text{ Hz}$)**. |
| **2. Stale Replay** | Absent timestamping / freshness tokens | Modbus FC 3 $\to$ FC 6 | $t=60\text{s} \to 180\text{s}$ | Replays recorded $50.00\text{ Hz}$ nominal data during a $+3\text{ MW}$ load surge; controller stays blind, leaving frequency **depressed below safe limits**. |
| **3. Latency Injection (Delay)** | Unbounded network latency / jitter | Linux `tc netem delay 800ms` | $t=60\text{s} \to 180\text{s}$ | Controller acts on delayed state; control response is out-of-phase, causing **sustained frequency hunting and oscillations**. |
| **4. Denial-of-Service (DoS)** | Lack of transport availability | Linux `tc netem loss 90%` | $t=60\text{s} \to 180\text{s}$ | Complete loss of sensor telemetry; controller freezes setpoint in open-loop while **grid frequency drifts unchecked**. |

---

## 📁 Repository Structure

```
.
├── examples/powergrid/
│   ├── attacks/
│   │   ├── spoof.py            # Modbus FC 6 unauthenticated register overwrite
│   │   ├── replay.py           # Modbus FC 3 record -> FC 6 replay loop
│   │   ├── delay.py            # Linux tc/netem latency injection
│   │   └── dos.py              # Linux tc/netem packet loss DoS
│   ├── controller_device.py    # MiniCPS Controller PLC (Threshold & Droop+PI)
│   ├── sensor_device.py        # MiniCPS Sensor PLC (Modbus/TCP Server)
│   ├── physics.py              # Scaled swing equation physics engine
│   ├── logger.py               # Standardized simulation CSV logger
│   ├── plots.py                # Dual-panel time-series plotting utility
│   ├── topo.py                 # Mininet 3-node star topology definition
│   ├── utils.py                # System constants, Modbus tags, DB schema
│   ├── run_baseline.py / .sh   # Normal baseline runner (no attacks)
│   ├── run_spoof.py / .sh      # Spoofing attack orchestrator
│   ├── run_replay.py / .sh     # Replay attack orchestrator
│   ├── run_delay.py / .sh      # Delay attack orchestrator
│   ├── run_dos.py / .sh        # DoS attack orchestrator
│   ├── tests.py                # Unit & integration test suite
│   ├── logs/                   # Simulation CSV logs and output plots (.png)
│   └── README.md               # Powergrid module documentation
├── minicps/                    # Core MiniCPS framework library
├── requirements.txt            # Python dependencies
└── README.md                   # Main project README
```

---

## 🚀 Getting Started

### Prerequisites
* Linux environment (Ubuntu 20.04+ / WSL2 / Debian).
* Python 3.8+ with pip.
* Mininet and Open vSwitch installed (`sudo apt install mininet openvswitch-switch`).
* Root / `sudo` privileges (required by Mininet for network namespace virtualization).

### Installation
```bash
git clone https://github.com/mythri-0306/CPS-Security-with-RL--Based-Attack-Mitigation-.git
cd CPS-Security-with-RL--Based-Attack-Mitigation-

# Install Python dependencies
pip install -r requirements.txt
```

### Running Unit & Integration Tests
```bash
python examples/powergrid/tests.py
```

---

## 📊 Running Simulations & Generating Results

### 1. Normal Baseline Run (300s)
```bash
cd examples/powergrid
./run_baseline.sh 300 baseline_run1
```
* Generates `logs/baseline_run1.csv` and `logs/baseline_run1.png`.
* Demonstrates stable frequency recovery following load disturbance at $t=10\text{ s}$.

### 2. Attack Simulation Runs
```bash
# Value Spoofing Attack
./run_spoof.sh 300 spoof_run1

# Stale-Value Replay Attack
./run_replay.sh 300 replay_run1

# Network Delay Injection Attack
./run_delay.sh 300 delay_run1

# Packet Loss DoS Attack
./run_dos.sh 300 dos_run1
```

### 3. Standalone Plotting
To re-render or customize plots for any simulation CSV:
```bash
python plots.py logs/spoof_run1.csv --run-label "Spoof Attack Experiment"
```

---

## 🔬 Experimental Artifacts

Outputs generated in `examples/powergrid/logs/`:
* **`*.csv`**: High-resolution time-series data with columns:
  `timestamp`, `sim_time_sec`, `true_frequency`, `sensor_reading_sent`, `controller_reading_received`, `generation_command`, `load_demand_mw`, `attack_active`.
* **`*.png`**: Dual-panel visualizations featuring:
  * **Top Panel**: Ground truth vs. received frequency, nominal $50.00\text{ Hz}$ reference, safe operating band ($49.80 - 50.20\text{ Hz}$), and red-hatched active attack intervals.
  * **Bottom Panel**: Active generator output ($P_{\text{gen}}$) vs. electrical load demand ($P_{\text{load}}$).

---

## 📚 References & Background Literature

1. **Mississippi State University / Oak Ridge National Laboratory**: *Industrial Control System (ICS) Cyber Attack Datasets for Power Systems*, IEEE PES.
2. **NIST SP 800-82 (Rev 2)**: *Guide to Industrial Control Systems (ICS) Security*.
3. **NISTIR 7628**: *Guidelines for Smart Grid Cyber Security*.
4. **IEC 62351**: *Power Systems Management and Associated Information Exchange – Data and Communications Security*.
5. **MiniCPS Framework**: M. Antonioli and N. O. Tippenhauer, *"MiniCPS: A toolkit for security research on cyber-physical systems,"* in ACM CPS-SPC, 2015.
6. **Power System Dynamics**: P. Kundur, *Power System Stability and Control*, McGraw-Hill, 1994.

---

## 📄 License
This project is open-source under the Apache 2.0 / BSD license. See [LICENSE](LICENSE) for details.
