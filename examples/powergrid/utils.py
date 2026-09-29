"""
Power Grid Frequency Control CPS Testbed - utils.py

Defines system parameters, network configuration, Modbus tags,
scaling factors, and SQLite database schema for state storage.
"""

from minicps.utils import build_debug_logger

logger = build_debug_logger(
    name=__name__,
    bytes_per_file=100000,
    rotating_files=2,
    lformat='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    ldir='logs/',
    suffix=''
)

# ---------------------------------------------------------
# Physical Process & Control Constants
# ---------------------------------------------------------
NOMINAL_FREQ = 50.00          # Nominal grid frequency in Hz
INERTIA_H = 5.0               # Inertia constant H in seconds (df/dt = (P_gen - P_load) / (2H))
PHYSICS_PERIOD_SEC = 0.1      # Physics loop timestep dt in seconds (100 ms)
CONTROLLER_PERIOD_SEC = 0.2   # Controller polling & actuation period in seconds (200 ms)
SENSOR_PERIOD_SEC = 0.1       # Sensor update period in seconds (100 ms)

# Power values (in MW or p.u.)
NOMINAL_LOAD = 100.00         # Base load demand P_load
INITIAL_GEN_SETPOINT = 100.00 # Initial generator setpoint P_gen

# Threshold Control Rule Boundaries
FREQ_LOW_THRESH = 49.80       # Lower threshold: ramp generation UP if f < 49.8 Hz
FREQ_HIGH_THRESH = 50.20      # Upper threshold: ramp generation DOWN if f > 50.2 Hz
RAMP_STEP = 0.50              # Generation setpoint increment/decrement per control cycle (MW)

# Modbus integer scaling factor (Modbus registers are 16-bit unsigned ints)
# e.g., 50.00 Hz -> 5000, 100.00 MW -> 10000 (preserves 2 decimal places)
SCALE_FACTOR = 100.0

# ---------------------------------------------------------
# Network Topology Configuration
# ---------------------------------------------------------
IP = {
    'sensor': '192.168.1.10',
    'controller': '192.168.1.20',
    'attacker': '192.168.1.77',
}

NETMASK = '/24'

MAC = {
    'sensor': '00:00:00:00:00:01',
    'controller': '00:00:00:00:00:02',
    'attacker': 'AA:AA:AA:AA:AA:AA',
}

MODBUS_PORT = 502

# ---------------------------------------------------------
# Modbus Protocol & Register Tag Mapping
# ---------------------------------------------------------
# Holding register 0: Frequency scaled by 100 (read by controller, written by sensor/physics)
# Holding register 1: Generation setpoint scaled by 100 (written by controller, read by physics)
FREQ_HR_TAG = ('HR', 0)
GEN_SETPOINT_HR_TAG = ('HR', 1)

SENSOR_ADDR = IP['sensor'] + ':' + str(MODBUS_PORT)
CONTROLLER_ADDR = IP['controller'] + ':' + str(MODBUS_PORT)

# Modbus Server specification for Sensor PLC (mode 1 = TCP Modbus Server)
SENSOR_SERVER = {
    'address': SENSOR_ADDR,
    'tags': (10, 10, 10, 10)  # (discrete_inputs, coils, input_registers, holding_registers)
}

SENSOR_PROTOCOL = {
    'name': 'modbus',
    'mode': 1,
    'server': SENSOR_SERVER
}

# Modbus Client specification for Controller PLC (mode 0 = Client only)
CONTROLLER_PROTOCOL = {
    'name': 'modbus',
    'mode': 0,
    'server': {}
}

# ---------------------------------------------------------
# State Database Configuration (SQLite)
# ---------------------------------------------------------
PATH = 'powergrid_db.sqlite'
NAME = 'powergrid'

STATE = {
    'name': NAME,
    'path': PATH
}

# Tags in SQLite state database: tuple of (name, pid)
FREQ_TAG = ('FREQ', 1)
GEN_SETPOINT_TAG = ('GEN_SETPOINT', 1)
LOAD_DEMAND_TAG = ('LOAD_DEMAND', 1)
SENSOR_FREQ_TAG = ('SENSOR_FREQ', 1)
CONTROLLER_FREQ_TAG = ('CONTROLLER_FREQ', 1)
ATTACK_ACTIVE_TAG = ('ATTACK_ACTIVE', 1)

SCHEMA = """
CREATE TABLE powergrid (
    name              TEXT NOT NULL,
    pid               INTEGER NOT NULL,
    value             TEXT,
    PRIMARY KEY (name, pid)
);
"""

SCHEMA_INIT = f"""
    INSERT INTO powergrid VALUES ('FREQ', 1, '{NOMINAL_FREQ:.2f}');
    INSERT INTO powergrid VALUES ('GEN_SETPOINT', 1, '{INITIAL_GEN_SETPOINT:.2f}');
    INSERT INTO powergrid VALUES ('LOAD_DEMAND', 1, '{NOMINAL_LOAD:.2f}');
    INSERT INTO powergrid VALUES ('SENSOR_FREQ', 1, '{NOMINAL_FREQ:.2f}');
    INSERT INTO powergrid VALUES ('CONTROLLER_FREQ', 1, '{NOMINAL_FREQ:.2f}');
    INSERT INTO powergrid VALUES ('ATTACK_ACTIVE', 1, '0');
"""
