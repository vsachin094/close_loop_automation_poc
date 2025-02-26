import time
import json
import logging
import yaml
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from netmiko import ConnectHandler
from netmiko.ssh_autodetect import SSHDetect
from pysnmp.hlapi import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configure Logging
logging.basicConfig(filename="closed_loop.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure logs directory exists
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
FLAP_TRACKING_FILE = os.path.join(LOG_DIR, "flap_tracking.json")
FLAP_THRESHOLD = 3  # Number of flaps before remediation
FLAP_TIME_WINDOW = 300  # Time window in seconds for counting flaps

OID_IF_OPER_STATUS = "1.3.6.1.2.1.2.2.1.8"

def load_flap_tracking():
    """Load interface status tracking data from a JSON file."""
    try:
        with open(FLAP_TRACKING_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return defaultdict(lambda: {"last_status": None, "flap_count": 0})

def save_flap_tracking(data):
    """Save interface status tracking data to a JSON file."""
    with open(FLAP_TRACKING_FILE, "w") as file:
        json.dump(data, file, indent=4)

interface_status_history = load_flap_tracking()

def load_device_config(filename="devices.yaml"):
    """Load device information from a YAML file."""
    with open(filename, "r") as file:
        return yaml.safe_load(file)

def autodetect_device_type(device):
    """Automatically detect device type using Netmiko's SSHDetect."""
    guesser = SSHDetect(**device)
    return guesser.autodetect()

def connect_to_device(device):
    """Establish SSH connection to the device."""
    if "device_type" not in device or not device["device_type"]:
        device["device_type"] = autodetect_device_type(device)
    return ConnectHandler(**device)

def snmp_get_v3(ip, user, authkey, privkey, oid):
    """Fetch SNMP v3 data from the device."""
    iterator = getCmd(SnmpEngine(),
                      UsmUserData(user, authkey, privkey, authProtocol=usmHMACSHAAuthProtocol,
                                  privProtocol=usmAesCfb128Protocol),
                      UdpTransportTarget((ip, 161)),
                      ContextData(),
                      ObjectType(ObjectIdentity(oid)))
    
    errorIndication, errorStatus, errorIndex, varBinds = next(iterator)
    if errorIndication:
        logging.error(f"SNMP v3 error: {errorIndication}")
        return None
    for varBind in varBinds:
        return int(varBind[1])
    return None

def get_interface_status_snmp(ip, snmp_config, interface_index):
    """Fetch interface status via SNMP v3."""
    status = snmp_get_v3(ip, snmp_config["user"], snmp_config["authkey"], snmp_config["privkey"], f"{OID_IF_OPER_STATUS}.{interface_index}")
    return "up" if status == 1 else "down"

def collect_data_snmp(device, snmp_config, samples=50, interval=5):
    """Collect data for ML training using SNMP v3."""
    ip = device["ip"]
    interfaces = device.get("interfaces", [])
    data = []
    for _ in range(samples):
        for interface_index in interfaces:
            status = get_interface_status_snmp(ip, snmp_config, interface_index)
            data.append([time.time(), interface_index, status])
            logging.info(f"Collected data: {data[-1]}")
        time.sleep(interval)
    return pd.DataFrame(data, columns=["timestamp", "interface", "status"])

def train_model(df):
    """Train ML model to detect interface flaps."""
    encoder = LabelEncoder()
    df["status_encoded"] = encoder.fit_transform(df["status"])
    X = df[["timestamp"]]
    y = df["status_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, encoder

def detect_anomaly(model, encoder, device, snmp_config):
    """Detect interface flaps and apply remediation if necessary."""
    global interface_status_history
    ip = device["ip"]
    interfaces = device.get("interfaces", [])
    for interface_index in interfaces:
        status = get_interface_status_snmp(ip, snmp_config, interface_index)
        history = interface_status_history.setdefault(interface_index, {"last_status": None, "flap_count": 0})
        if history["last_status"] and history["last_status"] != status:
            history["flap_count"] += 1
            history["last_flap_time"] = time.time()
            logging.warning(f"Flap detected: Interface {interface_index} status changed to {status}. Count: {history['flap_count']}")
        history["last_status"] = status
        if "last_flap_time" in history and time.time() - history["last_flap_time"] > FLAP_TIME_WINDOW:
            history["flap_count"] = 0
        save_flap_tracking(interface_status_history)
        if history["flap_count"] >= FLAP_THRESHOLD:
            logging.error(f"Anomaly detected: Interface {interface_index} exceeded flap threshold. Manual intervention required.")
            history["flap_count"] = 0  # Reset after logging
            save_flap_tracking(interface_status_history)

def main():
    """Main function to execute closed-loop automation."""
    config = load_device_config()
    snmp_config = config["snmp"]
    devices = config["devices"]
    for ip, device in devices.items():
        df = collect_data_snmp(device, snmp_config)
        model, encoder = train_model(df)
        while True:
            detect_anomaly(model, encoder, device, snmp_config)
            time.sleep(10)  # Poll every 10 seconds

if __name__ == "__main__":
    main()
