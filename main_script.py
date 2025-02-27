import time
import json
import logging
import yaml
import os
import numpy as np
import pandas as pd
import copy
from collections import defaultdict
from netmiko import ConnectHandler
from netmiko.ssh_autodetect import SSHDetect
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configure Logging
logging.basicConfig(filename="closed_loop.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure logs directory exists
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
FLAP_TRACKING_FILE = os.path.join(LOG_DIR, "flap_tracking.json")
STATUS_HISTORY_FILE = os.path.join(LOG_DIR, "status_history.csv")
FLAP_THRESHOLD = 3  # Number of flaps before remediation
FLAP_TIME_WINDOW = 300  # Time window in seconds for counting flaps

def load_flap_tracking():
    """Load interface status tracking data from a JSON file."""
    try:
        with open(FLAP_TRACKING_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return defaultdict(lambda: {"last_status": None, "flap_count": 0})

def save_flap_tracking(data):
    """Save interface status tracking data to a JSON file."""
    logging.debug(f"Saving flap tracking data: {data}")
    with open(FLAP_TRACKING_FILE, "w") as file:
        json.dump(data, file, indent=4)

def append_status_history(timestamp, ip, interface, status):
    """Append interface status history to a CSV file."""
    with open(STATUS_HISTORY_FILE, "a") as file:
        file.write(f"{timestamp},{ip},{interface},{status}\n")

interface_status_history = load_flap_tracking()

def load_device_config(filename="devices.yaml"):
    """Load device information from a YAML file."""
    with open(filename, "r") as file:
        return yaml.safe_load(file)

def autodetect_device_type(device):
    """Automatically detect device type using Netmiko's SSHDetect."""
    device["device_type"] = "autodetect"
    device_copy = copy.deepcopy(device)
    device_copy.pop("interfaces", None)  # Ensure 'interfaces' key is removed
    guesser = SSHDetect(**device_copy)
    return guesser.autodetect()

def connect_to_device(device):
    """Establish SSH connection to the device."""
    if "device_type" not in device or not device["device_type"]:
        device["device_type"] = autodetect_device_type(device)
    device_copy = copy.deepcopy(device)
    device_copy.pop("interfaces", None)  # Ensure 'interfaces' key is removed
    logging.debug(f"Device parameters before SSH connection: {device_copy}")
    return ConnectHandler(**device_copy)

def get_interface_status_ssh(conn, interface):
    """Fetch interface status via SSH."""
    try:
        output = conn.send_command(f"show interface {interface} brief")
        return "up" if "up" in output.lower() else "down"
    except Exception as e:
        logging.error(f"SSH Error: {e}")
        return None

def shutdown_interface(conn, interface):
    """Shut down the interface via SSH."""
    try:
        conn.send_config_set([f"interface {interface}", "shutdown", "description shut the interface by automation due to flapping"])
        logging.info(f"Interface {interface} has been shut down due to flapping.")
    except Exception as e:
        logging.error(f"Error shutting down interface {interface}: {e}")

def collect_data_ssh(device, ip, samples=50, interval=5):
    """Collect data for ML training using SSH."""
    interfaces = device.get("interfaces", [])
    data = []
    conn = connect_to_device(device)
    if not conn:
        return pd.DataFrame(columns=["timestamp", "ip", "interface", "status"])
    try:
        for _ in range(samples):
            for interface in interfaces:
                status = get_interface_status_ssh(conn, interface)
                timestamp = time.time()
                data.append([timestamp, ip, interface, status])
                append_status_history(timestamp, ip, interface, status)
                logging.info(f"Collected data: {data[-1]}")
            time.sleep(interval)
    except Exception as e:
        logging.error(f"SSH Connection Error: {e}")
    finally:
        conn.disconnect()
    return pd.DataFrame(data, columns=["timestamp", "ip", "interface", "status"])

def train_model():
    """Train ML model to detect interface flaps."""
    if not os.path.exists(STATUS_HISTORY_FILE):
        logging.error("Status history file does not exist. Skipping model training.")
        return None, None
    df = pd.read_csv(STATUS_HISTORY_FILE, names=["timestamp", "ip", "interface", "status"])
    if df.empty or len(df) < 10:  # Ensure there is enough data to train the model
        logging.error("Not enough training data. Skipping model training.")
        return None, None
    encoder = LabelEncoder()
    df["status_encoded"] = encoder.fit_transform(df["status"])
    X = df[["timestamp"]]
    y = df["status_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if X_train.empty or y_train.empty:
        logging.error("Training set is empty after train-test split. Skipping model training.")
        return None, None
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, encoder

def detect_anomaly(model, encoder, device, ip):
    """Detect interface flaps and apply remediation if necessary."""
    if model is None or encoder is None:
        logging.error("Model or encoder is not available. Skipping anomaly detection.")
        return
    global interface_status_history
    interfaces = device.get("interfaces", [])
    conn = connect_to_device(device)
    if not conn:
        return
    try:
        for interface in interfaces:
            status = get_interface_status_ssh(conn, interface)
            history = interface_status_history.setdefault(f"{ip}_{interface}", {"last_status": None, "flap_count": 0})
            if history["last_status"] and history["last_status"] != status:
                history["flap_count"] += 1
                history["last_flap_time"] = time.time()
                logging.warning(f"Flap detected: Interface {interface} on device {ip} status changed to {status}. Count: {history['flap_count']}")
            history["last_status"] = status
            if "last_flap_time" in history and time.time() - history["last_flap_time"] > FLAP_TIME_WINDOW:
                history["flap_count"] = 0
            save_flap_tracking(interface_status_history)
            if history["flap_count"] >= FLAP_THRESHOLD:
                if status == "up":
                    logging.error(f"Anomaly detected: Interface {interface} on device {ip} exceeded flap threshold. Shutting down interface.")
                    shutdown_interface(conn, interface)
                else:
                    logging.info(f"Interface {interface} on device {ip} is already shut down. No action taken.")
                save_flap_tracking(interface_status_history)
    finally:
        conn.disconnect()

def main():
    """Main function to execute closed-loop automation."""
    config = load_device_config()
    devices = config["devices"]
    while True:
        for ip, device in devices.items():
            collect_data_ssh(device, ip)
        model, encoder = train_model()
        if model and encoder:  # Ensure model and encoder are available before starting anomaly detection
            for ip, device in devices.items():
                detect_anomaly(model, encoder, device, ip)
        time.sleep(600)  # Collect data and retrain the model every 10 minutes

if __name__ == "__main__":
    main()