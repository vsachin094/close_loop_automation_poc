# Closed Loop Automation

This project implements a closed-loop automation system for network devices. It monitors interface statuses using SNMP, detects anomalies such as interface flaps, and logs the events for further analysis.

## Features

- Collects interface status data using SNMP v3
- Trains a machine learning model to detect interface flaps
- Detects anomalies and logs events
- Supports automatic device type detection using Netmiko

## Requirements

- Python 3.7+
- `numpy`
- `pandas`
- `netmiko`
- `pysnmp`
- `scikit-learn`
- `pyyaml`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/closed_loop_automation.git
    cd closed_loop_automation
    ```

2. Install the required packages:
    ```sh
    pip install numpy pandas netmiko pysnmp scikit-learn pyyaml
    ```

## Configuration

1. Create a `devices.yaml` file with the following structure:
    ```yaml
    snmp:
      user: "your_snmp_user"
      authkey: "your_snmp_authkey"
      privkey: "your_snmp_privkey"
    devices:
      "192.168.1.1":
        ip: "192.168.1.1"
        interfaces: [1, 2, 3]
      "192.168.1.2":
        ip: "192.168.1.2"
        interfaces: [1, 2, 3]
    ```

2. Ensure the `logs` directory exists:
    ```sh
    mkdir -p logs
    ```

## Usage

Run the main script:
```sh
python main_script.py
```

## Logging

Logs are saved to `closed_loop.log` in the project directory. Interface flap tracking data is saved to `logs/flap_tracking.json`.

## Functions

- `load_flap_tracking()`: Loads interface status tracking data from a JSON file.
- `save_flap_tracking(data)`: Saves interface status tracking data to a JSON file.
- `load_device_config(filename)`: Loads device information from a YAML file.
- `autodetect_device_type(device)`: Automatically detects device type using Netmiko's SSHDetect.
- `connect_to_device(device)`: Establishes SSH connection to the device.
- `snmp_get_v3(ip, user, authkey, privkey, oid)`: Fetches SNMP v3 data from the device.
- `get_interface_status_snmp(ip, snmp_config, interface_index)`: Fetches interface status via SNMP v3.
- `collect_data_snmp(device, snmp_config, samples, interval)`: Collects data for ML training using SNMP v3.
- `train_model(df)`: Trains ML model to detect interface flaps.
- `detect_anomaly(model, encoder, device, snmp_config)`: Detects interface flaps and applies remediation if necessary.
- `main()`: Main function to execute closed-loop automation.

## License

This project is licensed under the MIT License.