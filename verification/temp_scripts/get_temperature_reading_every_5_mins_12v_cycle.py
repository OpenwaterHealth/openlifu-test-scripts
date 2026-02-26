from __future__ import annotations

import sys
import time

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_toggle_12v.py

SLEEP_INTERVAL_SECONDS = 5*60  # 5 minutes

print("Starting LIFU Test Script...")
interface = LIFUInterface()
tx_connected, hv_connected = interface.is_device_connected()
if not tx_connected:
    interface.hvcontroller.turn_12v_on()
    time.sleep(2)
    interface.stop_monitoring()
    del interface
    time.sleep(1)  # Short delay before recreating

    print("Reinitializing LIFU interface after powering 12V...")
    interface =  LIFUInterface()

    # Re-check connection
    tx_connected, hv_connected = interface.is_device_connected()

print("Connecting device...")

# tx_connected, hv_connected = interface.is_device_connected()

# tx_connected, hv_connected = interface.is_device_connected()
if tx_connected and hv_connected:
    print("LIFU Device Fully connected.")
else:
    print(f'LIFU Device NOT Fully Connected. TX: {tx_connected}, HV: {hv_connected}')

if not hv_connected:
    print("HV Controller not connected.")
    sys.exit()

interface.hvcontroller.ping()

while True: 
    print(f"TX Device temperature: {interface.txdevice.get_temperature()}")
    print("Sleeping for 5 minutes before next reading...")
    interface.hvcontroller.turn_12v_off()
    time.sleep(SLEEP_INTERVAL_SECONDS)  # Sleep for 5 minutes (300 seconds)
    interface.hvcontroller.turn_12v_on()