#!/usr/bin/env python3
"""
LIFU Test Script to satisfy PRODREQS-85: Sonication Duration

A professional command-line tool for automated thermal stress testing of LIFU devices.
This script connects to the device, configures test parameters, monitors temperatures,
and logs data with automatic safety shutoffs.

Author: OpenLIFU Team
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import os
import threading
import time
import asyncio
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from packaging.version import Version

from serial.serialutil import SerialException

import numpy as np

# from openlifu_sdk import LIFUInterface
import openlifu_sdk
from openlifu_sdk.io import LIFUInterface
# from openlifu.bf.pulse import Pulse
# from openlifu.bf.sequence import Sequence
# from openlifu.db import Database
# from openlifu.geo import Point
# from openlifu_sdk.io import LIFUInterface
# from openlifu.plan.solution import Solution

try:
    from .config import *
except ImportError:
    from config import *

"""
Thermal Stress Test Script
- User selects a test case.
- Test runs for a fixed total duration or until a thermal shutdown occurs.
- Logs temperature and device status.
"""

__version__ = "1.0.3"
TEST_ID = Path(__file__).name.replace(".py", "")

# Constants for solution generation
SPEED_OF_SOUND = 1500  # Speed of sound in m/s, used for time-of-flight calculations
NUM_ELEMENTS_PER_MODULE = 64  # Assuming each module has 64 elements, adjust as needed

def _base_path():
    """Return the directory containing bundled data files.
    Works in both frozen (PyInstaller) and normal Python execution."""
    import sys as _sys
    import os as _os
    if getattr(_sys, 'frozen', False):
        return _sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

MIN_REQUIRED_CONSOLE_FW_VERSION = "1.2.4"
MIN_REQUIRED_TX_FW_VERSION = "2.0.5"

# ------------------- Test Case Definitions ------------------- #
TEST_CASES = [
    {"voltage": 65, "duty_cycle": 5,  "PRI_ms": 200, "max_starting_temperature": 32},
    {"voltage": 60, "duty_cycle": 10, "PRI_ms": 200, "max_starting_temperature": 32},
    {"voltage": 55, "duty_cycle": 15, "PRI_ms": 200, "max_starting_temperature": 32},
    {"voltage": 50, "duty_cycle": 20, "PRI_ms": 200, "max_starting_temperature": 32},
    {"voltage": 45, "duty_cycle": 25, "PRI_ms": 200, "max_starting_temperature": 32},
    {"voltage": 40, "duty_cycle": 30, "PRI_ms": 200, "max_starting_temperature": 32},
    {"voltage": 35, "duty_cycle": 35, "PRI_ms": 200, "max_starting_temperature": 32},
    {"voltage": 30, "duty_cycle": 40, "PRI_ms": 200, "max_starting_temperature": 32},
    {"voltage": 25, "duty_cycle": 45, "PRI_ms": 200, "max_starting_temperature": 32},
    {"voltage": 20, "duty_cycle": 50, "PRI_ms": 200, "max_starting_temperature": 60},
    {"voltage": 15, "duty_cycle": 50, "PRI_ms": 200, "max_starting_temperature": 60},
    {"voltage": 10, "duty_cycle": 50, "PRI_ms": 200, "max_starting_temperature": 60},
    {"voltage": 5,  "duty_cycle": 50, "PRI_ms": 200, "max_starting_temperature": 60},
]

default_coordinates = {"x": 0.0, "y": 0.0, "z": 50.0}

class SafeFormatter(logging.Formatter):
    """Formatter that handles Unicode characters safely on Windows."""

    def format(self, record):
        """Format the log record, removing unsupported Unicode characters."""
        try:
            msg = super().format(record)
            # Remove or replace emojis/non-ASCII characters for Windows compatibility
            return msg.encode("ascii", "ignore").decode("ascii")
        except Exception:
            return super().format(record)

def format_hhmmss(seconds: float) -> str:
    """Format a number of seconds into HH:MM:SS or MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

'''TODO:
def calculate_temperature_discharge_time_seconds(final_temp_C: float) -> float:
    """Estimate time to discharge temperature to safe levels (dummy implementation)."""
    # Placeholder: Implement a realistic model based on device thermal characteristics
    baseline_temp_C = 30.0
    minutes = 1
    return 60.0*minutes # temp value until formula figured out
'''

class AbortTest(Exception):
    """Custom exception to signal test abortion."""
    pass

class TestSonicationDurationBase:
    """Main class for Thermal Stress Test 5."""
    def __init__(
            self, 
            frequency_khz: Optional[float] = None,
            num_modules: Optional[int] = None,
            external_power: bool = False,
            simulate: bool = False,
            test_runthrough: bool = False,
            console_shutoff_temp: float = CONSOLE_SHUTOFF_TEMP_C_DEFAULT,
            tx_shutoff_temp: float = TX_SHUTOFF_TEMP_C_DEFAULT,
            ambient_shutoff_temp: float = AMBIENT_SHUTOFF_TEMP_C_DEFAULT,
            temperature_check_interval: float = TEMPERATURE_CHECK_INTERVAL_DEFAULT,
            temperature_log_interval: float = TEMPERATURE_LOG_INTERVAL_DEFAULT,
            log_dir: Optional[str] = None,
            verbose: bool = False,
            quiet: bool = False,
            skip_logfile: bool = False,
            bypass_console_fw: bool = False,
            bypass_tx_fw: bool = False,
            test_case: Optional[int] = None,
            interface: Optional[LIFUInterface] = None
    ) -> None:
            # args):
        # self.args = args

        # Derived paths
        self.openlifu_dir = Path(openlifu_sdk.__file__).parent.parent.parent.resolve()
        self.log_dir = Path(log_dir or (Path(__file__).resolve().parents[1] / "logs"))
        
        # Runtime attributes
        self.interface = interface

        self.shutdown_event = threading.Event()
        self.sequence_complete_event = threading.Event()
        self.temperature_shutdown_event = threading.Event()
        self.voltage_shutdown_event = threading.Event()
        self.bypass_transmitter = False
        
        # Threading locks
        self.mutex = threading.Lock()

        self.stop_logging = False
        # self.voltage_accuracy_test = True

        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time: float | None = None

        # Test configuration – set later via interactive selection
        self.frequency_khz = frequency_khz
        self.test_case_num: int | None = None
        self.voltage: float | None = None
        self.interval_msec: float | None = None
        self.duration_msec: int | None = None
        self.num_modules = num_modules

        self.test_status: str | None = "not started"
        self.sequence_duration: float = TEST_CASE_DURATION_SECONDS
        self.starting_test_case: int = test_case
        self.test_results: dict[int, TestCaseResult] = {}
        self.test_case_start_time: float | None = 0.0
        self.is_in_cooldown: bool = False
        self.cooldown_start_time: float | None = None

        # Flags from args
        self.use_external_power = external_power
        self.hw_simulate = simulate
        self.test_runthrough = test_runthrough
        self.bypass_console_fw = bypass_console_fw
        self.bypass_tx_fw = bypass_tx_fw

        # Safety parameters from args
        self.console_shutoff_temp_C = console_shutoff_temp
        self.tx_shutoff_temp_C = tx_shutoff_temp
        self.ambient_shutoff_temp_C = ambient_shutoff_temp
        self.voltage_deviation_percentage_limit = VOLTAGE_DEVIATION_PERCENTAGE_LIMIT
        self.temperature_check_interval = temperature_check_interval
        self.temperature_log_interval = temperature_log_interval

        # Solution loading state
        self._solution_loaded = False
        self._loaded_solution_data = None
        self._solution_name = ""

        # Logger
        self._file_handler_attached = False
        self.verbose = verbose
        self.quiet = quiet
        self.skip_logfile = skip_logfile
        self.log_file_path = ""
        self.logger = self._setup_logging()

        # self.logger.debug(f"{TEST_ID} initialized with arguments: {self.args}")

    def monitor_interface(self):
        asyncio.run(self.interface.start_monitoring(interval=1))

    def _get_test_id(self) -> str:
        module_name = getattr(self.__class__, "__module__", "")
        if module_name:
            return module_name.rsplit(".", 1)[-1]
        return self.__class__.__name__

    def _setup_logging(self) -> logging.Logger:
        """Configure root logger with console output; file handler added later."""
        logger = logging.getLogger(__name__)

        # Set log level based on verbosity
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        elif self.quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

        # Prevent duplicate handlers when re-run
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = SafeFormatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.propagate = False
        return logger

    ###### going to be different for each test, may want to move to child class ######
    def _attach_file_handler(self) -> None:
        """Attach a file handler for this run once test case is known."""
        if not self.skip_logfile:
            if self._file_handler_attached:
                return

            self.log_dir.mkdir(parents=True, exist_ok=True)

            test_id = self._get_test_id()
            filename = f"{self.run_timestamp}_{test_id}_{self.frequency_khz}kHz_{self.num_modules}x.log"

            log_path = self.log_dir / filename
            self.log_file_path = str(log_path)  # Store as instance attribute for access by connector

            formatter = SafeFormatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logging.getLogger(__name__).addHandler(file_handler)

            self._file_handler_attached = True
            # self.logger.info("Run log will be saved to: %s", log_path)
        else:
            self.logger.info("User elected to skip logging to file - logs will only be output to console.")



    # ------------------- User Input Section ------------------- #

    def _select_frequency(self) -> None:
        """Select TX frequency in kHz (100–500), CLI override or interactive."""
        # CLI override
        if self.frequency_khz is not None:
            # self.frequency_khz = self.frequency
            return

        while True:
            choice = input("Enter TX frequency in kHz (100–500): ").strip()
            try:
                freq = int(choice)
            except ValueError:
                self.logger.info("Invalid input. Enter an integer value.")
                continue

            if 100 <= freq <= 500:
                self.frequency_khz = freq
                return

            self.logger.info("Frequency must be between 100 and 500 kHz.")

    def _select_num_modules(self) -> None:
        """Interactively select number of modules."""
        # CLI override
        if self.num_modules is not None:
            # self.num_modules = self.args.num_modules
            return

        while True:
            choice = input(f"Select number of modules {list(NUM_MODULES)}: ")
            if choice.isdigit() and int(choice) in NUM_MODULES:
                self.num_modules = int(choice)
                break
            self.logger.info("Invalid selection. Please try again.")


    def _select_starting_test_case(self) -> None:
        valid_test_nums = list(range(1, len(TEST_CASES) + 1))

        self.starting_test_case = 1
        return

        # Test case selection
        if self.starting_test_case is not None:
            return
        else:
            self.logger.info("\nAvailable Test Cases:")
            for test_id, test_case in enumerate(TEST_CASES, start=1):
                self.logger.info(
                    f"Test Case {test_id}. {test_case['voltage']}V, "
                    f"{test_case['duty_cycle']}% Duty Cycle, total"
                )

            while True:
                choice = input(f"Press enter to run through all test cases or select a test case by number to start at: ")
                if choice == "":
                    self.starting_test_case = 1
                    break
                if choice.isdigit() and int(choice) in valid_test_nums:
                    self.starting_test_case = int(choice)
                    break
                self.logger.info("Invalid selection. Please try again.")

    def connect_device(self) -> None:
        """Connect to the LIFU device and verify connection."""
        
        if self.interface is not None:
            self.logger.info("Using provided LIFUInterface instance.")

            # Reconnect console UART
            if not self.use_external_power and self.interface.hvcontroller and not self.interface.hvcontroller.uart.is_connected:
                self.interface.hvcontroller.uart.connect()

            if not self.use_external_power and not self.interface.hvcontroller.get_12v_status():
                self.logger.info("TX device not powered. Turning 12V on...")
                self.interface.hvcontroller.turn_12v_on()
                time.sleep(5)  # Give the TX board time to boot and mount to the USB bus
                
            # Reconnect tx UART
            if hasattr(self.interface, 'txdevice') and not self.interface.txdevice.uart.is_connected:
                self.interface.txdevice.uart.connect()
            self.logger.info("TX uart port: %s, is_open: %s",
                self.interface.txdevice.uart._find_port(),
                self.interface.txdevice.uart._serial.is_open if (self.interface.txdevice.uart._serial is not None) else "None"
            )
            return
            
        self.interface = LIFUInterface(
            ext_power_supply=self.use_external_power,
            TX_test_mode=self.hw_simulate,
            HV_test_mode=self.hw_simulate,
            voltage_table_selection="evt0",
            sequence_time_selection="stress_test",
            run_async=True
        )
        time.sleep(2)

        if self.bypass_transmitter: tx_connected = True

        monitor_thread = threading.Thread(target=self.monitor_interface, daemon=True)
        monitor_thread.daemon = True
        monitor_thread.start()

        self.interface._tx_uart.signal_connect.connect(
            lambda desc, port: self.logger.info("TX connected on %s", port)
        )
        self.interface._tx_uart.signal_disconnect.connect(
            lambda desc, port: self.logger.info("TX disconnected from %s", port)
        )
        if self.interface._hv_uart is not None:
            self.interface._hv_uart.signal_connect.connect(
                lambda desc, port: self.logger.info("HV connected on %s", port)
            )
            self.interface._hv_uart.signal_disconnect.connect(
                lambda desc, port: self.logger.info("HV disconnected from %s", port)
            )
            
        time.sleep(2)
        
        tx_connected, hv_connected = self.interface.is_device_connected()
        if not self.use_external_power and not tx_connected:
            self.logger.warning("TX device not connected. Attempting to turn on 12V...")
            try:
                for attempt in range(3):
                    if self.interface.hvcontroller.get_12v_status():
                        self.logger.info("12V is now on.")
                        break
                    try:
                        self.interface.hvcontroller.turn_12v_on()
                        time.sleep(1)
                    except Exception as e:
                        self.logger.error("Error turning on 12V: %s", e)
            except Exception as e:
                self.logger.error("Error turning on 12V after 3 attempts: %s", e)
            time.sleep(2)

            # Reinitialize interface after powering 12V
            try:
                self.interface.stop_monitoring()
            except Exception as e:
                self.logger.warning("Error stopping monitoring during reinit: %s", e)

            # with contextlib.suppress(Exception):
            #     del self.interface

            # time.sleep(1)
            # self.logger.info("Reinitializing LIFU interface after powering 12V...")
            # self.interface = LIFUInterface(
            #     ext_power_supply=self.use_external_power,
            #     TX_test_mode=self.hw_simulate,
            #     HV_test_mode=self.hw_simulate,
            #     voltage_table_selection="evt0",
            #     sequence_time_selection="stress_test",
            #     run_async=True
            # )
            # tx_connected, hv_connected = self.interface.is_device_connected()
            tx_connected, hv_connected = self.interface.is_device_connected()

        if not self.use_external_power:
            if hv_connected:
                self.logger.info("  HV Connected: %s", hv_connected)
            else:
                self.logger.error("HV NOT fully connected.")
                # sys.exit(1)
        else:
            self.logger.info("  Using external power supply")

        if tx_connected:
            if not self.bypass_transmitter:
                self.logger.info("  TX Connected: %s", tx_connected)
                self.logger.info("LIFU Device fully connected.")
        else:
            self.logger.error("TX NOT fully connected.")
            sys.exit(1)

    def verify_communication(self) -> bool:
        """Verify communication with the LIFU device."""
        if self.interface is None:
            self.logger.error("Interface not connected for communication verification.")
            return False

        try:
            if not self.use_external_power and not self.interface.hvcontroller.ping():
                self.logger.error("Failed to ping the console device.")
            else:
                self.logger.info("Successfully pinged console device")
        except Exception as e:
            self.logger.error("Console Communication verification failed: %s", e)
            return False

        if not self.bypass_transmitter:
            try:
                if not self.interface.txdevice.ping():
                    self.logger.error("Failed to ping the transmitter device.")
                return True
            except Exception as e:
                self.logger.error("TX Device Communication verification failed: %s", e)
                return False
        else:
            return True
        
    def _parse_fw_version(self, version_str: str) -> Version:
        """Strip any local version suffix (e.g. '2.0.3-tsfdj') before parsing."""
        base = version_str.split("-")[0]
        return Version(base)

    def get_firmware_versions(self) -> None:
        """Retrieve and log firmware versions from the LIFU device."""
        if self.interface is None:
            self.logger.error("Interface not connected for firmware version retrieval.")
            return

        console_fw_mismatch = False
        tx_fw_mismatch = False
        
        try:
            if not self.use_external_power:
                console_fw = self.interface.hvcontroller.get_version()
                self.logger.info("Console Firmware Version: %s", console_fw)
            if not self.bypass_console_fw and self._parse_fw_version(console_fw) < self._parse_fw_version(MIN_REQUIRED_CONSOLE_FW_VERSION):
                self.logger.error("Console firmware version %s does not match required version %s.",
                                console_fw, MIN_REQUIRED_CONSOLE_FW_VERSION)
                console_fw_mismatch = True
        except Exception as e:
            self.logger.error("Error retrieving console firmware version: %s", e)

        if not self.bypass_transmitter:        
            try:
                for i in range(self.num_modules):
                    tx_fw = self.interface.txdevice.get_version(module=i)
                    self.logger.info("TX Device %d Firmware Version: %s", i, tx_fw)
                    if not self.bypass_tx_fw and self._parse_fw_version(tx_fw) < self._parse_fw_version(MIN_REQUIRED_TX_FW_VERSION):
                        self.logger.error("TX firmware version %s does not match required version %s.",
                                        tx_fw, MIN_REQUIRED_TX_FW_VERSION)
                        tx_fw_mismatch = True
            except Exception as e:
                self.logger.error("Error retrieving TX device firmware version: %s", e)        

        if console_fw_mismatch or tx_fw_mismatch:
            if console_fw_mismatch:
                self.logger.error("\n\n!! Incompatible console firmware version, please upgrade to %s !!\n\n", MIN_REQUIRED_CONSOLE_FW_VERSION)
            if tx_fw_mismatch:
                self.logger.error("\n\n!! Incompatible TX firmware version, please upgrade to %s !!\n\n", MIN_REQUIRED_TX_FW_VERSION)
            sys.exit()

    def enumerate_devices(self):
        """Enumerate TX7332 devices and verify count."""
        self.logger.info("Enumerate TX7332 chips")
        num_tx_devices = self.interface.txdevice.enum_tx7332_devices()

        if num_tx_devices == 0:
            raise ValueError("No TX7332 devices found.")
        elif num_tx_devices == self.num_modules * 2:
            self.logger.info(f"Number of TX7332 devices found: {num_tx_devices}")
            return 32 * num_tx_devices
        else:
            raise Exception(f"Number of TX7332 devices found: {num_tx_devices} != 2x{self.num_modules}")

    def get_solution(self, xInput, yInput, zInput, freq, voltage, pulseInterval, pulseCount, trainInterval, trainCount, durationS, validate=False):
        """Get or calculate a solution dictionary.
        
        If a solution is loaded, use that; otherwise calculate a new one based on the parameters.
        
        Args:
            xInput, yInput, zInput: Focus point coordinates
            freq: Frequency in kHz
            voltage: Voltage in volts
            pulseInterval: Pulse interval in ms
            pulseCount: Number of pulses
            trainInterval: Train interval
            trainCount: Number of trains
            durationS: Duration in microseconds
            validate: If True, validate the loaded solution against connected modules
            
        Returns:
            dict: Solution dictionary with delays, apodizations, pulse, sequence, and transducer data
        """
        # Validate parameter types
        try:
            xInput = float(xInput)
            yInput = float(yInput)
            zInput = float(zInput)
            freq = float(freq)
            voltage = float(voltage)
            pulseInterval = float(pulseInterval)
            pulseCount = int(pulseCount)
            trainInterval = float(trainInterval)
            trainCount = int(trainCount)
            durationS = float(durationS)
        except (TypeError, ValueError) as e:
            self.logger.error(f"Invalid parameter type in get_solution: {e}")
            raise ValueError(f"Invalid parameter type in get_solution: {e}")
        
        if self._solution_loaded:
            self.logger.info("Using loaded solution for configuration")
            solution = self._loaded_solution_data.copy()  # Make a copy to avoid modifying the original
            # Check if delays and apodizations match the number of elements in the loaded solution
            delays_arr = np.array(solution["delays"]).reshape(-1)  # Ensure it's a 1D array
            apodizations_arr = np.array(solution["apodizations"]).reshape(-1)  # Ensure it's a 1D array
            if validate:
                if delays_arr.ndim == 1:
                    n_delays = delays_arr.shape[0]
                else:
                    n_delays = delays_arr.shape[1]
                if n_delays != self.num_modules * NUM_ELEMENTS_PER_MODULE:
                    self.logger.error(
                        f"Loaded solution has {len(delays_arr)} delays, but expected {self.num_modules * NUM_ELEMENTS_PER_MODULE} for {self.num_modules} modules."
                    )
                    raise ValueError(
                        f"Loaded solution has {len(delays_arr)} delays, but expected {self.num_modules * NUM_ELEMENTS_PER_MODULE} for {self.num_modules} modules."
                    )
                if apodizations_arr.ndim == 1:
                    n_apodizations = apodizations_arr.shape[0]
                else:
                    n_apodizations = apodizations_arr.shape[1]
                if n_apodizations != self.num_modules * NUM_ELEMENTS_PER_MODULE:
                    self.logger.error(
                        f"Loaded solution has {len(apodizations_arr)} apodizations, but expected {self.num_modules * NUM_ELEMENTS_PER_MODULE} for {self.num_modules} modules."
                    )
                    raise ValueError(
                        f"Loaded solution has {len(apodizations_arr)} apodizations, but expected {self.num_modules * NUM_ELEMENTS_PER_MODULE} for {self.num_modules} modules."
                    )
            
            # Ensure pulse and sequence have correct types before returning
            if "pulse" in solution and isinstance(solution["pulse"], dict):
                solution["pulse"]["frequency"] = float(solution["pulse"].get("frequency", 0))
                solution["pulse"]["duration"] = float(solution["pulse"].get("duration", 0))
                solution["pulse"]["amplitude"] = float(solution["pulse"].get("amplitude", 1.0))
            if "sequence" in solution and isinstance(solution["sequence"], dict):
                solution["sequence"]["pulse_interval"] = float(solution["sequence"].get("pulse_interval", 0))
                solution["sequence"]["pulse_count"] = int(solution["sequence"].get("pulse_count", 1))
                solution["sequence"]["pulse_train_interval"] = float(solution["sequence"].get("pulse_train_interval", 0))
                solution["sequence"]["pulse_train_count"] = int(solution["sequence"].get("pulse_train_count", 1))
            if "voltage" in solution:
                solution["voltage"] = float(solution.get("voltage", 0))
            
            return solution
        
        # Calculate a new solution
        frequency_hz = float(freq) * 1e3
        duration_seconds = float(durationS) * 1e-6
        pulse_interval_seconds = float(pulseInterval) * 1e-3

        def load_element_positions_from_file(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            if "type" in data and data["type"] == "TransducerArray":
                modules = []
                for module in data['modules']:
                    module_transform = np.array(module['transform'])
                    element_positions = np.array([elem['position'] for elem in module['elements']])
                    element_positions = np.hstack((element_positions, np.ones((element_positions.shape[0], 1))))
                    world_positions = (np.linalg.inv(module_transform) @ element_positions.T).T[:, :3]  # drop the homogeneous coordinate
                    modules.append(world_positions)
                element_positions = np.vstack(modules)
            else:
                element_positions = np.array([elem['position'] for elem in data['elements']])
            return element_positions

        focus = np.array([xInput, yInput, zInput])
        
        try:
            element_positions = load_element_positions_from_file(os.path.join(_base_path(), f"pinmap_{self.num_modules}x.json"))
            if not isinstance(element_positions, np.ndarray):
                raise TypeError(f"Expected numpy array from load_element_positions_from_file, got {type(element_positions)}")
                
            numelements = element_positions.shape[0]
            self.logger.info(f"{self.num_modules}x config file loaded with {numelements} elements")
            
            distances = np.sqrt(np.sum((focus - element_positions)**2, 1))
            tof = distances * 1e-3 / SPEED_OF_SOUND
            delays = tof.max() - tof
            apodizations = np.ones(numelements)
        except Exception as e:
            self.logger.error(f"Error calculating solution arrays: {type(e).__name__}: {e}")
            raise RuntimeError(f"Error calculating solution arrays: {e}")
        
        sequence = {
            "pulse_interval": float(pulse_interval_seconds),
            "pulse_count": int(pulseCount),
            "pulse_train_interval": float(trainInterval),
            "pulse_train_count": int(trainCount)
        }
        transducer_dummy = {"elements": [{"position": pos.tolist()} for pos in element_positions]}
        
        # Ensure pulse dict has proper types (matching original implementation)
        pulse_dict = {
            "frequency": float(frequency_hz),
            "duration": float(duration_seconds),
            "amplitude": float(1.0)
        }
        
        solution = {
            "id": "solution",
            "name": "Solution",
            "delays": delays.tolist() if isinstance(delays, np.ndarray) else delays,
            "apodizations": apodizations.tolist() if isinstance(apodizations, np.ndarray) else apodizations,
            "pulse": pulse_dict,
            "sequence": sequence,
            "voltage": float(voltage),
            "transducer": transducer_dummy
        }
        
        # Log solution structure right after creation (before any modifications)
        self.logger.debug(f"get_solution() created solution with keys: {solution.keys()}")
        self.logger.debug(f"  pulse type: {type(solution['pulse'])}, value: {solution['pulse']}")
        self.logger.debug(f"  sequence type: {type(solution['sequence'])}, value: {solution['sequence']}")
        
        return solution

    def load_solution_from_file(self, file_path: str) -> bool:
        """Load a solution from a JSON file and store it in the instance.
        
        Args:
            file_path: The path to the solution JSON file
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            self.logger.info(f"Attempting to load solution from: {file_path}")
            
            # Normalize the path for the current OS
            normalized_path = os.path.normpath(file_path)
            self.logger.info(f"Normalized path: {normalized_path}")
            
            # Validate file exists and is readable
            if not os.path.exists(normalized_path):
                error_msg = f"File not found: {normalized_path}"
                self.logger.error(error_msg)
                return False
                
            if not os.path.isfile(normalized_path):
                error_msg = f"Path is not a file: {normalized_path}"
                self.logger.error(error_msg)
                return False
                
            with open(normalized_path, 'r', encoding='utf-8') as f:
                solution_data = json.load(f)
                
            self.logger.info(f"Successfully parsed JSON from {normalized_path}")
            self.logger.info(f"JSON data type: {type(solution_data)}")
            if isinstance(solution_data, dict):
                self.logger.info(f"JSON keys: {list(solution_data.keys())}")
            else:
                self.logger.warning(f"Unexpected JSON data type: {type(solution_data)}, value: {str(solution_data)[:100]}")
            
            # Validate solution structure
            if not self._validate_solution_format(solution_data):
                return False
                
            # If modules are known, verify element count matches
            if self.num_modules is not None:
                expected_elements = self.num_modules * NUM_ELEMENTS_PER_MODULE
                actual_elements = len(solution_data.get('transducer', {}).get('elements', []))
                
                if expected_elements != actual_elements:
                    error_msg = f"Element count mismatch! Expected: {expected_elements} elements ({self.num_modules} modules × {NUM_ELEMENTS_PER_MODULE}), Found in solution: {actual_elements} elements"
                    self.logger.error(error_msg)
                    return False
            
            # Store loaded solution data
            self._loaded_solution_data = solution_data
            self._solution_loaded = True
            self._solution_name = solution_data.get('name', 'Unnamed Solution')
            
            # Log success
            if "name" in solution_data:
                message = f"Loaded solution '{solution_data['name']}' from file"
            else:
                message = f"Loaded solution with {len(solution_data.get('transducer', {}).get('elements', []))} elements"
            self.logger.info(message)
            self.logger.info(f"Successfully loaded solution: {self._solution_name}")
            return True
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            self.logger.error(error_msg)
            return False
        except PermissionError as e:
            error_msg = f"Permission denied accessing file: {str(e)}"
            self.logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"Error loading solution: {str(e)}"
            self.logger.error(f"Error loading solution from {file_path}: {e}")
            return False

    def _validate_solution_format(self, solution_data) -> bool:
        """Validate that the solution file has the required structure.
        
        Args:
            solution_data: The parsed JSON solution data
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # First check if solution_data is actually a dict
            if not isinstance(solution_data, dict):
                self.logger.error(f"Invalid solution format: expected JSON object, got {type(solution_data).__name__}")
                return False
            
            self.logger.info(f"Validating solution with keys: {list(solution_data.keys())}")
            
            # Check for required top-level fields
            required_fields = ['transducer', 'pulse', 'sequence']
            for field in required_fields:
                if field not in solution_data:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate transducer structure
            transducer = solution_data['transducer']
            if not isinstance(transducer, dict):
                self.logger.error("Transducer field must be an object")
                return False
                
            if 'elements' not in transducer:
                self.logger.error("Missing 'elements' in transducer data")
                return False
                
            if not isinstance(transducer['elements'], list):
                self.logger.error("Transducer elements must be a list")
                return False
                
            # Validate pulse structure
            pulse = solution_data['pulse']
            if not isinstance(pulse, dict):
                self.logger.error("Pulse field must be an object")
                return False
                
            pulse_fields = ['frequency', 'duration']
            for field in pulse_fields:
                if field not in pulse:
                    self.logger.error(f"Missing pulse field: {field}")
                    return False
            
            # Validate sequence structure
            sequence = solution_data['sequence']
            if not isinstance(sequence, dict):
                self.logger.error("Sequence field must be an object")
                return False
                
            sequence_fields = ['pulse_interval', 'pulse_count']
            for field in sequence_fields:
                if field not in sequence:
                    self.logger.error(f"Missing sequence field: {field}")
                    return False
                    
            self.logger.info("Solution validation passed")
            return True
            
        except Exception as e:
            error_msg = f"Error validating solution format: {str(e)}"
            self.logger.error(error_msg)
            return False

    def configure_solution(self) -> None:
        """Configure the beamforming solution and load it into the device."""
        if self.interface is None:
            raise RuntimeError("Interface not connected.")

        # Validate required parameters
        if self.frequency_khz is None:
            raise RuntimeError("Frequency not set. Call _select_frequency() first.")
        if self.voltage is None:
            raise RuntimeError("Voltage not set. Call test case selection first.")
        if self.interval_msec is None:
            raise RuntimeError("Pulse interval not set. Call test case selection first.")
        if self.duration_msec is None:
            raise RuntimeError("Duration not set. Call test case selection first.")

        solution = self.get_solution(
            default_coordinates["x"], 
            default_coordinates["y"], 
            default_coordinates["z"], 
            freq=self.frequency_khz, 
            voltage=self.voltage, 
            pulseInterval=self.interval_msec, 
            pulseCount=1, 
            trainInterval=0, 
            trainCount=0, 
            durationS=self.duration_msec * 1000
        )        

        # Validate solution dict structure before passing to SDK
        if not isinstance(solution, dict):
            raise TypeError(f"get_solution() should return a dict, got {type(solution)}")
        if "pulse" not in solution or not isinstance(solution["pulse"], dict):
            raise ValueError(f"Solution missing 'pulse' dict. Got: {solution.get('pulse', 'MISSING')}")
        if "sequence" not in solution or not isinstance(solution["sequence"], dict):
            raise ValueError(f"Solution missing 'sequence' dict. Got: {solution.get('sequence', 'MISSING')}")

        # Coerce all numeric values to proper types before passing to SDK
        solution['pulse']['frequency'] = float(solution['pulse'].get('frequency', 0))
        solution['pulse']['duration'] = float(solution['pulse'].get('duration', 0))
        solution['pulse']['amplitude'] = float(solution['pulse'].get('amplitude', 1.0))
        solution['sequence']['pulse_interval'] = float(solution['sequence'].get('pulse_interval', 0))
        solution['sequence']['pulse_count'] = int(solution['sequence'].get('pulse_count', 1))
        solution['sequence']['pulse_train_interval'] = float(solution['sequence'].get('pulse_train_interval', 0))
        solution['sequence']['pulse_train_count'] = int(solution['sequence'].get('pulse_train_count', 1))
        solution['voltage'] = float(solution.get('voltage', 0))
        
        # Final verification before passing to SDK
        self.logger.info("Final solution values before SDK:")
        self.logger.info(f"  pulse['duration']: {solution['pulse']['duration']!r} (type={type(solution['pulse']['duration']).__name__})")
        self.logger.info(f"  sequence['pulse_interval']: {solution['sequence']['pulse_interval']!r} (type={type(solution['sequence']['pulse_interval']).__name__})")
        self.logger.info(f"  sequence['pulse_train_interval']: {solution['sequence']['pulse_train_interval']!r} (type={type(solution['sequence']['pulse_train_interval']).__name__})")
        
        # Test the calculation that SDK will do
        try:
            if solution['sequence']['pulse_train_interval'] == 0:
                test_duty_cycle = solution['pulse']['duration'] / solution['sequence']['pulse_interval']
            else:
                test_duty_cycle = (solution['pulse']['duration'] * solution['sequence']['pulse_count']) / solution['sequence']['pulse_train_interval']
            self.logger.info(f"  Calculated duty cycle: {test_duty_cycle!r} (type={type(test_duty_cycle).__name__})")
        except Exception as e:
            self.logger.error(f"  Error calculating duty cycle: {e}")

        # Check for any unexpected dict values in solution
        for key, value in solution.items():
            if isinstance(value, dict):
                if key not in ('pulse', 'sequence', 'transducer'):
                    self.logger.warning(f"  Unexpected dict found at solution['{key}']: {value}")
                else:
                    for k2, v2 in value.items():
                        if isinstance(v2, dict) and key in ('pulse', 'sequence'):
                            self.logger.warning(f"  Nested unexpected dict found at solution['{key}']['{k2}']: {v2}")

        trigger_mode = "Continuous"

        def _set_solution() -> None:
            self.interface.set_solution(
                solution=solution,
                trigger_mode=trigger_mode,
            )

        try:
            self._retry_operation("set_solution", _set_solution, final_log_level="error")
        except Exception as e:
            self.logger.error(f"Error calling set_solution(): {type(e).__name__}: {e}")
            import traceback
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            # Try to serialize solution to JSON to see what's actually being passed
            try:
                import json
                solution_json = json.dumps(solution, default=str)
                self.logger.error(f"Solution JSON: {solution_json}")
            except Exception as json_err:
                self.logger.error(f"Could not serialize solution to JSON: {json_err}")
                self.logger.error(f"Solution dict keys: {solution.keys()}")
                for key in solution:
                    try:
                        self.logger.error(f"  {key}: {solution[key]!r}")
                    except Exception as item_err:
                        self.logger.error(f"  {key}: <error getting value: {item_err}>")
            raise

        self.logger.info("Solution configured for Test Case %s.", self.test_case_num)

    def is_solution_loaded(self) -> bool:
        """Check if a solution is currently loaded.
        
        Returns:
            bool: True if a solution is loaded, False otherwise
        """
        return self._solution_loaded

    def get_loaded_solution_name(self) -> str:
        """Get the name of the currently loaded solution.
        
        Returns:
            str: The solution name, or empty string if no solution is loaded
        """
        return self._solution_name

    def unload_solution(self) -> None:
        """Unload the currently loaded solution and reset to generation mode."""
        self._solution_loaded = False
        self._loaded_solution_data = None
        self._solution_name = ""
        self.logger.info("Solution unloaded. Will generate solutions based on parameters.")

    def _retry_operation(
        self,
        operation_name: str,
        func,
        *,
        attempts: int = MAX_OPERATION_RETRIES,
        retry_delay_s: float = RETRY_DELAY_SECONDS,
        final_log_level: str = "error",
    ):
        """Run an operation with bounded retries and a shutdown-aware delay."""
        last_exception = None

        for attempt in range(1, attempts + 1):
            if self.shutdown_event.is_set():
                raise AbortTest()

            try:
                return func()
            except AbortTest:
                raise
            except Exception as exc:
                last_exception = exc
                log_method = self.logger.warning if attempt < attempts else getattr(self.logger, final_log_level, self.logger.error)
                log_method(
                    "%s failed on attempt %d/%d: %s: %s",
                    operation_name,
                    attempt,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
                if attempt < attempts and self.shutdown_event.wait(retry_delay_s):
                    raise AbortTest()

        if last_exception is not None:
            raise last_exception

    # def test_console_voltage_accuracy_no_load(self) -> None:
    #     """Test console voltage accuracy under no-load conditions."""
    #     if self.interface is None:
    #         self.logger.error("Interface not connected for voltage accuracy test.")
    #         return

    #     for test_case in range(0, len(TEST_CASES)//2):  # Run for 2 test cases as a quick check
    #         self.logger.info(f"Starting console voltage accuracy test for Test Case {test_case} with no load for {VOLTAGE_ACCURACY_NO_LOAD_TEST_DURATION_SECONDS} seconds.")
    #         try:
    #             self.interface.hvcontroller.set_voltage(self.voltage)
    #             self.interface.hvcontroller.turn_hv_on()
    #         except SerialException as e:
    #             self.logger.error("SerialException encountered while reading console voltage: %s", e)
    #         except Exception as e:
    #             self.logger.error("Unexpected error while reading console voltage: %s", e)


    def monitor_console_voltage(self) -> None:
        """Thread target: monitor console voltage."""
        if self.hw_simulate:
            self.logger.info("Console voltage monitoring skipped in hardware simulation mode.")
            while not self.shutdown_event.is_set():
                time.sleep(0.5)
            return
        
        if self.interface is None:
            self.logger.error("Interface is not initialized in monitor_console_voltage.")
            return

        # serial_failures = 0
        start_time = time.time()
        last_log_time = 0.0

        deviation_limit_percentage = self.voltage * self.voltage_deviation_percentage_limit / 100
        deviation_limit_absolute_value = VOLTAGE_DEVIATION_ABSOLUTE_VALUE_LIMIT
        deviation_limit_v = max(deviation_limit_percentage, deviation_limit_absolute_value)
        
        self.logger.info("  Voltage Deviation Limits: %.2f V (max of %.2f%% of %.2fV (%.2fV) or %.1f V) from expected %.2f V.",
                            deviation_limit_v,
                            self.voltage_deviation_percentage_limit,
                            self.voltage,
                            deviation_limit_percentage,
                            deviation_limit_absolute_value,
                            self.voltage,
                        )

        while not self.shutdown_event.is_set():

            time_elapsed = time.time() - start_time

            # Read temperatures
            try:
                if not self.use_external_power:
                    with self.mutex:
                        if self.shutdown_event.is_set():
                            return
                        console_voltage = self._read_with_retry(reading="voltage")

                        deviation_limit_percentage = self.voltage * self.voltage_deviation_percentage_limit / 100
                        deviation_limit_absolute_value = VOLTAGE_DEVIATION_ABSOLUTE_VALUE_LIMIT
                        deviation_limit_v = max(deviation_limit_percentage, deviation_limit_absolute_value)

                        delta_v = abs(console_voltage - self.voltage)
                        deviation_pct = delta_v / self.voltage * 100

                        if self.test_results[self.test_case_num].max_voltage_deviation_absolute is None or delta_v > self.test_results[self.test_case_num].max_voltage_deviation_absolute:
                            self.test_results[self.test_case_num].max_voltage_deviation_absolute = delta_v
                        if self.test_results[self.test_case_num].max_voltage_deviation_percentage is None or deviation_pct > self.test_results[self.test_case_num].max_voltage_deviation_percentage:
                            self.test_results[self.test_case_num].max_voltage_deviation_percentage = deviation_pct
                        # self.test_results[self.test_case_num].test_time_elapsed = time_elapsed
                        
                        if delta_v > deviation_limit_v:
                            self.logger.warning(
                                "Console voltage %.2f V deviates %.2f%% (%.2f V), exceeding limit %.2f V "
                                "(max of %.2f%% or %.1f V) from expected %.2f V.",
                                console_voltage,
                                deviation_pct,
                                delta_v,
                                deviation_limit_v,
                                self.voltage_deviation_percentage_limit,
                                deviation_limit_absolute_value,
                                self.voltage,
                            )
                            # break #uncomment when ADC error resolved

            except SerialException as e:
                self.logger.error("SerialException encountered while reading console voltage: %s", e)
                break
            except Exception as e:
                self.logger.error("Unexpected error while reading console voltage: %s", e)
                break

            # Periodic logging
            time_since_last_log = time_elapsed - last_log_time
            if time_since_last_log >= self.temperature_log_interval:
                last_log_time = time_elapsed
                if not self.use_external_power and console_voltage is not None:
                    self.logger.info(
                        "  Console Voltage: %6.2f V, Console Voltage Absolute Deviation: %3.2f V Percent Deviation: %2.2f%% ", 
                        console_voltage,
                        delta_v,
                        deviation_pct,
                    )

            time.sleep(self.temperature_check_interval)

        self.logger.info("Console voltage monitoring shutdown triggered.")
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()
            self.voltage_shutdown_event.set()

    def monitor_temperature(self) -> None:
        """Thread target: monitor temperatures and trigger shutdown on safety violations."""
        if self.hw_simulate:
            self.logger.info("Temperature monitoring skipped in hardware simulation mode.")
            while not self.shutdown_event.is_set():
                time.sleep(0.5)
            return
        
        if self.interface is None:
            self.logger.error("Interface is not initialized in monitor_temperature.")
            return

        # serial_failures = 0
        start_time = time.time()
        last_log_time = 0.0

        prev_tx_temp = None
        prev_amb_temp = None
        prev_con_temp = None

        tx_temp = None
        # time_elapsed = 0.0

        def _fmt_temp(value):
            return "N/A" if value is None else f"{value:.2f}C"

        while not self.shutdown_event.is_set():
            time_elapsed = time.time() - start_time

            # Read temperatures
            try:
                if not self.use_external_power:
                    if prev_con_temp is None:
                        with self.mutex:
                            if self.shutdown_event.is_set():
                                return
                            prev_con_temp = self._read_with_retry(reading="console")
                    with self.mutex:
                        if self.shutdown_event.is_set():
                            return
                        con_temp = self._read_with_retry(reading="console")
                else:
                    con_temp = None

                if prev_tx_temp is None:
                    with self.mutex:
                        if self.shutdown_event.is_set():
                            return
                        prev_tx_temp = self._read_with_retry(reading="tx")
                with self.mutex:
                    if self.shutdown_event.is_set():
                        return
                    tx_temp = self._read_with_retry(reading="tx")
                    self.test_results[self.test_case_num].final_temperature = tx_temp

                if prev_amb_temp is None:
                    with self.mutex:
                        if self.shutdown_event.is_set():
                            return
                        prev_amb_temp = self._read_with_retry(reading="tx_ambient")
                with self.mutex:  
                    if self.shutdown_event.is_set():
                        return  
                    amb_temp = self._read_with_retry(reading="tx_ambient")

            except SerialException as e:
                self.logger.error("SerialException encountered while reading temperatures: %s", e)
                break
            except Exception as e:
                self.logger.error("Unexpected error while reading temperatures: %s", e)
                break

            # Periodic logging
            time_since_last_log = time_elapsed - last_log_time
            if time_since_last_log >= self.temperature_log_interval:
                last_log_time = time_elapsed
                if not self.use_external_power and con_temp is not None:
                    self.logger.info(
                        "  Console Temp: %s, TX Temp: %s, Ambient Temp: %s",
                        _fmt_temp(con_temp),
                        _fmt_temp(tx_temp),
                        _fmt_temp(amb_temp),
                    )
                else:
                    self.logger.info("  TX Temp: %s, Ambient Temp: %s", _fmt_temp(tx_temp), _fmt_temp(amb_temp))

            # Absolute temperature thresholds
            if (not self.use_external_power and con_temp is not None and
                    con_temp > self.console_shutoff_temp_C):
                self.logger.warning(
                    "Console temperature %.2f°C exceeds shutoff threshold %.2f°C.",
                    con_temp,
                    self.console_shutoff_temp_C,
                )
                break

            if tx_temp is not None and tx_temp > self.tx_shutoff_temp_C:
                self.logger.warning(
                    "TX device temperature %.2f°C exceeds shutoff threshold %.2f°C.",
                    tx_temp,
                    self.tx_shutoff_temp_C,
                )
                break

            if amb_temp is not None and amb_temp > self.ambient_shutoff_temp_C:
                self.logger.warning(
                    "Ambient temperature %.2f°C exceeds shutoff threshold %.2f°C.",
                    amb_temp,
                    self.ambient_shutoff_temp_C,
                )
                break

            time.sleep(self.temperature_check_interval)
        
        self.logger.info("Temperature monitoring shutdown triggered.")
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()
            self.temperature_shutdown_event.set()

    def _read_with_retry(self, reading: str = "", max_attempts: int = 5, retry_delay_s: float = 2.0) -> float | None:
        """Attempt to read a temperature value, retrying on transient None returns or read failures.

        Args:
            reading: A string indicating which reading to read ("tx", "tx_ambient", "console", or "voltage") for logging purposes.
            max_attempts: Maximum number of read attempts before giving up.
            retry_delay_s: Delay in seconds between retry attempts.

        Returns:
            The reading value (float) on success, or None if all attempts fail.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                if reading == "tx":
                    value = self.interface.txdevice.get_temperature()
                elif reading == "tx_ambient":
                    value = self.interface.txdevice.get_ambient_temperature()
                elif reading == "console":
                    value = self.interface.hvcontroller.get_temperature1()
                elif reading == "voltage":
                    value = self.interface.hvcontroller.get_voltage()
                else:
                    self.logger.error(f"Invalid type specified for retry: '{reading}'")
                    return None
            except Exception as e:
                if attempt < max_attempts:
                    self.logger.warning(
                        "%s read failed (attempt %d/%d): %s. Retrying in %.0fs...",
                        reading or "Reading",
                        attempt,
                        max_attempts,
                        e,
                        retry_delay_s,
                    )
                    if self.shutdown_event.wait(retry_delay_s):
                        return None
                    continue

                self.logger.error(
                    "%s read failed after %d attempts: %s",
                    reading or "Reading",
                    max_attempts,
                    e,
                )
                return None

            if value is not None:
                return value
            if attempt < max_attempts:
                self.logger.warning(
                    "%s read returned None (attempt %d/%d). Retrying in %.0fs...",
                    reading or "Reading", attempt, max_attempts, retry_delay_s,
                )
                # Honour a shutdown request while waiting between retries
                if self.shutdown_event.wait(retry_delay_s):
                    return None
        self.logger.error(
            "%s read returned None after %d attempts - possible connection issue.",
            reading or "Reading",
            max_attempts,
        )
        return None

    def _verify_start_conditions(self, test_case, starting_temperature) -> None:
        """Monitor cooldown period before starting the test."""
        temp = self._read_with_retry(reading="tx")  # Initial read with retry for transient failures
        self.logger.info(f"Initial TX temperature: {temp}C")

        if self.test_runthrough:
            starting_temperature = TEST_RUNTHROUGH_TEMPERATURE_BYPASS_C
            self.logger.info(f"Test runthrough mode enabled - starting temp set to {starting_temperature}C.")

        counter = 0
        while temp is None or temp > starting_temperature:
            self.is_in_cooldown = True
            self.cooldown_start_time = time.time()
            if temp is None:
                self.logger.warning(
                    "TX temperature read returned None after retries before test case %s. Cooling/retrying in %s minutes.",
                    test_case,
                    TIME_BETWEEN_TESTS_TEMPERATURE_CHECK_SECONDS // 60,
                )
            else:
                self.logger.info(f"Current temperature of {temp}C is greater than max starting "
                                 f"temperature of {starting_temperature}C for test case {test_case}. "
                                 f"Transmitter will turn off for {TIME_BETWEEN_TESTS_TEMPERATURE_CHECK_SECONDS // 60} minutes to cool down and then check again.")
            self.turn_off_console_and_tx()
            self.cleanup_interface()

            if self.shutdown_event.wait(TIME_BETWEEN_TESTS_TEMPERATURE_CHECK_SECONDS):
                self.logger.info("Shutdown event set during cooldown wait. Exiting.")
                self.is_in_cooldown = False
                raise AbortTest()
            
            # time.sleep(TIME_BETWEEN_TESTS_TEMPERATURE_CHECK_SECONDS)  # Wait before rechecking
            
            self.connect_device()
            self.verify_communication()
            temp = self._read_with_retry(reading="tx")  # Retry after reconnect too
            counter += 1
        
        self.is_in_cooldown = False

        if counter > 0:
            self.logger.info(f"TX module took ~{counter * TIME_BETWEEN_TESTS_TEMPERATURE_CHECK_SECONDS // 60} minutes to cool down to starting temperature of {starting_temperature}C.")
            self.test_results[test_case].cooldown_time_elapsed = counter * TIME_BETWEEN_TESTS_TEMPERATURE_CHECK_SECONDS // 60

        self.logger.info(f"TX temperature of {temp}C is within the allowed starting temperature of {starting_temperature}C. Proceeding with test case {test_case}.")
        self.test_results[test_case].starting_temperature = temp 

    def exit_on_time_complete(self) -> None:
        """Thread target: stop test when total test time is reached."""
        if self.shutdown_event.wait(self.sequence_duration):
            return
        self.logger.info("Sequence complete: %s reached.", format_hhmmss(self.sequence_duration))
        self.sequence_complete_event.set()
        self.shutdown_event.set()
        # start = time.time()

        # while True:
        #     if self.shutdown_event.is_set():
        #         return

        #     time.sleep(.1)
        #     elapsed_time = time.time() - start
        #     # time_since_last_log = elapsed_time - last_log_time

        #     if elapsed_time >= self.sequence_duration:
        #         self.logger.info(
        #             "  Sequence complete: %s reached.",
        #             format_hhmmss(self.sequence_duration),
        #         )
        #         self.sequence_complete_event.set()
        #         self.shutdown_event.set()
        #         return

    # ------------------------------------------------------------------ #
    # Hardware shutdown & cleanup
    # ------------------------------------------------------------------ #

    def turn_off_console_and_tx(self) -> None:
        """Safely turn off HV and 12V if console is used."""
        if self.interface is None:
            return
        if self.use_external_power:
            return

        try:
            self.logger.info("Turning off HV and 12V...")
            with contextlib.suppress(Exception):
                self.interface.hvcontroller.turn_hv_off()
            with contextlib.suppress(Exception):
                self.interface.hvcontroller.turn_12v_off()
            self.logger.info("HV and 12V turned off.")
        except Exception as e:
            self.logger.warning("Error turning off HV/12V: %s", e)

    def cleanup_interface(self) -> None:
        if self.interface is None:
            return
        try:
            self.logger.info("Closing device interface...")
            # with contextlib.suppress(Exception):
            #     self.interface.stop_monitoring()
            # time.sleep(0.2)
            # Close the serial ports so Windows releases the handles
            # with contextlib.suppress(Exception):
            #     self.interface.txdevice.uart.disconnect()
            # with contextlib.suppress(Exception):
            #     if self.interface.hvcontroller:
            #         self.interface.hvcontroller.uart.disconnect()
        except Exception as e:
            self.logger.warning("Issue closing LIFU interface: %s", e)

    def print_banner(self) -> None:
        self.logger.info("Selected frequency: %dkHz", self.frequency_khz)
        self.logger.info("Number of modules: %d", self.num_modules)

        self.logger.info("--------------------------------------------------------------------------------")
        self.logger.info(
            "\n\nThis script will automatically cycle through all of the following test cases:\n\n"
            + "\n".join(
                f"Test Case {i:>2}: "
                f"{tc['voltage']:>3}V, "
                f"{tc['duty_cycle']:>3}% Duty Cycle, "
                f"{tc['PRI_ms']:>4}ms PRI, "
                f"Max Starting Temperature: {tc['max_starting_temperature']:>3}C"
                for i, tc in enumerate(TEST_CASES[self.starting_test_case-1:], start=self.starting_test_case)
            )
            + "\n\nThe script will account for cooldown periods as needed between test cases. \n" \
            f"Each test case will run for {self.sequence_duration/60:.2f} minutes. \n"
            f"The lower voltage tests starting at {LOW_VOLTAGE_VALUE}V and below will run for {LOW_VOLTAGE_VALUE_TEST_DURATION_SECONDS} seconds. \n"
            "Approximate test duration is 24hrs.\n"
        )
        self.logger.info("--------------------------------------------------------------------------------\n\n\n")

    def print_test_summary(self) -> None:
        self.logger.info("--------------------------------------------------------------------------------")
        self.logger.info("\n\nTest Case Summary:\n")

        for test_case, test_case_parameters in enumerate(TEST_CASES[self.starting_test_case - 1:], start=self.starting_test_case):
            r = self.test_results.get(test_case)
            act_start  = f"{r.starting_temperature:.2f}C" if r and r.starting_temperature is not None else " -   "
            final      = f"{r.final_temperature:.2f}C" if r and r.final_temperature is not None else " - "
            max_dv     = f"{r.max_voltage_deviation_absolute:.2f}V" if r and r.max_voltage_deviation_absolute is not None else " - "
            max_dv_pct = f"{r.max_voltage_deviation_percentage:.2f}%" if r and r.max_voltage_deviation_percentage is not None else " - "
            cooldown   = f"~{r.cooldown_time_elapsed}m" if r and r.cooldown_time_elapsed is not None else " - "
            dur        = format_hhmmss(r.test_time_elapsed) if r and r.test_time_elapsed is not None else " - "
            status     = r.status if r and getattr(r, "status", None) else "NOT RUN"
            
            
            self.logger.info(
                f"Test Case {test_case:>2}: "
                f"{test_case_parameters['voltage']:>2}V, "
                f"{test_case_parameters['duty_cycle']:>2}% DC, "
                f"{test_case_parameters['PRI_ms']:>2}ms PRI, "
                f"Max Start Temp: {test_case_parameters['max_starting_temperature']:>2}C, "
                f"Cooldown Time: {cooldown:>5}, "
                f"Actual Start Temp: {act_start:>6}, "
                f"Final Temp: {final:>6}, "
                f"Max Allowed Voltage Deviation: {VOLTAGE_DEVIATION_ABSOLUTE_VALUE_LIMIT:>3}V ({self.voltage_deviation_percentage_limit:>3}%), "
                f"Actual Voltage Deviation: {max_dv:>5} ({max_dv_pct:>5}) "
                f"Duration Run: {dur:>5}  --> "
                f"{status}" + ("\n" if test_case == len(TEST_CASES) / 2 else "")
            )

        passed_count = sum(1 for r in self.test_results.values() if getattr(r, 'status', None) == "PASSED")

        self.logger.info(
            f"{passed_count} out of {len(TEST_CASES)-self.starting_test_case+1} test cases passed."
        )

        self.logger.info(f"Script ran for a total of {format_duration(time.time() - self.start_time)}.")

        self.logger.info(
            "\n\nOVERALL RESULT: %s\n",
            "PASSED" if passed_count == len(TEST_CASES)-self.starting_test_case+1 else "FAILED",
        )

    def run(self) -> None:
        """Execute the thermal stress test with graceful shutdown."""
        self.test_status = "starting pre-test checks"

        try:
            self._select_num_modules()
            self._select_frequency()
            self._select_starting_test_case()
            self._attach_file_handler()
            self.print_banner()
        except Exception as e:
            self.logger.error("Error during initial selection: %s", e)
            sys.exit(1)

        self.logger.info("Starting automated test sequence from test case %d out of %d total test cases. " % (self.starting_test_case, len(TEST_CASES)))
        self.start_time = time.time()

        try:
            for test_case, test_case_parameters in enumerate(TEST_CASES[self.starting_test_case-1:], start=self.starting_test_case):
                if self.test_status in ("aborted by user", "error"):
                    self.logger.warning("Previous test case ended with status '%s'. Aborting remaining test cases.", self.test_status)
                    break
                self.test_case_num = test_case
                self.test_results[self.test_case_num] = TestCaseResult()
                self.voltage = float(test_case_parameters["voltage"])
                self.interval_msec = int(test_case_parameters["PRI_ms"])
                self.duration_msec = int(test_case_parameters["duty_cycle"] / 100 * self.interval_msec)
                
                self.logger.info(f"Starting test case {self.test_case_num} out of {len(TEST_CASES)}")
                self.logger.info("Test Case %d: %dV, %d%% Duty Cycle, %dms duration, %dms PRI, Max Starting Temperature: %dC",
                                self.test_case_num, 
                                self.voltage, 
                                test_case_parameters["duty_cycle"], 
                                self.duration_msec, 
                                self.interval_msec, 
                                test_case_parameters["max_starting_temperature"])

                self.test_case_start_time = 0.0

                try:
                    self.shutdown_event.clear()
                    self.sequence_complete_event.clear()
                    self.temperature_shutdown_event.clear()
                    self.voltage_shutdown_event.clear()
                
                    if self.shutdown_event.is_set():
                        self.test_status = "aborted by user"
                        raise AbortTest()

                    if not self.hw_simulate:
                        self.connect_device()
                        self.verify_communication()

                        # if test has already run at least once, skip
                        if test_case == self.starting_test_case: 
                            self.get_firmware_versions()
                            self.enumerate_devices()

                        self._verify_start_conditions(test_case, test_case_parameters["max_starting_temperature"])
                    else:
                        self.logger.info("Hardware simulation enabled; skipping device configuration.")

                    if self.test_runthrough:
                        self.sequence_duration = SHORT_TEST_DURATION_SECONDS
                    elif self.voltage is not None and self.voltage <= LOW_VOLTAGE_VALUE:
                        self.sequence_duration = LOW_VOLTAGE_VALUE_TEST_DURATION_SECONDS
                    else:
                        self.sequence_duration = TEST_CASE_DURATION_SECONDS
                    
                    self.configure_solution()

                    if self.shutdown_event.is_set():
                        self.test_status = "aborted by user"
                        raise AbortTest()

                    # Start sonication
                    if not self.hw_simulate:
                        self.logger.info("Starting Trigger...")
                        if not self.interface.start_sonication():
                            self.logger.error("Failed to start trigger.")
                            self.test_status = "error"
                            return
                        self.test_case_start_time = time.time()
                    else:
                        self.logger.info("Simulated Trigger start... (no hardware)")

                    self.logger.info("Trigger Running... (Press CTRL-C to stop early)")
                    self.test_status = "running"

                    # Start monitoring threads
                    # self.shutdown_event.clear()
                    # self.sequence_complete_event.clear()
                    # self.temperature_shutdown_event.clear()
                    # self.voltage_shutdown_event.clear()

                    temp_thread = threading.Thread(
                        target=self.monitor_temperature,
                        name="TemperatureMonitorThread",
                        # daemon=True,
                    )
                    completion_thread = threading.Thread(
                        target=self.exit_on_time_complete,
                        name="SequenceCompletionThread",
                        # daemon=True,
                    )
                    voltage_thread = threading.Thread(
                        target=self.monitor_console_voltage,
                        name="ConsoleVoltageMonitorThread",
                        # daemon=True,
                    )
                    
                    voltage_thread.start()
                    temp_thread.start()
                    completion_thread.start()

                    # Wait for threads or user interrupt
                    try:
                        while not self.shutdown_event.is_set():
                            time.sleep(0.1)
                    except KeyboardInterrupt:
                        self.logger.warning("Test aborted by user KeyboardInterrupt.")
                        self.test_status = "aborted by user"
                        self.shutdown_event.set()
                        raise

                    # Ensure shutdown event set
                    if not self.shutdown_event.is_set():
                        self.logger.warning("A thread exited without setting shutdown event; forcing shutdown.")
                        self.shutdown_event.set()

                    # Stop sonication
                    if not self.hw_simulate and self.interface is not None:
                        try:
                            if self.interface.stop_sonication():
                                self.logger.info("Trigger stopped successfully.")
                            else:
                                self.logger.error("Failed to stop trigger.")
                        except Exception as e:
                            self.logger.error("Error stopping trigger: %s", e)

                    # Wait for threads to exit gracefully
                    temp_thread.join()
                    voltage_thread.join()
                    completion_thread.join()

                    # Determine final status
                    if self.test_status not in ("aborted by user", "error"):
                        if self.sequence_complete_event.is_set():
                            self.test_status = "passed"
                        elif self.temperature_shutdown_event.is_set():
                            self.test_status = "temperature shutdown"
                        elif self.voltage_shutdown_event.is_set():
                            self.test_status = "voltage deviation"
                        else:
                            self.test_status = "error"
                finally:
                    # Record test time
                    # self.test_results[self.test_case_num].test_time_elapsed = time.time() - self.test_case_start_time if self.test_case_start_time else 0
                    duration = time.time() - self.test_case_start_time if self.test_case_start_time else 0.0
                    self.test_results[self.test_case_num].test_time_elapsed = duration

                    # Power down and cleanup
                    if not self.hw_simulate:
                        with contextlib.suppress(Exception):
                            self.turn_off_console_and_tx()
                        self.cleanup_interface()

                    # Final status log
                    if self.test_status == "passed":
                        self.logger.info("TEST CASE %d PASSED.", self.test_case_num)
                        self.test_results[self.test_case_num].status = "PASSED"
                    elif self.test_status == "temperature shutdown":
                        self.logger.info("TEST CASE %d FAILED.", self.test_case_num)
                        self.test_results[self.test_case_num].status = "FAILED (temperature shutdown)"
                    elif self.test_status == "aborted by user":
                        self.logger.info("TEST CASE %d ABORTED by user.", self.test_case_num)
                        self.test_results[self.test_case_num].status = "ABORTED"
                    elif self.test_status == "voltage deviation":
                        self.logger.info("TEST CASE %d FAILED.", self.test_case_num)
                        self.test_results[self.test_case_num].status = "FAILED (voltage deviation)"
                    elif self.test_status == "error":
                        self.logger.info("TEST CASE %d FAILED due to error.", self.test_case_num)
                        self.test_results[self.test_case_num].status = "FAILED (error)"
                    elif self.test_status == "not started":
                        self.logger.info("TEST CASE %d NOT RUN.", self.test_case_num)
                        self.test_results[self.test_case_num].status = "NOT RUN"
                    else:
                        self.logger.info(
                            "TEST CASE %d FAILED due to unexpected error.",
                            self.test_case_num,
                        )
                        self.test_results[self.test_case_num] = "FAILED (unexpected error)"
                    
                    self.logger.info("TEST CASE %d ran for a total of %s.", self.test_case_num, format_duration(duration))

                    if self.test_status == "aborted by user":
                        self.logger.info("Aborting remaining test cases due to user interrupt.")
                        # self.print_test_summary()
                        # return
                    # self.test_results[self.test_case_num].cooldown_time_elapsed = 0.0
                    self.logger.info("Run log will be saved to: %s", self.log_file_path)
        except AbortTest:
            self.logger.warning("Test sequence aborted by user. Exiting remaining test cases.")
        finally:
            self.print_test_summary()    

@dataclass
class TestCaseResult:
    # self.test_case : int | None = None
    starting_temperature: float | None = None
    final_temperature: float | None = None
    max_voltage_deviation_absolute: float | None = None
    max_voltage_deviation_percentage: float | None = None
    test_time_elapsed: float | None = None
    cooldown_time_elapsed: int | None = None
    status: str | None = None

def frequency_khz(value: str) -> int:
    ivalue = int(value)
    if not 100 <= ivalue <= 500:
        raise argparse.ArgumentTypeError(
            "frequency (kHz) must be between 100 and 500 kHz"
        )
    return ivalue

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LIFU Thermal Stress Burn-in Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
This script interactively prompts for:
* TX frequency (kHz)
* Predefined burn-in test case (voltage, duty, duration)

Examples:
# Run with default settings
%(prog)s

# Run using external power supply
%(prog)s --external-power

# Run with more aggressive console shutoff
%(prog)s --console-shutoff-temp 65

# Run in quiet mode writing logs to ./logs
%(prog)s --quiet --log-dir ./logs

# Run without "Press ENTER to start" prompt
%(prog)s --no-prompt

# Run a specific test case and frequency non-interactively
    %(prog)s --frequency 400 --test-case 2 --no-prompt
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Power & behavior
    behavior_group = parser.add_argument_group("Power & Behavior")
    behavior_group.add_argument(
        "--external-power",
        action="store_true",
        help="Use external power supply instead of console 12V/HV.",
    )
    behavior_group.add_argument(
        "--simulate",
        action="store_true",
        help="Hardware simulation mode (no actual device I/O or HV changes).",
    )
    behavior_group.add_argument(
        "--test-runthrough",
        action="store_true",
        help="Run through all test cases with short duration for testing script functionality.",
    )
    behavior_group.add_argument(
        "--bypass-console-fw",
        action="store_true",
        help="Bypass console firmware version check.",
    )
    behavior_group.add_argument(
        "--bypass-tx-fw",
        action="store_true",
        help="Bypass TX firmware version check.",
    )
    behavior_group.add_argument(
        "--num-modules",
        type=int,
        default=None,
        choices=NUM_MODULES,
        metavar="N",
        help=f"Number of modules connected.",
    )
    behavior_group.add_argument(
        "--frequency",
        type=frequency_khz,
        default=None,
        metavar="KHZ",
        help="TX frequency in kHz (overrides interactive selection).",
    )
    behavior_group.add_argument(
        "--test-case",
        type=int,
        choices=range(1, len(TEST_CASES) + 1),
        default=None,
        metavar="N",
        help="Starting test case number (overrides interactive selection).",
    )

    # Safety thresholds
    safety_group = parser.add_argument_group("Safety Thresholds")
    safety_group.add_argument(
        "--console-shutoff-temp",
        type=float,
        default=CONSOLE_SHUTOFF_TEMP_C_DEFAULT,
        metavar="C",
        help=f"Console shutoff temperature in Celsius (default: {CONSOLE_SHUTOFF_TEMP_C_DEFAULT}).",
    )
    safety_group.add_argument(
        "--tx-shutoff-temp",
        type=float,
        default=TX_SHUTOFF_TEMP_C_DEFAULT,
        metavar="C",
        help=f"TX device shutoff temperature in Celsius (default: {TX_SHUTOFF_TEMP_C_DEFAULT}).",
    )
    safety_group.add_argument(
        "--ambient-shutoff-temp",
        type=float,
        default=AMBIENT_SHUTOFF_TEMP_C_DEFAULT,
        metavar="C",
        help=f"Ambient shutoff temperature in Celsius (default: {AMBIENT_SHUTOFF_TEMP_C_DEFAULT}).",
    )

    # Timing / logging
    timing_group = parser.add_argument_group("Timing & Logging")
    timing_group.add_argument(
        "--temperature-check-interval",
        type=float,
        default=TEMPERATURE_CHECK_INTERVAL_DEFAULT,
        metavar="S",
        help=(
            "Temperature check interval in seconds "
            f"(default: {TEMPERATURE_CHECK_INTERVAL_DEFAULT})."
        ),
    )
    timing_group.add_argument(
        "--temperature-log-interval",
        type=float,
        default=TEMPERATURE_LOG_INTERVAL_DEFAULT,
        metavar="S",
        help=(
            "Temperature log interval in seconds "
            f"(default: {TEMPERATURE_LOG_INTERVAL_DEFAULT})."
        ),
    )
    timing_group.add_argument(
        "--log-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for log files (default: <openlifu_root>/logs).",
    )

    # Verbosity
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level).",
    )
    verbosity_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational messages (WARNING level only).",
    )

    # Automation
    automation_group = parser.add_argument_group("Automation")
    automation_group.add_argument(
        "--skip-logfile",
        action="store_true",
        help="Do not attach file handler to logger (console only).",
    )

    return parser.parse_args()

# def main() -> None:
#     """Main entry point for the script."""
#     args = parse_arguments()
#     voltage_test = VoltageAccuracyTest(args)
#     temp_and_voltage_stability_test = TransmitterHeatingAndVoltageStabilityTest(args)

#     try:
#         voltage_test.run()
#         temp_and_voltage_stability_test.run()
#     except KeyboardInterrupt:
#         for test in (voltage_test, temp_and_voltage_stability_test):
#             test.logger.warning("Test aborted by user KeyboardInterrupt. Shutting down...")
#             test.shutdown_event.set()
#             test.stop_logging = True
#             time.sleep(0.5)
#             with contextlib.suppress(Exception):
#                 test.print_test_summary()
#                 test.turn_off_console_and_tx()
#             with contextlib.suppress(Exception):
#                 test.cleanup_interface()
#         sys.exit(0)
#     except Exception as e:
#         test.logger.error(f"\nFatal error: {e}")
#         for test in (voltage_test, temp_and_voltage_stability_test):
#             with contextlib.suppress(Exception):
#                 test.print_test_summary()
#             with contextlib.suppress(Exception):
#                 test.turn_off_console_and_tx()
#             with contextlib.suppress(Exception):
#                 test.cleanup_interface()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()


