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
import threading
import time
import inspect
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from serial.serialutil import SerialException

import numpy as np

import openlifu
from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.xdc import Transducer, TransducerArray
from openlifu.xdc.transducerarray import get_angle_from_gap
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

from config import *
from openlifu.xdc.transducer import Transducer

"""
Thermal Stress Test Script
- User selects a test case.
- Test runs for a fixed total duration or until a thermal shutdown occurs.
- Logs temperature and device status.
"""

__version__ = "1.0.3"
REQUIRED_CONSOLE_FW_VERSION = "v1.2.2"
REQUIRED_TX_FW_VERSION = "v2.0.3"

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


class TestSonicationDurationBase:
    """Main class for Thermal Stress Test 5."""

    def __init__(self, args):
        self.args = args

        # Derived paths
        self.openlifu_dir = Path(openlifu.__file__).parent.parent.parent.resolve()
        self.log_dir = Path(self.args.log_dir or (Path(__file__).resolve().parents[1] / "logs"))
        
        # Runtime attributes
        self.interface: LIFUInterface | None = None
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
        self.frequency_khz: float | None = None
        self.test_case_num: int | None = None
        self.voltage: float | None = None
        self.interval_msec: float | None = None
        self.duration_msec: int | None = None
        self.num_modules: int | None = None
        self.max_allowed_voltage_deviation_percentage: float | None = None

        self.test_status: str | None = "not started"
        self.sequence_duration: float = TEST_CASE_DURATION_SECONDS
        self.starting_test_case: int = 1
        self.test_results: dict[int, TestCaseResult] = {}

        # Flags from args
        self.use_external_power = self.args.external_power
        self.hw_simulate = self.args.simulate
        self.test_runthrough = self.args.test_runthrough
        self.bypass_console_fw = self.args.bypass_console_fw
        self.bypass_tx_fw = self.args.bypass_tx_fw

        # Safety parameters from args
        self.console_shutoff_temp_C = self.args.console_shutoff_temp
        self.tx_shutoff_temp_C = self.args.tx_shutoff_temp
        self.ambient_shutoff_temp_C = self.args.ambient_shutoff_temp
        self.temperature_check_interval = self.args.temperature_check_interval
        self.temperature_log_interval = self.args.temperature_log_interval

        # Logger
        self.logger = self._setup_logging()
        self._file_handler_attached = False
        self.test_id = Path(inspect.getfile(type(self))).name.replace(".py", "")

        self.logger.debug(f"{self.test_id} initialized with arguments: {self.args}")

    def _setup_logging(self) -> logging.Logger:
        """Configure root logger with console output; file handler added later."""
        logger = logging.getLogger(__name__)

        # Set log level based on verbosity
        if self.args.verbose:
            logger.setLevel(logging.DEBUG)
        elif self.args.quiet:
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
        if not self.args.skip_logfile:
            if self._file_handler_attached:
                return

            self.log_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{self.run_timestamp}_{self.test_id}_{self.frequency_khz}kHz_{self.num_modules}x.log"

            log_path = self.log_dir / filename

            formatter = SafeFormatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logging.getLogger(__name__).addHandler(file_handler)

            self._file_handler_attached = True
            self.logger.info("Run log will be saved to: %s", log_path)
        else:
            self.logger.info("User elected to skip logging to file - logs will only be output to console.")



    # ------------------- User Input Section ------------------- #

    def _select_frequency(self) -> None:
        """Select TX frequency in kHz (100–500), CLI override or interactive."""
        # CLI override
        if self.args.frequency is not None:
            self.frequency_khz = self.args.frequency
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
        if self.args.num_modules is not None:
            self.num_modules = self.args.num_modules
            return

        while True:
            choice = input(f"Select number of modules {list(NUM_MODULES)}: ")
            if choice.isdigit() and int(choice) in NUM_MODULES:
                self.num_modules = int(choice)
                break
            self.logger.info("Invalid selection. Please try again.")


    def _select_starting_test_case(self) -> None:
        valid_test_nums = list(range(1, len(TEST_CASES) + 1))

        # Test case selection
        if self.args.test_case is not None:
            self.starting_test_case = int(self.args.test_case)
        else:
            self.logger.info("\nAvailable Test Cases:")
            for test_id, test_case in enumerate(TEST_CASES, start=1):
                self.logger.info(
                    f"Test Case {test_id:>2}. {test_case['voltage']:>2}V, "
                    f"{test_case['duty_cycle']:>2}% Duty Cycle, total"
                )

            while True:
                choice = input(f"Press enter to run through all test cases or select a test case by number to start at: ")
                if choice == "":
                    break
                if choice.isdigit() and int(choice) in valid_test_nums:
                    self.starting_test_case = int(choice)
                    break
                self.logger.info("Invalid selection. Please try again.")

    def connect_device(self) -> None:
        """Connect to the LIFU device and verify connection."""
        self.logger.info("Starting test...")
        self.interface = LIFUInterface(
            ext_power_supply=self.use_external_power,
            TX_test_mode=self.hw_simulate,
            HV_test_mode=self.hw_simulate,
            voltage_table_selection="evt0",
            sequence_time_selection="stress_test"
        )
        tx_connected, hv_connected = self.interface.is_device_connected()
        if self.bypass_transmitter: tx_connected = True

        if not self.use_external_power and not tx_connected:
            self.logger.warning("TX device not connected. Attempting to turn on 12V...")
            try:
                self.interface.hvcontroller.turn_12v_on()
            except Exception as e:
                self.logger.error("Error turning on 12V: %s", e)
            time.sleep(2)

            # Reinitialize interface after powering 12V
            try:
                self.interface.stop_monitoring()
            except Exception as e:
                self.logger.warning("Error stopping monitoring during reinit: %s", e)

            with contextlib.suppress(Exception):
                del self.interface

            time.sleep(1)
            self.logger.info("Reinitializing LIFU interface after powering 12V...")
            self.interface = LIFUInterface(
                ext_power_supply=self.use_external_power,
                TX_test_mode=self.hw_simulate,
                HV_test_mode=self.hw_simulate,
                voltage_table_selection="evt0",
                sequence_time_selection="stress_test"
            )
            tx_connected, hv_connected = self.interface.is_device_connected()

        if not self.use_external_power:
            if hv_connected:
                self.logger.info("  HV Connected: %s", hv_connected)
            else:
                self.logger.error("HV NOT fully connected.")
                sys.exit(1)
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
            if not self.args.external_power and not self.interface.hvcontroller.ping():
                self.logger.error("Failed to ping the console device.")
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

    def get_firmware_versions(self) -> None:
        """Retrieve and log firmware versions from the LIFU device."""
        if self.interface is None:
            self.logger.error("Interface not connected for firmware version retrieval.")
            return

        console_fw_mismatch = False
        tx_fw_mismatch = False
        
        try:
            if not self.args.external_power:
                console_fw = self.interface.hvcontroller.get_version()
                self.logger.info("Console Firmware Version: %s", console_fw)
            if not self.args.bypass_console_fw and console_fw != REQUIRED_CONSOLE_FW_VERSION:
                self.logger.error("Console firmware version %s does not match required version %s.",
                                console_fw, REQUIRED_CONSOLE_FW_VERSION)
                console_fw_mismatch = True
        except Exception as e:
            self.logger.error("Error retrieving console firmware version: %s", e)

        if not self.bypass_transmitter:        
            try:
                for i in range(1, self.num_modules+1):
                    tx_fw = self.interface.txdevice.get_version(module=i)
                    self.logger.info("TX Device %d Firmware Version: %s", i, tx_fw)
                    if not self.args.bypass_tx_fw and tx_fw != REQUIRED_TX_FW_VERSION:
                        self.logger.error("TX firmware version %s does not match required version %s.",
                                        tx_fw, REQUIRED_TX_FW_VERSION)
                        tx_fw_mismatch = True
            except Exception as e:
                self.logger.error("Error retrieving TX device firmware version: %s", e)        

        if console_fw_mismatch or tx_fw_mismatch:
            if console_fw_mismatch:
                self.logger.error("\n\n!! Incompatible console firmware version, please upgrade to %s !!\n\n", REQUIRED_CONSOLE_FW_VERSION)
            if tx_fw_mismatch:
                self.logger.error("\n\n!! Incompatible TX firmware version, please upgrade to %s !!\n\n", REQUIRED_TX_FW_VERSION)
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

    def configure_solution(self) -> None:
        """Configure the beamforming solution and load it into the device."""
        if self.interface is None:
            raise RuntimeError("Interface not connected.")

        trans0_id = "openlifu_1x400_evt1"
        trans0 = Transducer.gen_matrix_array(nx=8, ny=8, pitch=5, kerf=0.3, id=trans0_id, name="OpenLIFU 1x 400kHz EVT1")

        if self.num_modules == 1:
            arr=trans0
        elif self.num_modules == 2:
            gap0 = 10
            roc0 = 100
            dth = get_angle_from_gap(40, gap0, roc0)
            gap = 12.9
            transducers= TransducerArray.get_concave_cylinder(trans=trans0, rows=1, cols=2, width=40, dth=dth, roc=None, gap=gap)
            arr = transducers.to_transducer()
        
        # Focus at (0, 0, 50 mm)
        x_input, y_input, z_input = 0, 0, 50
        target = Point(position=(x_input, y_input, z_input), units="mm")
        focus = target.get_position(units="mm")

        distances = np.sqrt(
            np.sum((focus - arr.get_positions(units="mm")) ** 2, axis=1)
        ).reshape(1, -1)
        tof = distances * 1e-3 / 1500  # mm to m, divide by 1500 m/s
        delays = tof.max() - tof
        apodizations = np.ones((1, arr.numelements()))

        pulse = Pulse(
            frequency=self.frequency_khz * 1e3,
            duration=self.duration_msec * 1e-3,
        )

        sequence = Sequence(
            pulse_interval=self.interval_msec * 1e-3,
            pulse_count=int(self.sequence_duration / (self.interval_msec * 1e-3)),
            pulse_train_interval=0,
            pulse_train_count=1,
        )

        pin_order = np.argsort([el.pin for el in arr.elements])
        solution = Solution(
            delays=delays[:, pin_order],
            apodizations=apodizations[:, pin_order],
            transducer=arr,
            pulse=pulse,
            voltage=self.voltage,
            sequence=sequence,
        )

        profile_index = 1
        profile_increment = True
        trigger_mode = "continuous"

        self.interface.set_solution(
            solution=solution,
            profile_index=profile_index,
            profile_increment=profile_increment,
            trigger_mode=trigger_mode,
        )

        self.logger.info("Solution configured for Test Case %s.", self.test_case_num)

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

        deviation_limit_percentage = self.voltage * self.max_allowed_voltage_deviation_percentage / 100
        deviation_limit_absolute_value = VOLTAGE_DEVIATION_ABSOLUTE_VALUE_LIMIT
        deviation_limit_v = max(deviation_limit_percentage, deviation_limit_absolute_value)
        
        self.logger.info("  Voltage Deviation Limits: %.2f V (max of %.2f%% of %.2fV (%.2fV) or %.1f V) from expected %.2f V.",
                            deviation_limit_v,
                            self.max_allowed_voltage_deviation_percentage,
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
                        console_voltage = self.interface.hvcontroller.get_voltage()

                        deviation_limit_percentage = self.voltage * self.max_allowed_voltage_deviation_percentage / 100
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
                                self.max_allowed_voltage_deviation_percentage,
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

        self.logger.warning("Console voltage shutdown triggered.")
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

        while not self.shutdown_event.is_set():
            time_elapsed = time.time() - start_time

            # Read temperatures
            try:
                if not self.use_external_power:
                    if prev_con_temp is None:
                        with self.mutex:
                            if self.shutdown_event.is_set():
                                return
                            prev_con_temp = self.interface.hvcontroller.get_temperature1()
                    with self.mutex:
                        if self.shutdown_event.is_set():
                            return
                        con_temp = self.interface.hvcontroller.get_temperature1()
                else:
                    con_temp = None

                if prev_tx_temp is None:
                    with self.mutex:
                        if self.shutdown_event.is_set():
                            return
                        prev_tx_temp = self.interface.txdevice.get_temperature()
                with self.mutex:
                    if self.shutdown_event.is_set():
                        return
                    tx_temp = self.interface.txdevice.get_temperature()
                    self.test_results[self.test_case_num].final_temperature = tx_temp

                if prev_amb_temp is None:
                    with self.mutex:
                        if self.shutdown_event.is_set():
                            return
                        prev_amb_temp = self.interface.txdevice.get_ambient_temperature()
                with self.mutex:  
                    if self.shutdown_event.is_set():
                        return  
                    amb_temp = self.interface.txdevice.get_ambient_temperature()

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
                        "  Console Temp: %.2f°C, TX Temp: %.2f°C, Ambient Temp: %.2f°C",
                        con_temp,
                        tx_temp,
                        amb_temp,
                    )
                else:
                    self.logger.info(
                        "  TX Temp: %.2f°C, Ambient Temp: %.2f°C",
                        tx_temp,
                        amb_temp,
                    )

            # Absolute temperature thresholds
            if (not self.use_external_power and con_temp is not None and
                    con_temp > self.console_shutoff_temp_C):
                self.logger.warning(
                    "Console temperature %.2f°C exceeds shutoff threshold %.2f°C.",
                    con_temp,
                    self.console_shutoff_temp_C,
                )
                break

            if tx_temp > self.tx_shutoff_temp_C:
                self.logger.warning(
                    "TX device temperature %.2f°C exceeds shutoff threshold %.2f°C.",
                    tx_temp,
                    self.tx_shutoff_temp_C,
                )
                break

            if amb_temp > self.ambient_shutoff_temp_C:
                self.logger.warning(
                    "Ambient temperature %.2f°C exceeds shutoff threshold %.2f°C.",
                    amb_temp,
                    self.ambient_shutoff_temp_C,
                )
                break

            time.sleep(self.temperature_check_interval)
        
        self.logger.warning("Temperature shutdown triggered.")
        self.shutdown_event.set()
        self.temperature_shutdown_event.set()

    def _verify_start_conditions(self, test_case, starting_temperature) -> None:
        """Monitor cooldown period before starting the test."""
        temp = self.interface.txdevice.get_temperature()  # Initial read to populate temperature
        self.logger.info(f"Initial TX temperature: {temp}C")

        if self.test_runthrough:
            starting_temperature = TEST_RUNTHROUGH_TEMPERATURE_BYPASS_C
            self.logger.info(f"Test runthrough mode enabled - starting temp set to {starting_temperature}C.")

        counter = 0
        while temp > starting_temperature:
            self.logger.info(f"Current temperature of {temp}C is greater than max starting "
                             f"temperature of {starting_temperature}C for test case {test_case}. "
                             f"Transmitter will turn off for {TIME_BETWEEN_TESTS_TEMPERATURE_CHECK_SECONDS // 60} minutes to cool down and then check again.")
            self.turn_off_console_and_tx()
            self.cleanup_interface()
            time.sleep(TIME_BETWEEN_TESTS_TEMPERATURE_CHECK_SECONDS)  # Wait before rechecking
            self.connect_device()
            self.verify_communication()
            temp = self.interface.txdevice.get_temperature()  # Update temperature after cooldown
            counter += 1

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
        """Safely cleanup the LIFU interface."""
        if self.interface is None:
            return
        try:
            self.logger.info("Closing device interface...")
            with contextlib.suppress(Exception):
                self.interface.stop_monitoring()
            time.sleep(0.2)
            del self.interface
        except Exception as e:
            self.logger.warning("Issue closing LIFU interface: %s", e)
        finally:
            self.interface = None

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
                + ("\n" if i == len(TEST_CASES)/2 else "")
                for i, tc in enumerate(TEST_CASES[self.starting_test_case-1:], start=self.starting_test_case)
            )
            + "\n\nThe script will account for cooldown periods as needed between test cases. \n" \
            f"Each test case will run for {self.sequence_duration/60:.2f} minutes. \n"
            f"The lower voltage tests starting at {LOW_VOLTAGE_VALUE}V and below will run for {LOW_VOLTAGE_VALUE_TEST_DURATION_SECONDS} seconds. \n"
            f"Maximum allowed console voltage deviation is {self.max_allowed_voltage_deviation_percentage:.2f}% or {VOLTAGE_DEVIATION_ABSOLUTE_VALUE_LIMIT:.2f}V, whichever is greater.\n"
            f"The temperature shutoff limits are: \n"
            f"  Console: {self.console_shutoff_temp_C:>3}C"
            f"  TX Device: {self.tx_shutoff_temp_C:>3}C" 
            f"  Ambient: {self.ambient_shutoff_temp_C:>3}C \n"
            "Approximate test duration is <24hrs.\n"
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
                f"Max Allowed Voltage Deviation: {VOLTAGE_DEVIATION_ABSOLUTE_VALUE_LIMIT:>3}V ({self.max_allowed_voltage_deviation_percentage:>3}%), "
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
        self.test_status = "not started"

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

        for test_case, test_case_parameters in enumerate(TEST_CASES[self.starting_test_case-1:], start=self.starting_test_case):
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

            test_case_start_time = 0

            try:
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

                # Start sonication
                if not self.hw_simulate:
                    self.logger.info("Starting Trigger...")
                    if not self.interface.start_sonication():
                        self.logger.error("Failed to start trigger.")
                        self.test_status = "error"
                        return
                    test_case_start_time = time.time()
                else:
                    self.logger.info("Simulated Trigger start... (no hardware)")

                self.logger.info("Trigger Running... (Press CTRL-C to stop early)")
                self.test_status = "running"

                # Start monitoring threads
                self.shutdown_event.clear()
                self.sequence_complete_event.clear()
                self.temperature_shutdown_event.clear()
                self.voltage_shutdown_event.clear()

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
                # self.test_results[self.test_case_num].test_time_elapsed = time.time() - test_case_start_time if test_case_start_time else 0
                duration = time.time() - test_case_start_time if test_case_start_time else 0.0
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
                # self.test_results[self.test_case_num].cooldown_time_elapsed = 0.0

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


