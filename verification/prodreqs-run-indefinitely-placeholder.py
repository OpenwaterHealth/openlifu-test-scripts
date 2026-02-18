from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from serial.serialutil import SerialException

import numpy as np

import openlifu
from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.db import Database
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution
from prodreqs_base_class import *
from config import *

SELECTED_TEST_CASE_FOR_INDEFINITE_RUN = 10
TEST_CASE_DURATION_SECONDS = 20 * 60
TEST_CASE_COOLDOWN_SECONDS = 10 * 60

class transmitter_indefinite_run(TestSonicationDurationBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.logger = self._setup_logging()
        self.sequence_duration = TEST_CASE_DURATION_SECONDS
        self.test_case = SELECTED_TEST_CASE_FOR_INDEFINITE_RUN

    def print_banner(self) -> None:
        self.logger.info("Selected frequency: %dkHz", self.frequency_khz)
        self.logger.info("Number of modules: %d", self.num_modules)
        test_case_parameters = TEST_CASES[self.test_case]

        self.logger.info("--------------------------------------------------------------------------------")
        self.logger.info(
            "\n\nThis script will run the following test case indefinitely:\n\n" + 
                f"Test Case {self.test_case:>2}: "
                f"{test_case_parameters['voltage']:>3}V, "
                f"{test_case_parameters['duty_cycle']:>3}% Duty Cycle, "
                f"{test_case_parameters['PRI_ms']:>4}ms PRI, "
                f"Max Starting Temperature: {test_case_parameters['max_starting_temperature']:>3}C"
            + "\n\nThe script will account for cooldown periods as needed between test cases. \n" \
            f"Each test case will run for {self.sequence_duration/60:.2f} minutes. \n")
        self.logger.info("--------------------------------------------------------------------------------\n\n\n")

    def run(self) -> None:
        self.test_status = "not started"

        try:
            self._select_num_modules()
            self._select_frequency()
            # self._select_starting_test_case()
            self._attach_file_handler()
            self.print_banner()
        except Exception as e:
            self.logger.error("Error during initial selection: %s", e)
            sys.exit(1)

        self.logger.info(f"Starting indefinite run of test case {self.test_case}. ")
        self.start_time = time.time()

        # for test_case, test_case_parameters in enumerate(TEST_CASES[self.starting_test_case-1:], start=self.starting_test_case):

        self.test_case_num = self.test_case
        test_case_parameters = TEST_CASES[self.test_case_num-1]

        self.test_results[self.test_case_num] = TestCaseResult()
        self.voltage = float(test_case_parameters["voltage"])
        self.interval_msec = int(test_case_parameters["PRI_ms"])
        self.duration_msec = int(test_case_parameters["duty_cycle"] / 100 * self.interval_msec)
        
        self.logger.info(f"Starting test case {self.test_case_num}")
        self.logger.info("Test Case %d: %dV, %d%% Duty Cycle, %dms duration, %dms PRI, Max Starting Temperature: %dC",
                            self.test_case_num, 
                            self.voltage, 
                            test_case_parameters["duty_cycle"], 
                            self.duration_msec, 
                            self.interval_msec, 
                            test_case_parameters["max_starting_temperature"])

        test_case_start_time = 0

        while True:
            try:
                if not self.hw_simulate:
                    self.connect_device()
                    self.verify_communication()
                    self.get_firmware_versions()
                    self.enumerate_devices()
                    self._verify_start_conditions(self.test_case_num, test_case_parameters["max_starting_temperature"])
                else:
                    self.logger.info("Hardware simulation enabled; skipping device configuration.")

                # if self.test_runthrough:
                #     self.sequence_duration = SHORT_TEST_DURATION_SECONDS
                # elif self.voltage is not None and self.voltage <= LOW_VOLTAGE_VALUE:
                #     self.sequence_duration = LOW_VOLTAGE_VALUE_TEST_DURATION_SECONDS
                # else:
                #     self.sequence_duration = TEST_CASE_DURATION_SECONDS
                
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
                temp_thread.join(timeout=5.0)
                voltage_thread.join(timeout=5.0)
                completion_thread.join(timeout=5.0)

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
            
            self.logger.info(f"{self.sequence_duration//60} minutes of sonication complete. Will cool down for {TEST_CASE_COOLDOWN_SECONDS//60} minutes and then continue looping this test case.\n\n")
            # self.print_test_summary()   

def main() -> None:
    """Main entry point for the script."""

    args = parse_arguments()
    temp_and_voltage_stability_test = transmitter_indefinite_run(args)

    try:
        temp_and_voltage_stability_test.run()
    except KeyboardInterrupt:
        temp_and_voltage_stability_test.logger.warning("Indefinitely running test aborted by user KeyboardInterrupt. Shutting down...")
        temp_and_voltage_stability_test.shutdown_event.set()
        temp_and_voltage_stability_test.stop_logging = True
        time.sleep(0.5)
        with contextlib.suppress(Exception):
            temp_and_voltage_stability_test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            temp_and_voltage_stability_test.cleanup_interface()
        sys.exit(0)
    except Exception as e:
        temp_and_voltage_stability_test.logger.error(f"\nFatal error: {e}")
        with contextlib.suppress(Exception):
            temp_and_voltage_stability_test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            temp_and_voltage_stability_test.cleanup_interface()
        sys.exit(1)

if __name__ == "__main__":
    main()