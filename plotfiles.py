"""
Usage
-----
python plot_logs.py run1.log run2.log run3.log
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

TESTCASE_RE = re.compile(r"Test Case (\d+)")
CONFIG_RE   = re.compile(r"Solution configured for Test Case (\d+)")

TEMP_RE = re.compile(
    r"Console Temp:\s*([\d.]+)C,\s*TX Temp:\s*([\d.]+)C,\s*Ambient Temp:\s*([\d.]+)C"
)

VOLT_RE = re.compile(
    r"Console Voltage:\s*([\d.]+)\s*V"
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

# data[file][test_case] = {
#     "temp":    [(t, console, tx, ambient), ...]
#     "voltage": [(t, v), ...]
# }

def parse_log_file(path: Path):

    data = defaultdict(lambda: {"temp": [], "voltage": []})
    current_tc = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:

            # timestamp
            ts_match = TIMESTAMP_RE.match(line)
            if not ts_match:
                continue

            ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")

            # test case transitions
            m = CONFIG_RE.search(line)
            if m:
                current_tc = int(m.group(1))
                continue

            m = TESTCASE_RE.search(line)
            if m and "Starting test case" in line:
                current_tc = int(m.group(1))
                continue

            if current_tc is None:
                continue

            # temperature line
            tm = TEMP_RE.search(line)
            if tm:
                c = float(tm.group(1))
                tx = float(tm.group(2))
                amb = float(tm.group(3))
                data[current_tc]["temp"].append((ts, c, tx, amb))
                continue

            # voltage line
            vm = VOLT_RE.search(line)
            if vm:
                v = float(vm.group(1))
                if v > 100 or v < 0:
                    continue # sanity check
                data[current_tc]["voltage"].append((ts, v))
                continue

    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_time(samples):
    """
    Convert absolute timestamps to seconds from first sample.
    """
    if not samples:
        return []

    t0 = samples[0][0]
    return [((s[0] - t0).total_seconds(), *s[1:]) for s in samples]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# def plot_all(files_data):

#     # collect all test cases across all files
#     all_test_cases = set()
#     for fd in files_data.values():
#         all_test_cases.update(fd.keys())

#     for tc in sorted(all_test_cases):

#         # ----------------------------
#         # Temperature plot
#         # ----------------------------
#         plt.figure(figsize=(10, 6))

#         for file_idx, (fname, file_data) in enumerate(files_data.items()):

#             if tc not in file_data:
#                 continue

#             samples = normalize_time(file_data[tc]["temp"])
#             if not samples:
#                 continue

#             t = [s[0] for s in samples]
#             console = [s[1] for s in samples]
#             tx      = [s[2] for s in samples]
#             amb     = [s[3] for s in samples]

#             base_label = f"{fname.name}"

#             plt.plot(t, console, label=f"{base_label} – Console")
#             plt.plot(t, tx,      label=f"{base_label} – TX",      linestyle="--")
#             plt.plot(t, amb,     label=f"{base_label} – Ambient", linestyle=":")

#         plt.title(f"Test Case {tc} – Temperatures")
#         plt.xlabel("Time (s)")
#         plt.ylabel("Temperature (C)")
#         plt.grid(True)
#         plt.legend(fontsize=8)
#         plt.tight_layout()


#         # ----------------------------
#         # Voltage plot
#         # ----------------------------
#         plt.figure(figsize=(10, 6))

#         for fname, file_data in files_data.items():

#             if tc not in file_data:
#                 continue

#             samples = normalize_time(file_data[tc]["voltage"])
#             if not samples:
#                 continue

#             t = [s[0] for s in samples]
#             v = [s[1] for s in samples]

#             plt.plot(t, v, label=fname.name)

#         plt.title(f"Test Case {tc} – Console Voltage")
#         plt.xlabel("Time (s)")
#         plt.ylabel("Voltage (V)")
#         plt.grid(True)
#         plt.legend(fontsize=8)
#         plt.tight_layout()

#     plt.show()

def plot_all(files_data):
    # Create an output directory
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    all_test_cases = set()
    for fd in files_data.values():
        all_test_cases.update(fd.keys())

    for tc in sorted(all_test_cases):
        # ----------------------------
        # Temperature plot
        # ----------------------------
        plt.figure(figsize=(12, 7))
        has_temp_data = False

        for fname, file_data in files_data.items():
            if tc not in file_data: continue
            
            samples = normalize_time(file_data[tc]["temp"])
            if not samples: continue
            has_temp_data = True

            t = [s[0] for s in samples]
            c, tx, amb = [s[1] for s in samples], [s[2] for s in samples], [s[3] for s in samples]

            base_label = fname.name
            # plt.plot(t, c, label=f"{base_label} - Console")
            plt.plot(t, tx, label=f"{base_label} - TX", linestyle="--")
            # plt.plot(t, amb, label=f"{base_label} - Ambient", linestyle=":")

        if has_temp_data:
            plt.title(f"Test Case {tc} – Temperatures")
            plt.xlabel("Seconds from Start")
            plt.ylabel("Temperature (C)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f"test_case_{tc}_temp.png")
        plt.close() # Important: close the figure!

        # ----------------------------
        # Voltage plot
        # ----------------------------
        plt.figure(figsize=(12, 7))
        has_volt_data = False

        for fname, file_data in files_data.items():
            if tc not in file_data: continue
            
            samples = normalize_time(file_data[tc]["voltage"])
            if not samples: continue
            has_volt_data = True

            t, v = [s[0] for s in samples], [s[1] for s in samples]
            plt.plot(t, v, label=fname.name)

        if has_volt_data:
            plt.title(f"Test Case {tc} – Voltage")
            plt.xlabel("Seconds from Start")
            plt.ylabel("Voltage (V)")
            plt.legend(loc='best', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f"test_case_{tc}_voltage.png")
        plt.close()

    print(f"Finished! Plots saved to: {output_dir.absolute()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    if len(sys.argv) != 2:
        print("Usage: python plot_logs.py <log_folder>")
        sys.exit(1)

    log_dir = Path(sys.argv[1])

    if not log_dir.is_dir():
        print(f"{log_dir} is not a directory")
        sys.exit(1)

    paths = sorted(log_dir.glob("*.log"))

    if not paths:
        print("No .log files found in folder.")
        sys.exit(1)

    files_data = {}

    for p in paths:
        files_data[p] = parse_log_file(p)

    plot_all(files_data)



if __name__ == "__main__":
    main()
