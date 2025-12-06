import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob

# Add scripts dir to path to import io_utils
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from io_utils import read_signal_file, read_events

def find_file(folder, patterns):
    """Find first file matching any of the patterns (case-insensitive, partial match)."""
    files = sorted(glob.glob(os.path.join(folder, '*')))
    for f in files:
        fname_lower = os.path.basename(f).lower()
        for pattern in patterns:
            if pattern.lower() in fname_lower:
                return f
    return None

def plot_signals(nasal_df, thor_df, spo2_df, events_df, out_pdf, participant):
    """Plot signals and overlay events."""
    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(15, 10))
        plt.suptitle(f"{participant} – Sleep Signals", fontsize=18)

        # Nasal Airflow
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(nasal_df.index, nasal_df["value"], linewidth=0.5, label="Nasal Airflow")
        ax1.set_title("Nasal Airflow")
        ax1.set_ylabel("Flow (a.u.)")
        
        # Overlay events: support both 'onset'/'offset' (from events) and 'start'/'end' (legacy)
        if not events_df.empty:
            onset_col = 'onset' if 'onset' in events_df.columns else ('start' if 'start' in events_df.columns else None)
            offset_col = 'offset' if 'offset' in events_df.columns else ('end' if 'end' in events_df.columns else None)
            if onset_col and offset_col:
                for _, row in events_df.iterrows():
                    ax1.axvspan(row[onset_col], row[offset_col], color="red", alpha=0.2)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Thoracic Movement
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(thor_df.index, thor_df["value"], color="orange", linewidth=0.5, label="Thoracic")
        ax2.set_title("Thoracic Movement")
        ax2.set_ylabel("Movement (a.u.)")
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # SpO2
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(spo2_df.index, spo2_df["value"], color="green", linewidth=1, label="SpO2")
        ax3.set_title("SpO₂ (Blood Oxygen Saturation)")
        ax3.set_ylabel("SpO2 (%)")
        ax3.set_xlabel("Time")
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", required=True, help="Folder path for participant, e.g. Data/AP01")
    parser.add_argument("--out", default="Visualizations", help="Output directory for PDFs")
    parser.add_argument("--no-filter", action="store_true", help="Skip bandpass filtering")
    parser.add_argument("--downsample", type=str, default="1s", help="Downsample frequency (e.g., '1s', '5s')")
    parser.add_argument("--max-hours", type=float, default=None, help="Limit to first N hours")
    args = parser.parse_args()

    folder = args.name
    participant = os.path.basename(folder)

    # Find signal files by pattern matching
    nasal_file = find_file(folder, ["flow", "nasal"])
    thor_file = find_file(folder, ["thorac"])
    spo2_file = find_file(folder, ["spo2"])
    events_file = find_file(folder, ["event", "flow event"])
    sleep_file = find_file(folder, ["sleep"])

    if not nasal_file or not thor_file or not spo2_file:
        print(f"Error: Missing signal files in {folder}")
        print(f"  Nasal: {nasal_file}, Thorac: {thor_file}, SpO2: {spo2_file}")
        return

    print(f"Files found:")
    print(f" Nasal: {os.path.basename(nasal_file)}")
    print(f" Thorac: {os.path.basename(thor_file)}")
    print(f" SpO2: {os.path.basename(spo2_file)}")
    if events_file:
        print(f" Events: {os.path.basename(events_file)}")
    if sleep_file:
        print(f" Sleep: {os.path.basename(sleep_file)}")

    # Load signals using robust reader
    nasal_df = read_signal_file(nasal_file)
    thor_df = read_signal_file(thor_file)
    spo2_df = read_signal_file(spo2_file)

    # Load events if available
    events_df = pd.DataFrame()
    if events_file:
        events_df = read_events(events_file)

    # Optionally limit to first N hours
    if args.max_hours:
        end_time = nasal_df.index[0] + pd.Timedelta(hours=args.max_hours)
        nasal_df = nasal_df[nasal_df.index <= end_time]
        thor_df = thor_df[thor_df.index <= end_time]
        spo2_df = spo2_df[spo2_df.index <= end_time]
        if not events_df.empty:
            events_df = events_df[(events_df['onset'] <= end_time)]

    # Optionally downsample
    if args.downsample and not args.no_filter:
        nasal_df = nasal_df.resample(args.downsample).mean()
        thor_df = thor_df.resample(args.downsample).mean()
        spo2_df = spo2_df.resample(args.downsample).mean()

    # Compute sampling rates
    if len(nasal_df) > 1:
        fs_nasal = 1.0 / (nasal_df.index[1] - nasal_df.index[0]).total_seconds()
    else:
        fs_nasal = 0
    if len(spo2_df) > 1:
        fs_spo2 = 1.0 / (spo2_df.index[1] - spo2_df.index[0]).total_seconds()
    else:
        fs_spo2 = 0

    print(f"Sampling rates (Hz) - nasal: {fs_nasal:.2f} thor: {fs_nasal:.2f} spo2: {fs_spo2:.2f}")

    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    out_pdf = os.path.join(args.out, f"{participant}_visualization.pdf")

    plot_signals(nasal_df, thor_df, spo2_df, events_df, out_pdf, participant)
    print(f"Saved PDF: {out_pdf}")

if __name__ == "__main__":
    main()
