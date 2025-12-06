import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import glob

# Add parent scripts dir to path to import io_utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_scripts_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_scripts_dir)
from io_utils import read_signal_file, read_events

def butter_bandpass(low, high, fs, order=4):
    """Design a bandpass Butterworth filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return b, a

def bandpass_filter(data, low, high, fs):
    """Apply bandpass filter to signal."""
    if len(data) < 4 * 4:  # minimum length for filtfilt with order=4
        return data
    b, a = butter_bandpass(low, high, fs)
    return filtfilt(b, a, data)

def find_file(folder, patterns):
    """Find first file matching any of the patterns (case-insensitive)."""
    files = sorted(glob.glob(os.path.join(folder, '*')))
    for f in files:
        fname_lower = os.path.basename(f).lower()
        for pattern in patterns:
            if pattern.lower() in fname_lower:
                return f
    return None

def windows_from_timeseries(ts, window_sec=30, overlap=0.5):
    """
    Create fixed-length windows from a time series by resampling.
    
    Args:
        ts: pandas Series or DataFrame with datetime index
        window_sec: window duration in seconds
        overlap: overlap fraction (0.5 = 50%)
    
    Returns:
        List of (window_array, start_time, end_time) tuples
    """
    windows = []
    if len(ts) < 2:
        return windows
    
    fs = 1.0 / (ts.index[1] - ts.index[0]).total_seconds()
    window_samples = int(window_sec * fs)
    step = int(window_samples * (1 - overlap))
    
    for start in range(0, len(ts) - window_samples + 1, step):
        end = start + window_samples
        segment = ts.iloc[start:end]
        if len(segment) == window_samples:
            # Extract values as 1D array
            if isinstance(segment, pd.DataFrame):
                window_array = segment.iloc[:, 0].values
            else:
                window_array = segment.values
            start_time = segment.index[0]
            end_time = segment.index[-1]
            windows.append((window_array, start_time, end_time))
    
    return windows

def label_window(start_time, end_time, events_df):
    """Label a window based on event overlap."""
    if events_df.empty:
        return 'Normal'
    
    window_duration = (end_time - start_time).total_seconds()
    
    for _, event in events_df.iterrows():
        event_start = event.get('onset') or event.get('start')
        event_end = event.get('offset') or event.get('end')
        
        if pd.isna(event_start) or pd.isna(event_end):
            continue
        
        overlap_start = max(start_time, event_start)
        overlap_end = min(end_time, event_end)
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        
        if overlap_duration > 0.5 * window_duration:
            label = event.get('event') or 'Unknown'
            return str(label)
    
    return 'Normal'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', required=True, help='Input data directory')
    parser.add_argument('-out_dir', required=True, help='Output directory')
    parser.add_argument('--participant', default=None, help='Process single participant')
    parser.add_argument('--dry-run', action='store_true', help='Do not save, just report')
    parser.add_argument('--verbose', action='store_true', help='Print progress details')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []

    participants = [args.participant] if args.participant else sorted(os.listdir(args.in_dir))
    
    for participant in participants:
        folder = os.path.join(args.in_dir, participant)

        if not os.path.isdir(folder):
            if args.verbose:
                print(f"Skipping {folder} (not a directory)")
            continue

        # Find signal files
        nasal_file = find_file(folder, ['flow', 'nasal'])
        thor_file = find_file(folder, ['thorac'])
        spo2_file = find_file(folder, ['spo2'])
        events_file = find_file(folder, ['event', 'flow event'])

        if not nasal_file or not thor_file or not spo2_file:
            print(f"Skipping {participant}: missing signal files")
            continue

        try:
            # Load signals
            nasal_ts = read_signal_file(nasal_file)
            thor_ts = read_signal_file(thor_file)
            spo2_ts = read_signal_file(spo2_file)

            # Load events
            events_df = pd.DataFrame()
            if events_file:
                events_df = read_events(events_file)

            # Compute sampling rates
            fs_nasal = 1.0 / (nasal_ts.index[1] - nasal_ts.index[0]).total_seconds() if len(nasal_ts) > 1 else 32
            fs_spo2 = 1.0 / (spo2_ts.index[1] - spo2_ts.index[0]).total_seconds() if len(spo2_ts) > 1 else 4

            # Apply bandpass filter
            nasal_ts['value'] = bandpass_filter(nasal_ts['value'].values, 0.17, 0.4, fs_nasal)
            thor_ts['value'] = bandpass_filter(thor_ts['value'].values, 0.17, 0.4, fs_nasal)

            # Create 30-second windows with 50% overlap
            windows = windows_from_timeseries(nasal_ts[['value']], window_sec=30, overlap=0.5)

            if args.verbose:
                print(f"{participant}: {len(windows)} windows created, fs={fs_nasal:.2f} Hz")

            for window_array, start_time, end_time in windows:
                window_len = len(window_array)
                
                # Get corresponding segments from other signals
                thor_segment = thor_ts.loc[start_time:end_time, 'value']
                spo2_segment = spo2_ts.loc[start_time:end_time, 'value']
                
                # Ensure consistent 1D arrays by resampling to fixed length
                # Resample all to the nasal window length
                nasal_array = np.array(window_array, dtype=np.float32)
                
                # Resample thor and spo2 to match nasal length
                if len(thor_segment) > 0:
                    thor_array = np.interp(
                        np.linspace(0, 1, window_len),
                        np.linspace(0, 1, len(thor_segment)),
                        thor_segment.values
                    ).astype(np.float32)
                else:
                    thor_array = np.zeros(window_len, dtype=np.float32)
                
                if len(spo2_segment) > 0:
                    spo2_array = np.interp(
                        np.linspace(0, 1, window_len),
                        np.linspace(0, 1, len(spo2_segment)),
                        spo2_segment.values
                    ).astype(np.float32)
                else:
                    spo2_array = np.zeros(window_len, dtype=np.float32)

                # Label the window
                label = label_window(start_time, end_time, events_df)

                row = {
                    'participant': participant,
                    'start': start_time,
                    'end': end_time,
                    'label': label,
                    'nasal': nasal_array,
                    'thor': thor_array,
                    'spo2': spo2_array
                }
                all_rows.append(row)

        except Exception as e:
            print(f"Error processing {participant}: {e}")
            continue

    if args.dry_run:
        print(f"Dry-run: would save {len(all_rows)} windows")
        if all_rows:
            row = all_rows[0]
            print(f"  Nasal shape: {row['nasal'].shape}")
            print(f"  Thor shape: {row['thor'].shape}")
            print(f"  SPO2 shape: {row['spo2'].shape}")
        return

    # Save as parquet for easy loading in training
    try:
        df = pd.DataFrame(all_rows)
        out_file = os.path.join(args.out_dir, 'breathing_dataset.csv')
        df.to_csv(out_file, index=False)
        print(f"Dataset saved at: {out_file}")
    except Exception as e:
        print(f"Error saving dataset: {e}")

if __name__ == "__main__":
    main()
