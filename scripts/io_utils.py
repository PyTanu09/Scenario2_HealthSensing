"""
Robust I/O utilities for reading health sensing data.
"""
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

def read_signal_file(path):
    """
    Read a semicolon-separated signal file with European timestamps.
    
    Expected format:
      [Optional headers/blank lines]
      Data:
      timestamp; value
      30.05.2024 20:59:00,031; 0.123
      ...
    
    Returns:
        DataFrame indexed by datetime with column 'value'
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Skip to "Data:" line
    data_start = 0
    for i, line in enumerate(lines):
        if 'Data:' in line or 'data:' in line.lower():
            data_start = i + 1
            break
    
    data_lines = lines[data_start:]
    
    timestamps = []
    values = []
    
    for line in data_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Handle semicolon, comma, or whitespace separators
        parts = None
        if ';' in line:
            parts = line.split(';')
        elif ',' in line and len(line.split(',')) == 2:
            # check if it looks like "; ," pattern or standalone comma
            parts = line.split(',')
        else:
            parts = line.split()
        
        if len(parts) < 2:
            continue
        
        ts_str = parts[0].strip()
        val_str = parts[1].strip()
        
        try:
            # Normalize European timestamp: "30.05.2024 20:59:00,031" -> datetime
            # Handle comma as decimal separator for milliseconds
            ts_str = ts_str.replace(',', '.')
            dt = pd.to_datetime(ts_str, format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
            if pd.isna(dt):
                # Try alternative formats
                for fmt in ['%d.%m.%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                    dt = pd.to_datetime(ts_str, format=fmt, errors='coerce')
                    if not pd.isna(dt):
                        break
            if pd.isna(dt):
                continue
            
            val = float(val_str)
            timestamps.append(dt)
            values.append(val)
        except (ValueError, TypeError):
            continue
    
    if not timestamps:
        raise ValueError(f"No valid data found in {path}")
    
    df = pd.DataFrame({'value': values}, index=pd.DatetimeIndex(timestamps))
    df.index.name = 'timestamp'
    return df

def read_events(path):
    """
    Read an events file with ranged events and/or discrete sleep profiles.
    
    Expected formats:
      1. Full-timestamp ranged events: "date start_time-end_time; dur; label; stage"
         e.g., "29.05.2024 21:33:57,246-21:34:33,496; 36; Hypopnea; N1"
      2. Time-only ranged events: "start-end; dur; label; stage"
         e.g., "20:59:00-21:00:00; 60; Obstructive Apnea; NREM"
      3. Discrete sleep profile: "timestamp; stage"
         e.g., "20:59:00; NREM"
      4. Optional header: "Rate: N s"
    
    Returns:
        DataFrame with columns ['onset', 'offset', 'event']
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    
    rate_s = 1  # default rate in seconds
    events_list = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('Signal'):
            continue
        
        # Check for rate header
        if line.lower().startswith('rate:'):
            try:
                rate_s = int(line.split(':')[1].split('s')[0].strip())
            except (ValueError, IndexError):
                pass
            continue
        
        # Skip metadata lines
        if any(x in line.lower() for x in ['start time:', 'unit:', 'signal']):
            continue
        
        # Parse event line
        parts = [p.strip() for p in line.split(';')]
        
        if len(parts) >= 1:
            time_part = parts[0]
            
            # Check if it's a ranged event (contains '-')
            if '-' in time_part:
                try:
                    # Split into date and times if full timestamp provided
                    if ',' in time_part or any(c.isdigit() for c in time_part.split('-')[0]):
                        # Could be "29.05.2024 21:33:57,246-21:34:33,496"
                        # or "21:33:57,246-21:34:33,496"
                        
                        # Try to parse as full datetime range
                        if any(c in '.-/' for c in time_part.split('-')[0]):
                            # Full date in first part
                            parts_dash = time_part.split('-')
                            start_str = parts_dash[0].strip()
                            end_str = '-'.join(parts_dash[1:]).strip()  # in case end has dash
                            
                            # Normalize timestamps
                            start_str = start_str.replace(',', '.')
                            end_str = end_str.replace(',', '.')
                            
                            start_time = pd.to_datetime(start_str, format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
                            end_time = pd.to_datetime(end_str, format='%H:%M:%S.%f', errors='coerce')
                            
                            if pd.isna(start_time):
                                start_time = pd.to_datetime(start_str, format='%d.%m.%Y %H:%M:%S', errors='coerce')
                            if pd.isna(end_time):
                                end_time = pd.to_datetime(end_str, format='%H:%M:%S', errors='coerce')
                            
                            # If end_time has no date, inherit from start_time
                            if not pd.isna(end_time) and pd.isna(end_time.year) or end_time.year == 1900:
                                end_time = start_time.replace(hour=end_time.hour, minute=end_time.minute, second=end_time.second, microsecond=end_time.microsecond)
                        else:
                            # Time-only range
                            start_str, end_str = time_part.split('-')
                            start_str = start_str.strip().replace(',', '.')
                            end_str = end_str.strip().replace(',', '.')
                            
                            start_time = pd.to_datetime(start_str, format='%H:%M:%S.%f', errors='coerce')
                            end_time = pd.to_datetime(end_str, format='%H:%M:%S.%f', errors='coerce')
                            
                            if pd.isna(start_time):
                                start_time = pd.to_datetime(start_str, format='%H:%M:%S', errors='coerce')
                            if pd.isna(end_time):
                                end_time = pd.to_datetime(end_str, format='%H:%M:%S', errors='coerce')
                        
                        if not pd.isna(start_time) and not pd.isna(end_time):
                            label = parts[2] if len(parts) > 2 else 'Unknown'
                            events_list.append({
                                'onset': start_time,
                                'offset': end_time,
                                'event': label.strip()
                            })
                except (ValueError, IndexError):
                    pass
            else:
                # Discrete event: timestamp and stage/label
                ts_str = time_part.replace(',', '.')
                label = parts[1] if len(parts) > 1 else 'Unknown'
                try:
                    ts = pd.to_datetime(ts_str, format='%H:%M:%S.%f', errors='coerce')
                    if pd.isna(ts):
                        ts = pd.to_datetime(ts_str, format='%H:%M:%S', errors='coerce')
                    if not pd.isna(ts):
                        events_list.append({
                            'onset': ts,
                            'offset': ts + pd.Timedelta(seconds=rate_s),
                            'event': label.strip()
                        })
                except (ValueError, IndexError):
                    pass
    
    if not events_list:
        return pd.DataFrame(columns=['onset', 'offset', 'event'])
    
    df = pd.DataFrame(events_list)
    return df
