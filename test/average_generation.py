import os
from datetime import datetime

def get_creation_times(log_dir):
    try:
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        timestamps = []

        for file in log_files:
            file_path = os.path.join(log_dir, file)
            stat_info = os.stat(file_path)
            creation_time = stat_info.st_birthtime  # macOS-specific birth time
            timestamps.append(datetime.fromtimestamp(creation_time))

        timestamps.sort()
        return timestamps

    except Exception as e:
        print(f"Error: {e}")
        return []

def calculate_intervals(timestamps, max_allowed_gap=3600):
    """
    Calculate intervals between consecutive timestamps and filter out large gaps.

    :param timestamps: List of datetime objects sorted in ascending order.
    :param max_allowed_gap: Maximum allowable gap in seconds (default is 3600 seconds = 1 hour).
    :return: Average interval and total time in valid intervals.
    """
    if len(timestamps) < 2:
        return None, None

    intervals = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if delta <= max_allowed_gap:  # Only consider reasonable gaps
            intervals.append(delta)
        else:
            print(f"Interval {i}: {format_time(delta)}")

    if intervals:
        avg_interval = sum(intervals) / len(intervals)
        total_time = sum(intervals)
        return avg_interval, total_time
    return None, None

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

log_dir = 'log_google'

timestamps = get_creation_times(log_dir)
average_interval, total_time = calculate_intervals(timestamps, max_allowed_gap=80)

if average_interval is not None:
    print(f"Average interval (seconds): {average_interval:.2f}")
    print(f"Total time: {format_time(total_time)}")
else:
    print("Not enough valid intervals to calculate an average.")