from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import wfdb


def normalize_signal(x, eps=1e-8):
    x = x.astype(np.float32)
    return (x - np.mean(x)) / (np.std(x) + eps)


def estimate_fs(timestamps):
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        raise ValueError("Invalid timestamp range.")
    return (len(timestamps) - 1) / duration


def detect_r_peaks(ecg_values, fs):
    x = normalize_signal(ecg_values)

    min_distance = max(1, int(0.45 * fs))

    peaks, _ = find_peaks(
        x,
        distance=min_distance,
        prominence=1.0,
    )

    return peaks


def save_wfdb_record(csv_path, output_dir, record_name, force_fs=None):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if "raw" not in df.columns or "ecg" not in df.columns:
        raise ValueError("CSV must contain 'raw' and 'ecg' columns.")

    raw = df["raw"].to_numpy(dtype=np.float32)
    ecg = df["ecg"].to_numpy(dtype=np.float32)

    if "timestamp_sec" in df.columns and force_fs is None:
        timestamps = df["timestamp_sec"].to_numpy(dtype=np.float32)
        fs = estimate_fs(timestamps)
    elif force_fs is not None:
        fs = float(force_fs)
    else:
        raise ValueError("CSV needs timestamp_sec column or use --fs.")

    print(f"Estimated/used fs: {fs:.2f} Hz")

    # Two-channel WFDB record:
    # channel 0 = processed ECG
    # channel 1 = raw signal
    signal = np.column_stack([ecg, raw]).astype(np.float32)

    record_path = output_dir / record_name

    wfdb.wrsamp(
        record_name=record_name,
        fs=fs,
        units=["adu", "adu"],
        sig_name=["ECG", "RAW"],
        p_signal=signal,
        fmt=["16", "16"],
        write_dir=str(output_dir),
    )

    r_peaks = detect_r_peaks(ecg, fs)

    symbols = ["N"] * len(r_peaks)

    wfdb.wrann(
        record_name=record_name,
        extension="atr",
        sample=r_peaks.astype(np.int64),
        symbol=symbols,
        write_dir=str(output_dir),
    )

    print("\n========== WFDB Record Saved ==========")
    print("Input CSV:", csv_path)
    print("Output directory:", output_dir)
    print("Record name:", record_name)
    print("Signal length:", len(ecg))
    print("Sampling rate:", fs)
    print("Detected beats:", len(r_peaks))
    print("All beat labels: N")
    print("Files:")
    print(f"  {record_path}.dat")
    print(f"  {record_path}.hea")
    print(f"  {record_path}.atr")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        default="/home/icsl/esp32_wifi_ecg.csv",
    )

    parser.add_argument(
        "--output_dir",
        default="/home/icsl/synthetic_mitbih",
    )

    parser.add_argument(
        "--record_name",
        default="synthetic_normal",
    )

    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="Optional fixed sampling rate. If omitted, estimate from timestamp_sec.",
    )

    args = parser.parse_args()

    save_wfdb_record(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        record_name=args.record_name,
        force_fs=args.fs,
    )


if __name__ == "__main__":
    main()