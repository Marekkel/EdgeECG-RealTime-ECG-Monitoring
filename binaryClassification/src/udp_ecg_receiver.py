import socket
import csv
import time
from pathlib import Path


UDP_IP = "0.0.0.0"
UDP_PORT = 5005
OUTPUT_CSV = "/home/icsl/esp32_wifi_ecg.csv"


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"Listening for UDP ECG data on port {UDP_PORT}...")
    print(f"Saving to {OUTPUT_CSV}")

    output_path = Path(OUTPUT_CSV)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_sec", "raw", "ecg", "y_min", "y_max"])

        start_time = time.perf_counter()
        packet_count = 0

        while True:
            data, addr = sock.recvfrom(1024)
            line = data.decode("utf-8", errors="ignore").strip()

            parts = line.split(",")

            if len(parts) < 2:
                continue

            try:
                raw = float(parts[0])
                ecg = float(parts[1])
                y_min = float(parts[2]) if len(parts) > 2 else None
                y_max = float(parts[3]) if len(parts) > 3 else None
            except ValueError:
                continue

            timestamp_sec = time.perf_counter() - start_time

            writer.writerow([timestamp_sec, raw, ecg, y_min, y_max])
            f.flush()

            packet_count += 1

            if packet_count % 50 == 0:
                print(
                    f"packet={packet_count}, "
                    f"from={addr[0]}, "
                    f"raw={raw:.1f}, ecg={ecg:.1f}"
                )


if __name__ == "__main__":
    main()