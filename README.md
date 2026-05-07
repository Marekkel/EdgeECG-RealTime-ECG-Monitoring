# EdgeECG-RealTime-ECG-Monitoring

Real-time ECG monitoring and arrhythmia classification with an ESP32, a Raspberry Pi, and a pruned INT8 TensorFlow Lite model.

This repository combines three parts of the workflow:

1. Data collection from an ECG sensor on ESP32 over Wi-Fi.
2. Local recording and conversion on a Raspberry Pi.
3. Binary ECG classification with a compact 1D CNN exported to TFLite.

The code is organized so you can either reproduce the offline experiments from the notebook or run the edge deployment pipeline end to end.

## Repository Layout

```text
binaryClassification/
	model/
		ecg_1dcnn_pruned_int8.tflite
	send_data_to_Pi/
		send_data_to_Pi.ino
	src/
		benchmark_ecg_tflite.py
		convert_wifi_csv_to_wfdb.py
		ecg_edge_arrhythmia_starter.ipynb
		run_ecg_tflite.py
		udp_ecg_receiver.py
```

## What Each Part Does

- `send_data_to_Pi.ino`: reads ECG samples on ESP32 and sends them to the Raspberry Pi over UDP.
- `udp_ecg_receiver.py`: listens for UDP packets and saves them as a CSV file.
- `convert_wifi_csv_to_wfdb.py`: converts the collected CSV into a WFDB record and annotation file.
- `run_ecg_tflite.py`: runs one-window inference with the TFLite model and prints the prediction.
- `benchmark_ecg_tflite.py`: measures average TFLite inference latency.
- `ecg_edge_arrhythmia_starter.ipynb`: notebook for preprocessing, training, pruning, quantization, and exporting the model.

## Requirements

### Software

- Python 3.10+ recommended.
- Jupyter Notebook or JupyterLab for the training notebook.
- Python packages used by the notebook and scripts:
	- `numpy`
	- `pandas`
	- `matplotlib`
	- `scipy`
	- `scikit-learn`
	- `wfdb`
	- `tensorflow`
	- `tensorflow-model-optimization`
	- `tf_keras`
	- one of `ai_edge_litert`, `tflite_runtime`, or `tensorflow` for the TFLite interpreter

### Hardware

- ESP32 board.
- ECG sensor front-end, such as AD8232.
- Raspberry Pi or another machine that can receive UDP packets on the same network.

### Network Setup

- ESP32 and the Raspberry Pi must be on the same Wi-Fi network.
- The Pi must allow incoming UDP traffic on port `5005` by default.

## Quick Start

### 1. Reproduce the notebook workflow

Open `binaryClassification/src/ecg_edge_arrhythmia_starter.ipynb` and run it from top to bottom.

The notebook covers:

- loading ECG data,
- preprocessing and windowing,
- training a baseline 1D CNN,
- pruning and fine-tuning,
- INT8 quantization,
- exporting the final TFLite model,
- benchmarking the generated model.

The notebook downloads MIT-BIH data through `wfdb` and therefore needs internet access on first run.

### 2. Run local TFLite inference

The inference scripts currently expect the model at:

`/home/icsl/model/ecg_1dcnn_pruned_int8.tflite`

If you use the model stored in this repository, either copy it to that location or update `MODEL_PATH` in:

- `binaryClassification/src/run_ecg_tflite.py`
- `binaryClassification/src/benchmark_ecg_tflite.py`

Then run:

```bash
python binaryClassification/src/run_ecg_tflite.py
```

### 3. Measure latency

```bash
python binaryClassification/src/benchmark_ecg_tflite.py
```

This script warms up the interpreter and reports average latency per sample.

### 4. Start UDP recording on the Raspberry Pi

```bash
python binaryClassification/src/udp_ecg_receiver.py
```

By default, it writes to:

`/home/icsl/esp32_wifi_ecg.csv`

The CSV contains these columns:

- `timestamp_sec`
- `raw`
- `ecg`
- `y_min`
- `y_max`

### 5. Convert the recorded CSV to WFDB

```bash
python binaryClassification/src/convert_wifi_csv_to_wfdb.py \
	--csv_path /home/icsl/esp32_wifi_ecg.csv \
	--output_dir /home/icsl/synthetic_mitbih \
	--record_name synthetic_normal
```

You can also provide `--fs` if you want to override the sampling rate instead of estimating it from timestamps.

### 6. Flash the ESP32 sketch

Open `binaryClassification/send_data_to_Pi/send_data_to_Pi.ino` in the Arduino IDE or PlatformIO, then update:

- `WIFI_SSID`
- `WIFI_PASSWORD`
- `PI_IP`
- `PI_PORT`

The sketch samples the ECG input roughly at 250 Hz and sends each sample as a UDP packet.

## Data Flow

```text
ECG sensor -> ESP32 -> UDP -> Raspberry Pi CSV -> WFDB conversion -> training / analysis / evaluation
```

For inference, the pruned INT8 model can be tested independently on any 256-sample ECG window.

## Configuration Notes

- `run_ecg_tflite.py` and `benchmark_ecg_tflite.py` use a fixed `WINDOW_SIZE` of `256`.
- The TFLite model expects input shaped like `[1, 256, 1]`.
- The inference scripts normalize each window with z-score normalization before quantization or casting.
- `udp_ecg_receiver.py` binds to `0.0.0.0:5005` by default.
- `convert_wifi_csv_to_wfdb.py` requires the CSV to contain `raw` and `ecg` columns, and it can estimate sampling rate from `timestamp_sec`.

## Generated Outputs

Depending on which workflow you run, the project can generate:

- a trained baseline Keras model,
- a pruned Keras model,
- an INT8 TFLite model,
- benchmark logs or CSV results from the notebook,
- a raw ECG capture CSV from the UDP receiver,
- a WFDB record with `.hea`, `.dat`, and `.atr` files.

## Troubleshooting

- If TFLite import fails, install one of `ai_edge_litert`, `tflite_runtime`, or `tensorflow`.
- If the Pi does not receive packets, check the ESP32 Wi-Fi credentials, the Pi IP address, and firewall rules for UDP port `5005`.
- If the model file is not found, update the hardcoded `MODEL_PATH` values or copy the model to the expected path.
- If WFDB conversion fails, verify that the CSV contains `raw`, `ecg`, and ideally `timestamp_sec`.

## Notes

- The repository currently keeps the model artifact under `binaryClassification/model/`.
- The deployment scripts still use an absolute model path by default, so the README intentionally documents that mismatch.
- This project is focused on a binary classification demo: normal vs. arrhythmia.