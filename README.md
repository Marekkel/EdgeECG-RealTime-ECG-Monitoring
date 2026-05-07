# EdgeECG-RealTime-ECG-Monitoring

EdgeECG is a real-time ECG monitoring and arrhythmia detection project designed for edge deployment. The models are trained in Google Colab using MIT-BIH ECG data, compressed with pruning and INT8 quantization, and exported as lightweight TensorFlow Lite models for devices such as Raspberry Pi.

This repository includes:

- Colab training notebooks for binary and multi-class ECG classification.
- A trained INT8 TensorFlow Lite model.
- Edge-side TFLite inference and benchmark scripts.
- An ESP32 UDP sender example for streaming ECG samples to a Raspberry Pi.
- A utility for converting WiFi-captured CSV ECG data into WFDB format.
- A multi-class AAMI-style classification workflow with compression comparison analysis.

## Repository Structure

```text
.
├── README.md
├── binaryClassification
│   ├── model
│   │   └── ecg_1dcnn_pruned_int8.tflite
│   ├── send_data_to_Pi
│   │   └── send_data_to_Pi.ino
│   └── src
│       ├── benchmark_ecg_tflite.py
│       ├── convert_wifi_csv_to_wfdb.py
│       ├── ecg_edge_arrhythmia_starter.ipynb
│       ├── run_ecg_tflite.py
│       └── udp_ecg_receiver.py
└── multiClassification
    └── ecg_multiclass_quant_pruning.ipynb
```

## Project Overview

The repository now contains two Colab training workflows:

```text
binaryClassification/src/ecg_edge_arrhythmia_starter.ipynb
multiClassification/ecg_multiclass_quant_pruning.ipynb
```

Both notebooks are intended to be run on Google Colab. The binary notebook trains a normal-vs-arrhythmia classifier, while the multi-class notebook trains an AAMI-style 5-class classifier.

The shared training flow is:

1. Load ECG records and annotations from the MIT-BIH Arrhythmia Database using `wfdb`.
2. Extract 256-sample ECG windows centered around annotated R-peaks.
3. Convert beat annotations into task-specific labels.
4. Train a lightweight 1D-CNN baseline model.
5. Apply model pruning with TensorFlow Model Optimization Toolkit.
6. Convert the pruned model to a fully INT8 TensorFlow Lite model.
7. Evaluate accuracy, model size, inference latency, and compression trade-offs.

Binary labels:

```text
0 -> normal
1 -> arrhythmia
```

Multi-class labels:

```text
N -> Normal
S -> Supraventricular ectopic beat
V -> Ventricular ectopic beat
F -> Fusion beat
Q -> Unknown, paced, or other beat
```

The notebooks were trained and exported in Colab, so this repository is mainly used for storing trained artifacts and running deployment-side scripts.

## Training in Google Colab

Open one of the notebooks below in Google Colab:

```text
binaryClassification/src/ecg_edge_arrhythmia_starter.ipynb
multiClassification/ecg_multiclass_quant_pruning.ipynb
```

Install dependencies in a fresh Colab runtime:

```bash
pip install numpy pandas matplotlib scipy scikit-learn wfdb tensorflow tensorflow-model-optimization tf_keras
```

The notebooks stream MIT-BIH records from PhysioNet through `wfdb`, so internet access is required during the first run.

Default training configuration:

```text
RECORDS = ['100','101','102','103','104','105','106','107','108','109']
WINDOW_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 10
```

After training, each notebook writes artifacts to the Colab `artifacts/` directory:

```text
baseline_1dcnn.keras
pruned_1dcnn.keras
ecg_1dcnn_pruned_int8.tflite
benchmark_results.csv
```

Copy the exported TFLite model into this repository:

```text
binaryClassification/model/ecg_1dcnn_pruned_int8.tflite
```

## Binary Model Results

The following results come from one Colab run of the current notebook. They may change depending on the selected records, random seed, Colab runtime, and train/test split.

| Model | Accuracy | Size | Latency |
| --- | ---: | ---: | ---: |
| Baseline FP32 1D-CNN | 0.9928 | 194.03 KB | 72.045 ms/sample |
| Pruned 1D-CNN | 0.9948 | 87.46 KB | 56.678 ms/sample |
| Pruned + INT8 TFLite | 0.8965 | 25.01 KB | 0.056 ms/sample |

The deployment model currently stored in this repository is:

```text
binaryClassification/model/ecg_1dcnn_pruned_int8.tflite
```

## Multi-Class Classification

The multi-class notebook adds AAMI-style arrhythmia classification:

```text
multiClassification/ecg_multiclass_quant_pruning.ipynb
```

It maps MIT-BIH beat annotations into five classes:

| Class | Meaning | MIT-BIH symbols used |
| --- | --- | --- |
| N | Normal and bundle branch block beats | `N`, `L`, `R`, `e`, `j` |
| S | Supraventricular ectopic beats | `A`, `a`, `J`, `S` |
| V | Ventricular ectopic beats | `V`, `E` |
| F | Fusion beats | `F` |
| Q | Unknown, paced, or other beats | `/`, `f`, `Q` |

The notebook uses the same 256-sample beat window format and the same baseline/pruning/INT8 export structure as the binary workflow.

### Multi-Class Model Results

The following values come from one Colab run of the multi-class notebook:

| Model | Accuracy | Size | Latency |
| --- | ---: | ---: | ---: |
| Baseline FP32 1D-CNN | 0.9926 | 221.42 KB | 75.632 ms/sample |
| Pruned 1D-CNN | 0.9917 | 87.84 KB | 56.057 ms/sample |
| Pruned + INT8 TFLite | 0.7879 | 24.94 KB | 0.081 ms/sample |

The notebook also computes compression trade-offs relative to the baseline:

| Model | Size Reduction | Latency Change | Accuracy Drop |
| --- | ---: | ---: | ---: |
| Pruned 1D-CNN | 60.33% | 25.88% faster | 0.09% |
| Pruned + INT8 TFLite | 88.74% | 99.89% faster | 20.48% |

The multi-class task is more sensitive to quantization than the binary task, especially because minority classes such as `S` and `F` have very few samples in the current record subset. For stronger multi-class results, expand the MIT-BIH record list and use a more balanced split or class reweighting.

## Edge Deployment

On Raspberry Pi or another Linux edge device, create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas wfdb
```

Install one TFLite interpreter backend depending on your device:

```bash
pip install ai-edge-litert
```

or:

```bash
pip install tflite-runtime
```

If neither package is available, TensorFlow can be used as a fallback:

```bash
pip install tensorflow
```

### Model Path

The inference scripts currently use this default model path:

```text
/home/icsl/model/ecg_1dcnn_pruned_int8.tflite
```

You can either copy the model to that location:

```bash
mkdir -p /home/icsl/model
cp binaryClassification/model/ecg_1dcnn_pruned_int8.tflite /home/icsl/model/
```

or update `MODEL_PATH` in:

```text
binaryClassification/src/run_ecg_tflite.py
binaryClassification/src/benchmark_ecg_tflite.py
```

## Run Inference

Run a simple sanity check with a random ECG window:

```bash
python3 binaryClassification/src/run_ecg_tflite.py
```

Example output:

```text
Prediction: normal
Output scores: [...]
Latency: ... ms
```

The script normalizes each 256-sample ECG window with z-score normalization, quantizes the input when the model expects INT8 data, runs the TFLite interpreter, and maps the output to:

```text
0 -> normal
1 -> arrhythmia
```

## Benchmark Inference Latency

Run the benchmark script:

```bash
python3 binaryClassification/src/benchmark_ecg_tflite.py
```

The script performs 100 warm-up runs and 1000 measured runs, then prints the average latency per sample.

## Real-Time ECG Streaming

The repository includes an ESP32 example that streams ECG samples over UDP to a Raspberry Pi.

Example hardware setup:

- ESP32
- AD8232 ECG module
- Raspberry Pi or another Linux receiver
- Shared WiFi network

### ESP32 Sender

Open:

```text
binaryClassification/send_data_to_Pi/send_data_to_Pi.ino
```

Update the WiFi and receiver settings:

```cpp
const char* WIFI_SSID = "XXXXXXXX";
const char* WIFI_PASSWORD = "XXXXXXXX";
const char* PI_IP = "XXX.XXX.XXX.XXX";
const int PI_PORT = 5005;
```

The default sampling delay is:

```cpp
const int SAMPLE_DELAY_MS = 4;   // approximately 250 Hz
```

The ESP32 sends UDP packets in this format:

```text
raw,ecg,y_min,y_max
```

### Raspberry Pi Receiver

Run the UDP receiver:

```bash
python3 binaryClassification/src/udp_ecg_receiver.py
```

Default listener:

```text
0.0.0.0:5005
```

Default CSV output:

```text
/home/icsl/esp32_wifi_ecg.csv
```

CSV columns:

```text
timestamp_sec,raw,ecg,y_min,y_max
```

## Convert Captured CSV to WFDB

To convert WiFi-captured CSV data into a WFDB record:

```bash
python3 binaryClassification/src/convert_wifi_csv_to_wfdb.py \
  --csv_path /home/icsl/esp32_wifi_ecg.csv \
  --output_dir /home/icsl/synthetic_mitbih \
  --record_name synthetic_normal
```

If the CSV does not contain usable timestamps, or if you want to force a fixed sampling rate, add:

```bash
--fs 250
```

The script generates:

```text
synthetic_normal.dat
synthetic_normal.hea
synthetic_normal.atr
```

The `.atr` annotation file is generated by automatic R-peak detection using `scipy.signal.find_peaks`. All detected beats are labeled as `N` by default.

## Dependencies

Colab training dependencies:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
wfdb
tensorflow
tensorflow-model-optimization
tf_keras
```

Edge inference dependencies:

```text
numpy
ai-edge-litert or tflite-runtime or tensorflow
```

CSV-to-WFDB dependencies:

```text
pandas
numpy
scipy
wfdb
```

## Future Work

- Train and evaluate on more MIT-BIH records.
- Use patient-wise train/test splitting to reduce data leakage.
- Improve the multi-class workflow with class balancing, class weights, and per-class sensitivity analysis.
- Connect the UDP receiver to a sliding-window inference loop for live arrhythmia alerts.
- Benchmark latency and power consumption on Raspberry Pi 5, ESP32-S3, or other edge devices.

## Disclaimer

This project is for coursework, research, and prototyping only. It is not a medical device and must not be used for clinical diagnosis. Real medical use would require rigorous validation, clinical testing, and regulatory review.
