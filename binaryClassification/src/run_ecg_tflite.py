import time
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter


MODEL_PATH = "/home/icsl/model/ecg_1dcnn_pruned_int8_binary.tflite"
WINDOW_SIZE = 256


def load_interpreter(model_path):
    # Load the TFLite model and allocate tensors.
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def normalize_window(window):
    # Use the same z-score normalization as training.
    window = window.astype(np.float32)
    window = (window - np.mean(window)) / (np.std(window) + 1e-8)
    return window


def prepare_input(window, input_details):
    # Expected model input shape: [1, 256, 1]
    x = normalize_window(window)
    x = x.reshape(1, WINDOW_SIZE, 1).astype(np.float32)

    scale, zero_point = input_details["quantization"]

    if input_details["dtype"] == np.int8:
        x = np.round(x / scale + zero_point).astype(np.int8)
    else:
        x = x.astype(input_details["dtype"])

    return x


def dequantize_output(output, output_details):
    # Convert INT8 output back to float scores if needed.
    scale, zero_point = output_details["quantization"]

    if output_details["dtype"] == np.int8:
        output = (output.astype(np.float32) - zero_point) * scale

    return output


def predict(interpreter, window):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    x = prepare_input(window, input_details)

    interpreter.set_tensor(input_details["index"], x)

    start = time.perf_counter()
    interpreter.invoke()
    end = time.perf_counter()

    output = interpreter.get_tensor(output_details["index"])
    output = dequantize_output(output, output_details)

    pred = int(np.argmax(output, axis=1)[0])
    latency_ms = (end - start) * 1000

    label = "normal" if pred == 0 else "arrhythmia"

    return label, output, latency_ms


if __name__ == "__main__":
    interpreter = load_interpreter(MODEL_PATH)

    # Dummy ECG window for deployment sanity check.
    dummy_window = np.random.randn(WINDOW_SIZE).astype(np.float32)

    label, scores, latency_ms = predict(interpreter, dummy_window)

    print("Prediction:", label)
    print("Output scores:", scores)
    print(f"Latency: {latency_ms:.3f} ms")
