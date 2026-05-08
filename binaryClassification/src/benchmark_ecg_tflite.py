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
NUM_RUNS = 1000
WARMUP = 100


def main():
    # Load the TFLite model.
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Create one dummy ECG window.
    x = np.random.randn(1, WINDOW_SIZE, 1).astype(np.float32)

    # Quantize input if the model expects int8 input.
    input_scale, input_zero_point = input_details["quantization"]
    if input_details["dtype"] == np.int8:
        x = np.round(x / input_scale + input_zero_point).astype(np.int8)
    else:
        x = x.astype(input_details["dtype"])

    # Warm-up runs. These are not counted.
    for _ in range(WARMUP):
        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])

    start = time.perf_counter()

    for _ in range(NUM_RUNS):
        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])

    end = time.perf_counter()

    avg_latency_ms = (end - start) * 1000 / NUM_RUNS

    print(f"Average latency: {avg_latency_ms:.4f} ms/sample")
    print(f"Number of measured runs: {NUM_RUNS}")


if __name__ == "__main__":
    main()
