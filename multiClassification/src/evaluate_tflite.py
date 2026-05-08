import numpy as np
import time
import ai_edge_litert.interpreter as tflite

def evaluate_on_raspberry_pi():
    print("Loading stress test data...")
    X_test = np.load('/home/icsl/pi_eval_dataset/X_stress.npy')
    y_test = np.load('/home/icsl/pi_eval_dataset/y_stress.npy')
    
    print("Initializing TFLite model...")
    interpreter = tflite.Interpreter(model_path="/home/icsl/model/ecg_1dcnn_pruned_int8.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_dtype = input_details[0]['dtype']
    scale, zero_point = input_details[0]['quantization']
        
    correct_predictions = 0
    total_samples = len(X_test)
    
    print("Starting inference testing...")
    start_time = time.time()
    
    for i in range(total_samples):
        input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
        
        # 只保留 INT8 的量化缩放
        if input_dtype == np.int8:
            input_data = np.round((input_data / scale) + zero_point)
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        
        if prediction == y_test[i]:
            correct_predictions += 1
            
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / total_samples) * 1000 
    accuracy = (correct_predictions / total_samples) * 100
    
    print("\n======= Raspberry Pi Performance Report =======")
    print(f"Total test samples:     {total_samples}")
    print(f"Model accuracy:         {accuracy:.2f}%")
    print(f"Total inference time:   {total_time:.4f} seconds")
    print(f"Average latency/sample: {avg_latency:.2f} ms")
    print("===============================================")

if __name__ == "__main__":
    evaluate_on_raspberry_pi()