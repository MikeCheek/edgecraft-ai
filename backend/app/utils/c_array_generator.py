import struct
from typing import List

class CArrayGenerator:
    """Generate C-array representations of model files"""
    
    @staticmethod
    def binary_to_c_array(binary_data: bytes, model_name: str = "model") -> str:
        """Convert binary model data to C array format"""
        
        # Generate hex string from binary
        hex_values = []
        for i, byte in enumerate(binary_data):
            hex_values.append(f"0x{byte:02x}")
            
            # Add newline every 16 values for readability
            if (i + 1) % 16 == 0:
                hex_values.append("\n  ")
        
        hex_string = ", ".join(hex_values).rstrip(",\n  ")
        
        c_header = f"""#ifndef {model_name.upper()}_MODEL_H
#define {model_name.upper()}_MODEL_H

#include <stdint.h>

// Model data array
// Size: {len(binary_data)} bytes
// Generated automatically for TinyML deployment

#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(8)))

const uint8_t g_{model_name}[] DATA_ALIGN_ATTRIBUTE = {{
  {hex_string}
}};

const uint32_t g_{model_name}_len = {len(binary_data)};

#endif // {model_name.upper()}_MODEL_H
"""
        return c_header
    
    @staticmethod
    def generate_cpp_inference_wrapper(model_name: str, input_shape: tuple, 
                                       output_shape: tuple, task_type: str) -> str:
        """Generate C++ wrapper for model inference"""
        
        cpp_wrapper = f"""#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "{model_name}_model.h"

namespace tinyml_{{

class {model_name.upper()}Model {{
private:
    tflite::MicroInterpreter* interpreter_;
    TfLiteTensor* input_;
    TfLiteTensor* output_;
    
public:
    {model_name.upper()}Model() : interpreter_(nullptr), input_(nullptr), output_(nullptr) {{}}
    
    bool Initialize() {{
        // Load model
        const tflite::Model* model = tflite::GetModel(g_{model_name});
        if (model->version() != TFLITE_SCHEMA_VERSION) {{
            return false;
        }}
        
        // Create resolver
        tflite::MicroMutableOpResolver<10> resolver;
        // Add operators as needed
        
        // Create interpreter
        static uint8_t tensor_arena[10 * 1024];
        interpreter_ = new tflite::MicroInterpreter(
            model, resolver, tensor_arena, sizeof(tensor_arena));
        
        if (interpreter_->AllocateTensors() != kTfLiteOk) {{
            return false;
        }}
        
        input_ = interpreter_->input(0);
        output_ = interpreter_->output(0);
        return true;
    }}
    
    bool Invoke(const float* input_data, float* output_data) {{
        // Copy input
        std::copy(input_data, input_data + {input_shape[0] * input_shape[1] if len(input_shape) >= 2 else input_shape[0]}, input_->data.f);
        
        // Run inference
        if (interpreter_->Invoke() != kTfLiteOk) {{
            return false;
        }}
        
        // Copy output
        std::copy(output_->data.f, output_->data.f + {output_shape[0]}, output_data);
        return true;
    }}
    
    ~{model_name.upper()}Model() {{
        delete interpreter_;
    }}
}};

}} // namespace tinyml_
"""
        return cpp_wrapper
    
    @staticmethod
    def generate_arduino_sketch(model_name: str, task_type: str) -> str:
        """Generate an Arduino sketch template for model inference"""
        
        sketch = f"""#include <TensorFlowLite_ESP32.h>
#include "{model_name}_model.h"

// Model inference engine
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {{
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("EdgeCraft AI - {model_name} Model");
  Serial.println("Task: {task_type}");
  
  // Initialize model
  if (!initializeModel()) {{
    Serial.println("ERROR: Failed to initialize model");
    while(1) delay(1000);
  }}
  
  Serial.println("Model initialized successfully!");
}}

void loop() {{
  // Read input from sensor/camera/microphone
  // (Implementation depends on {task_type} task type)
  
  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {{
    Serial.println("ERROR: Inference failed");
    return;
  }}
  
  // Process output
  float* predictions = output->data.f;
  int num_classes = output->dims->data[output->dims->size - 1];
  
  Serial.print("Predictions: ");
  for (int i = 0; i < num_classes; i++) {{
    Serial.print(predictions[i], 4);
    Serial.print(" ");
  }}
  Serial.println();
  
  delay(1000);
}}

bool initializeModel() {{
  // Implementation of model initialization
  return true;
}}
"""
        return sketch
