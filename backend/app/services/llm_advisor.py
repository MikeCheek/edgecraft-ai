from typing import Dict, List
import json

class LLMAdvisor:
    """LLM-powered advisor for model optimization suggestions"""
    
    def __init__(self):
        # In a real implementation, this would connect to Ollama or another LLM
        self.llm_suggestions_db = {}
    
    def get_suggestions(self, training_id: str, metrics: dict) -> List[dict]:
        """Get LLM-powered suggestions for model improvement"""
        suggestions = []
        
        # Rule-based suggestions (simulating LLM responses)
        loss = metrics.get("loss", 1.0)
        accuracy = metrics.get("accuracy", 0.0)
        val_loss = metrics.get("val_loss", 1.0)
        val_accuracy = metrics.get("val_accuracy", 0.0)
        
        # Check for overfitting
        if (val_accuracy - accuracy) < -0.05:
            suggestions.append({
                "suggestion": "Model shows signs of overfitting",
                "reasoning": f"Training accuracy ({accuracy:.2%}) is significantly higher than validation accuracy ({val_accuracy:.2%})",
                "parameters_to_adjust": {
                    "dropout": 0.3,
                    "l2_regularization": 0.0001,
                    "batch_size": "increase by 20%"
                },
                "estimated_improvement": "3-5% reduction in overfitting"
            })
        
        # Check for underfitting
        if accuracy < 0.7:
            suggestions.append({
                "suggestion": "Model accuracy is low - may be underfitting",
                "reasoning": f"Training accuracy is only {accuracy:.2%}. Model may need more capacity or better features",
                "parameters_to_adjust": {
                    "learning_rate": "increase to 0.005",
                    "epochs": "increase by 30",
                    "base_model": "try larger architecture like EfficientNet"
                },
                "estimated_improvement": "5-10% accuracy improvement"
            })
        
        # Check for convergence
        if loss > 0.5:
            suggestions.append({
                "suggestion": "Loss is still high - model hasn't converged well",
                "reasoning": "Loss should ideally be < 0.3 after training",
                "parameters_to_adjust": {
                    "learning_rate": "try 0.0001 for finer tuning",
                    "epochs": "double the training epochs"
                },
                "estimated_improvement": "Loss reduction to 0.2-0.3"
            })
        
        # Data quality suggestions
        if accuracy > 0.95:
            suggestions.append({
                "suggestion": "Excellent model performance!",
                "reasoning": "Model has achieved high accuracy on both training and validation sets",
                "parameters_to_adjust": {
                    "next_step": "proceed to optimization and deployment"
                },
                "estimated_improvement": "Model ready for production"
            })
        
        return suggestions
    
    def get_optimization_advice(self, optimization_id: str, board: str) -> Dict:
        """Get LLM-powered advice on optimization for specific board deployment"""
        advice = {
            "board": board,
            "optimization_strategies": [],
            "deployment_tips": [],
            "potential_issues": []
        }
        
        # Board-specific optimization advice
        if board == "ESP32_S3_N16R8":
            advice["optimization_strategies"] = [
                "INT8 quantization is recommended for this powerful board",
                "Take advantage of the 8MB PSRAM for larger models",
                "Use esp-nn acceleration library components for optimized 8-bit inference"
            ]
            advice["deployment_tips"] = [
                "Store model in external PSRAM to preserve on-chip RAM",
                "Use hardware acceleration for matrix operations",
                "Consider using WiFi for OTA model updates"
            ]
            advice["potential_issues"] = [
                "Thermal management at high inference rates",
                "Power consumption with continuous inference"
            ]
        
        elif board == "RASPBERRY_PI_PICO_2_W":
            advice["optimization_strategies"] = [
                "INT8 quantization is mandatory due to 520KB RAM limit",
                "Consider FLOAT16 if accuracy is critical",
                "Implement quantization-aware training for better post-quantization performance"
            ]
            advice["deployment_tips"] = [
                "Use the ARM Cortex-M33 DSP unit for inference acceleration",
                "Load model from flash during initialization",
                "Implement batching for multiple inferences"
            ]
            advice["potential_issues"] = [
                "Very limited on-chip RAM - streaming inference may be necessary",
                "Slow clock speed requires efficient quantization"
            ]
        
        elif board == "ARDUINO_NANO_33_BLE":
            advice["optimization_strategies"] = [
                "Aggressive INT8 quantization required (256KB RAM limit)",
                "Consider model pruning to 70%+ sparsity",
                "Use lightweight architectures like MobileNet v1"
            ]
            advice["deployment_tips"] = [
                "Implement block-wise inference to process model in chunks",
                "Use BLE for wireless model updates",
                "Consider storing very small models (< 50KB) for real-time inference"
            ]
            advice["potential_issues"] = [
                "Severe memory constraints may require model splitting",
                "Slow inference due to limited computational resources",
                "Limited model complexity possible"
            ]
        
        advice["next_steps"] = [
            f"1. Apply INT8 quantization optimization for {board}",
            "2. Test inference latency and accuracy on actual hardware",
            "3. Implement error handling and fallback strategies",
            "4. Consider edge caching for frequently used models"
        ]
        
        return advice
