from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from app.services.converter import Converter
from app.services.mcu_advisor import MCUAdvisor
from app.services.llm_advisor import LLMAdvisor

router = APIRouter()
converter = Converter()
mcu_advisor = MCUAdvisor()
llm_advisor = LLMAdvisor()

class OptimizationRequest(BaseModel):
    training_id: str
    method: str  # INT8_QUANTIZATION, FLOAT16_QUANTIZATION, PRUNING
    sparsity_level: float = 0.5
    representative_dataset_size: int = 100

class BoardEvaluationRequest(BaseModel):
    training_id: str
    board: str

@router.post("/quantize")
async def quantize_model(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Quantize a trained model"""
    try:
        optimization_id = converter.create_optimization_session(
            training_id=request.training_id,
            method=request.method,
            sparsity_level=request.sparsity_level
        )
        
        # Run optimization in background
        background_tasks.add_task(converter.optimize, optimization_id)
        
        return {
            "status": "success",
            "optimization_id": optimization_id,
            "message": f"Optimization started with method: {request.method}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/status/{optimization_id}")
async def get_optimization_status(optimization_id: str):
    """Get the status of an optimization"""
    try:
        status = converter.get_optimization_status(optimization_id)
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/result/{optimization_id}")
async def get_optimization_result(optimization_id: str):
    """Get the result of an optimization"""
    try:
        result = converter.get_optimization_result(optimization_id)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/to-c-array/{optimization_id}")
async def export_as_c_array(optimization_id: str):
    """Export optimized model as C-array"""
    try:
        c_array = converter.generate_c_array(optimization_id)
        return {
            "status": "success",
            "c_array": c_array,
            "message": "C-array generated successfully"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/evaluate-board")
async def evaluate_for_board(request: BoardEvaluationRequest):
    """Evaluate an optimized model for a specific board"""
    try:
        recommendation = mcu_advisor.evaluate_model(
            optimization_id=request.training_id,
            board=request.board
        )
        return {
            "status": "success",
            "recommendation": recommendation
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/boards")
async def get_supported_boards():
    """Get list of supported boards with specifications"""
    try:
        boards = mcu_advisor.get_supported_boards()
        return {
            "status": "success",
            "boards": boards
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/llm-suggest")
async def get_llm_suggestions(training_id: str, metrics: dict):
    """Get LLM-powered suggestions for model improvement"""
    try:
        suggestions = llm_advisor.get_suggestions(
            training_id=training_id,
            metrics=metrics
        )
        return {
            "status": "success",
            "suggestions": suggestions
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/llm-optimize")
async def get_llm_optimization_advice(optimization_id: str, board: str):
    """Get LLM-powered advice on optimization and board deployment"""
    try:
        advice = llm_advisor.get_optimization_advice(
            optimization_id=optimization_id,
            board=board
        )
        return {
            "status": "success",
            "advice": advice
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
