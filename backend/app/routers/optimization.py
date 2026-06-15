from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict
from app.services.shared_state import converter
from app.services.mcu_advisor import MCUAdvisor
from app.services.llm_advisor import LLMAdvisor

router = APIRouter()
mcu_advisor = MCUAdvisor()
llm_advisor = LLMAdvisor()

# --- Request Schemas ---
class OptimizationRequest(BaseModel):
    training_id: str
    method: str
    sparsity_level: float = 0.5
    representative_dataset_size: int = 100

class BoardEvaluationRequest(BaseModel):
    optimization_id: str  # FIXED: Was training_id
    board: str

class LLMSuggestRequest(BaseModel):  # NEW: For 422 Fix
    training_id: str
    metrics: Dict

class LLMOptimizeRequest(BaseModel): # NEW: For 422 Fix
    optimization_id: str
    board: str
# -----------------------

@router.post("/quantize")
async def quantize_model(request: OptimizationRequest, background_tasks: BackgroundTasks):
    try:
        optimization_id = converter.create_optimization_session(
            training_id=request.training_id,
            method=request.method,
            sparsity_level=request.sparsity_level
        )
        background_tasks.add_task(converter.optimize, optimization_id)
        return {"status": "success", "optimization_id": optimization_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/status/{optimization_id}")
async def get_optimization_status(optimization_id: str):
    try:
        return {"status": "success", "data": converter.get_optimization_status(optimization_id)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/result/{optimization_id}")
async def get_optimization_result(optimization_id: str):
    try:
        return {"status": "success", "data": converter.get_optimization_result(optimization_id)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/to-c-array/{optimization_id}")
async def export_as_c_array(optimization_id: str):
    try:
        return {"status": "success", "c_array": converter.generate_c_array(optimization_id)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/evaluate-board")
async def evaluate_for_board(request: BoardEvaluationRequest):
    try:
        recommendation = mcu_advisor.evaluate_model(
            optimization_id=request.optimization_id,
            board=request.board
        )
        return {"status": "success", "recommendation": recommendation}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/boards")
async def get_supported_boards():
    try:
        return {"status": "success", "boards": mcu_advisor.get_supported_boards()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# FIXED: Now expects a JSON body (LLMSuggestRequest) instead of URL parameters
@router.post("/llm-suggest")
async def get_llm_suggestions(request: LLMSuggestRequest):
    try:
        suggestions = llm_advisor.get_suggestions(
            training_id=request.training_id,
            metrics=request.metrics
        )
        return {"status": "success", "suggestions": suggestions}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# FIXED: Now expects a JSON body (LLMOptimizeRequest)
@router.post("/llm-optimize")
async def get_llm_optimization_advice(request: LLMOptimizeRequest):
    try:
        advice = llm_advisor.get_optimization_advice(
            optimization_id=request.optimization_id,
            board=request.board
        )
        return {"status": "success", "advice": advice}
    except Exception as e:
        return {"status": "error", "message": str(e)}