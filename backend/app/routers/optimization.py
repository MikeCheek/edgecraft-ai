from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict
from app.services.shared_state import converter
from app.services.mcu_advisor import MCUAdvisor
from app.services.llm_advisor import LLMAdvisor
from app.services.local_llm_advisor import LocalLLMAdvisor

router = APIRouter()
mcu_advisor = MCUAdvisor()
llm_advisor = LLMAdvisor()          # rule-based (always available)
local_llm = LocalLLMAdvisor()       # real Ollama-backed advisor


# --- Request Schemas ---
class OptimizationRequest(BaseModel):
    training_id: str
    method: str
    sparsity_level: float = 0.5
    representative_dataset_size: int = 100

class BoardEvaluationRequest(BaseModel):
    optimization_id: str
    board: str

class LLMSuggestRequest(BaseModel):
    training_id: str
    metrics: Dict
    use_local_llm: bool = False     # NEW: opt-in to real LLM

class LLMOptimizeRequest(BaseModel):
    optimization_id: str
    board: str
    use_local_llm: bool = False     # NEW: opt-in to real LLM
# -----------------------


@router.post("/quantize")
async def quantize_model(request: OptimizationRequest, background_tasks: BackgroundTasks):
    try:
        optimization_id = converter.create_optimization_session(
            training_id=request.training_id,
            method=request.method,
            sparsity_level=request.sparsity_level,
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
            board=request.board,
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


@router.get("/llm-status")
async def get_llm_status():
    """Check whether a local Ollama LLM is available."""
    try:
        return {"status": "success", "llm": local_llm.get_status()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/llm-suggest")
async def get_llm_suggestions(request: LLMSuggestRequest):
    """
    Return training improvement suggestions.
    When use_local_llm=True the request is sent to Ollama; otherwise
    the deterministic rule-based advisor is used.
    """
    try:
        if request.use_local_llm:
            suggestions = local_llm.get_training_suggestions(metrics=request.metrics)
        else:
            suggestions = llm_advisor.get_suggestions(
                training_id=request.training_id,
                metrics=request.metrics,
            )
        return {"status": "success", "suggestions": suggestions}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/llm-optimize")
async def get_llm_optimization_advice(request: LLMOptimizeRequest):
    """
    Return board-specific optimisation advice.
    When use_local_llm=True the request is sent to Ollama.
    """
    try:
        if request.use_local_llm:
            # Resolve model size from converter state when possible
            opt_data = converter.get_optimization_result(request.optimization_id) or {}
            size_kb = (opt_data.get("optimized_size_bytes", 0) or
                       opt_data.get("original_size_bytes", 0)) // 1024
            advice = local_llm.get_optimization_advice(
                board=request.board,
                model_size_kb=size_kb,
            )
        else:
            advice = llm_advisor.get_optimization_advice(
                optimization_id=request.optimization_id,
                board=request.board,
            )
        return {"status": "success", "advice": advice}
    except Exception as e:
        return {"status": "error", "message": str(e)}