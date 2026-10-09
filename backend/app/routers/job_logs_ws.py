"""
app/routers/job_logs_ws.py
---------------------------
GET (ws) /ws/logs/{job_id}

Streams the live console output of a training or optimization job
(job_id = training_id or optimization_id) to the frontend: sends the
buffered history immediately on connect, then pushes new lines as they
happen. Purely additive/read-only - closing the connection has no effect
on the job itself.
"""

import hmac

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import config
from app.services.job_logs import job_log_broker

router = APIRouter()


@router.websocket("/ws/logs/{job_id}")
async def stream_job_logs(websocket: WebSocket, job_id: str):
    # HTTP middleware doesn't cover WebSockets; browsers can't set headers
    # on them, so the optional API token travels as ?api_key=.
    if config.API_TOKEN:
        provided = websocket.query_params.get("api_key", "")
        if not (provided and hmac.compare_digest(provided, config.API_TOKEN)):
            await websocket.close(code=4401)
            return
    await websocket.accept()

    # Send buffered history first so a client connecting mid-job (or
    # reconnecting after a network blip) immediately sees everything so far.
    for entry in job_log_broker.get_history(job_id):
        await websocket.send_json(entry)

    job_log_broker.subscribe(job_id, websocket)
    try:
        while True:
            # We don't expect the client to send anything meaningful - this
            # just keeps the connection open and lets us detect disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        job_log_broker.unsubscribe(job_id, websocket)
