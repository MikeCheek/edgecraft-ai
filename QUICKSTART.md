# Quick Start Guide for EdgeCraft AI

## Prerequisites

- Python 3.10+
- Node.js 18+
- pip and npm

## Installation & Setup

### 1. Clone the Repository

```bash
cd edgecraft-ai
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the backend
cp .env.example .env
uvicorn app.main:app --port 8000
```

The backend will be available at `http://localhost:8000`.

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## Using EdgeCraft AI

### 1. **Data Collection**

- Select your task (e.g., IMAGE_CLASSIFICATION, KEYWORD_SPOTTING)
- Upload samples individually, or bulk-import a labeled ZIP with the visual folder→label mapping tool
- Manage classes and train/val/test splits from the Dataset Manager

### 2. **Model Training**

- Configure training parameters (epochs, batch size, learning rate) — or click **"✨ Suggest Optimal Config"** for an LLM-generated starting point based on your actual dataset and target board
- Choose CPU, GPU, or Auto for the training device
- Click "Start Training" to begin
- Monitor progress in real-time, including a live console log (auto-expands on failure with the full error, not just a generic message)
- Archive, cancel, or resume training sessions — navigating away or reloading mid-run reattaches to the live job automatically

### 3. **Optimization**

- Create variants (they queue behind any running job):
  - **Full INT8**: ~4× smaller, calibrated on real training samples (recommended for MCUs)
  - **Dynamic range** / **Float16**
  - **Pruning + fine-tune** and **Clustering + fine-tune**, followed by INT8 or dynamic quantization. Their gain shows in the compressed size.
- Each variant's report compares it with a float32 TFLite baseline (accuracy/F1, size, latency, per-class metrics, confusion matrix) and shows whether TensorFlow Lite Micro can run it
- Try the original and optimized models side by side with an upload, URL, webcam or microphone

### 4. **Models**

- Browse the Dataset → Model → Optimized Variant tree to jump straight into optimizing or deploying any artifact

### 5. **Deployment**

- Select your target board in the header (ESP32-S3, ESP32-CAM, Raspberry Pi Pico, Arduino Nano)
- Evaluate model compatibility (memory usage, deployment warnings)
- Configure camera pins (image models) or I2S microphone pins (audio models, with on-device MFCC) and an optional SPI status display
- Preview the generated `sketch.ino` live as you edit pins
- **Export Arduino Project** — a ready-to-flash zip (`model_data.h`, `sketch.ino`, README) — or export just the raw C-array if you're integrating into your own sketch

## Supported Tasks

- **IMAGE_CLASSIFICATION**: Classify images into categories
- **OBJECT_DETECTION**: Detect objects within images
- **VISUAL_WAKE_WORDS**: Binary person detection
- **KEYWORD_SPOTTING**: Detect spoken keywords
- **AUDIO_CLASSIFICATION**: Classify audio into categories

## Supported Boards

| Board                  | RAM       | Flash | Export Support                                |
| ---------------------- | --------- | ----- | --------------------------------------------- |
| ESP32-S3 N16R8         | 8MB PSRAM | 16MB  | Full Arduino export, optional external camera |
| ESP32-CAM (AI-Thinker) | ~PSRAM    | ~4MB  | Full Arduino export, integrated camera        |
| Raspberry Pi Pico 2 W  | 520KB     | 4MB   | Generic Serial test harness only              |
| Arduino Nano 33 BLE    | 256KB     | 1MB   | Generic Serial test harness only              |

## Advanced Features

### LLM-Powered Suggestions

EdgeCraft AI can use an LLM at three points in the workflow: pre-training config recommendations, post-training diagnosis, and board-specific deployment advice. Two providers are supported:

**OpenRouter** (cloud, no local install): set `OPENROUTER_API_KEY` in `backend/.env`, then pick a model (or type any model id) on the **Settings** page.

**Ollama** (fully offline):

```bash
# Install Ollama (https://ollama.ai)
# Pull a model
ollama pull phi3

# Enable in backend .env
OLLAMA_ENABLED=true
```

If neither is reachable, EdgeCraft AI falls back to rule-based suggestions — you'll always get an answer, and a real error message (not silent fake advice) if a configured provider fails.

### Docker Deployment

```bash
cp backend/.env.example backend/.env
docker compose up --build

# App: http://localhost (nginx proxies /api and /ws to the backend)
```

## Troubleshooting

### Backend won't start

- Check if port 8000 is available: `netstat -an | grep 8000`
- Verify Python 3.10+ is installed: `python --version`
- Try reinstalling dependencies: `pip install --upgrade -r requirements.txt`

### Frontend won't load

- Check if port 5173 is available
- If the backend runs elsewhere, set its URL on the Settings page
- Clear npm cache: `npm cache clean --force`
- Reinstall dependencies: `rm -rf node_modules && npm install`

### Models not training

- Ensure backend is running: `http://localhost:8000/api/health`
- Check the live job console (WebSocket log panel) for the full error traceback, not just the browser console
- Verify dataset samples are uploaded

### Out of memory

- Reduce batch size
- Use a smaller base model (`MobileNetV1_0.25` or `Custom3LayerCNN`)
- Apply more aggressive quantization

## Project Structure

```
edgecraft-ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── routers/             # API endpoints (datasets, training, optimization, inference, job_logs_ws)
│   │   ├── services/            # preprocessing, trainer, job_queue, optimizer, evaluator, exporter, mcu_advisor, ...
│   │   └── utils/                # Utilities (c_array_generator, mfcc_codegen, zip_processor)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/           # React components (ModelTrainer, OptimizationStudio, DeploymentPanel,
│   │   │                          #   ModelTree, TerminalLogPanel, DatasetManager/...)
│   │   ├── types/                # TypeScript types
│   │   ├── hooks/                # Custom hooks
│   │   └── context/              # Context providers
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```

## Next Steps

1. Train your first model
2. Review the real accuracy/size comparison in Optimization
3. Evaluate compatibility on your target board
4. Export the ready-to-flash Arduino project (or just the C-array)
5. Integrate into your microcontroller project

## Support & Documentation

- API Documentation: `http://localhost:8000/docs` (Swagger UI)
- GitHub Issues: [Report bugs or request features]
- Examples: Check `/examples` for complete project walkthroughs

## License

MIT License - See LICENSE file for details
