# EdgeCraft AI - Local TinyML Studio

A comprehensive, private alternative to Edge Impulse for collecting data, training TinyML models, optimizing them for microcontrollers, and deploying as C-arrays.

## Tech Stack

- **Frontend:** React 18+, Vite, TypeScript, Tailwind CSS, Lucide React, Recharts
- **Backend:** Python 3.10+, FastAPI, TensorFlow 2.15+, TensorFlow Lite, OpenCV, Librosa
- **ML Tasks:** Image Classification, Visual Wake Words, Object Detection, Keyword Spotting, Audio Classification
- **Target Boards:** ESP32-S3 N16R8, Raspberry Pi Pico 2 W, Arduino Nano 33 BLE
- **LLM Integration:** Ollama for intelligent model recommendations

## Project Structure

```
edgecraft-ai/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── routers/      # API endpoints
│   │   ├── services/     # Core ML services
│   │   └── utils/        # Helper functions
│   ├── datasets/         # Sample data storage
│   └── requirements.txt
├── frontend/             # React frontend
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── types/        # TypeScript types
│   │   ├── hooks/        # Custom hooks
│   │   └── context/      # React context
│   └── package.json
└── README.md
```

## Quick Start

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Running the Application

```bash
# Terminal 1 - Backend
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173` and the API at `http://localhost:8000`.

## Features (Planned)

- ✅ Data collection from webcam/microphone
- ✅ Multi-task TensorFlow training pipeline
- ✅ Post-training optimization (quantization, pruning)
- ✅ C-array generation for microcontroller deployment
- ✅ Board-specific hardware advisor
- ✅ LLM-powered optimization suggestions
- ✅ Real-time training metrics visualization
- ✅ Multi-board target support

## License

MIT
