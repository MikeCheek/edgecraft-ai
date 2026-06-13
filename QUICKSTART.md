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
uvicorn app.main:app --reload --port 8000
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
- Upload training samples via the Data Collection tab
- Label each sample appropriately

### 2. **Model Training**

- Configure training parameters (epochs, batch size, learning rate)
- Click "Start Training" to begin
- Monitor progress in real-time
- Receive LLM-powered suggestions for optimization

### 3. **Optimization**

- Select quantization method:
  - **INT8 Quantization**: 75% size reduction (recommended)
  - **FLOAT16 Quantization**: 50% size reduction (balanced)
  - **Pruning**: 35% size reduction (sparse)
- View optimization results and compression ratio

### 4. **Board Evaluation**

- Select your target board (ESP32, Raspberry Pi, Arduino)
- Evaluate model compatibility
- View memory usage and deployment warnings

### 5. **Export & Deploy**

- Export as C-array
- Copy to your microcontroller project
- Use in your embedded application

## Supported Tasks

- **IMAGE_CLASSIFICATION**: Classify images into categories
- **OBJECT_DETECTION**: Detect objects within images
- **VISUAL_WAKE_WORDS**: Binary person detection
- **KEYWORD_SPOTTING**: Detect spoken keywords
- **AUDIO_CLASSIFICATION**: Classify audio into categories

## Supported Boards

| Board                 | RAM   | Flash | Best For                             |
| --------------------- | ----- | ----- | ------------------------------------ |
| ESP32-S3 N16R8        | 8MB   | 16MB  | Powerful, resource-rich applications |
| Raspberry Pi Pico 2 W | 520KB | 4MB   | Moderate complexity models           |
| Arduino Nano 33 BLE   | 256KB | 1MB   | Ultra-lightweight models only        |

## Advanced Features

### LLM-Powered Suggestions

EdgeCraft AI can integrate with local LLM (Ollama) for intelligent suggestions:

```bash
# Install Ollama (https://ollama.ai)
# Pull a model
ollama pull neural-chat

# Enable in backend .env
OLLAMA_ENABLED=true
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up

# Backend: http://localhost:8000
# Frontend: http://localhost
```

## Troubleshooting

### Backend won't start

- Check if port 8000 is available: `netstat -an | grep 8000`
- Verify Python 3.10+ is installed: `python --version`
- Try reinstalling dependencies: `pip install --upgrade -r requirements.txt`

### Frontend won't load

- Check if port 5173 is available
- Clear npm cache: `npm cache clean --force`
- Reinstall dependencies: `rm -rf node_modules && npm install`

### Models not training

- Ensure backend is running: `http://localhost:8000/api/health`
- Check browser console for API errors
- Verify dataset samples are uploaded

### Out of memory

- Reduce batch size
- Use smaller base model (Custom3LayerCNN)
- Apply more aggressive quantization

## Project Structure

```
edgecraft-ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── routers/             # API endpoints
│   │   ├── services/            # ML services
│   │   └── utils/               # Utilities
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── types/               # TypeScript types
│   │   ├── hooks/               # Custom hooks
│   │   └── context/             # Context providers
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```

## Next Steps

1. Train your first model
2. Evaluate on your target board
3. Optimize for deployment
4. Export as C-array
5. Integrate into your microcontroller project

## Support & Documentation

- API Documentation: `http://localhost:8000/docs` (Swagger UI)
- GitHub Issues: [Report bugs or request features]
- Examples: Check `/examples` for complete project walkthroughs

## License

MIT License - See LICENSE file for details
