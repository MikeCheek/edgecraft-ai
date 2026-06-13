#!/bin/bash
# EdgeCraft AI Startup Script

echo "🚀 EdgeCraft AI - Local TinyML Studio"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check uv
echo "Checking uv installation..."
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv not found. Please install it first!${NC}"
    echo "Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo -e "${GREEN}✅ uv found: $(uv --version)${NC}"
echo ""

# Check Node.js
echo "Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Node.js found: $(node --version)${NC}"
echo ""

# Backend Setup
echo "Setting up backend..."
cd backend

# Create venv with uv if it doesn't exist (forcing Python 3.11)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with Python 3.11..."
    uv venv --python 3.11 venv
fi

# Activate venv
source venv/bin/activate || source venv/Scripts/activate 2>/dev/null

# Install dependencies using uv's lightning-fast pip
echo "Installing Python dependencies..."
uv pip install -r requirements.txt -q
echo -e "${GREEN}✅ Backend dependencies installed${NC}"

# Backend is ready
echo -e "${GREEN}✅ Backend ready${NC}"
echo ""

# Frontend Setup
echo "Setting up frontend..."
cd ../frontend

# Install npm dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install -q
fi
echo -e "${GREEN}✅ Frontend dependencies installed${NC}"
echo ""

echo -e "${GREEN}======================================"
echo "🎉 Setup Complete!"
echo "=====================================${NC}"
echo ""
echo -e "${YELLOW}To start the application:${NC}"
echo ""
echo "Terminal 1 - Backend:"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --reload --port 8000"
echo ""
echo "Terminal 2 - Frontend:"
echo "  cd frontend"
echo "  npm run dev"
echo ""
echo "Then open:"
echo "  Frontend: http://localhost:5173"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""