#!/bin/bash
# =============================================================================
# EdgeCraft AI — WSL2 Setup Script
# RTX 4050 / CUDA 12.6 / Ubuntu on WSL2
# Run this once from inside your WSL2 terminal:
#   bash setup.sh
# =============================================================================

set -e  # exit on first error

PYTHON_VERSION="3.10"
BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/backend" && pwd)"
FRONTEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/frontend" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}▸ $*${NC}"; }
success() { echo -e "${GREEN}✔ $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠ $*${NC}"; }
error()   { echo -e "${RED}✘ $*${NC}"; exit 1; }
header()  { echo -e "\n${BOLD}$*${NC}"; echo "$(echo "$*" | sed 's/./-/g')"; }

# =============================================================================
header "EdgeCraft AI — WSL2 Setup"
# =============================================================================

# ── 0. Sanity: must be inside WSL2 ───────────────────────────────────────────
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    error "This script must be run inside WSL2, not native Linux or Windows CMD."
fi

# ── 1. GPU passthrough check ──────────────────────────────────────────────────
header "1. Checking GPU passthrough"
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    success "GPU detected: $GPU_NAME"
else
    error "nvidia-smi not found or failed.\n  • Make sure your Windows NVIDIA driver is ≥ 525.\n  • Re-run: wsl --update  (in PowerShell as admin)\n  • Then reopen WSL and try again."
fi

# ── 2. System packages ────────────────────────────────────────────────────────
header "2. Installing system dependencies"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    curl \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0
success "System packages ready"

# ── 3. uv ────────────────────────────────────────────────────────────────────
header "3. Installing uv (Python package manager)"
if ! command -v uv &>/dev/null; then
    info "Downloading uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$PATH"
    success "uv installed"
else
    UV_VER=$(uv --version 2>&1)
    success "uv already installed ($UV_VER)"
fi

# Persist uv in PATH for future sessions
if ! grep -q 'uv' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

export UV_LINK_MODE=copy

# ── 4. Python 3.10 via uv ────────────────────────────────────────────────────
header "4. Installing Python $PYTHON_VERSION via uv"
uv python install "$PYTHON_VERSION"
success "Python $PYTHON_VERSION available"

# ── 5. Backend virtual environment ───────────────────────────────────────────
header "5. Setting up backend"
cd "$BACKEND_DIR"

if [ -d ".venv" ]; then
    warn ".venv already exists — recreating it cleanly"
    rm -rf .venv
fi

info "Creating venv with Python $PYTHON_VERSION..."
uv venv --python "$PYTHON_VERSION" .venv
success "Virtual environment created"

info "Installing standard Python dependencies (FastAPI, OpenCV, etc.)..."
# Install everything EXCEPT tensorflow first, so you can see if something is actually compiling
grep -v "tensorflow" requirements.txt > req_fast.txt
uv pip install -v --python .venv/bin/python -r req_fast.txt
success "Standard dependencies installed"

info "Downloading TensorFlow and CUDA wheels (~2 GB, please wait)..."
# Now install TensorFlow with verbose logs so you can watch the download progress
uv pip install -v --python .venv/bin/python "tensorflow[and-cuda]==2.17.1"
success "Backend dependencies completely installed"

# ── 6. CUDA library path fix ─────────────────────────────────────────────────
# tensorflow[and-cuda] installs CUDA libs as Python packages under site-packages.
# TF needs LD_LIBRARY_PATH to point at them. We write an activation hook so
# it's set automatically every time the venv is activated.
header "6. Configuring CUDA library paths"

VENV_ACTIVATE="$BACKEND_DIR/.venv/bin/activate"
HOOK_MARKER="# EdgeCraft CUDA path hook"

if ! grep -q "$HOOK_MARKER" "$VENV_ACTIVATE" 2>/dev/null; then
    cat >> "$VENV_ACTIVATE" << 'CUDA_HOOK'

# EdgeCraft CUDA path hook
# Points LD_LIBRARY_PATH at the CUDA/cuDNN wheels bundled by tensorflow[and-cuda]
_set_cuda_paths() {
    local PYTHON_BIN
    PYTHON_BIN="$(dirname "${BASH_SOURCE[0]}")/python"
    local NVIDIA_DIR
    NVIDIA_DIR="$(dirname "$("$PYTHON_BIN" -c 'import nvidia.cudnn; print(nvidia.cudnn.__file__)' 2>/dev/null || echo '')")"
    if [ -n "$NVIDIA_DIR" ]; then
        # Walk up two levels: nvidia/cudnn/__init__.py -> nvidia/ -> site-packages/nvidia/
        local SITE_NVIDIA
        SITE_NVIDIA="$(dirname "$NVIDIA_DIR")"
        # Collect all lib dirs under nvidia/ packages
        local CUDA_LIBS
        CUDA_LIBS="$(find "$SITE_NVIDIA" -name 'lib' -type d 2>/dev/null | tr '\n' ':')"
        export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
}
_set_cuda_paths
unset -f _set_cuda_paths
CUDA_HOOK
    success "CUDA path hook added to venv activation"
else
    success "CUDA path hook already present"
fi

# ── 7. Verify TensorFlow sees the GPU ────────────────────────────────────────
header "7. Verifying TensorFlow GPU detection"
GPU_CHECK=$("$BACKEND_DIR/.venv/bin/python" - << 'EOF'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Replicate the LD_LIBRARY_PATH fix inline for this check
import importlib.util, pathlib
try:
    import nvidia.cudnn
    nvidia_root = pathlib.Path(nvidia.cudnn.__file__).parent.parent
    lib_dirs = ':'.join(str(p) for p in nvidia_root.rglob('lib') if p.is_dir())
    if lib_dirs:
        old = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f"{lib_dirs}:{old}" if old else lib_dirs
except Exception:
    pass

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    details = tf.config.experimental.get_device_details(gpus[0])
    print(f"OK:{details.get('device_name', gpus[0].name)}")
else:
    print("NOGPU")
EOF
)

if [[ "$GPU_CHECK" == OK:* ]]; then
    success "TensorFlow detected GPU: ${GPU_CHECK#OK:}"
elif [[ "$GPU_CHECK" == "NOGPU" ]]; then
    warn "TensorFlow installed but GPU not detected."
    warn "Try opening a new WSL terminal and run:"
    warn "  cd backend && source .venv/bin/activate"
    warn "  python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
else
    warn "GPU check returned unexpected output: $GPU_CHECK"
fi

# ── 8. Frontend (Node / npm) ──────────────────────────────────────────────────
header "8. Setting up frontend"

# Install Node.js via nvm if not present
if ! command -v node &>/dev/null; then
    info "Node.js not found — installing via nvm..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    # shellcheck source=/dev/null
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    nvm install 20
    nvm use 20
    nvm alias default 20
else
    NODE_VER=$(node --version)
    success "Node.js $NODE_VER already installed"
fi

cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
    info "Installing npm dependencies..."
    npm install --silent
fi
success "Frontend dependencies installed"

# ── 9. Print launch instructions ─────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}============================================${NC}"
echo -e "${BOLD}${GREEN}  ✔  Setup complete!${NC}"
echo -e "${BOLD}${GREEN}============================================${NC}"
echo ""
echo -e "${BOLD}To launch the app, open two WSL terminals:${NC}"
echo ""
echo -e "${CYAN}Terminal 1 — Backend (GPU-accelerated):${NC}"
echo "  cd backend"
echo "  source .venv/bin/activate"
echo "  python -m uvicorn app.main:app --reload --port 8000"
echo ""
echo -e "${CYAN}Terminal 2 — Frontend:${NC}"
echo "  cd frontend"
echo "  npm run dev"
echo ""
echo -e "${CYAN}Then open in your Windows browser:${NC}"
echo "  Frontend : http://localhost:5173"
echo "  API docs : http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Tip: access your Windows files from WSL at /mnt/c/Users/...${NC}"
echo -e "${YELLOW}     access WSL files from Windows Explorer: \\\\wsl.localhost\\Ubuntu${NC}"