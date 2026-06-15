#!/bin/bash

# Define colors for clean terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting EdgeCraft AI export process...${NC}\n"

# ----------------------------------------
# 1. Process Frontend
# ----------------------------------------
if [ -d "frontend/src" ]; then
    echo -e "${YELLOW}📦 Zipping frontend/src/ into frontend.zip...${NC}"
    # -r makes it recursive, -q makes it quiet (no spamming the console)
    zip -r frontend.zip frontend/src/ -q
    
    echo -e "${YELLOW}📄 Converting frontend.zip to frontend.md using markitdown...${NC}"
    markitdown frontend.zip > frontend.md
    
    echo -e "${GREEN}✅ Frontend processed successfully!${NC}\n"
else
    echo -e "${RED}❌ Directory frontend/src/ not found! Skipping frontend...${NC}\n"
fi

# ----------------------------------------
# 2. Process Backend
# ----------------------------------------
if [ -d "backend/app" ]; then
    echo -e "${YELLOW}📦 Zipping backend/app/ into backend.zip...${NC}"
    zip -r backend.zip backend/app/ -q
    
    echo -e "${YELLOW}📄 Converting backend.zip to backend.md using markitdown...${NC}"
    markitdown backend.zip > backend.md
    
    echo -e "${GREEN}✅ Backend processed successfully!${NC}\n"
else
    echo -e "${RED}❌ Directory backend/app/ not found! Skipping backend...${NC}\n"
fi

echo -e "${GREEN}🎉 All exports complete! Check your directory for the new .md files.${NC}"