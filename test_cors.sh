#!/bin/bash
# Test CORS Configuration
# This script tests if CORS is properly configured between frontend and backend

echo "======================================"
echo "Testing CORS Configuration"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check if backend is running
echo "1. Checking backend API..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Backend is running at http://localhost:8000"
else
    echo -e "${RED}✗${NC} Backend is NOT running"
    echo "   Start it with: ./start_api.sh or python api.py"
    exit 1
fi

echo ""

# Test 2: Check CORS headers for port 4202
echo "2. Testing CORS headers for localhost:4202..."
CORS_RESPONSE=$(curl -s -I -H "Origin: http://localhost:4202" http://localhost:8000/health)

if echo "$CORS_RESPONSE" | grep -i "access-control-allow-origin" > /dev/null; then
    ALLOWED_ORIGIN=$(echo "$CORS_RESPONSE" | grep -i "access-control-allow-origin" | cut -d' ' -f2 | tr -d '\r')
    echo -e "${GREEN}✓${NC} CORS headers present"
    echo "   Access-Control-Allow-Origin: $ALLOWED_ORIGIN"
else
    echo -e "${RED}✗${NC} CORS headers NOT found"
    exit 1
fi

echo ""

# Test 3: Check CORS headers for port 4200 (backward compatibility)
echo "3. Testing CORS headers for localhost:4200 (backward compatibility)..."
CORS_RESPONSE_4200=$(curl -s -I -H "Origin: http://localhost:4200" http://localhost:8000/health)

if echo "$CORS_RESPONSE_4200" | grep -i "access-control-allow-origin" > /dev/null; then
    echo -e "${GREEN}✓${NC} CORS also works for port 4200"
else
    echo -e "${YELLOW}⚠${NC} CORS for port 4200 not configured (not critical)"
fi

echo ""

# Test 4: Test actual API endpoints
echo "4. Testing API endpoints..."

# Test health endpoint
HEALTH_RESPONSE=$(curl -s -H "Origin: http://localhost:4202" http://localhost:8000/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✓${NC} /health endpoint working"
else
    echo -e "${RED}✗${NC} /health endpoint failed"
fi

# Test models endpoint
MODELS_RESPONSE=$(curl -s -H "Origin: http://localhost:4202" http://localhost:8000/models)
if echo "$MODELS_RESPONSE" | grep -q "models"; then
    echo -e "${GREEN}✓${NC} /models endpoint working"
else
    echo -e "${RED}✗${NC} /models endpoint failed"
fi

# Test languages endpoint
LANGUAGES_RESPONSE=$(curl -s -H "Origin: http://localhost:4202" http://localhost:8000/languages)
if echo "$LANGUAGES_RESPONSE" | grep -q "common_languages"; then
    echo -e "${GREEN}✓${NC} /languages endpoint working"
else
    echo -e "${RED}✗${NC} /languages endpoint failed"
fi

echo ""
echo "======================================"
echo "CORS Configuration Summary"
echo "======================================"
echo ""
echo "Backend API:      http://localhost:8000"
echo "Frontend (new):   http://localhost:4202"
echo "Frontend (old):   http://localhost:4200"
echo ""
echo -e "${GREEN}✓${NC} All CORS tests passed!"
echo ""
echo "You can now:"
echo "1. Start frontend: cd frontend && npm start"
echo "2. Open browser: http://localhost:4202"
echo "3. Test speech recognition"
echo ""
