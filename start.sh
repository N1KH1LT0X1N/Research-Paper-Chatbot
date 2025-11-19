#!/bin/bash

# Quick Start Script for Research Paper Chatbot v2.0
# This script helps you get started quickly

set -e

echo "=================================="
echo "Research Paper Chatbot v2.0"
echo "Quick Start Setup"
echo "=================================="
echo ""

# Check Python version
echo "1️⃣  Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "❌ Python 3.11+ required. Found: $python_version"
    exit 1
fi
echo "✅ Python $python_version"

# Check if .env exists
echo ""
echo "2️⃣  Checking configuration..."
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from example..."
    cp .env.async.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your credentials:"
    echo "   - TWILIO_ACCOUNT_SID"
    echo "   - TWILIO_AUTH_TOKEN"
    echo "   - GEMINI_API_KEY"
    echo ""
    read -p "Press Enter after you've edited .env..."
else
    echo "✅ .env file exists"
fi

# Create virtual environment
echo ""
echo "3️⃣  Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Created virtual environment"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "4️⃣  Activating virtual environment..."
source venv/bin/activate
echo "✅ Activated"

# Install dependencies
echo ""
echo "5️⃣  Installing dependencies..."
pip install -r requirements-async.txt --quiet
echo "✅ Dependencies installed"

# Check Redis
echo ""
echo "6️⃣  Checking Redis..."
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "⚠️  Redis is not running"
    echo "   Start Redis with: redis-server"
    echo "   Or use Docker: docker run -d -p 6379:6379 redis:7-alpine"
    echo ""
    read -p "Press Enter when Redis is running..."
fi

# Run migration if old database exists
echo ""
echo "7️⃣  Checking for old database..."
if [ -f "whatsapp_bot.db" ]; then
    echo "📦 Old database found. Running migration..."
    python migrate_to_async.py
    echo "✅ Migration complete"
else
    echo "✅ No old database to migrate"
    # Create fresh database
    python migrate_to_async.py --fresh
fi

echo ""
echo "=================================="
echo "✅ SETUP COMPLETE!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Make sure Redis is running:"
echo "   redis-server"
echo ""
echo "2. Start the application:"
echo "   python async_research_bot.py"
echo ""
echo "3. (Optional) Start Celery workers in separate terminals:"
echo "   celery -A celery_worker worker --loglevel=info"
echo "   celery -A celery_worker beat --loglevel=info"
echo ""
echo "4. Configure Twilio webhook:"
echo "   - For local dev: Use ngrok to expose port 8000"
echo "     ngrok http 8000"
echo "   - Set Twilio webhook to: https://your-ngrok-url.ngrok.io/whatsapp"
echo ""
echo "5. Test by sending a WhatsApp message to your Twilio number:"
echo "   'help'"
echo ""
echo "For production deployment, see README-ASYNC.md"
echo ""
