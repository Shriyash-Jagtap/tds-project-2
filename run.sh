#!/bin/bash

# Check if AIPIPE_API_KEY is set
if [ -z "$AIPIPE_API_KEY" ]; then
    echo "Error: AIPIPE_API_KEY is not set!"
    echo "Please set it in .env file or export it:"
    echo "  export AIPIPE_API_KEY=your_key_here"
    echo ""
    echo "Get your API key from: https://aipipe.org"
    exit 1
fi

echo "Starting Data Analyst Agent API..."
echo "API Key configured: ${AIPIPE_API_KEY:0:10}..."
echo "Server will be available at: http://localhost:8000"
echo ""

python main.py