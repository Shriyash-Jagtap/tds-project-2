# Data Analyst Agent API

A FastAPI-based data analysis agent that uses Gemini Flash via aipipe.org to source, prepare, analyze, and visualize data.

## Features

- Web scraping (Wikipedia tables)
- DuckDB integration for large-scale data analysis
- Data visualization with matplotlib
- LLM-powered analysis using Gemini Flash via aipipe.org
- CSV and other file format support
- Automatic regression analysis and plotting
- Serverless deployment on Vercel

## Setup

### Prerequisites

1. Get your API key from [aipipe.org](https://aipipe.org)
2. Create a `.env` file from the example:
```bash
cp .env.example .env
```
3. Add your API key to `.env`:
```
AIPIPE_API_KEY=your_api_key_here
```

## Deployment Options

### Option 1: Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Option 2: Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

### Option 3: Vercel Deployment (Recommended)

#### Prerequisites
- Install [Vercel CLI](https://vercel.com/docs/cli): `npm i -g vercel`
- Create a [Vercel account](https://vercel.com/signup)

#### Deploy to Vercel

1. **Install Vercel CLI** (if not already installed):
```bash
npm install -g vercel
```

2. **Login to Vercel**:
```bash
vercel login
```

3. **Set up environment variables** in Vercel:
```bash
vercel env add AIPIPE_API_KEY
# Enter your aipipe.org API key when prompted
```

4. **Deploy to Vercel**:
```bash
vercel --prod
```

5. **Alternative: Deploy via GitHub**
   - Push your code to GitHub
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import your GitHub repository
   - Add environment variable: `AIPIPE_API_KEY = your_api_key_here`
   - Click "Deploy"

Your API will be deployed at: `https://your-project-name.vercel.app/api/`

#### Vercel Configuration Details

The project includes:
- `vercel.json` - Configures routing and function settings
- `api/index.py` - Serverless function endpoint
- Maximum function duration: 180 seconds (3 minutes)

#### Testing Vercel Deployment

```bash
# Replace with your Vercel URL
curl "https://your-project-name.vercel.app/api/" \
  -F "files=@test_questions.txt"
```

## API Usage

Send a POST request to `/api/` with:
- `questions.txt`: Required file containing analysis questions
- Additional files: Optional data files (CSV, etc.)

Example using curl:
```bash
curl "http://localhost:8000/api/" \
  -F "files=@questions.txt" \
  -F "files=@data.csv"
```

## Testing

Run the test script:
```bash
python test_api.py
```

## Response Format

The API returns responses based on the question format:
- Array format: Returns JSON array
- Dictionary format: Returns JSON object
- Mixed content: Analyzes with Gemini Flash

## Deployment Options

- **Local**: Run directly with Python
- **Docker**: Use provided Dockerfile
- **Cloud**: Deploy to any platform supporting Docker (AWS, GCP, Azure, Heroku, etc.)

## Environment Variables

- `AIPIPE_API_KEY`: **Required** - API key from aipipe.org for Gemini Flash access
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)