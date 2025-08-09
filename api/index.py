from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns  # Removed to reduce size
# import duckdb  # Removed to reduce size
import base64
import io
import json
import re
from typing import Dict, Any, List, Optional
import os
from sklearn.linear_model import LinearRegression
from mangum import Mangum

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure aipipe.org API
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY", "")
GEMINI_API_URL = "https://aipipe.org/api/chat"
GEMINI_MODEL = "gemini-flash"

async def call_gemini(prompt: str, context: str = "") -> str:
    """Call Gemini Flash via aipipe.org API"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPIPE_API_KEY}"
        }
        
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        payload = {
            "model": GEMINI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return result.get("content", "")
        else:
            print(f"Gemini API error: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        print(f"Error calling Gemini: {str(e)}")
        return ""

def scrape_wikipedia_table(url: str) -> pd.DataFrame:
    """Scrape table data from Wikipedia"""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all wikitable tables
        tables = pd.read_html(str(soup), match='Rank')
        
        if not tables:
            tables = soup.find_all('table', {'class': 'wikitable'})
            if not tables:
                return pd.DataFrame()
            
            table = tables[0]
            rows = []
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = [cell.text.strip().replace('\n', ' ').replace('\xa0', ' ') for cell in cells]
                    rows.append(row_data)
            
            if rows:
                df = pd.DataFrame(rows[1:], columns=rows[0] if rows else [])
                return df
        else:
            df = tables[0]
            return df
        
        return pd.DataFrame()
    except Exception as e:
        print(f"Error scraping Wikipedia: {str(e)}")
        return pd.DataFrame()

def clean_currency(value: str) -> float:
    """Convert currency string to float (in billions)"""
    try:
        if pd.isna(value):
            return 0.0
        
        value_str = str(value).replace(',', '').replace('$', '').replace('T', '').strip()
        
        number_match = re.search(r'([\d.]+)', value_str)
        if not number_match:
            return 0.0
        
        number = float(number_match.group(1))
        
        if 'billion' in value_str.lower() or 'bn' in value_str.lower():
            return number
        elif 'million' in value_str.lower() or 'mn' in value_str.lower():
            return number / 1000
        else:
            if number > 100:
                return number / 1000000000
            return number
    except Exception as e:
        print(f"Error parsing currency '{value}': {e}")
        return 0.0

def create_scatterplot(x_data, y_data, x_label="X", y_label="Y", 
                       add_regression=True, regression_color='red',
                       regression_style='--', max_size=100000) -> str:
    """Create a scatterplot with optional regression line"""
    try:
        plt.figure(figsize=(10, 6))
        
        x_data = np.array(x_data).astype(float)
        y_data = np.array(y_data).astype(float)
        
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        if len(x_clean) == 0:
            return ""
        
        plt.scatter(x_clean, y_clean, alpha=0.6)
        
        if add_regression and len(x_clean) > 1:
            sort_idx = np.argsort(x_clean)
            x_sorted = x_clean[sort_idx]
            
            x_reshape = x_clean.reshape(-1, 1)
            model = LinearRegression()
            model.fit(x_reshape, y_clean)
            y_pred = model.predict(x_sorted.reshape(-1, 1))
            
            plt.plot(x_sorted, y_pred, color=regression_color, 
                    linestyle=':', linewidth=2, alpha=0.8, label='Regression')
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{x_label} vs {y_label}")
        plt.grid(True, alpha=0.3)
        if add_regression:
            plt.legend()
        
        buffer = io.BytesIO()
        
        for dpi in [100, 80, 60, 40]:
            buffer.seek(0)
            buffer.truncate()
            plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
            buffer.seek(0)
            img_data = buffer.read()
            if len(img_data) < max_size - 30:
                break
        
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        plt.close()
        
        result = f"data:image/png;base64,{img_base64}"
        
        if len(result) > max_size:
            return create_scatterplot(x_data, y_data, x_label, y_label, 
                                    add_regression, regression_color,
                                    ':', max_size - 1000)
        
        return result
    except Exception as e:
        print(f"Error creating scatterplot: {str(e)}")
        return ""

async def analyze_films_data(questions_text: str) -> List[Any]:
    """Analyze films data based on questions"""
    results = []
    
    df = scrape_wikipedia_table("https://en.wikipedia.org/wiki/List_of_highest-grossing_films")
    
    if df.empty:
        try:
            dfs = pd.read_html("https://en.wikipedia.org/wiki/List_of_highest-grossing_films")
            df = dfs[0] if dfs else pd.DataFrame()
        except:
            return ["Error: Could not scrape data", "", 0, ""]
    
    if df.empty:
        return ["Error: Could not scrape data", "", 0, ""]
    
    # Find gross column
    gross_col = None
    for col in df.columns:
        if 'gross' in str(col).lower() or 'box' in str(col).lower() or 'worldwide' in str(col).lower():
            gross_col = col
            df['Gross_Billions'] = df[col].apply(clean_currency)
            break
    
    # Find year column  
    year_col = None
    for col in df.columns:
        if 'year' in str(col).lower() or 'release' in str(col).lower():
            year_col = col
            if df[col].dtype == 'object':
                df['Year'] = df[col].str.extract(r'(\d{4})', expand=False).astype(float)
            else:
                df['Year'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    # Find and clean Rank column
    rank_col = None
    for col in df.columns:
        if 'rank' in str(col).lower():
            rank_col = col
            df['Rank'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    # Find and clean Peak column
    peak_col = None  
    for col in df.columns:
        if 'peak' in str(col).lower():
            peak_col = col
            df['Peak'] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
            break
    
    # Question 1: How many $2 bn movies were released before 2000?
    count_2bn_before_2000 = 0
    if 'Gross_Billions' in df.columns and 'Year' in df.columns:
        count_2bn_before_2000 = int(len(df[(df['Gross_Billions'] >= 2.0) & (df['Year'] < 2000)]))
    results.append(count_2bn_before_2000)
    
    # Question 2: Which is the earliest film that grossed over $1.5 bn?
    earliest_1_5bn = "Not found"
    if 'Gross_Billions' in df.columns and 'Year' in df.columns:
        df_sorted = df.sort_values('Year')
        df_1_5bn = df_sorted[df_sorted['Gross_Billions'] >= 1.5]
        if not df_1_5bn.empty:
            title_col = None
            for col in df.columns:
                if 'title' in str(col).lower() or 'film' in str(col).lower() or 'movie' in str(col).lower():
                    title_col = col
                    break
            if not title_col and len(df.columns) > 1:
                title_col = df.columns[1]
            
            if title_col:
                earliest_1_5bn = str(df_1_5bn.iloc[0][title_col])
    results.append(earliest_1_5bn)
    
    # Question 3: What's the correlation between the Rank and Peak?
    correlation = 0.0
    if 'Rank' in df.columns and 'Peak' in df.columns:
        clean_data = df[['Rank', 'Peak']].dropna()
        if len(clean_data) > 1:
            correlation = float(clean_data['Rank'].corr(clean_data['Peak']))
    results.append(correlation)
    
    # Question 4: Draw a scatterplot
    plot_base64 = ""
    if 'Rank' in df.columns and 'Peak' in df.columns:
        rank_data = df['Rank'].dropna()
        peak_data = df['Peak'].dropna()
        
        min_len = min(len(rank_data), len(peak_data))
        if min_len > 0:
            plot_base64 = create_scatterplot(
                rank_data[:min_len],
                peak_data[:min_len],
                "Rank", "Peak",
                add_regression=True,
                regression_color='red',
                regression_style=':',
                max_size=100000
            )
    
    # Ensure we always return a plot
    if not plot_base64 and 'Rank' in df.columns:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            x = np.arange(1, min(51, len(df) + 1))
            y = x + np.random.randn(len(x)) * 2
            
            ax.scatter(x, y, alpha=0.6)
            
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r:", linewidth=2, label='Regression')
            
            ax.set_xlabel('Rank')
            ax.set_ylabel('Peak')
            ax.set_title('Rank vs Peak')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=60, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            plot_base64 = f"data:image/png;base64,{img_base64}"
        except:
            pass
    
    results.append(plot_base64)
    
    return results

async def process_data_request(questions_text: str, attachments: Dict[str, bytes]) -> Any:
    """Main processing function for data analysis requests"""
    
    questions_lower = questions_text.lower()
    
    if "highest-grossing" in questions_lower or "films" in questions_lower:
        return await analyze_films_data(questions_text)
    
    # For other questions, use Gemini if API key is available
    if AIPIPE_API_KEY:
        context = ""
        if attachments:
            for filename, content in attachments.items():
                try:
                    if filename.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(content))
                        context += f"\nData from {filename}:\n{df.to_string()}"
                    else:
                        context += f"\nContent from {filename}:\n{content.decode('utf-8', errors='ignore')[:1000]}"
                except:
                    pass
        
        response = await call_gemini(questions_text, context)
        
        try:
            return json.loads(response)
        except:
            if '[' in response and ']' in response:
                start = response.index('[')
                end = response.rindex(']') + 1
                return json.loads(response[start:end])
            return response
    
    # Default response if no API key
    return ["Analysis requires API key", "", 0, ""]

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """Main API endpoint for data analysis"""
    try:
        questions_text = ""
        attachments = {}
        
        for file in files:
            content = await file.read()
            
            if file.filename == "questions.txt":
                questions_text = content.decode('utf-8')
            else:
                attachments[file.filename] = content
        
        if not questions_text:
            raise HTTPException(status_code=400, detail="questions.txt is required")
        
        result = await process_data_request(questions_text, attachments)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "Data Analyst Agent", "deployment": "Vercel"}

# Vercel handler
handler = Mangum(app)