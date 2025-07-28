# Fraud Detector

This Streamlit app uses a Large Language Model (LLM) to analyze transcribed insurance claim calls and detect potential fraud.

## Features
- Paste or upload a transcript of a call
- LLM highlights suspicious statements, gives a deception probability, and explains its reasoning
- Suspicious lines are highlighted in the transcript
- Easy to re-analyze or load new transcripts

## Installation
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install streamlit openai python-dotenv
   ```
3. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env and set your OPENAI_API_KEY
   ```

## Usage
Run the app with:
```bash
streamlit run app.py
```

## Notes
- Requires Python 3.8+
- Uses OpenAI GPT-4 via API (set your API key in `.env`)
- For demonstration only; do not use real customer data 
