import streamlit as st
import openai
import os
import json
import math
from dotenv import load_dotenv

# --- Configuration ---
st.set_page_config(page_title="Forensic Linguistic Analyzer", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LOGIN_USER = os.getenv('LOGIN_USER', 'admin')
LOGIN_PASSWORD = os.getenv('LOGIN_PASSWORD', 'claim-x')

# --- Initialization and Error Handling ---
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set. Please add it to your .env file or Streamlit secrets.")
    st.stop()
openai.api_key = OPENAI_API_KEY

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'active_filters' not in st.session_state:
    st.session_state.active_filters = []

# --- Simple Login Logic ---
def login_form():
    st.title("Login Required")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == LOGIN_USER and password == LOGIN_PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password.")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_form()
    st.stop()

# --- Core LLM Logic ---

def create_forensic_prompt(transcript):
    """
    Creates a prompt instructing the LLM to perform ONLY a line-by-line forensic analysis.
    The overall score will be calculated separately by our application.
    """
    prompt = f"""
    You are a world-class forensic linguist. Your task is to analyze an insurance claim transcript by synthesizing basic linguistic cues with advanced narrative and psychological indicators. You will only comment on lines that contain potential deception cues. Your SOLE FOCUS is the line-by-line analysis.

    **Analysis Instructions:**
    1.  **Line-by-Line Analysis:** Process the transcript sequentially.
    2.  **Suspicion Score:** Assign a float from 0.0 (benign) to 1.0 (highly deceptive). A score of 0.0 means the line is completely normal.
    3.  **Conditional Commenting & Tagging:**
        -   **If a line is suspicious (score > 0.0):** You MUST provide a `reason` and a list of `cues_triggered` from the Official Cue List below.
        -   **If a line is NOT suspicious (score = 0.0):** The `reason` MUST be an empty string (`""`) and `cues_triggered` MUST be an empty list (`[]`).

    **Official Cue List (Use these exact strings for tagging):**
    `Time`, `Space`, `Motion`, `I-Pronouns`, `Personal Pronouns`, `Long Words (>6 letters)`, `Negations`, `Focus: Future`, `Focus: Present`, `Focus: Past`, `Risk Language`, `Cognitive Process`, `Sadness`, `Anger`, `Anxiety`, `Negative Emotion`, `Positive Emotion`, `Narrative Imbalance`, `Lack of Context`, `Passive Voice Usage`, `High Cognitive Load`, `Question Evasion`, `Statement Against Interest`, `Inappropriate Emotion`, `Overly Formal`, `Contradiction`, `Vague Language`

    **Required Output Format (Strict JSON):**
    You MUST produce a JSON object containing a brief summary and the detailed line-by-line analysis. Do NOT calculate an overall score yourself.
    {{
      "analysis_summary": "<A 2-3 sentence summary of key findings, referencing the cues.>",
      "all_detected_cues": ["<List of all unique cue strings found in the transcript>"],
      "analyzed_transcript": [
        {{
          "speaker": "<'Agent' or 'Claimant'>",
          "line_number": <Integer>,
          "text": "<The exact text of the line>",
          "suspicion_score": <Float>,
          "reason": "<Brief justification ONLY if score > 0.0, otherwise ''>",
          "cues_triggered": ["<List of cues from the Official Cue List ONLY if score > 0.0, otherwise []>"]
        }}
      ]
    }}

    **Transcript to Analyze:**
    ---
    {transcript}
    ---
    """
    return prompt

def analyze_transcript(transcript):
    """Sends transcript to LLM and returns parsed JSON data."""
    prompt = create_forensic_prompt(transcript)
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        response_content = response.choices[0].message.content
        analysis_result = json.loads(response_content)
        return analysis_result, None
    except Exception as e:
        return None, f"An error occurred during analysis: {e}"


# --- UI & Styling Helper Functions ---

def score_to_heatmap_color(score):
    """Maps a score (0-1) to a green-yellow-orange-red heatmap color."""
    score = max(0, min(1, score))
    if score == 0.0: return "#b7e4c7"
    elif score < 0.15: return "#fff9db"
    elif score < 0.3: return "#ffe066"
    elif score < 0.45: return "#ffd166"
    elif score < 0.6: return "#ffb347"
    elif score < 0.75: return "#ff8800"
    elif score < 0.9: return "#ff704d"
    else: return "#ff4d4d"

def calculate_overall_deception_probability(analyzed_transcript):
    """
    Calculates a nuanced overall deception probability based on line-by-line scores.
    """
    suspicious_scores = []
    claimant_line_count = 0

    for line in analyzed_transcript:
        if line.get("speaker", "").lower() == 'claimant':
            claimant_line_count += 1
            score = line.get("suspicion_score", 0.0)
            if score > 0:
                suspicious_scores.append(score)

    if not suspicious_scores: return 0.0
    if claimant_line_count == 0: return 0.0

    power = 2.0
    raw_deception_sum = sum([s ** power for s in suspicious_scores])

    scaling_factor = 4.0
    normalized_score = (raw_deception_sum / math.sqrt(claimant_line_count)) * scaling_factor

    probability = math.tanh(normalized_score)
    return probability

def get_recommendation_from_probability(probability):
    """Maps a probability score to a categorical risk recommendation."""
    if probability <= 0.01: return "No Red Flags" # Allow for tiny floating point values
    elif probability <= 0.35: return "Low Risk"
    elif probability <= 0.75: return "Medium Risk"
    else: return "High Risk"


# --- Streamlit UI ---

st.title("Dolos AI")

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    transcript_text = st.text_area("Paste transcript here:", height=150)
    
    st.header("Debug Tools")
    imported_file = st.file_uploader("Import LLM Analysis (JSON)", type=["json"])

    if st.button("Analyze Transcript", type="primary", use_container_width=True):
        st.session_state.analysis_result = None
        st.session_state.active_filters = []
        if imported_file:
            try:
                result = json.load(imported_file)
                st.session_state.analysis_result = result
                st.success("Successfully loaded analysis from file.")
            except Exception as e:
                st.error(f"Failed to read JSON file: {e}")
        elif transcript_text.strip():
            with st.spinner("Performing forensic analysis..."):
                result, error = analyze_transcript(transcript_text)
                if error: st.error(error)
                if result:
                    st.session_state.analysis_result = result
        else:
            st.warning("Please provide a transcript or import a file.")
    
    if st.session_state.analysis_result:
        json_string = json.dumps(st.session_state.analysis_result, indent=2)
        st.download_button(
            label="Export Analysis (JSON)", data=json_string,
            file_name="forensic_analysis.json", mime="application/json", use_container_width=True
        )

    if st.session_state.analysis_result:
        st.divider()
        st.subheader("Highlight Cues")
        all_cues = st.session_state.analysis_result.get("all_detected_cues", [])
        if all_cues and isinstance(all_cues, list):
            st.session_state.active_filters = st.multiselect(
                "Select cues to highlight:", options=sorted(all_cues), default=st.session_state.active_filters
            )

# --- Main Display Area ---
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    analyzed_lines = result.get("analyzed_transcript", [])
    
    # THIS IS THE CRITICAL SECTION:
    # We calculate the probability and recommendation here in Python,
    # completely ignoring any summary values from the LLM.
    prob = calculate_overall_deception_probability(analyzed_lines)
    rec = get_recommendation_from_probability(prob)
    
    # We add the calculated values to the session state so they are included
    # in the JSON export for a complete record.
    st.session_state.analysis_result['calculated_deception_probability'] = prob
    st.session_state.analysis_result['calculated_final_recommendation'] = rec

    active_filters = st.session_state.active_filters

    st.header("Analysis Dashboard", divider='rainbow')
    
    # Display the calculated values.
    st.markdown(f"**Overall Deception Probability:** `{prob:.2%}`")
    st.markdown(f"**Final Recommendation:** `{rec}`")
    summary = result.get("analysis_summary", None)
    if summary: st.markdown(f"**LLM Summary:** *{summary}*")

    st.markdown("**Deception Timeline**")
    
    timeline_html = "<div style='display: flex; width: 100%; height: 20px; gap: 2px;'>"
    if analyzed_lines:
        for line in analyzed_lines:
            score = line.get('suspicion_score', 0)
            color = score_to_heatmap_color(score)
            text_length = len(line.get('text', ''))
            flex = max(10, min(150, text_length))
            line_num = line.get('line_number', 0)
            timeline_html += f"<a href='#line-{line_num}' style='flex-grow: {flex}; background-color: {color}; border-radius: 2px;' title='Line {line_num} | Score: {score:.2f}'></a>"
    timeline_html += "</div>"
    st.markdown(timeline_html, unsafe_allow_html=True)

    st.subheader("Interactive Transcript")
    
    chat_container = st.container()

    for line in analyzed_lines:
        with chat_container:
            line_num = line.get('line_number')
            st.markdown(f"<div id='line-{line_num}' style='position: relative; top: -60px;'></div>", unsafe_allow_html=True)
            
            speaker = line.get("speaker", "Unknown").lower()
            text = line.get("text", "")
            score = line.get("suspicion_score", 0)
            cues = line.get("cues_triggered", [])
            is_suspicious = score > 0
            is_filtered = any(cue in active_filters for cue in cues)
            
            bubble_style = f"""
                border-radius: 20px; padding: 10px 15px; max-width: 100%;
                border: {'2px solid #00BFFF' if is_filtered else 'none'};
                margin-bottom: 10px; display: inline-block; word-wrap: break-word;
            """
            
            if speaker == 'agent':
                bubble_style += "background-color: #F1F3F5; color: #212529;"
                st.markdown(f"<div style='display: flex; justify-content: flex-start;'><div style='{bubble_style}'>{text}</div></div>", unsafe_allow_html=True)
            else:
                bubble_color = score_to_heatmap_color(score)
                bubble_style += f"background-color: {bubble_color}; color: #212529;"
                
                cols = st.columns([1, 12, 1], gap="small")
                with cols[1]:
                    st.markdown(f"<div style='display: flex; justify-content: flex-end;'><div style='{bubble_style}'>{text}</div></div>", unsafe_allow_html=True)

                if is_suspicious:
                    with cols[2]:
                        with st.popover("❗"):
                            st.markdown(f"**Suspicion Score:** `{score:.2f}`")
                            st.markdown(f"**Reason:** {line.get('reason', 'N/A')}")
                            if cues: st.markdown("**Cues:**\n- " + "\n- ".join(f"`{cue}`" for cue in cues))
else:
    st.info("Upload or paste a transcript and click 'Analyze Transcript' in the sidebar to begin.")