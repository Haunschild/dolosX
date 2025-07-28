import streamlit as st
import openai
import os
import json
from dotenv import load_dotenv

# --- Configuration ---
st.set_page_config(page_title="Forensic Linguistic Analyzer", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# --- Initialization and Error Handling ---
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set. Please add it to your .env file or Streamlit secrets.")
    st.stop()
openai.api_key = OPENAI_API_KEY

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'active_filters' not in st.session_state:
    st.session_state.active_filters = []

# --- Core LLM Logic (Unchanged) ---

def create_forensic_prompt(transcript):
    """
    Creates a comprehensive prompt instructing the LLM to perform a line-by-line forensic analysis,
    considering BOTH the initial linguistic cues and the advanced narrative cues.
    """
    prompt = f"""
    You are a world-class forensic linguist. Your task is to analyze an insurance claim transcript by synthesizing basic linguistic cues with advanced narrative and psychological indicators. You will only comment on lines that contain potential deception cues.

    **Analysis Instructions:**
    1.  **Line-by-Line Analysis:** Process the transcript sequentially.
    2.  **Suspicion Score:** Assign a float from 0.0 (benign) to 1.0 (highly deceptive). A score of 0.0 means the line is completely normal.
    3.  **Conditional Commenting & Tagging:**
        -   **If a line is suspicious (score > 0.0):** You MUST provide a `reason` and a list of `cues_triggered` from the Official Cue List below. The reason should explain how the cues interact.
        -   **If a line is NOT suspicious (score = 0.0):** The `reason` MUST be an empty string (`""`) and `cues_triggered` MUST be an empty list (`[]`).

    **Official Cue List (Use these exact strings for tagging):**

    **Part 1: Foundational Linguistic Cues**
    - `Time`: General reference to time (e.g., "yesterday", "in an hour").
    - `Space`: Mentions of places or distances.
    - `Motion`: Words indicating movement (e.g., "go", "move", "drive").
    - `I-Pronouns`: Over or under-use of "I", "my", "me".
    - `Personal Pronouns`: Use of "we", "you", "they".
    - `Long Words (>6 letters)`: Proportion of longer, more formal words.
    - `Negations`: Words like "not", "no", "never".
    - `Focus: Future`: Statements about future events.
    - `Focus: Present`: Statements about the present moment.
    - `Focus: Past`: Statements about past events.
    - `Risk Language`: Language expressing uncertainty (e.g., "might", "risk", "maybe").
    - `Cognitive Process`: Words about thinking (e.g., "consider", "think", "understand").
    - `Sadness`: Language expressing sadness.
    - `Anger`: Language expressing anger.
    - `Anxiety`: Language expressing fear or insecurity.
    - `Negative Emotion`: General negative tone.
    - `Positive Emotion`: General positive tone.

    **Part 2: Advanced Narrative & Deception Cues**
    - `Narrative Imbalance`: Excessive detail in some areas, amnesia in others.
    - `Lack of Context`: Story lacks a natural prologue or epilogue.
    - `Passive Voice Usage`: Using passive voice to deflect agency (e.g., "the window was broken").
    - `High Cognitive Load`: High density of fillers ('um', 'uh'), hesitations, or stuttering.
    - `Question Evasion`: Deflecting, repeating, or not directly answering questions.
    - `Statement Against Interest`: The *absence* of minor, self-critical details which adds suspicion.
    - `Inappropriate Emotion`: The emotional tone does not match the described events.
    - `Overly Formal`: A sudden shift to overly formal or polite language.
    - `Contradiction`: Statement contradicts previous information.
    - `Vague Language`: Using non-specific words to avoid commitment.

    **Required Output Format (Strict JSON):**
    {{
      "overall_deception_probability": <Float>,
      "final_recommendation": "<'No Red Flags', 'Low Risk', 'Medium Risk', or 'High Risk'>",
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
        return None, f"An unexpected error occurred: {e}"

# --- UI & Styling Helper Functions ---

def score_to_heatmap_color(score):
    """Maps a score (0-1) to a green-yellow-orange-red heatmap color."""
    score = max(0, min(1, score))
    if score == 0.0:
        return "#b7e4c7"  # green
    elif score < 0.15:
        return "#fff9db"  # very light yellow
    elif score < 0.3:
        return "#ffe066"  # light yellow
    elif score < 0.45:
        return "#ffd166"  # yellow
    elif score < 0.6:
        return "#ffb347"  # light orange
    elif score < 0.75:
        return "#ff8800"  # orange
    elif score < 0.9:
        return "#ff704d"  # orange-red
    else:
        return "#ff4d4d"  # red

# --- Streamlit UI ---

st.title("V-Claim")

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    transcript_text = st.text_area("Paste transcript here:", height=150)
    
    st.header("Debug Tools")
    imported_file = st.file_uploader("Import LLM Analysis (JSON)", type=["json"])

    if st.button("Analyze Transcript", type="primary", use_container_width=True):
        st.session_state.analysis_result = None
        if imported_file:
            try:
                result = json.load(imported_file)
                st.session_state.analysis_result = result
                st.session_state.active_filters = []
                st.success("Successfully loaded analysis from file.")
            except Exception as e:
                st.error(f"Failed to read JSON file: {e}")
        elif transcript_text.strip():
            with st.spinner("Performing forensic analysis..."):
                result, error = analyze_transcript(transcript_text)
                if error: st.error(error)
                else:
                    st.session_state.analysis_result = result
                    st.session_state.active_filters = []
        else:
            st.warning("Please provide a transcript or import a file.")
    
    if st.session_state.analysis_result:
        json_string = json.dumps(st.session_state.analysis_result, indent=2)
        st.download_button(
            label="Export LLM Analysis (JSON)", data=json_string,
            file_name="llm_analysis.json", mime="application/json", use_container_width=True
        )

    if st.session_state.analysis_result:
        st.divider()
        st.subheader("Highlight Cues")
        all_cues = st.session_state.analysis_result.get("all_detected_cues", [])
        if all_cues:
            st.session_state.active_filters = st.multiselect(
                "Select cues to highlight:", options=all_cues, default=st.session_state.active_filters
            )

# --- Main Display Area ---
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    analyzed_lines = result.get("analyzed_transcript", [])
    active_filters = st.session_state.active_filters

    st.header("Analysis Dashboard", divider='rainbow')
    prob = result.get("overall_deception_probability", None)
    rec = result.get("final_recommendation", None)
    if prob is not None:
        st.markdown(f"**Overall Deception Probability:** {prob:.2%}")
    if rec:
        st.markdown(f"**Final Recommendation:** {rec}")
    summary = result.get("analysis_summary", None)
    if summary: st.markdown(f"**Summary:** {summary}")
    
    st.markdown("**Deception Timeline**")
    
    # --- FIXED: Proportional Timeline with Min/Max Width ---
    timeline_html = "<div style='display: flex; width: 100%; height: 20px; gap: 2px;'>"
    MIN_FLEX = 10 # Set a minimum width basis
    MAX_FLEX = 150 # Set a maximum width basis to prevent extreme dominance
    for line in analyzed_lines:
        score = line.get('suspicion_score', 0)
        color = score_to_heatmap_color(score)
        text_length = len(line.get('text', ''))
        # Clamp the flex value to ensure visibility
        flex = max(MIN_FLEX, min(MAX_FLEX, text_length))
        
        timeline_html += f"<a href='#line-{line.get('line_number', 0)}' style='flex-grow: {flex}; background-color: {color}; border-radius: 2px;' title='Line {line.get('line_number', 0)} | Score: {score:.2f}'></a>"
    timeline_html += "</div>"
    st.markdown(timeline_html, unsafe_allow_html=True)

    # --- FIXED: Two-Sided Interactive Chat with Reliable Popover ---
    st.subheader("Interactive Transcript")
    
    chat_container = st.container()

    for line in analyzed_lines:
        with chat_container:
            # Add invisible anchor div with negative offset for accurate scrolling
            st.markdown(f"<div id='line-{line.get('line_number')}' style='position: relative; top: -40px; height: 0; margin: 0; padding: 0;'></div>", unsafe_allow_html=True)
            speaker = line.get("speaker", "Unknown").lower()
            text = line.get("text", "")
            score = line.get("suspicion_score", 0)
            cues = line.get("cues_triggered", [])
            is_suspicious = score > 0
            is_filtered = any(cue in active_filters for cue in cues)
            
            # Create an invisible anchor for accurate scrolling
            st.markdown(f"<div id='line-{line.get('line_number')}'></div>", unsafe_allow_html=True)

            bubble_style = f"""
                border-radius: 20px;
                padding: 10px 15px;
                max-width: 100%; /* The column will handle the max-width */
                border: {'2px solid #00BFFF' if is_filtered else 'none'};
                margin-bottom: 10px;
                display: inline-block; /* Important for alignment */
            """
            
            if speaker == 'agent':
                # Agent bubble on the left
                bubble_style += f"background-color: #F1F3F5; color: #212529;"
                st.markdown(f"<div style='display: flex; justify-content: flex-start;'><div style='{bubble_style}'>{text}</div></div>", unsafe_allow_html=True)
            
            else: # Claimant bubble on the right
                bubble_color = score_to_heatmap_color(score)
                bubble_style += f"background-color: {bubble_color}; color: #212529;"
                
                cols = st.columns([1, 12, 1], gap="small") # Use columns for stable right-alignment
                with cols[1]: # The middle, expanding column
                    st.markdown(f"<div style='display: flex; justify-content: flex-end;'><div style='{bubble_style}'>{text}</div></div>", unsafe_allow_html=True)

                if is_suspicious:
                    with cols[2]: # The narrow column for the icon
                        with st.popover("❗"):
                            st.markdown(f"**Suspicion Score:** {score:.2f}")
                            st.markdown(f"**Reason:** {line.get('reason', '')}")
                            if cues: st.markdown("**Cues:**\n- " + "\n- ".join(cues))

else:
    st.info("Upload or paste a transcript and click 'Analyze Transcript' in the sidebar to begin.")