import streamlit as st
import cv2
import pickle
import numpy as np
from PIL import Image
import os
import base64

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Hearing Impairment Assistant",
    page_icon="🤟",
    layout="centered"
)

# ===============================
# DESIGN TOKENS (matching main1.py exactly)
# ===============================
BG_TOP  = (232, 148, 178)   # warm rose
BG_MID  = (198, 128, 198)   # mauve
BG_BOT  = (158, 138, 222)   # soft lavender

PILL_BG  = "#FFFFFF"
PILL_FG  = "#2B7FFF"

TITLE_FG = "#FFFFFF"
SUB_FG   = "#F2E6FF"
DIM_FG   = "#D0C0E8"

LOG_BG   = "#F7F0FF"
LOG_FG   = "#2D1F4E"
LOG_DIM  = "#9880B8"
LOG_BOR  = "#DDD0F0"

# ===============================
# GLOBAL CSS — faithful to main1.py
# ===============================
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

  /* ── Background gradient + blobs */
  .stApp {{
    background:
      radial-gradient(ellipse 520px 340px at 12% 18%, rgba(245,120,155,0.35) 0%, transparent 70%),
      radial-gradient(ellipse 440px 280px at 82% 14%, rgba(175,95,215,0.30) 0%, transparent 70%),
      radial-gradient(ellipse 400px 260px at 58% 78%, rgba(130,125,235,0.28) 0%, transparent 70%),
      radial-gradient(ellipse 340px 220px at 22% 82%, rgba(215,145,185,0.24) 0%, transparent 70%),
      radial-gradient(ellipse 300px 200px at 70% 45%, rgba(200,110,210,0.20) 0%, transparent 70%),
      linear-gradient(180deg,
        rgb({BG_TOP[0]},{BG_TOP[1]},{BG_TOP[2]}) 0%,
        rgb({BG_MID[0]},{BG_MID[1]},{BG_MID[2]}) 50%,
        rgb({BG_BOT[0]},{BG_BOT[1]},{BG_BOT[2]}) 100%
      );
    min-height: 100vh;
    font-family: 'DM Sans', sans-serif;
  }}

  /* hide default streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 740px;
  }}

  /* ── TITLE BLOCK */
  .hero-title {{
    text-align: center;
    margin-bottom: 0.2rem;
  }}
  .hero-title h1 {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: {TITLE_FG} !important;
    letter-spacing: -0.5px;
    margin: 0;
    text-shadow: 0 2px 16px rgba(0,0,0,0.18);
  }}
  .hero-title p {{
    color: {SUB_FG} !important;
    font-size: 0.92rem;
    margin: 0.3rem 0 0;
    letter-spacing: 0.3px;
  }}

  /* ── STATUS PILL */
  .status-pill {{
    display: flex;
    justify-content: center;
    margin: 0.9rem 0 1.6rem;
  }}
  .status-pill span {{
    background: {PILL_BG};
    color: #888;
    font-size: 0.82rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 6px 22px;
    border: 1px solid {LOG_BOR};
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    letter-spacing: 0.2px;
  }}

  /* ── PILL BUTTONS ROW */
  .pill-row {{
    display: flex;
    gap: 18px;
    justify-content: center;
    margin-bottom: 1.8rem;
    flex-wrap: wrap;
  }}
  .pill-btn {{
    background: {PILL_BG};
    color: {PILL_FG};
    border: none;
    border-radius: 999px;
    padding: 14px 34px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 4px 0 #C0C0C0, 0 6px 18px rgba(0,0,0,0.12);
    transition: transform 0.08s, box-shadow 0.08s, background 0.12s;
    letter-spacing: 0.1px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }}
  .pill-btn:hover {{
    background: #EEF4FF;
    transform: translateY(-1px);
    box-shadow: 0 5px 0 #B0B0B0, 0 8px 22px rgba(0,0,0,0.14);
  }}
  .pill-btn:active {{
    transform: translateY(2px);
    box-shadow: 0 2px 0 #C0C0C0, 0 2px 8px rgba(0,0,0,0.10);
  }}
  .pill-btn.danger {{ color: #E03060; }}

  /* ── FROSTED LOG CARD */
  .log-card {{
    background: {LOG_BG};
    border: 1px solid {LOG_BOR};
    border-radius: 18px;
    padding: 0 0 12px;
    margin-top: 0.5rem;
    box-shadow: 0 4px 24px rgba(80,40,120,0.10);
  }}
  .log-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px 10px;
    border-bottom: 1px solid {LOG_BOR};
    margin-bottom: 2px;
  }}
  .log-header .log-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #7050A8;
    letter-spacing: 0.3px;
  }}
  .log-header .log-clear {{
    font-size: 0.78rem;
    color: {PILL_FG};
    cursor: pointer;
    font-weight: 600;
  }}
  .log-entries {{
    padding: 6px 20px 0;
    max-height: 200px;
    overflow-y: auto;
  }}
  .log-row {{
    display: flex;
    gap: 10px;
    align-items: baseline;
    padding: 3px 0;
    border-bottom: 1px solid #EDE0FF;
    font-size: 0.84rem;
    line-height: 1.5;
  }}
  .log-row:last-child {{ border-bottom: none; }}
  .log-ts {{
    color: {LOG_DIM};
    font-size: 0.75rem;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
    flex-shrink: 0;
  }}
  .log-msg {{ color: {LOG_FG}; }}
  .log-msg.ok  {{ color: #207840; }}
  .log-msg.warn {{ color: #D04040; }}
  .log-msg.blue {{ color: {PILL_FG}; }}
  .log-msg.dim  {{ color: {LOG_DIM}; }}

  /* ── TABS */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: transparent;
    border-bottom: 2px solid rgba(255,255,255,0.25);
    padding-bottom: 0;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: rgba(255,255,255,0.18) !important;
    border-radius: 999px 999px 0 0 !important;
    color: {SUB_FG} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 8px 22px !important;
    border: none !important;
    transition: background 0.15s;
  }}
  .stTabs [aria-selected="true"] {{
    background: rgba(255,255,255,0.42) !important;
    color: #2D1F4E !important;
  }}
  .stTabs [data-baseweb="tab-panel"] {{
    background: rgba(247,240,255,0.82);
    border-radius: 0 18px 18px 18px;
    padding: 1.4rem 1.6rem;
    border: 1px solid {LOG_BOR};
    box-shadow: 0 4px 24px rgba(80,40,120,0.10);
  }}

  /* ── SECTION HEADER inside tabs */
  .section-hd {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #3D1F6E;
    margin-bottom: 0.8rem;
    letter-spacing: -0.2px;
  }}

  /* ── RESULT BADGE */
  .result-badge {{
    display: inline-block;
    background: linear-gradient(135deg, #2B7FFF, #9B4DCA);
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 12px 40px;
    margin: 1rem auto;
    box-shadow: 0 4px 18px rgba(43,127,255,0.30);
    text-align: center;
    display: block;
    width: fit-content;
  }}

  /* ── WARNING / SUCCESS */
  .msg-warn {{
    background: #FFF0F0; color: #D04040;
    border: 1px solid #F5CACA;
    border-radius: 10px; padding: 10px 16px;
    font-size: 0.85rem; font-weight: 600;
    margin-top: 0.6rem;
  }}
  .msg-ok {{
    background: #F0FFF4; color: #207840;
    border: 1px solid #B8EAC8;
    border-radius: 10px; padding: 10px 16px;
    font-size: 0.85rem; font-weight: 600;
    margin-top: 0.6rem;
  }}

  /* ── INPUT OVERRIDE */
  .stTextInput > div > div > input {{
    border-radius: 999px !important;
    border: 1.5px solid {LOG_BOR} !important;
    background: white !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 10px 20px !important;
    color: #2D1F4E !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  .stTextInput > div > div > input:focus {{
    border-color: {PILL_FG} !important;
    box-shadow: 0 0 0 3px rgba(43,127,255,0.15) !important;
  }}

  /* ── CAMERA INPUT */
  .stCameraInput label {{ color: {LOG_FG} !important; font-weight: 600; }}

  /* ── SIGN GRID */
  .sign-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
    margin-top: 1rem;
  }}
  .sign-cell {{
    background: white;
    border-radius: 14px;
    border: 1.5px solid {LOG_BOR};
    padding: 8px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(80,40,120,0.08);
    min-width: 90px;
  }}
  .sign-cell .letter-label {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: {PILL_FG};
    margin-top: 4px;
  }}

  /* ── FOOTER */
  .footer-txt {{
    text-align: center;
    color: {DIM_FG};
    font-size: 0.78rem;
    margin-top: 2rem;
    letter-spacing: 0.3px;
  }}

  /* scrollbar in log entries */
  .log-entries::-webkit-scrollbar {{ width: 5px; }}
  .log-entries::-webkit-scrollbar-track {{ background: #EAD8F8; border-radius: 4px; }}
  .log-entries::-webkit-scrollbar-thumb {{ background: #C8A8E8; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)


# ===============================
# SESSION STATE — activity log
# ===============================
if "log" not in st.session_state:
    from datetime import datetime
    st.session_state.log = [
        {"ts": datetime.now().strftime("%H:%M:%S"), "msg": "System ready.", "tag": "ok"}
    ]

def add_log(msg, tag=""):
    from datetime import datetime
    st.session_state.log.append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "msg": msg,
        "tag": tag
    })


# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    if not os.path.exists('./model.p'):
        return None
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict['model']

model = load_model()


# ===============================
# LOAD MEDIAPIPE
# ===============================
@st.cache_resource
def load_mediapipe():
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
    drawing = mp.solutions.drawing_utils
    hands_module = mp.solutions.hands
    return hands, drawing, hands_module

hands, mp_drawing, mp_hands_module = load_mediapipe()


# ===============================
# ISL DATA
# ===============================
ISL_PHRASES = [
    'hello','hii','goodmorning','goodafternoon','goodevening',
    'goodnight','howareyou','iamfine','please','sorry',
    'thankyou','welcome','areyouokay','whatisyourage','whatisyourname'
]


# ===============================
# HERO TITLE
# ===============================
st.markdown("""
<div class="hero-title">
  <h1>🤟 Hearing Impairment Assistant</h1>
  <p>Indian Sign Language &nbsp;·&nbsp; Speech to Sign &nbsp;·&nbsp; Live Detection</p>
</div>
""", unsafe_allow_html=True)


# ===============================
# STATUS PILL
# ===============================
st.markdown("""
<div class="status-pill">
  <span>● Idle</span>
</div>
""", unsafe_allow_html=True)


# ===============================
# TABS — styled to match pill buttons
# ===============================
tab1, tab2 = st.tabs(["📷  Camera Sign Detection", "🔤  Text → Sign"])


# ───────────────────────────────
# TAB 1 — Camera
# ───────────────────────────────
with tab1:
    st.markdown('<div class="section-hd">Show a hand sign to your camera</div>', unsafe_allow_html=True)

    img_file = st.camera_input("", label_visibility="collapsed")

    if img_file:
        add_log("Photo captured — processing…", "blue")

        img = Image.open(img_file)
        img_array = np.array(img)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        results = hands.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

        data_aux, x_, y_ = [], [], []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_array, hand_landmarks,
                    mp_hands_module.HAND_CONNECTIONS
                )
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x); y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            st.image(img_array, use_container_width=True)

            if len(data_aux) == 42 and model is not None:
                predicted_char = model.predict([np.asarray(data_aux)])[0]
                st.markdown(
                    f'<div class="result-badge">✅ Detected Sign: {predicted_char}</div>',
                    unsafe_allow_html=True
                )
                add_log(f'✦  Detected: "{predicted_char}"', "ok")
            elif model is None:
                st.markdown('<div class="msg-warn">⚠ model.p not found — place it in the app directory.</div>', unsafe_allow_html=True)
                add_log("model.p not found.", "warn")
            else:
                st.markdown('<div class="msg-warn">Could not read hand landmarks cleanly. Try again with better framing.</div>', unsafe_allow_html=True)
                add_log("Landmark extraction failed.", "warn")
        else:
            st.image(img_array, use_container_width=True)
            st.markdown('<div class="msg-warn">No hand detected. Try better lighting or move closer.</div>', unsafe_allow_html=True)
            add_log("No hand detected in photo.", "warn")


# ───────────────────────────────
# TAB 2 — Text → Sign
# ───────────────────────────────
with tab2:
    st.markdown('<div class="section-hd">Type a word or letter to see its ISL sign</div>', unsafe_allow_html=True)

    user_input = st.text_input(
        "", placeholder="e.g.  hello  or  ABC  or  thankyou",
        label_visibility="collapsed"
    )

    if user_input:
        phrase = user_input.lower().replace(" ", "")
        add_log(f'🔤  Input: "{user_input}"', "blue")

        # ── Whole-phrase GIF
        if phrase in ISL_PHRASES:
            gif_path = f"isl_gif/{phrase}.gif"
            if os.path.exists(gif_path):
                add_log(f"▶  ISL phrase: {phrase}", "ok")

                # Read and encode gif for display
                with open(gif_path, "rb") as f:
                    gif_bytes = f.read()
                b64 = base64.b64encode(gif_bytes).decode()
                st.markdown(
                    f'''<div style="text-align:center;margin-top:0.8rem;">
                          <img src="data:image/gif;base64,{b64}"
                               style="max-width:320px;border-radius:18px;
                                      box-shadow:0 4px 24px rgba(80,40,120,0.18);" />
                          <div style="color:{LOG_DIM};font-size:0.8rem;margin-top:6px;">
                            ISL phrase: <strong style="color:{PILL_FG};">{phrase}</strong>
                          </div>
                        </div>''',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f'<div class="msg-warn">⚠ GIF not found: isl_gif/{phrase}.gif</div>', unsafe_allow_html=True)
                add_log(f"GIF missing: {phrase}.gif", "warn")

        # ── Spell letter-by-letter
        else:
            add_log(f"🔤  Spelling: {phrase}", "")
            letters = [ch for ch in phrase if ch.isalpha()]

            if letters:
                cells_html = ""
                for ch in letters:
                    img_path = f"isl_alphabets/{ch.lower()}.jpeg"
                    if os.path.exists(img_path):
                        with open(img_path, "rb") as f:
                            img_b64 = base64.b64encode(f.read()).decode()
                        cells_html += f"""
                        <div class="sign-cell">
                          <img src="data:image/jpeg;base64,{img_b64}"
                               style="width:80px;height:80px;object-fit:cover;border-radius:10px;" />
                          <div class="letter-label">{ch.upper()}</div>
                        </div>"""
                    else:
                        cells_html += f"""
                        <div class="sign-cell">
                          <div style="width:80px;height:80px;display:flex;align-items:center;
                                      justify-content:center;font-size:2rem;color:#C8A8E8;">?</div>
                          <div class="letter-label">{ch.upper()}</div>
                        </div>"""

                st.markdown(
                    f'<div class="sign-grid">{cells_html}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div class="msg-warn">Please enter alphabetic characters.</div>', unsafe_allow_html=True)


# ===============================
# ACTIVITY LOG — frosted card
# ===============================
st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

clear_col, _ = st.columns([1, 4])
with clear_col:
    if st.button("Clear Log", key="clear_log"):
        st.session_state.log = []
        st.rerun()

log_rows_html = ""
for entry in reversed(st.session_state.log[-30:]):
    tag_class = entry.get("tag", "")
    log_rows_html += f"""
    <div class="log-row">
      <span class="log-ts">[{entry['ts']}]</span>
      <span class="log-msg {tag_class}">{entry['msg']}</span>
    </div>"""

st.markdown(f"""
<div class="log-card">
  <div class="log-header">
    <span class="log-title">Activity Log</span>
  </div>
  <div class="log-entries">
    {log_rows_html if log_rows_html else '<div style="color:#9880B8;font-size:0.82rem;padding:8px 0;">No entries yet.</div>'}
  </div>
</div>
""", unsafe_allow_html=True)


# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="footer-txt">
  Powered by MediaPipe &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)