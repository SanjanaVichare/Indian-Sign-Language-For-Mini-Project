import streamlit as st
import cv2
import pickle
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="ISL Assistant", page_icon="🤟", layout="centered")
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #E894B2, #C680C6, #9E8ADE); }
h1, h2, h3, p { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("🤟 Hearing Impairment Assistant")
st.markdown("**Indian Sign Language · Speech to Sign · Live Detection**")

@st.cache_resource
def load_model():
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict['model']

model = load_model()

@st.cache_resource
def load_mediapipe():
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2
    hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
    drawing = mp.solutions.drawing_utils
    hands_module = mp.solutions.hands
    return hands, drawing, hands_module

hands, mp_drawing, mp_hands_module = load_mediapipe()

labels_dict = {i: chr(65+i) for i in range(26)}

tab1, tab2 = st.tabs(["📷 Camera Sign Detection", "🔤 Text → Sign"])

with tab1:
    st.subheader("Show a hand sign to your camera")
    img_file = st.camera_input("Take a photo of your hand sign")
    if img_file:
        img = Image.open(img_file)
        img_array = np.array(img)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        data_aux, x_, y_ = [], [], []
        results = hands.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img_array, hand_landmarks, mp_hands_module.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x); y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))
            if len(data_aux) == 42:
                predicted_char = model.predict([np.asarray(data_aux)])[0]
                st.image(img_array, use_column_width=True)
                st.success(f"✅ Detected Sign: **{predicted_char}**")
            else:
                st.image(img_array, use_column_width=True)
                st.warning("Could not read hand landmarks. Try again.")
        else:
            st.image(img_array, use_column_width=True)
            st.warning("No hand detected. Try better lighting.")

with tab2:
    st.subheader("Type a word or letter to see its ISL sign")
    isl_phrases = ['hello','hii','goodmorning','goodafternoon','goodevening',
        'goodnight','howareyou','iamfine','please','sorry',
        'thankyou','welcome','areyouokay','whatisyourage','whatisyourname']
    user_input = st.text_input("Enter text:", placeholder="e.g. hello or ABC")
    if user_input:
        phrase = user_input.lower().replace(" ", "")
        if phrase in isl_phrases:
            gif_path = f"isl_gif/{phrase}.gif"
            if os.path.exists(gif_path):
                st.image(gif_path, caption=f"ISL: {phrase}", use_column_width=True)
            else:
                st.error(f"GIF not found for: {phrase}")
        else:
            cols = st.columns(min(len(phrase), 6))
            for i, ch in enumerate(phrase):
                if ch.isalpha():
                    img_path = f"isl_alphabets/{ch.lower()}.jpeg"
                    with cols[i % len(cols)]:
                        if os.path.exists(img_path):
                            st.image(img_path, caption=ch.upper(), use_column_width=True)
                        else:
                            st.write(f"**{ch.upper()}**")