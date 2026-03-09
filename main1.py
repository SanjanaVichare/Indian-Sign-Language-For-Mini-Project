import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image, ImageTk, ImageDraw, ImageFilter
from itertools import count
import tkinter as tk
import string
import pickle
import mediapipe as mp
from collections import deque
import threading
import time


# ===============================
# LOAD SIGN DETECTION MODEL
# ===============================
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']


# ===============================
# DESIGN TOKENS
# ===============================
BG_TOP   = (232, 148, 178)   # warm rose
BG_MID   = (198, 128, 198)   # mauve
BG_BOT   = (158, 138, 222)   # soft lavender

PILL_BG  = "#FFFFFF"
PILL_FG  = "#2B7FFF"
PILL_HOV = "#EEF4FF"

TITLE_FG = "#FFFFFF"
SUB_FG   = "#F2E6FF"
DIM_FG   = "#D0C0E8"

LOG_BG   = "#F7F0FF"
LOG_FG   = "#2D1F4E"
LOG_DIM  = "#9880B8"
LOG_BOR  = "#DDD0F0"

WIN_W = 760
WIN_H = 570


# ===============================
# IMAGE LABEL FOR GIF
# ===============================
class ImageLabel(tk.Label):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.frames = []
        self.loc = 0
        self.delay = 100

    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        self.frames = []
        self.loc = 0
        try:
            while True:
                self.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(len(self.frames))
        except EOFError:
            pass
        self.delay = im.info.get("duration", 100)
        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()

    def next_frame(self):
        if not self.winfo_exists():
            return

        if self.frames:
            self.loc = (self.loc + 1) % len(self.frames)
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)


# ===============================
# GRADIENT + BLOB BACKGROUND
# ===============================
def make_background(w, h):
    img = Image.new("RGB", (w, h))
    px = img.load()
    half = h // 2
    for y in range(h):
        if y < half:
            t = y / half
            r = int(BG_TOP[0] + (BG_MID[0]-BG_TOP[0])*t)
            g = int(BG_TOP[1] + (BG_MID[1]-BG_TOP[1])*t)
            b = int(BG_TOP[2] + (BG_MID[2]-BG_TOP[2])*t)
        else:
            t = (y-half)/(h-half)
            r = int(BG_MID[0] + (BG_BOT[0]-BG_MID[0])*t)
            g = int(BG_MID[1] + (BG_BOT[1]-BG_MID[1])*t)
            b = int(BG_MID[2] + (BG_BOT[2]-BG_MID[2])*t)
        for x in range(w):
            px[x, y] = (r, g, b)

    base = img.convert("RGBA")
    blobs = [
        (int(w*0.12), int(h*0.18), 260, (245, 120, 155, 90)),
        (int(w*0.82), int(h*0.14), 220, (175, 95, 215, 80)),
        (int(w*0.58), int(h*0.78), 200, (130, 125, 235, 70)),
        (int(w*0.22), int(h*0.82), 170, (215, 145, 185, 60)),
        (int(w*0.70), int(h*0.45), 150, (200, 110, 210, 50)),
    ]
    for bx, by, rad, col in blobs:
        blob = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(blob)
        bd.ellipse([bx-rad, by-rad, bx+rad, by+rad], fill=col)
        blob = blob.filter(ImageFilter.GaussianBlur(rad//2))
        base = Image.alpha_composite(base, blob)
    return base.convert("RGB")


# ===============================
# PILL BUTTON (Canvas-drawn)
# ===============================
class PillButton(tk.Canvas):
    def __init__(self, parent, text, icon="", command=None,
                 w=200, h=60, fg=PILL_FG, bg=PILL_BG, hov=PILL_HOV,
                 font_size=14, bg_color="#C896C8", **kwargs):
        super().__init__(parent, width=w, height=h+6,
                         highlightthickness=0, bd=0,
                         bg=bg_color, **kwargs)
        self.width = w
        self.height = h
        self._label = (f"{icon}  {text}") if icon else text
        self._command = command
        self._fg = fg
        self._bg = bg
        self._hov = hov
        self._font = ("Helvetica", font_size, "bold")
        self.after(10, self._draw)
        self.bind("<Enter>",    lambda e: (self._draw(True),  self.configure(cursor="hand2")))
        self.bind("<Leave>",    lambda e:  self._draw(False))
        self.bind("<Button-1>", lambda e:  self._command() if self._command else None)

    def _draw(self, hover=False):
        if not self.winfo_exists():
            return

        try:
            self.delete("all")
        except tk.TclError:
            return

        w, h = self.width, self.height
        bg = self._hov if hover else self._bg
        r = h // 2

        # shadow
        self.create_arc(2, 5, h+2, h+5, start=90, extent=180,
                        fill="#C0C0C0", outline="")
        self.create_arc(w-h+2, 5, w+2, h+5, start=270, extent=180,
                        fill="#C0C0C0", outline="")
        self.create_rectangle(r+2, 5, w-r+2, h+5,
                            fill="#C0C0C0", outline="")

        # pill
        self.create_arc(0, 0, h, h, start=90, extent=180, fill=bg, outline="")
        self.create_arc(w-h, 0, w, h, start=270, extent=180, fill=bg, outline="")
        self.create_rectangle(r, 0, w-r, h, fill=bg, outline="")

        self.create_text(w//2, h//2,
                        text=self._label,
                        fill=self._fg,
                        font=self._font)
# ===============================
# CAMERA SIGN DETECTION
# ===============================
def start_camera():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5)
    prediction_buffer = deque(maxlen=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux, x_, y_ = [], [], []
                mp_drawing.draw_landmarks(frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x); y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))
                if len(data_aux) != 42:
                    continue
                x1 = int(min(x_)*W)-10; y1 = int(min(y_)*H)-10
                x2 = int(max(x_)*W)+10; y2 = int(max(y_)*H)+10
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]
                prediction_buffer.append(predicted_character)
                predicted_character = max(set(prediction_buffer), key=prediction_buffer.count)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(198,128,198),3)
                cv2.putText(frame, predicted_character, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,1.3,(198,128,198),3,cv2.LINE_AA)

        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(W,44),(50,30,70),-1)
        cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
        cv2.putText(frame,"SIGN DETECTION  |  Q to quit",
                    (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(198,128,198),1,cv2.LINE_AA)
        cv2.imshow("Sign Language Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# ===============================
# SPEECH → SIGN SYSTEM
# ===============================
def speech_to_sign(log_callback=None, status_callback=None):
    r = sr.Recognizer()

    isl_gif = [
        'hello','hii','goodmorning','goodafternoon','goodevening',
        'goodnight','howareyou','iamfine','please','sorry',
        'thankyou','welcome','areyouokay','whatisyourage','whatisyourname'
    ]

    arr = list(string.ascii_lowercase)

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def status(msg):
        if status_callback:
            status_callback(msg)

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)

        while True:
            status("listening")
            log("🎙  Listening…")

            audio = r.listen(source)

            try:
                a = r.recognize_google(audio)
                log(f'✦  Recognized: "{a}"')

                a = a.lower()
                phrase = a.replace(" ", "")

                # exit command
                if phrase in ['goodbye','bye','exit']:
                    log("⏹  Exiting voice mode.")
                    status("idle")
                    break

                # ======================
                # GIF PHRASE
                # ======================
                if phrase in isl_gif:

                    status(f"showing: {phrase}")
                    log(f"▶  ISL phrase: {phrase}")

                    win = tk.Toplevel()
                    win.title(f"ISL — {phrase}")
                    win.configure(bg="#EAD8F8")

                    lbl = ImageLabel(win)
                    lbl.pack(padx=20, pady=20)

                    lbl.load(f"isl_gif/{phrase}.gif")

                    win.after(4000, win.destroy)

                # ======================
                # SPELL LETTERS
                # ======================
                else:

                    status(f"spelling: {a}")
                    log(f"🔤  Spelling: {a}")

                    win = tk.Toplevel()
                    win.title("ISL Alphabet")
                    win.configure(bg="#EAD8F8")

                    img_label = tk.Label(win, bg="#EAD8F8")
                    img_label.pack(padx=20, pady=20)

                    def show_letter(i=0):

                        if i >= len(a):
                            win.after(1200, win.destroy)
                            return

                        ch = a[i]

                        # SPACE BETWEEN WORDS
                        if ch == " ":
                            img_label.config(image="", text=" ")
                            win.after(700, lambda: show_letter(i+1))
                            return

                        if ch in arr:
                            img = Image.open(f"isl_alphabets/{ch}.jpeg")
                            img = img.resize((300, 300), Image.LANCZOS)

                            photo = ImageTk.PhotoImage(img)

                            img_label.config(image=photo)
                            img_label.image = photo

                        # slower animation
                        win.after(1000, lambda: show_letter(i+1))

                    show_letter()

            except Exception as e:
                log(f"⚠  Error: {e}")


# ===============================
# MAIN APPLICATION
# ===============================
class HearingAssistantApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Hearing Impairment Assistant")
        self.resizable(False, False)
        self._voice_thread = None
        self._bg_photo = None
        self._build_ui()
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{WIN_W}x{WIN_H}+{(sw-WIN_W)//2}+{(sh-WIN_H)//2}")

    def _build_ui(self):
        avg_r = (BG_TOP[0]+BG_BOT[0])//2
        avg_g = (BG_TOP[1]+BG_BOT[1])//2
        avg_b = (BG_TOP[2]+BG_BOT[2])//2
        avg_hex = f"#{avg_r:02x}{avg_g:02x}{avg_b:02x}"
        self.configure(bg=avg_hex)

        # ── Full-window gradient canvas
        self._cv = tk.Canvas(self, width=WIN_W, height=WIN_H,
                             highlightthickness=0, bd=0)
        self._cv.place(x=0, y=0)

        bg_img = make_background(WIN_W, WIN_H)
        self._bg_photo = ImageTk.PhotoImage(bg_img)
        self._cv.create_image(0, 0, anchor="nw", image=self._bg_photo)

        # ── Title
        self._cv.create_text(WIN_W//2, 72,
            text="Hearing Impairment Assistant",
            fill=TITLE_FG, font=("Helvetica", 26, "bold"), anchor="center")
        self._cv.create_text(WIN_W//2, 106,
            text="Indian Sign Language  ·  Speech to Sign  ·  Live Detection",
            fill=SUB_FG, font=("Helvetica", 11), anchor="center")

        # ── Status pill
        self._status_var = tk.StringVar(value="● Idle")
        status_wrap = tk.Frame(self._cv, bg=PILL_BG,
                               highlightbackground=LOG_BOR, highlightthickness=1)
        self._status_lbl = tk.Label(status_wrap,
                                    textvariable=self._status_var,
                                    fg="#888888", bg=PILL_BG,
                                    font=("Helvetica", 10, "bold"),
                                    padx=18, pady=6)
        self._status_lbl.pack()
        self._cv.create_window(WIN_W//2, 146, window=status_wrap)

        # ── Three pill buttons
        btn_y  = 240
        gap    = 215
        bg_col = avg_hex   # canvas bg passthrough

        self._btn_voice = PillButton(
            self._cv, text="Live Voice", icon="🎙",
            command=self._on_voice, w=188, h=60,
            font_size=14, bg_color=bg_col)
        self._cv.create_window(WIN_W//2 - gap, btn_y, window=self._btn_voice)

        self._btn_camera = PillButton(
            self._cv, text="Camera Sign", icon="📷",
            command=self._on_camera, w=188, h=60,
            font_size=14, bg_color=bg_col)
        self._cv.create_window(WIN_W//2, btn_y, window=self._btn_camera)

        self._btn_exit = PillButton(
            self._cv, text="Exit", icon="⏹",
            command=self.quit, w=188, h=60,
            font_size=14, fg="#E03060", bg_color=bg_col)
        self._cv.create_window(WIN_W//2 + gap, btn_y, window=self._btn_exit)

        # ── Activity Log (frosted card)
        log_outer = tk.Frame(self._cv, bg=LOG_BG,
                             highlightbackground=LOG_BOR, highlightthickness=1)

        lh = tk.Frame(log_outer, bg=LOG_BG)
        lh.pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(lh, text="Activity Log", fg="#7050A8",
                 bg=LOG_BG, font=("Helvetica", 10, "bold")).pack(side="left")
        clr = tk.Label(lh, text="Clear", fg=PILL_FG, bg=LOG_BG,
                       font=("Helvetica", 10), cursor="hand2")
        clr.pack(side="right")
        clr.bind("<Button-1>", lambda e: self._clear_log())

        tk.Frame(log_outer, bg=LOG_BOR, height=1).pack(fill="x", pady=(8, 0))

        self._log_text = tk.Text(
            log_outer, bg=LOG_BG, fg=LOG_FG,
            font=("Helvetica", 10), relief="flat", bd=0,
            height=7, padx=16, pady=8,
            state="disabled", wrap="word",
            insertbackground=PILL_FG,
            selectbackground="#C8B0F0"
        )
        self._log_text.pack(fill="both", expand=True)
        self._log_text.tag_configure("blue", foreground=PILL_FG)
        self._log_text.tag_configure("dim",  foreground=LOG_DIM)
        self._log_text.tag_configure("warn", foreground="#D04040")
        self._log_text.tag_configure("ok",   foreground="#207840")

        sb = tk.Scrollbar(self._log_text, command=self._log_text.yview,
                          bg="#EAD8F8", troughcolor="#EAD8F8",
                          activebackground=PILL_FG)
        self._log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        self._cv.create_window(WIN_W//2, 420, window=log_outer,
                               width=WIN_W - 80, height=190)

        # ── Footer text
        self._cv.create_text(WIN_W//2, WIN_H - 14,
            text="Powered by MediaPipe · SpeechRecognition · Tkinter",
            fill=DIM_FG, font=("Helvetica", 9), anchor="center")

        self._log("System ready.", "ok")

    # ── log helpers
    def _log(self, msg, tag=""):
        ts = time.strftime("%H:%M:%S")
        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"[{ts}]  ", "dim")
        self._log_text.insert("end", msg + "\n", tag or None)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _set_status(self, text):
        m = {
            "idle":      ("● Idle",        "#888888"),
            "listening": ("◉ Listening…",  "#2B7FFF"),
            "camera":    ("◉ Camera On",   "#9B59B6"),
        }
        label, color = m.get(text, (f"◎ {text.title()}", "#D06020"))
        self._status_var.set(label)
        self._status_lbl.configure(fg=color)

    # ── button handlers
    def _on_voice(self):
        if self._voice_thread and self._voice_thread.is_alive():
            self._log("Voice mode already active.", "warn"); return
        self._log("Starting voice mode…", "blue")
        self._set_status("listening")
        def run():
            speech_to_sign(
                log_callback=lambda m: self.after(0, self._log, m, ""),
                status_callback=lambda s: self.after(0, self._set_status, s))
            self.after(0, self._set_status, "idle")
        self._voice_thread = threading.Thread(target=run, daemon=True)
        self._voice_thread.start()

    def _on_camera(self):
        self._log("Launching camera…", "blue")
        self._set_status("camera")
        def run():
            start_camera()
            self.after(0, self._set_status, "idle")
            self.after(0, self._log, "Camera closed.", "dim")
        threading.Thread(target=run, daemon=True).start()


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    app = HearingAssistantApp()
    app.mainloop()