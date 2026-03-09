import os
import pickle
import mediapipe as mp
import cv2

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './dataset/Indian'
MAX_IMAGES = 100   # limit per class

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):

    class_path = os.path.join(DATA_DIR, dir_)

    # Skip files that are not folders
    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing class: {dir_}")

    img_count = 0

    for img_path in os.listdir(class_path):

        if img_count >= MAX_IMAGES:
            break

        full_path = os.path.join(class_path, img_path)

        img = cv2.imread(full_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                data_aux = []
                x_ = []
                y_ = []

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                # Ensure correct feature size
                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(dir_)
                    img_count += 1

        print(f"{dir_}: {img_count}/{MAX_IMAGES}", end="\r")

print("\nSaving dataset...")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

hands.close()

print("Dataset created successfully!")