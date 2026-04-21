import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import pandas as pd

# Load model
model_dict = pickle.load(open("model.p", "rb"))
model = model_dict["model"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

dataset_path = "dataset/testing_dataset"

correct = {}
total = {}

for label in os.listdir(dataset_path):

    label_path = os.path.join(dataset_path, label)

    correct[label] = 0
    total[label] = 0

    for img_name in os.listdir(label_path):

        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:

                prediction = model.predict([np.asarray(data_aux)])

                predicted_letter = prediction[0]

                total[label] += 1

                if predicted_letter == label:
                    correct[label] += 1


rows = []

print("\nAlphabet Accuracy Table\n")
print("{:<10} {:<10} {:<10} {:<10}".format("Alphabet", "Correct", "Total", "Accuracy"))

for label in sorted(total.keys()):

    if total[label] > 0:
        acc = (correct[label] / total[label]) * 100

        print("{:<10} {:<10} {:<10} {:.2f}%".format(
            label,
            correct[label],
            total[label],
            acc
        ))

        rows.append({
            "Alphabet": label,
            "Correct Predictions": correct[label],
            "Total Samples": total[label],
            "Accuracy (%)": round(acc, 2)
        })

# Create dataframe
df = pd.DataFrame(rows)

# Save to Excel
df.to_excel("alphabet_accuracy.xlsx", index=False)

print("\nExcel file saved as alphabet_accuracy.xlsx")