import os
import cv2

DATA_DIR = './dataset/Indian'

# Create dataset directory if not exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = [
'0','1','2','3','4','5','6','7','8','9',
'A','B','C','D','E','F','G','H','I','J',
'K','L','M','N','O','P','Q','R','S','T',
'U','V','W','X','Y','Z'
]

dataset_size = 100

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

for label in classes:

    class_path = os.path.join(DATA_DIR, label)

    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print("Collecting data for:", label)

    # Wait until user presses Q
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        cv2.putText(
            frame,
            f"Show sign '{label}' - Press Q to start",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("frame", frame)

        key = cv2.waitKey(25)

        if key == ord('q'):
            break
        elif key == 27:   # ESC key
            cap.release()
            cv2.destroyAllWindows()
            exit()

    counter = 0

    while counter < dataset_size:

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        print(f"{label} image {counter+1}/{dataset_size}")

        cv2.imshow("frame", frame)

        cv2.imwrite(os.path.join(class_path, f"{counter}.jpg"), frame)

        counter += 1

        key = cv2.waitKey(100)

        if key == 27:   # ESC to stop anytime
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()