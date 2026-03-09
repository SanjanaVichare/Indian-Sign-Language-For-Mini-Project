import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Filter samples with correct feature size
filtered_data = []
filtered_labels = []

for sample, label in zip(data, labels):
    if len(sample) == 42:  # 21 landmarks × (x,y)
        filtered_data.append(sample)
        filtered_labels.append(label)

data = np.array(filtered_data)
labels = np.array(filtered_labels)

print(f"Total valid samples: {len(data)}")

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)

# Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training model...")
model.fit(x_train, y_train)

# Evaluate model
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"Accuracy: {score * 100:.2f}%")

# Save trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as model.p")