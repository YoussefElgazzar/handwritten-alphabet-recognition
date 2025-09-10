import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
# import tensorflow as tf
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
# from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import os

# 1. Load the dataset
file_path = "A_Z Handwritten Data.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)

# 2. Explore the dataset
print("Dataset Head:\n", df.head())
print("\nDataset Info:")
print(df.info())

# Extract labels and image data
labels = df.iloc[:, 0].values  # First column contains the labels
images = df.iloc[:, 1:].values  # Remaining columns contain pixel values

# Check dataset properties
print(f"\nNumber of classes: {len(np.unique(labels))}")
print(f"Class distribution:\n{pd.Series(labels).value_counts()}")

# Visualize class distribution
sns.countplot(x=labels)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")

# Replace x-axis labels with A to Z
alphabet = [chr(i) for i in range(65, 91)]  # Generate list of characters from A to Z
plt.xticks(ticks=np.arange(len(alphabet)), labels=alphabet)

plt.show()

# 3. Normalize images
images_normalized = images / 255.0  # Scale pixel values to [0, 1]

# 4. Reshape images
image_shape = (28, 28)  # Adjust dimensions if necessary
images_reshaped = images_normalized.reshape(-1, *image_shape)

# Display all 26 characters, each in 3 shapes
alphabet = [chr(i) for i in range(65, 91)]  # Generate list of characters from A to Z
unique_labels = np.unique(labels)  # Unique labels in the dataset
print("labels: ")
print(labels)
# Find 3 images for each label
images_per_label = {}
for label in unique_labels:
    indices = np.where(labels == label)[0][:3]  # Get the first 3 indices for each label
    images_per_label[label] = images_reshaped[indices]

# Create a figure with subplots (3 rows, 26 columns)
fig, axes = plt.subplots(3, 26, figsize=(26, 6))  # Adjust figsize as needed

for col, label in enumerate(unique_labels):
    for row in range(3):
        # Display the image
        image = images_per_label[label][row]
        axes[row, col].imshow(image, cmap='gray')
        if row == 0:  # Add a title for the top row
            axes[row, col].set_title(alphabet[label], fontsize=10)
        axes[row, col].axis('off')  # Turn off the axis for better visualization

plt.tight_layout()
plt.show()


# 5. Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(images_normalized, labels, test_size = 0.98, random_state=42, stratify=labels)
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# 6. Train SVM models
# Linear kernel SVM
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
# RBF kernel SVM
svm_rbf = SVC(kernel='rbf', random_state=42)#non-linear
svm_rbf.fit(X_train, y_train)
# 7. Evaluate models
# Predictions
y_pred_linear = svm_linear.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)
# Confusion matrices
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
# F1 Scores
f1_linear = f1_score(y_test, y_pred_linear, average='weighted')
f1_rbf = f1_score(y_test, y_pred_rbf, average='weighted')

# Display results
print(f"Linear SVM - F1 Score: {f1_linear:.2f}")
print(f"RBF SVM - F1 Score: {f1_rbf:.2f}")

ConfusionMatrixDisplay(conf_matrix_linear).plot()
plt.title("Linear SVM Confusion Matrix")
plt.show()

ConfusionMatrixDisplay(conf_matrix_rbf).plot()
plt.title("RBF SVM Confusion Matrix")
plt.show()



X_train, X_test, y_train, y_test = train_test_split(
    images_normalized, labels, test_size = 0.2, random_state=42, stratify=labels
    )

# 8. Further split training data into training and validation datasets
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Final Training Data Shape: {X_train_final.shape}, Validation Data Shape: {X_val.shape}")


def logistic_regression_one_vs_all(X, y, num_classes):
    # Initialize the logistic regression models
    models = []
    # Binarize the labels for one-vs-all
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)

    # Train a separate logistic regression model for each class
    for i in range(num_classes):
        print(f"Training model for class {i} vs all...")
        model = LogisticRegression(solver='liblinear', max_iter=4000, random_state=42)
        model.fit(X, y_bin[:, i])  # Fit each model to the binary labels for class i vs all
        models.append(model)

    return models, lb


def predict_one_vs_all(X, models):
    # Predict probabilities for each class
    probs = np.zeros((X.shape[0], len(models)))
    for i, model in enumerate(models):
        probs[:, i] = model.predict_proba(X)[:, 1]  # Get the probability for the positive class
    # Get the class with the highest probability for each sample
    y_pred = np.argmax(probs, axis=1)
    return y_pred


# Train logistic regression models (one-vs-all)
num_classes = len(np.unique(labels))  # In your case, it should be 26 for A-Z
models, lb = logistic_regression_one_vs_all(X_val, y_val, num_classes)

# Test the model
y_pred = predict_one_vs_all(X_test, models)

# Confusion matrix and F1 score
conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# Display confusion matrix
ConfusionMatrixDisplay(conf_matrix, display_labels=lb.classes_).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score (Weighted): {f1:.2f}")

#----------------Neural Network---------------------------------------------------

# Convert labels to categorical (one-hot encoding)
y_train_cat = to_categorical(y_train_final, num_classes=26)
y_val_cat = to_categorical(y_val, num_classes=26)
y_test_cat = to_categorical(y_test, num_classes=26)

# Define two neural networks
def build_model_1():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_2():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Reshape data for TensorFlow
X_train_tf = X_train_final.reshape(-1, 28, 28, 1)
X_val_tf = X_val.reshape(-1, 28, 28, 1)
X_test_tf = X_test.reshape(-1, 28, 28, 1)

# Train and evaluate models
models = [build_model_1(), build_model_2()]
histories = []
for i, model in enumerate(models):
    print(f"Training Model {i+1}...")
    history = model.fit(X_train_tf, y_train_cat, validation_data=(X_val_tf, y_val_cat), epochs=10, batch_size=32)
    histories.append(history)

# Plot training and validation curves
for i, history in enumerate(histories):
    plt.figure(figsize=(12, 5))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Model {i+1} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Model {i+1} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

# Save the best model
best_model = models[1]  # Assume Model 2 is better; adjust based on results
model_path = "best_model.h5"
best_model.save(model_path)
print(f"Best model saved to {model_path}")

# Reload the model
loaded_model = tf.keras.models.load_model(model_path)

# Evaluate the best model on the test set
test_loss, test_accuracy = loaded_model.evaluate(X_test_tf, y_test_cat)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Confusion matrix and F1 score
y_test_pred = loaded_model.predict(X_test_tf)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_test_pred_classes)
f1 = f1_score(y_test, y_test_pred_classes, average='weighted')

ConfusionMatrixDisplay(conf_matrix, display_labels=alphabet).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print(f"F1 Score (Weighted): {f1:.2f}")

# Test with custom images for A, M, O, Y, Z
custom_letters = ['A', 'M', 'O', 'Y', 'Z']
custom_images = []  # Load or generate images for these letters

for i, letter in enumerate(custom_letters):
    # Preprocess and predict
    img = custom_images[i].reshape(1, 28, 28, 1) / 255.0  # Replace with actual image loading code
    pred = loaded_model.predict(img)
    pred_class = alphabet[np.argmax(pred)]
    print(f"Prediction for {letter}: {pred_class}")


import cv2  # For image loading and processing

custom_letters = ['A', 'M', 'O', 'Y', 'Z']
custom_images = []

# Load images for each letter
for letter in custom_letters:
    # Replace 'path_to_images' with the actual directory containing your images
    image_path = f"path_to_images/{letter}.png"  # Example: "images/A.png"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if img is None:
        print(f"Error: Image for {letter} not found at {image_path}")
        continue
    img = cv2.resize(img, (28, 28))  # Resize to 28x28
    custom_images.append(img)

# Ensure custom_images has the same number of elements as custom_letters
if len(custom_images) < len(custom_letters):
    raise ValueError("Not all custom letter images were loaded successfully.")

# Test with the loaded images
for i, letter in enumerate(custom_letters):
    img = custom_images[i].reshape(1, 28, 28, 1) / 255.0  # Normalize and reshape
    pred = loaded_model.predict(img)
    pred_class = alphabet[np.argmax(pred)]
    print(f"Prediction for {letter}: {pred_class}")


