import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
DATASET_PATH = "D:\ARTIFICAL INTELLIGENCE\SEM 2\wavepointer\gesturedata"
PROCESSED_PATH = "D:\ARTIFICAL INTELLIGENCE\SEM 2\wavepointer\processed_data"
IMAGE_SIZE = (128, 128)
TEST_SIZE = 0.2

# Create processed dataset folder if it doesn't exist
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Step 1: Load Images
def load_images():
    """Load images from dataset and return images, labels, and class names."""
    images, labels = [], []
    class_labels = sorted(os.listdir(DATASET_PATH))

    for label in class_labels:
        label_path = os.path.join(DATASET_PATH, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                image = cv2.imread(img_path)
                if image is not None:
                    images.append(image)
                    labels.append(label)
    
    return np.array(images), np.array(labels), class_labels

# Step 2: Preprocess Images
def preprocess_images(images):
    """Resize, convert to grayscale, and normalize images."""
    processed_images = []
    
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, IMAGE_SIZE)  # Resize to 128x128
        img = img / 255.0  # Normalize pixel values
        processed_images.append(img)
    
    return np.array(processed_images)

# Step 3: Split Dataset into Train & Test
def split_data(images, labels):
    """Split images and labels into training and testing sets."""
    return train_test_split(images, labels, test_size=TEST_SIZE, stratify=labels, random_state=42)

# Step 4: Data Augmentation
def augment_images(X_train):
    """Apply data augmentation to training images."""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    return datagen.flow(X_train, batch_size=32)

# Step 5: Save Processed Images
def save_processed_data(X_train, y_train, X_test, y_test, class_labels):
    """Save processed images into 'processed_data' directory."""
    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)

    for img, label in zip(X_train, y_train):
        class_folder = os.path.join(PROCESSED_PATH, label_encoder.inverse_transform([label])[0])
        os.makedirs(class_folder, exist_ok=True)
        img_path = os.path.join(class_folder, f"train_{np.random.randint(10000)}.png")
        cv2.imwrite(img_path, (img * 255).astype(np.uint8))

    for img, label in zip(X_test, y_test):
        class_folder = os.path.join(PROCESSED_PATH, label_encoder.inverse_transform([label])[0])
        os.makedirs(class_folder, exist_ok=True)
        img_path = os.path.join(class_folder, f"test_{np.random.randint(10000)}.png")
        cv2.imwrite(img_path, (img * 255).astype(np.uint8))

# Execute Preprocessing Steps
if __name__ == "__main__":
    print("🔄 Loading images...")
    images, labels, class_labels = load_images()

    print("🔄 Preprocessing images...")
    images = preprocess_images(images)

    print("🔄 Splitting dataset into train and test...")
    X_train, X_test, y_train, y_test = split_data(images, labels)

    print("🔄 Applying data augmentation...")
    augmented_data = augment_images(X_train)

    print("🔄 Saving processed images...")
    save_processed_data(X_train, y_train, X_test, y_test, class_labels)

    print("✅ Preprocessing complete! Processed data saved in 'processed_data'.")
