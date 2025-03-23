from preprocessing.load_data import load_images
from preprocessing.preprocess import augment_images
from preprocessing.split_data import split_dataset
from sklearn.preprocessing import LabelEncoder

# Load dataset
DATASET_PATH = "D:\ARTIFICAL INTELLIGENCE\SEM 2\wavepointer\gesturedata"
IMAGE_SIZE = (128, 128)

images, labels, class_labels = load_images(DATASET_PATH, image_size=IMAGE_SIZE, grayscale=False)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split dataset
X_train, X_test, y_train, y_test = split_dataset(images, encoded_labels)

# Apply augmentation (for training images only)
datagen = augment_images()
datagen.fit(X_train)

print("Preprocessing complete! Ready for model training.")
