import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight


dataset_path = "/Users/eswar/Downloads/dataset"

train_path = "/Users/eswar/Downloads/dataset/train"
test_path = "/Users/eswar/Downloads/dataset/test"

model_save_path = "/Users/eswar/Downloads/Non-Invasive-Blood-Group-Prediction-Using-Fingerprint-Analysis/model/new_model.keras"
import os
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)


if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("Splitting dataset into train/test...")

    categories = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
    train_ratio = 0.8

    for category in categories:
        os.makedirs(os.path.join(train_path, category), exist_ok=True)
        os.makedirs(os.path.join(test_path, category), exist_ok=True)

        images = os.listdir(os.path.join(dataset_path, category))
        random.shuffle(images)
        train_size = int(len(images) * train_ratio)
        train_images = images[:train_size]
        test_images = images[train_size:]

        for img in train_images:
            shutil.copy(os.path.join(dataset_path, category, img), os.path.join(train_path, category, img))
        for img in test_images:
            shutil.copy(os.path.join(dataset_path, category, img), os.path.join(test_path, category, img))

    print(" Dataset split complete.")
else:
    print(" Dataset already split.")


def preprocess_image(image_path, target_size=(224, 224)):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")




    img = cv2.GaussianBlur(img,(3,3),0)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    img = clahe.apply(img)

    kernel = np.array([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]
    ])

    img = cv2.filter2D(img,-1,kernel)


    img = cv2.resize(img, (224, 224)) 


    # 5. Normalize

    img = img.astype("float32")/255.0

    img = np.repeat(
        np.expand_dims(img,-1),
        3,
        axis=-1
    )

    return img

def load_data_from_directory(directory, target_size=(224, 224)):
    X, y = [], []
    class_labels = [
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]
    class_labels.sort()
    label_map = {label: idx for idx, label in enumerate(class_labels)}
    print(f"Class mapping: {label_map}")

    for label in class_labels:
        label_path = os.path.join(directory, label)

        if not os.path.isdir(label_path):
            continue

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)

            if not os.path.isfile(img_path):
                continue

            try:
                X.append(preprocess_image(img_path, target_size))
                y.append(label_map[label])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    return np.array(X), np.array(y), label_map


X_train, y_train, train_label_map = load_data_from_directory(train_path)
X_test, y_test, test_label_map = load_data_from_directory(test_path)
num_classes = len(set(y_train))


class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
print("Class Weights:", class_weights_dict)

# === Model Architecture ===
input_shape = (224, 224, 3)

def lr_schedule(epoch):
    initial_lr = 1e-3
    drop = 0.5
    epochs_drop = 5
    return initial_lr * (drop ** (epoch // epochs_drop))

def build_model(num_classes):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = True

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x) 
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

# === Train Model ===
def build_and_train_model(X_train, y_train, X_test, y_test, num_classes):
    model = build_model(num_classes)

    callbacks = [
        LearningRateScheduler(lr_schedule),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,  
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights_dict 
    )

    model.evaluate(X_test, y_test, verbose=1)
    return model, history

model, history = build_and_train_model(X_train, y_train, X_test, y_test, num_classes)

# === Plotting ===
def plot_training_curves(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_curves(history)

# === Evaluation ===
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

conf_matrix = confusion_matrix(y_test, y_pred_classes)
ConfusionMatrixDisplay(conf_matrix, display_labels=test_label_map.keys()).plot(cmap="Blues")
plt.show()

def resize_with_padding(img, size=224):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h))

    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    # Add padding
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return padded