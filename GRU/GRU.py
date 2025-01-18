import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ฟังก์ชันสำหรับการดึงคุณลักษณะจากไฟล์เสียง (MFCC)
def extract_features(file_path, max_pad_len=100):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((14, max_pad_len))


# ฟังก์ชันสำหรับโหลดข้อมูลจากโฟลเดอร์
def load_data(audio_folder):
    filenames = []
    labels = []
    for emotion in os.listdir(audio_folder):
        emotion_folder = os.path.join(audio_folder, emotion)
        if os.path.isdir(emotion_folder):
            for filename in os.listdir(emotion_folder):
                if filename.endswith(".wav"):
                    file_path = os.path.join(emotion_folder, filename)
                    filenames.append(file_path)
                    labels.append(emotion)
    return filenames, labels


# ฟังก์ชันสำหรับแปลงข้อมูลเสียงทั้งหมดเป็นฟีเจอร์ MFCC
def extract_features_from_all(filenames, max_pad_len=100):
    features = []
    for file in filenames:
        mfccs = extract_features(file, max_pad_len)
        features.append(mfccs)
    return np.array(features)


# สร้างโมเดล RNN
def create_rnn_model(input_shape, n_classes):
    model = Sequential()
    model.add(GRU(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(GRU(128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# โหลดข้อมูลจากโฟลเดอร์
train_folder = '/Users/gam/Desktop/DEEP/Woekshop#3/DATASET/Dataset/train'
test_folder = '/Users/gam/Desktop/DEEP/Woekshop#3/DATASET/Dataset/test'

# โหลดข้อมูล
train_filenames, train_labels = load_data(train_folder)
test_filenames, test_labels = load_data(test_folder)

# แปลงป้ายกำกับให้เป็นตัวเลขด้วย LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_labels)
y_test_encoded = label_encoder.transform(test_labels)

# แปลงป้ายกำกับเป็น one-hot encoding
y_train_encoded = to_categorical(y_train_encoded)
y_test_encoded = to_categorical(y_test_encoded)

# แปลงข้อมูลเสียงเป็นฟีเจอร์ MFCC
X_train_features = extract_features_from_all(train_filenames)
X_test_features = extract_features_from_all(test_filenames)

# เพิ่มมิติข้อมูลสำหรับ RNN
X_train = X_train_features  # ไม่เพิ่มมิติ
X_test = X_test_features  # ไม่เพิ่มมิติ

# สร้างโมเดล RNN
n_classes = len(label_encoder.classes_)
input_shape = (X_train.shape[1], X_train.shape[2])
model = create_rnn_model(input_shape, n_classes)

# ฝึกโมเดลพร้อม EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train_encoded, epochs=20, batch_size=50, validation_data=(X_test, y_test_encoded), callbacks=[early_stop])

# กราฟ Loss และ Accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ประเมินผลการทดสอบ
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# ทำนายข้อมูลทดสอบ
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)  # หาคลาสที่ทำนาย
true_classes = np.argmax(y_test_encoded, axis=1)  # หาคลาสจริง

# คำนวณความแม่นยำในการทำนาย
accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print(f"Accuracy: {accuracy*100:.2f}%")

# สร้าง Confusion Matrix Heatmap
cm = confusion_matrix(true_classes, predicted_classes)
class_labels = label_encoder.classes_
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# บันทึกโมเดล
model.save('emotion_recognition_model.h5')
