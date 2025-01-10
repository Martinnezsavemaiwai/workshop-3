
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ฟังก์ชันสำหรับการดึงคุณลักษณะจากไฟล์เสียง (MFCC)
def extract_features(file_path, max_pad_len=100):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((13, max_pad_len))

# ฟังก์ชันสำหรับโหลดข้อมูลจากโฟลเดอร์
def load_data_from_folder(audio_folder):
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

# ฟังก์ชันสำหรับแปลงข้อมูลเสียงทั้งหมดเป็นฟีเจอร์
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# ฟังก์ชันสำหรับดึงฟีเจอร์ MFCC จากไฟล์เสียงทั้งหมด
def extract_features_from_all(filenames):
    features = []
    for file in filenames:
        mfccs = extract_features(file)  # ดึง MFCC สำหรับแต่ละไฟล์
        features.append(mfccs)  # เพิ่มผลลัพธ์ลงใน features
    return features  # ส่งกลับข้อมูลในรูปแบบ array


# สร้างโมเดล RNN
def create_rnn_model(input_shape, n_classes):
    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(SimpleRNN(128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



audio_folder = '/Users/gam/Desktop/DEEP/Woekshop#3/DATASET/TESS Toronto emotional speech set data'
filenames, labels = load_data_from_folder(audio_folder)

# แปลงป้ายกำกับให้เป็นตัวเลขด้วย LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
y_encoded = to_categorical(y_encoded)

# แปลงข้อมูลเสียงเป็นฟีเจอร์ MFCC
X_features = extract_features_from_all(filenames)
X = np.expand_dims(X_features, axis=-1)
# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# สร้างโมเดล RNN
n_classes = len(label_encoder.classes_)
model = create_rnn_model(X_train.shape[1:], n_classes)

# ฝึกโมเดลพร้อม EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=20, batch_size=50, validation_data=(X_test, y_test), callbacks=[early_stop])

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
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# ทำนายข้อมูลทดสอบ
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)  # หาคลาสที่ทำนาย
true_classes = np.argmax(y_test, axis=1)  # หาคลาสจริง
predicted_emotions = label_encoder.inverse_transform(predicted_classes)  # แปลงคลาสทำนายเป็นอารมณ์
true_emotions = label_encoder.inverse_transform(true_classes)  # แปลงคลาสจริงเป็นอารมณ์

# คำนวณความแม่นยำในการทำนาย
accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print(f"Accuracy: {accuracy*100:.2f}%")

# สร้าง confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)


# สร้าง Confusion Matrix Heatmap
class_labels = label_encoder.classes_
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# บันทึกโมเดล
model.save('emotion_recognition_model.h5')
