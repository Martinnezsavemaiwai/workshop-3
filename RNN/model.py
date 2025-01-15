import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import librosa

# ฟังก์ชันสำหรับการดึงคุณลักษณะจากไฟล์เสียง (เช่น MFCC)
def extract_features(file_path, max_pad_len=100):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # ปรับเป็น 40
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = load_model('emotion_recognition_model.h5')

# โหลด LabelEncoder ที่ใช้ในตอนฝึก
label_encoder = LabelEncoder()
label_encoder.fit(['angry', 'disgust', 'surprise', 'neutral', 'sad', 'happy', 'fear'])

# ฟังก์ชันทำนายอารมณ์จากไฟล์เสียง
def predict_emotion(file_path):
    try:
        # แปลงไฟล์เสียงเป็นฟีเจอร์
        feature = extract_features(file_path)
        
        # ตรวจสอบว่าฟีเจอร์ถูกสร้างสำเร็จหรือไม่
        if feature is None or feature.shape[0] == 0:
            print("ไม่สามารถดึงฟีเจอร์จากไฟล์เสียงได้")
            return "Unknown", None

        # Reshape ให้ตรงกับโมเดล
        feature = feature.reshape(1, 40, 100, 1)

        # ทำนายอารมณ์
        prediction = model.predict(feature)
        print(f"Prediction Probabilities: {prediction[0]}")

        # แปลงค่าทำนายเป็นคลาส
        predicted_index = np.argmax(prediction)
        if predicted_index >= len(label_encoder.classes_):
            print(f"คลาสที่ทำนาย {predicted_index} อยู่นอกขอบเขต!")
            predicted_class = "Unknown"
        else:
            predicted_class = label_encoder.inverse_transform([predicted_index])[0]

        return predicted_class, prediction[0]

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        return "Error", None

# ตัวอย่างการใช้งาน
file_path = '/Users/gam/Desktop/DEEP/Woekshop#3/DATASET/TESS Toronto emotional speech set data/happy/YAF_merge_happy.wav'
predicted_emotion, probabilities = predict_emotion(file_path)
print(f"Predicted Emotion: {predicted_emotion}")
print(f"Class Probabilities: {probabilities}")
