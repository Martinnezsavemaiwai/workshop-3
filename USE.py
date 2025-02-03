import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# โหลดโมเดลที่บันทึกไว้
model = load_model('/Users/gam/Desktop/DEEP/Woekshop#3/emotion_recognition_model.h5')

# โหลด LabelEncoder เดิม (ต้องใช้ encoder ตัวเดียวกับตอนเทรน)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']  # ปรับตาม Dataset
label_encoder = LabelEncoder()
label_encoder.fit(emotion_labels)


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

# ปรับขนาด input ตอนทำนายให้ตรงกับโมเดล
def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Error", None

    features = np.expand_dims(features, axis=0)  # จาก (100, 60) → (1, 100, 60)

    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_emotion, prediction[0]

# เรียกใช้ฟังก์ชันทำนาย
audio_file_path = '/Users/gam/Desktop/DEEP/Woekshop#3/DATASET/TESS Toronto emotional speech set data/disgust/OAF_bar_disgust.wav'
emotion, probabilities = predict_emotion(audio_file_path)

if emotion != "Error":
    print(f"อารมณ์ที่ทำนายได้: {emotion}")
else:
    print("ไม่สามารถทำนายอารมณ์ได้")