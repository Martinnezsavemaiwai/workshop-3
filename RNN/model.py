# from tensorflow.keras.models import load_model
# import numpy as np
# import librosa
# import soundfile as sf
# import glob
# import os

# # โหลดโมเดลที่บันทึกไว้
# model = load_model('/Users/gam/Desktop/DEEP/Woekshop#3/vocal_separation_model.h5')

# # ฟังก์ชันการโหลดไฟล์เสียง
# def load_audio(file_path):
#     y, sr = librosa.load(file_path, sr=None)
#     return y, sr

# # แปลง Mel-Spectrogram กลับเป็นเสียง
# def mel_to_audio(mel_spec, sr=22050, n_iter=32, hop_length=512):
#     # แปลงกลับจาก dB เป็น power ก่อน
#     mel_spec = librosa.db_to_power(mel_spec)
#     return librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr, n_iter=n_iter, hop_length=hop_length)

# # แปลงเสียงเป็น Mel-Spectrogram
# def audio_to_melspectrogram(y, sr, n_mels=128, hop_length=512, n_fft=2048):
#     mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
#     mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
#     return mel_spectrogram

# # ฟังก์ชันการ padding Mel-Spectrogram
# def pad_mel_spectrogram(mel_spec, max_len):
#     current_len = mel_spec.shape[1]
#     if current_len < max_len:
#         pad_width = max_len - current_len
#         mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
#     elif current_len > max_len:
#         mel_spec = mel_spec[:, :max_len]
#     return mel_spec

# # ฟังก์ชันแยกเสียงร้องจากไฟล์เพลง
# def separate_vocals(music_file, model, max_len=200):
#     # โหลดไฟล์เพลง
#     music, sr = load_audio(music_file)

#     # แปลงเสียงเป็น Mel-Spectrogram
#     mel_music = audio_to_melspectrogram(music, sr)

#     # Padding หรือ Truncating Mel-Spectrogram
#     mel_music = pad_mel_spectrogram(mel_music, max_len)

#     # เพิ่มมิติที่จำเป็นสำหรับ input ของโมเดล
#     mel_music = np.expand_dims(mel_music, axis=0)  # ทำให้ขนาดเป็น (1, time_steps, n_mels)

#     # ใช้โมเดลในการทำนายเสียงร้อง
#     predicted_vocals = model.predict(mel_music)

#     # ตรวจสอบค่า output จากโมเดล
#     print(f"predicted_vocals shape: {predicted_vocals.shape}")
#     print(f"Min: {np.min(predicted_vocals)}, Max: {np.max(predicted_vocals)}")

#     # แปลง Mel-Spectrogram กลับเป็นเสียง
#     predicted_audio = mel_to_audio(predicted_vocals[0])

#     # ตรวจสอบและบันทึกเสียงที่แยกออกมา
#     if predicted_audio is not None and len(predicted_audio) > 0:
#         sf.write('predicted_vocals.wav', predicted_audio, sr)
#         print("เสียงร้องถูกแยกออกมาแล้วและบันทึกเป็น predicted_vocals.wav")
#     else:
#         print("ไม่สามารถแยกเสียงร้องได้.")

# # ทดสอบกับไฟล์เพลงใหม่
# music_file = '/Users/gam/Desktop/DEEP/Woekshop#3/DATASET/Mixtures/mix/mixture_1.wav'
# separate_vocals(music_file, model)


import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import librosa

# ฟังก์ชั่นสำหรับการดึงคุณลักษณะจากไฟล์เสียง (เช่น MFCC)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = load_model('emotion_recognition_model.h5')

# โหลด LabelEncoder ที่ใช้ในตอนฝึก
label_encoder = LabelEncoder()
label_encoder.fit(['angry', 'disgust', 'surprise', 'neutral', 'sad', 'happy', 'fear'])


# ฟังก์ชั่นทำนายอารมณ์จากไฟล์เสียง
def predict_emotion(file_path):
    try:
        # แปลงไฟล์เสียงเป็นฟีเจอร์
        feature = extract_features(file_path)
        
        # ตรวจสอบว่าฟีเจอร์ถูกสร้างสำเร็จหรือไม่
        if feature is None or feature.shape[0] == 0:
            print("ไม่สามารถดึงฟีเจอร์จากไฟล์เสียงได้")
            return "Unknown", None

        # Normalize ข้อมูลฟีเจอร์
        feature = (feature - np.mean(feature)) / np.std(feature)  # Z-score Normalization

        # Reshape ข้อมูลให้เหมาะสมกับ LSTM
        feature = feature.reshape(1, feature.shape[0], 1)

        # ทำนายอารมณ์
        prediction = model.predict(feature)
        print(f"Prediction Probabilities: {prediction[0]}")  # แสดงค่าความน่าจะเป็นของแต่ละคลาส

        # ตรวจสอบว่าคลาสที่ทำนายอยู่ในขอบเขตหรือไม่
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
file_path = '/Users/gam/Desktop/DEEP/Woekshop#3/DATASET/TESS Toronto emotional speech set data/surprised/YAF_seize_ps.wav'  # เส้นทางไปยังไฟล์เสียงใหม่
predicted_emotion, probabilities = predict_emotion(file_path)
print(f"Predicted Emotion: {predicted_emotion}")
print(f"Class Probabilities: {probabilities}")
