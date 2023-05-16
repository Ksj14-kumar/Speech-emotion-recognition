from flask import Flask, render_template, request
import librosa
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("mymodel.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio']
    file.save('audio.wav')
    audio, _ = librosa.load('audio.wav', sr=22050)
    mfccs = librosa.feature.mfcc(audio, sr=22050, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    emotion = model.predict([mfccs_scaled])[0]
    emotions = ['Angry', 'Fearful', 'Happy', 'Sad', 'Neutral']
    predicted_emotion = emotions[emotion]
    
    return render_template('index.html', emotion=predicted_emotion)

if __name__ == '__main__':
    app.run(debug=True)
