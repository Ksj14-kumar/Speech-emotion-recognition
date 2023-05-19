from flask import  Flask, render_template,request
import tensorflow as tf
import os,urllib
import librosa 
import numpy as np    

app= Flask(__name__)
def load_model():
    model=tf.keras.models.load_model('mymodel.h5')
    return model


def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    return mfccs

def predict(model,wav_filepath):
    emotions={1 : 'neutral', 2 : 'calm', 3 : 'happy', 4 : 'sad', 5 : 'angry', 6 : 'fearful', 7 : 'disgust', 8 : 'surprised'}
    print(wav_filepath)
    test_point=extract_mfcc(wav_filepath)
    print("extract features")
    test_point=np.reshape(test_point,newshape=(1,40,1))
    print("tst points")
    predictions=model.predict(test_point)
    print(emotions[np.argmax(predictions[0])+1])
    
    return emotions[np.argmax(predictions[0])+1]

@app.route('/')
def index():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predictionHandler():
    file = request.files['audio']
    file.save(f'audios/{file.filename}')
    print("loading models")
    start_model_loading="model loading start..."
    model = load_model()
    start_model_loading="model loading completed"
    print("loading model end")
    emotionDetect=predict(model,file)
    print(emotionDetect)
    print("loaded success")

    # audio, _ = librosa.load('audio.wav', sr=22050)
    # mfccs = librosa.feature.mfcc(audio, sr=22050, n_mfcc=13)
    # mfccs_scaled = np.mean(mfccs.T, axis=0)
    # emotion = model.predict([mfccs_scaled])[0]
    # emotions = ['Angry', 'Fearful', 'Happy', 'Sad', 'Neutral']
    # predicted_emotion = emotions[emotion]
    
    return render_template('index.html', emotion=True,start_model_loading=start_model_loading)

if __name__ == '__main__':
    app.run(debug=True)