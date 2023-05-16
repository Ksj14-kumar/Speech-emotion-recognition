from flask import  Flask, render_template,request
import tensorflow as tf

app= Flask(__name__)


def load_model():
    model=tf.keras.models.load_model('mymodel.h5')
    return model

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio']
    file.save(f'audios/{file.filename}')
    print("loading models")
    start_model_loading="loading models"
    # load_model()
    start_model_loading="model loading completed"

    # audio, _ = librosa.load('audio.wav', sr=22050)
    # mfccs = librosa.feature.mfcc(audio, sr=22050, n_mfcc=13)
    # mfccs_scaled = np.mean(mfccs.T, axis=0)
    # emotion = model.predict([mfccs_scaled])[0]
    # emotions = ['Angry', 'Fearful', 'Happy', 'Sad', 'Neutral']
    # predicted_emotion = emotions[emotion]
    
    return render_template('index.html', emotion=file,start_model_loading=start_model_loading)

if __name__ == '__main__':
    app.run(debug=True)