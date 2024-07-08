from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.models import load_model
from keras.applications.densenet import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
import numpy as np
import pickle
import io
from keras.models import Sequential, Model
from PIL import Image
from keras.layers import Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional
from keras.applications import VGG16, ResNet50, DenseNet201
model1 = load_model('model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# def preprocess_image(image):
#     img = Image.open(io.BytesIO(image))
#     img = img.convert('RGB')
#     img = img.resize((224, 224))
#     img = img_to_array(img)
#     img = img/255
#     #img = preprocess_input(img)
#     img = np.expand_dims(img, axis=0)
# #     return img
model = DenseNet201()
fe = Model(inputs=model.input, outputs=model.layers[-2].output)

# img_size = (224, 224)
# #features = {}

# def preprocess_image(img, img_size=(224, 224)):
#     #img = load_img(path, target_size=img_size)    
#     #img = img.convert('RGB')
#     img = img.resize(img_size)
#     #img = load_img(img,color_mode='rgb',target_size=(img_size,img_size))
     
#     img = img_to_array(img)
#     img = img/255.
#     img = np.expand_dims(img,axis=0)
#     feature = fe.predict(img, verbose=0)
#     #features[1] = feature
#     return feature
def preprocess_image(img, img_size=(224, 224)):
    #path=r'C:\Users\AHMII\OneDrive\Desktop\pic.jpg'
    #img = load_img(path, target_size=img_size)    
    img = Image.open(io.BytesIO(img))
    img = img.resize(img_size) 
    img = img_to_array(img)
    img = img/255.
    img = np.expand_dims(img, axis=0)
    feature = fe.predict(img, verbose=0)
    return feature

def generate_caption(image):
    photo = preprocess_image(image)
    in_text = 'start:'
    for i in range(0, 34):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=34)
        yhat = model1.predict([photo, sequence])
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

app = Flask(__name__)
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         image_path = request.form['image_path']
#         caption = generate_caption(image_path)
#         return render_template('result.html', caption=caption)
#     else:
#         return render_template('index.html')
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form['feedback']
    with open('feedback.txt', 'a') as f:
        f.write(feedback + '\n')
    return jsonify({'message': 'Feedback submitted successfully!'})
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        Fname = request.form['first_name_hidden']
        Lame = request.form['last_name_hidden']
        Email = request.form['email_hidden']

        # Save user input data to file
        with open('feedback.txt', 'a') as f:
            f.write(f"{Fname}, {Lame}, {Email}\n")
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded!'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected!'}), 400

        image = file.read()
        caption = generate_caption(image)

        return render_template('result.html', caption=caption)
    else:
        return render_template('index.html')
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
