from flask import Flask, render_template, url_for, request, redirect, session, flash
from datetime import datetime
from flask_mysqldb import MySQL

# I/O python
import os, shutil, glob, time, pathlib, zipfile, shutil
from pathlib import Path
import os.path
import io
import base64

# Preproses Data
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders
import numpy as np
import pandas as pd
from tensorflow import keras
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
import math
import json
from werkzeug.utils import secure_filename
from  io import BytesIO
import base64

# Pembuatan Model
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Visualisasi
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import plotly.express as px 
from PIL import Image
import plotly
from keras.utils.vis_utils import plot_model
import pydot_ng as pydot
pydot.find_graphviz()

# EVALUASI
# from sklearn.metrics import confusion_matrix, classification_report
import itertools
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Program Local

from database import dictionary
from connection import connect
from feature_extractor import FeatureExtractor


app = Flask(__name__)
app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "ta_resep"
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']

STATIC_NAME = "static"
TEMPLATE_NAME = "templates"
BASE_URL_DATASET = 'static/dataset_ta'
URL_DATASET_COBA = 'static/dataset_ta_coba'
BASE_URL_DATA_RESEP = 'static/resep/dataset_resep.csv'
MODEL = keras.models.load_model("static/model/mobilenet_model_Adam__0.0001.h5")
DATASET = "static/dataset"
TOTAL_IMAGE_TRAIN = 4208
TOTAL_IMAGE_VAL = 1816
WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 10
STEP_EPOCH = (TOTAL_IMAGE_TRAIN - TOTAL_IMAGE_VAL) / BATCH_SIZE
TRAIN_PATH = DATASET + '/train'
VALID_PATH = DATASET + '/val'
RASIO_DATA_TRAIN = 0
RASIO_DATA_VAL = 0

mysql = MySQL(app)
fe = FeatureExtractor()
conn = connect()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        cur = mysql.connection.cursor()
        
        cur.execute(f"SELECT * FROM admin WHERE username='{username}' AND password='{password}'")   
        mysql.connection.commit()
        
        cur.close
        
        return redirect(url_for('dashboard'))
    else:
        return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    cur = mysql.connection.cursor()
        
    cur.execute("SELECT * FROM resep")   
    mysql.connection.commit()
    rows = cur.fetchall()
    cur.close
    total_dataset = cek_jumlah_data(URL_DATASET_COBA)
    return render_template('dashboard/index.html', total_dataset=total_dataset, total_resep = len(rows))

@app.route('/dashboard/train_model_page')
def train_model_page():
    return render_template('train_model/index.html')

@app.route('/dashboard/preprocessing_data_resep')
def preprocessing_data_resep():
    return render_template('preprocessing_data_resep/index.html')

@app.route('/dashboard/preprocessing_data_resep/prepro_data_resep')
def prepro_data_resep():
    
    data_title = []
    data_bahan = []
    data_langkah = []
    #membaca dataset
    data = pd.read_csv(BASE_URL_DATA_RESEP)

    #membuat regex untuk membersihkan karakter yang tidak penting di dataset
    pola1 = re.compile(r'--?|\\s')
    pola2 = re.compile(r'\.\.+')

    # Menampung nama resep agar bisa dijadikan resep utuh
    for nama in data['Title']:
        if isinstance(nama, str):
            nama = nama.encode("ascii", "ignore")
            nama = nama.decode()
            nama = re.sub(pola1, "\n", str(nama))
            nama = re.sub(pola2, " ", str(nama)) 
            nama = nama.replace('.', '')
            nama = nama.replace("'", "")
            nama = nama.replace('"', '')
            data_title.append(nama)

    # Melakukan preprocessing pada dataset Bahan menggunakan looping  
    for bahan in data['Ingredients']:
        if isinstance(bahan, str):
            bahan = bahan.encode("ascii", "ignore")
            bahan = bahan.decode()
            bahan = re.sub(pola1, "\n", str(bahan))
            bahan = re.sub(pola2, " ", str(bahan))
            bahan = bahan.lower()
            bahan = bahan.replace('bahan', '')
            bahan = bahan.replace(' : ', '')
            bahan = bahan.replace(':', '')
            bahan = bahan.replace('hanya', '')
            bahan = bahan.replace('(', '')
            bahan = bahan.replace(')', '') 
            bahan = bahan.replace('1/2', 'setengah') 
            bahan = bahan.replace('/', '')
            bahan = bahan.replace('sdt', 'satu sendok teh')
            bahan = bahan.replace('cabe', 'cabai')
            bahan = bahan.replace("'", "")
            bahan = bahan.replace('"', '')  
            
            bahan = bahan.strip()
            data_bahan.append(bahan)

    # Melakukan preprocessing pada dataset Langkah menggunakan looping  
    for steps in data['Steps']:
        if isinstance(steps, str):
            steps = steps.encode("ascii", "ignore")
            steps = steps.decode()
            steps = re.sub(pola1, "\n", str(steps))
            steps = re.sub(pola2, " ", str(steps))
            steps = steps.lower()
            steps = steps.replace('b.', 'bawang')
            steps = steps.replace('taraaaaaaaaaaaaaa', 'tara')
            steps = steps.replace("'", "")
            steps = steps.replace('"', '')  
            data_langkah.append(steps)
            
    # untuk menampilkan banyak data resep
    total_data = len(data)
    
    # untuk menampilkan proses 1
    prepro1 = data['Steps'][0]
    prepro2 = re.sub(pola1, "\n", str(prepro1))
    
    
    # untuk menampilkan proses 2
    prepro3 = re.sub(pola2, " ", str(prepro2))
    
    
    # untuk menampilkan proses 3
    prepro4 = prepro3.encode("ascii", "ignore")
    prepro4 = prepro4.decode()
    
    # Untuk menampilkan proses 4
    prepro5 = prepro4.replace('taraaaaaaaaaaaaaa', 'tara')
    
    # Untuk list gambar
    list_gambar = []
    for i in range(1, 80+1):
        list_gambar.append("resep"+str(i)+".jpg")

    # Mengubah hasil pengeolahan menjadi dataframe
    df = pd.DataFrame(list(zip(data_title, data_bahan, data_langkah, list_gambar)),
                columns =['nama resep', 'bahan', 'langkah', 'gambar'])
    
    cur = mysql.connection.cursor()
        
    
    # looping untuk menginput data ke database
    cur.execute("SELECT COUNT(*) FROM resep")
    data = cur.fetchall()
    mysql.connection.commit()
    
    if data[0][0] < 79:
        for index in df.index:
            cur.execute(f"INSERT INTO resep VALUES('', '{df['nama resep'][index]}', '{df['bahan'][index]}', '{df['langkah'][index]}', '{df['gambar'][index]}')") 
            mysql.connection.commit()
    
    
    
    return render_template('preprocessing_data_resep/index.html', 
                           total_data = total_data,
                           prepro1 = prepro1,
                           prepro2 = prepro2,
                           prepro3 = prepro3,
                           prepro4 = prepro4,
                           prepro5 = prepro5)

# DATA USER
@app.route('/dashboard/user')
def list_user():
    cur = mysql.connection.cursor()
    cur.execute(f"SELECT * FROM users WHERE username != 'admin'")
    data = cur.fetchall()
    mysql.connection.commit()
    return render_template("user/index.html", data=data)
    

# DATA RESEP
@app.route('/dashboard/resep')
def list_resep():
    cur = mysql.connection.cursor()
    cur.execute(f"SELECT * FROM resep")
    data = cur.fetchall()
    mysql.connection.commit()
    return render_template("resep/index.html", data=data)



@app.route('/dashboard/train_model_page/load_train_model', methods=['GET', 'POST'])
def load_train_model():
    if request.method == 'POST':
        rasio1 = request.form['rasio_data_train'] 
        rasio2 = request.form['rasio_data_testing']
        
        if ( int(rasio1) + int(rasio2) != 100 ):
            return render_template('train_model/index.html', error=True)
        else:
            shutil.rmtree(DATASET+'/'+"train")
            shutil.rmtree(DATASET+'/'+"val")
            
            RASIO_DATA_TRAIN = int(rasio1) / 100
            RASIO_DATA_VAL = int(rasio2) / 100
            total_dataset = cek_jumlah_data(URL_DATASET_COBA)
            cnn_model(RASIO_DATA_TRAIN, RASIO_DATA_VAL)
            
            total_dataset_train , total_dataset_val = cek_jumlah_data_train_test(DATASET)
            ratio = f"{rasio1} & {rasio2}"
            
            # Visualisasi Dataset
            label = []
            count = []
            for dir in os.listdir(URL_DATASET_COBA):
                label.append(dir)
                count.append(len(os.listdir(f'{URL_DATASET_COBA}/{dir}')))
                
            fig = px.bar(x=label, y=count, title="Jumlah Data pada Kategori")
            graph1JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            
            return render_template('train_model/index.html', total_dataset = total_dataset, train_data=total_dataset_train, val_data=total_dataset_val, ratio=ratio, graph1JSON = graph1JSON)


@app.route('/dashboard/train_model_page/config_cnn', methods=['GET', 'POST'])
def config_cnn():
    if request.method == 'POST':
        conv1       = int(request.form['conv1'])
        conv2       = int(request.form['conv2'])
        conv3       = int(request.form['conv3'])
        lr          = float(request.form['lr'])
        optimizer   = request.form['optimizer']
        EPOCH       = int(request.form['epoch'])
        
        model = tf.keras.models.Sequential([
                    tf.keras.layers.Conv2D(conv1, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, 3)),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Conv2D(conv2, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Conv2D(conv3, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.20),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512,activation = "relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.20),
                    tf.keras.layers.Dense(512,activation = "relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.20),
                    tf.keras.layers.Dense(14, activation='softmax')
                    ])
        if optimizer == "Adam":
            model.compile(optimizer=Adam(learning_rate=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        else:
            model.compile(optimizer=RMSprop(learning_rate=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        
        save_model = f'my_model.h5'
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        modelCheckpoint = ModelCheckpoint(save_model, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
        reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_delta=1e-5, mode='min')

        train_images, valid_images = cnn_model(0.9, 0.1)
        
        results = model.fit(
            train_images,
            # steps_per_epoch=STEP_EPOCH,
            validation_data=valid_images,
            epochs=EPOCH,
            callbacks=[earlyStopping, modelCheckpoint, reducelr],
            verbose = 1
            )
        
        # menyimpan hasil pelatihan ke variabel
        epochs = results.epoch
        loss = results.history['loss']
        accuracy = results.history['accuracy']
        best_accuracy = round( max(results.history['accuracy']) * 100,2)
        best_accuracy = f'{best_accuracy} %'
        best_loss = round(min(results.history['loss']) * 100, 2)
        best_loss = f'{best_loss} %'

        # menggabungkan metrics kedalam satu array
        data = np.column_stack((epochs, loss, accuracy))

        # Simpan data kedalam txt
        np.savetxt('training_history.txt', data, fmt='%.4f', header='Epoch\tLoss\tAccuracy', delimiter='\t')
        
        # Untuk Diagram Error
        plt.subplots()
        plt.title("Grafik kesalahan pelatihan model")
        plt.plot(results.history["loss"], label="train_loss")
        plt.plot(results.history["val_loss"], label="val_loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Save the plot to a temporary buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Encode the plot image as a base64 string
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        
        plt.subplots()
        plt.title("Grafik akurasi pelatihan model")
        plt.plot(results.history["accuracy"], label="train_accuracy")
        plt.plot(results.history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("accuracy")
        plt.legend()

        # Save the plot to a temporary buffer
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)

        # Encode the plot image as a base64 string
        plot_data2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
        
        return render_template('train_model/plot.html', data="yes", conv1=conv1, conv2=conv2, conv3=conv3, lr=lr, optimizer=optimizer, EPOCH=EPOCH, plot_data = plot_data, plot_data2 = plot_data2, best_accuracy = best_accuracy, best_loss= best_loss)
    else:
        return render_template('train_model/index.html')

def cek_jumlah_data(path):
    total_gambar = 0
    for item in os.listdir(path):
        if len(item) > 7 :
            total_gambar += len(os.listdir(path + '/' + item))
        else:
            total_gambar += len(os.listdir(path + '/' + item))
    return total_gambar

def cek_jumlah_data_train_test(path):
    total_gambar_train = 0
    total_gambar_val = 0
    for item in os.listdir(path):
        if item == 'train':
            for data in os.listdir(path+'/'+item):
                total_gambar_train += len(os.listdir(f'{path}/{item}/{data}'))
        else:
            for data in os.listdir(path+'/'+item):
                total_gambar_val += len(os.listdir(f'{path}/{item}/{data}'))
    
    return total_gambar_train, total_gambar_val


def pembagian_dataset(path, ratio1, ratio2):
    splitfolders.ratio(URL_DATASET_COBA, DATASET, seed=1337, ratio=(ratio1, ratio2))


def cnn_model(ratio1, ratio2):
    pembagian_dataset(URL_DATASET_COBA, ratio1, ratio2)

    train_datagen = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input,
        shear_range=0.2, 
        zoom_range=0.2, 
        rotation_range=10,
        vertical_flip=True,
        horizontal_flip=True
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
    )
    train_images = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(WIDTH, HEIGHT),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=0
    )

    valid_images = valid_datagen.flow_from_directory(
        VALID_PATH,
        target_size=(WIDTH, HEIGHT),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    return train_images, valid_images

@app.route('/dashboard/rekomendasi')
def rekomendasi():
    return render_template('rekomendasi/index.html')


def cosine_score(X, Y):
    # tokenisasi data bahan
    X_list = word_tokenize(X) 
    Y_list = word_tokenize(Y)
        
    # daftar list filter bahan penelitian untuk menghemat komputasi
    bahan = ['bawang','merah', 'bawang', 'putih', 'bombai', 'cabai','hijau', 'cabai', 'jagung', 'jahe', 'kembang', 'kol','kentang', 'kubis', 'terong', 'timun', 'tomat','wortel']
    l1 =[];l2 =[]
        
    # mengfilter kata yang ada pada
    X_set = {w for w in X_list if w in bahan} 
    Y_set = {w for w in Y_list if w in bahan}
        
    # form a set containing keywords of both strings 
    rvector = X_set.union(Y_set) 
    for w in rvector:
        if w in X_set: l1.append(1) # membuat vektor dokumen 1
        else: l1.append(0)
        if w in Y_set: l2.append(1) # membuat vektor dokumen 2
        else: l2.append(0)
    X_list = np.array(l1)
    Y_list = np.array(l2)
    
    cosine = np.dot(X_list,Y_list)/(norm(X_list)*norm(Y_list))
    dot = np.dot(X_list,Y_list)
    norm1 = norm(X_list)
    norm2 = norm(Y_list)
    normal = (norm(X_list)*norm(Y_list))
    return cosine, dot, normal, norm1, norm2, X_list, Y_list

def transform_image(pillow_image):
    data = np.asarray(pillow_image)
    data = np.expand_dims(data, axis=0)
    images = np.vstack([data])
    images = tf.image.resize(data, [224, 224])
    return images

def predict(x):
    predictions = MODEL(x)
    pred = predictions[0]
    label = np.argmax(pred)

    sort = np.argsort(pred)
    largest_indices = sort[::-1]
    ranked = largest_indices[:1]

    class_dictionary = dictionary(conn)

    # print(class_dictionary)
    key_list = list(class_dictionary.keys())
    val_list = list(class_dictionary.values())

    prob = []
    for x in ranked:
        position = val_list.index(x)
        prob.append(key_list[position])

    return label, ', '.join(prob), predictions, largest_indices, class_dictionary

def convert_dist(old_value):
    old_min = 1
    old_max = 0
    new_min = 0
    new_max = 100
    return round(((old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min, 2)

# FITUR UTAMA
@app.route('/dashboard/rekomendasi_resep')
def rekomendasi_resep():
    return render_template('rekomendasi_resep/index.html')

@app.route('/dashboard/rekomendasi_resep/req', methods=['GET', 'POST'])
def req():
    if request.method == 'POST':

        uploaded_file = request.files["query_img"]

        filename = secure_filename(uploaded_file.filename)
        # check condition submit empty file or not
        if filename != '':
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return render_template("index.html", ext='Uploaded file is not a valid image. Only JPG, JPEG and PNG files are allowed')
            else:
                # open image PIL
                img = Image.open(uploaded_file.stream)
                rgb_pred = img.convert('RGB')
                rgb_blob = img.convert('RGB')

                # image to blob
                buffered = BytesIO()
                rgb_blob.save(buffered, format="JPEG")
                img_encode = base64.b64encode(buffered.getvalue())
                # decode image base64 to load on HTML
                img_decode = img_encode.decode("utf-8")
            
                # Prediction class
                tensor = transform_image(rgb_pred)
                # return id category and probability classes
                category, prob, pred, sort, dictionary = predict(tensor)
                

                Y = prob + " " + "jagung kentang"
                cur = mysql.connection.cursor()
                cur.execute(f"DELETE FROM rekomendasi")
                mysql.connection.commit()  
                
                cur.execute(f"SELECT id_resep, nama_resep, bahan FROM resep")   
                data = cur.fetchall()
                mysql.connection.commit()
                for item in data:
                    cosine, dot, normal, norm1, norm2, X_list, Y_list = cosine_score(item[2], Y)
                    if math.isnan(cosine):
                        pass
                    elif cosine == 0:
                        pass
                    else:
                        cur.execute(f"INSERT INTO rekomendasi VALUES('', '{item[0]}', 1 , '{item[1]}' , '{round(100*cosine)}')")
                        mysql.connection.commit()
                cur.execute(f"SELECT nama_resep, nilai_cosine FROM rekomendasi ORDER BY nilai_cosine DESC limit 5")
                data = cur.fetchall()
                mysql.connection.commit()
                cur.close
                
                req = mysql.connection.cursor()
                req.execute(f"SELECT resep.id_resep, rekomendasi.nama_resep, resep.bahan, resep.langkah, resep.gambar FROM rekomendasi INNER JOIN resep ON rekomendasi.id_resep = resep.id_resep ORDER BY nilai_cosine DESC limit 1;")
                
                mycosine = req.fetchall()
                mysql.connection.commit()
                
                list_hitung_cosing = []
                for item in mycosine:
                    cosine, dot, normal, norm1, norm2, X_list, Y_list = cosine_score(item[2], Y)
                    values = {
                        "nama_resep" : item[1],
                        "cosine" : cosine,
                        "ubah" : round(cosine*100),
                        "dot" : dot,
                        "normal" : round(normal,2),
                        "norm1" : round(norm1,2),
                        "norm2" : round(norm2,2),
                        "X_list" : X_list,
                        "Y_list" : Y_list
                    }
                    
                
                return render_template("rekomendasi_resep/index.html", query_path=img_decode, probability=prob, tensor=tensor.numpy(), pred=pred.numpy(), sort=sort, dictionary=dictionary, category=category, data=data, values=values)
        else: 
            return render_template("rekomendasi_resep/index.html", ext = "Please select image file to upload")
    else: 
        return render_template("rekomendasi_resep/index.html")



# API
# GET RESEP
@app.route('/api/get_resep', methods=['GET', 'POST'])
def get_resep():
    if request.method == 'POST':
        bahan1 = request.form['bahan1']
        bahan2 = request.form['bahan2']
        bahan3 = request.form['bahan3']
        bahan4 = request.form['bahan4']
        bahan5 = request.form['bahan5']
        bahan6 = request.form['bahan6']
        kategori = request.form['kategori']
        mybahan = [bahan1, bahan2, bahan3, bahan4, bahan5, bahan6]
        Y = bahan1 + " " + bahan2 + " " + bahan3 + " " + bahan4 + " " + bahan5 + " " + bahan6
        Y = Y.lower()

        cur = mysql.connection.cursor()
        cur.execute(f"DELETE FROM rekomendasi")
        mysql.connection.commit()  
        
        cur.execute(f"SELECT id_resep, nama_resep, bahan FROM resep WHERE kategori='{kategori}'")   
        data = cur.fetchall()
        mysql.connection.commit()
        for item in data:
            cosine, dot, normal, norm1, norm2, X_list, Y_list = cosine_score(item[2], Y)
            if math.isnan(cosine):
                pass
            elif cosine == 0:
                pass
            else:
                
                cur.execute(f"INSERT INTO rekomendasi VALUES('', '{item[0]}', 1,'{item[1]}' , '{round(100*cosine)}')")
                mysql.connection.commit()
        cur.execute(f"SELECT resep.id_resep, rekomendasi.nama_resep, resep.bahan, resep.langkah, resep.gambar FROM rekomendasi INNER JOIN resep ON rekomendasi.id_resep = resep.id_resep ORDER BY nilai_cosine DESC limit 5;")
        
        data = cur.fetchall()
        mysql.connection.commit()
        cur.close
        list_resep = []
        
        for item in data:
            resep = {
                "id_resep" : item[0],
                "nama_resep" : item[1],
                "bahan" : item[2],
                "langkah" : item[3],
                "gambar" : item[4],
                
            }
            list_resep.append(resep)
        
        
        return json.dumps(list_resep)

# LOGIN SERVICE
@app.route('/api/login_service', methods=['GET', 'POST'])
def login_service():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute(f"SELECT * FROM users WHERE username='{username}' AND password='{password}'")
        data = cur.fetchall()
        mysql.connection.commit()
        # response = []
        if data:
            for item in data:
                akun = {
                    "response" : True,
                    "username" : item[1]
                }

                
        return json.dumps(akun)

# REGISTRASI ACCOUNT SERVICE
@app.route('/api/register_service', methods=['GET', 'POST'])
def register_service():
    if request.method == 'POST':
        
        username = request.form['username']
        password = request.form['password']
        
        cur = mysql.connection.cursor()
        cur.execute(f"SELECT * FROM users WHERE username='{username}' AND password='{password}'")
        data = cur.fetchall()
        if data:
            akun = {
                "response" : False
            }
            return json.dumps(akun)
        else:
            insert = mysql.connection.cursor()
            insert.execute(f"INSERT INTO users VALUES('', '{username}', '{password}')")
            mysql.connection.commit()
            akun = {
                "response" : True
            }
                    
            return json.dumps(akun)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")




# @app.route('/dashboard/rekomendasi/cosine_similarity', methods=['GET', 'POST'])
# def cosine_similarity():
#     if request.method == 'POST':
#         bahan1 = request.form['bahan1']
#         bahan2 = request.form['bahan2']
#         bahan3 = request.form['bahan3']
#         bahan4 = request.form['bahan4']
#         bahan5 = request.form['bahan5']
#         bahan6 = request.form['bahan6']
#         Y = bahan1 + " " + bahan2 + " " + bahan3 + " " + bahan4 + " " + bahan5 + " " + bahan6
#         Y = Y.lower()
        
#         cur = mysql.connection.cursor()
#         cur.execute(f"DELETE FROM rekomendasi")
#         mysql.connection.commit()  
        
#         cur.execute(f"SELECT id_resep, nama_resep, bahan FROM resep")   
#         data = cur.fetchall()
#         mysql.connection.commit()
#         for item in data:
#             cosine, dot, normal, norm1, norm2, X_list, Y_list = cosine_score(item[2], Y)
#             if math.isnan(cosine):
#                 pass
#             elif cosine == 0:
#                 pass
#             else:
#                 cur.execute(f"INSERT INTO rekomendasi VALUES('', '{item[0]}', 1 , '{item[1]}' , '{round(100*cosine)}')")
#                 mysql.connection.commit()
#         cur.execute(f"SELECT nama_resep, nilai_cosine FROM rekomendasi ORDER BY nilai_cosine DESC limit 5")
#         data = cur.fetchall()
#         mysql.connection.commit()
#         cur.close
        
#         req = mysql.connection.cursor()
#         req.execute(f"SELECT resep.id_resep, rekomendasi.nama_resep, resep.bahan, resep.langkah, resep.gambar FROM rekomendasi INNER JOIN resep ON rekomendasi.id_resep = resep.id_resep ORDER BY nilai_cosine DESC limit 1;")
        
#         mycosine = req.fetchall()
#         mysql.connection.commit()
        
#         list_hitung_cosing = []
#         for item in mycosine:
#             cosine, dot, normal, norm1, norm2, X_list, Y_list = cosine_score(item[2], Y)
#             values = {
#                 "nama_resep" : item[1],
#                 "cosine" : cosine,
#                 "ubah" : round(cosine*100),
#                 "dot" : dot,
#                 "normal" : round(normal, 2),
#                 "norm1" : round(norm1, 2),
#                 "norm2" : round(norm2, 2),
#                 "X_list" : X_list,
#                 "Y_list" : Y_list
#             }
#         return render_template('rekomendasi/index.html', data=data, values=values)