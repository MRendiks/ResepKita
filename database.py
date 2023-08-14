import mysql.connector as mc
import pickle
import numpy as np


def  dictionary(db):
    # membuat dictionary
    try:
        if db.is_connected():
            cursor = db.cursor()

            sql = f"SELECT id_cat, label FROM kategori_label"
            cursor.execute(sql)
            result = cursor.fetchall()
            
            d = {}
            for index in range(len(result)):
                d[f'{result[index][1]}'] = result[index][0]
            return d

    except mc.Error as e:
        print("Gagal saat menghubungkan ke MySQL", e)


def check(label, db):
    # Check ketersediaan kategori 
    try:
        if db.is_connected():
            cursor = db.cursor()
            sql = f"SELECT id_cat FROM kategori_label WHERE label = '{label}'"
            cursor.execute(sql)

            result = cursor.fetchone()

            return result

    except mc.Error as e:
        print("Gagal saat menghubungkan ke MySQL", e)