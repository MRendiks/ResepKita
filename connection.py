import mysql.connector as mc

def connect():
    # custom database
    return mc.connect(host="localhost",
                        user="root",
                        passwd="",
                        database="ta_resep")