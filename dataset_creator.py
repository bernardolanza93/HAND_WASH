import pandas as pd
from glob import glob
import os
import csv
#___________COFIG
path = "/home/bernardo/PycharmProjects/lava_mani1.1/"



#___________________APERTURA DATI UTENTI E SALVATAGGIO SU DICTIONARY__________
users = pd.read_csv(path + "user_data/dati_utenti.csv")
print(len(users))
user_dict = {}
for index, row in users.iterrows():



    element = str(row.values[0])
    element = element.split(";")
    user_dict[element[0]] = int(element[1])
#print(user_dict)



#_________________APERTURA FILE DATI PER CONCATENAZIONE_____________

# setting the path for joining multiple files
files = glob(path + "data/*.csv")
all_csv_files = []
for file in files:

    #____NAME____#
    file_name = (str(file).split("/"))[-1]
    filename = file_name.split("_")
    name = filename[0] +"_" + filename[1]


    #___user extract-----
    try:
        current_user = user_dict[name]
    except Exception as e:
        print(e)
        current_user = 100

    #print(current_user)

    #___OPEN DOC_____

    df = pd.read_csv(file)
    column_lenght = len(df)


    new_column = pd.DataFrame({'user': [current_user] * column_lenght})
    df.insert(loc=0, column='user', value=([current_user] * column_lenght))
    all_csv_files.append(df)

DATABASE = pd.concat(all_csv_files, ignore_index=False)
print(DATABASE)
DATABASE.to_csv('database.csv')