
import face_recognition
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
import svm as s
import main as m

def get_gropped_face(image_file):
    image = face_recognition.load_image_file(image_file)
    face_locations = face_recognition.face_locations(image)
    a, b, c, d = face_locations[0]
    cropped_face = image[a:c,d:b,:]

    return cropped_face

def get_face_embedding(face):
    width = face.shape[1]
    height = face.shape[0]
    return face_recognition.face_encodings(face, known_face_locations=[(0, width, height, 0)])

# def DB(animal,tableName):
#     dir_path = f'./images/{tableName}'
#     embedding_dict = get_face_embedding_dict(dir_path)
#     HOST = 'localhost'
#     PORT = 3306
#     USER = 'root'
#     PASSWORD = 'mysql'
#     DATABASE = 'animal'
#
#     db = pymysql.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, database=DATABASE)
#     cursor = db.cursor()
#     for key,value in embedding_dict.items():
#         for i, v in enumerate(value, 1):
#             if i < 129:
#                 v = str(v)
#                 cursor.execute(
#                 f'INSERT INTO woman_{animal}_{tableName} (no, {key}) VALUES ({i}, {v}) ON DUPLICATE KEY UPDATE {key} = {v}')
#
#
#     db.commit()
#     db.close()

# images 디렉토리 안에 있는 모든 이미지 파일의 임베딩을 구해서 dict 구조에 담아 리턴하는 함수
def get_face_embedding_dict(dir_path):
    file_list = os.listdir(dir_path)
    embedding_dict = {}

    for file in file_list:
        img_path = os.path.join(dir_path, file)
        face = get_gropped_face(img_path)
        embedding = get_face_embedding(face)
        if len(embedding) > 0:  # 얼굴영역 face가 제대로 detect되지 않으면  len(embedding)==0인 경우가 발생하므로
                    # os.path.splitext(file)[0]에는 이미지파일명에서 확장자를 제거한 이름이 담깁니다.
            embedding_dict[os.path.splitext(file)[0]] = embedding[0]

    return embedding_dict


image_path = m.MainApp.name
print(image_path)
man = False
woman = False
cropped_face = get_gropped_face(image_path)

embedding = get_face_embedding(cropped_face)
pd_embedding = pd.DataFrame(embedding, columns=range(1,129))
if man == True:
    s.man(pd_embedding)
elif woman == True:
    s.woman(pd_embedding)

# DB("snake","yunju")

# dir_path = f'./images/test'
# embedding_dict = get_face_embedding_dict(dir_path)
#
# HOST = 'localhost'
# PORT = 3306
# USER = 'root'
# PASSWORD = 'mysql'
# DATABASE = 'animal'
#
# db = pymysql.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, database=DATABASE)
# cursor = db.cursor()
#
# for key in embedding_dict.keys():
#     for value in embedding_dict.values():
#         for i, v in enumerate(value, 1):
#             if i < 129:
#                 v = str(v)
#                 cursor.execute(
#                     f'INSERT INTO test (no, {key}) VALUES ({i}, {v}) ON DUPLICATE KEY UPDATE {key} = {v}')
#         break
#
# db.commit()
# db.close()