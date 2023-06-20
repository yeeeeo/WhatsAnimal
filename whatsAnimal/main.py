import math

from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager,Screen
from kivy.clock import Clock
import face_recognition
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
Window.size = (390, 843)

class get_animal:
    def __init__(self, pd_embedding):
        self.pd_embedding = pd_embedding

    def man(self):
        man_embedding = pd.read_csv('man_embedding.csv')
        vector = range(1, 129)
        manEmbeddingVector = man_embedding.iloc[:, vector]
        manEmbeddingResult = man_embedding[['result']]
        man_feature_train, man_feature_test, man_target_train, man_target_test = train_test_split(manEmbeddingVector,
                                                                                                  manEmbeddingResult,
                                                                                                  test_size=0.34)
        man_svm_model = SVC(kernel="rbf", C=80, gamma=0.1)
        man_svm_model.fit(man_feature_train, man_target_train)  # 학습모델
        print(self.pd_embedding)
        man_target_pred = man_svm_model.predict(self.pd_embedding)  # 테스트
        man_percent = man_svm_model.decision_function(self.pd_embedding)
        man_percent = {"cat_man": man_percent[0,0], "dino_man": man_percent[0,1],
                       "dog_man": man_percent[0,2],"monkey_man": man_percent[0,3],
                       "snake_man": man_percent[0,4]}
        man_target_pred = man_target_pred.tolist()
        man_target_pred_dic = {1: ''.join(man_target_pred)}
        man_percent_result = {"cat_man": 0, "dino_man": 0, "dog_man": 0,"monkey_man": 0, "snake_man": 0}
        for i in man_percent.keys():
            if(man_percent[i]<0):
                man_percent[i] = 0
            else:
                man_percent[i] = round(man_percent[i],2)
                if man_target_pred_dic[1] == i:
                    man_percent_result[i] = 100
                else:
                    man_percent_result[i] = math.trunc(man_percent[i]/man_percent[man_target_pred_dic[1]]*100)

        print(man_percent_result)


        return man_target_pred, man_percent_result

    def woman(self):
        girl_embedding = pd.read_csv('girl_embedding.csv')
        vector = range(1, 129)
        girlEmbeddingVector = girl_embedding.iloc[:, vector]
        girlEmbeddingResult = girl_embedding[['result']]
        girl_feature_train, girl_feature_test, girl_target_train, girl_target_test = train_test_split(
            girlEmbeddingVector,
            girlEmbeddingResult,
            test_size=0.34)
        girl_svm_model = SVC(kernel="rbf", C=80, gamma=0.1)
        girl_svm_model.fit(girl_feature_train, girl_target_train)  # 학습모델
        girl_target_pred = girl_svm_model.predict(self.pd_embedding)  # 테스트
        girl_percent = girl_svm_model.decision_function(self.pd_embedding)
        print(girl_percent)
        girl_percent = {"cat_girl": girl_percent[0, 0], "dog_girl": girl_percent[0, 1],
                       "fox_girl": girl_percent[0, 2], "fish_girl": girl_percent[0, 3],
                       "snake_girl": girl_percent[0, 4]}
        girl_target_pred = girl_target_pred.tolist()
        girl_target_pred_dic = {1: ''.join(girl_target_pred)}
        girl_percent_result = {"cat_girl": 0, "dog_girl": 0, "fox_girl": 0, "fish_girl": 0, "snake_girl": 0}

        for i in girl_percent.keys():
            if (girl_percent[i] < 0):
                girl_percent[i] = 0
            else:
                girl_percent[i] = round(girl_percent[i], 2)
                if girl_target_pred_dic[1] == i:
                    girl_percent_result[i] = 100
                else:
                    girl_percent_result[i] = math.trunc(girl_percent[i] / girl_percent[girl_target_pred_dic[1]] * 100)

        print(girl_percent_result)
        return girl_target_pred, girl_percent_result


class get_embedding:
    def __init__(self, image_path):
        self.image_path = image_path

    def start(self):
        cropped_face = self.get_gropped_face(self.image_path)
        embedding = self.get_face_embedding(cropped_face)
        pd_embedding = pd.DataFrame(embedding, columns=range(1, 129))
        return pd_embedding

    def get_gropped_face(self, image_file):
        image = face_recognition.load_image_file(image_file)
        face_locations = face_recognition.face_locations(image)
        a, b, c, d = face_locations[0]
        cropped_face = image[a:c, d:b, :]

        return cropped_face

    def get_face_embedding(self, face):
        width = face.shape[1]
        height = face.shape[0]
        return face_recognition.face_encodings(face, known_face_locations=[(0, width, height, 0)])


class GallaryScreen(Screen):
    pass
class LoadingScreen(Screen):
    def main(self, *args):
        screen_manager.current = "resultScreen"
    def on_enter(self):
        self.ids.loadingImage.anim_delay = 0.10
        Clock.schedule_once(self.main, 3)

class ResultScreen(Screen):
    pass

class DetailScreen(Screen):
    pass
class MainApp(MDApp):
    def manIcon_Button_clicked(self):
        if self.root.get_screen('main').ids.genderClick.text == "":
            self.root.get_screen('main').ids.manIcon_Button.source = "manIconSelected.png"
            self.root.get_screen('main').ids.genderClick.text = "manClick"
        else:
            self.root.get_screen('main').ids.manIcon_Button.source = "manIcon.png"
            self.root.get_screen('main').ids.genderClick.text = ""

    def womanIcon_Button_clicked(self):
        if self.root.get_screen('main').ids.genderClick.text == "":
            self.root.get_screen('main').ids.womanIcon_Button.source = "womanIconSelected.png"
            self.root.get_screen('main').ids.genderClick.text = "womanClick"
        else:
            self.root.get_screen('main').ids.womanIcon_Button.source = "womanIcon.png"
            self.root.get_screen('main').ids.genderClick.text = ""

    def gallary_clicked(self, name):
        self.root.get_screen('resultScreen').ids.selectedImage.source = name
        self.root.get_screen('detailScreen').ids.resultSelectedImage.source = name

        g = get_embedding(name)
        e = g.start()
        if self.root.get_screen('main').ids.genderClick.text == "manClick":
            man = get_animal(e)
            man_animal, man_percent_result = man.man()
            self.root.get_screen('detailScreen').ids.animalName1.text = "강아지상"
            self.root.get_screen('detailScreen').ids.animalName2.text = "고양이상"
            self.root.get_screen('detailScreen').ids.animalName3.text = "공룡상"
            self.root.get_screen('detailScreen').ids.animalName4.text = "원숭이상"
            self.root.get_screen('detailScreen').ids.animalName5.text = "뱀상"
            for i in man_percent_result.keys():
                if i == "dog_man":
                    if man_percent_result[i] == 0:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "0percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 20:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "10percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 30:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "20percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 40:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "30percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 50:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "40percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 60:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "50percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 70:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "60percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 80:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "70percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 90:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "80percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 100:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "90percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] == 100:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "100percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(man_percent_result[i])+"%"

                if i == "cat_man":
                    if man_percent_result[i] == 0:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "0percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 20:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "10percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 30:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "20percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 40:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "30percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 50:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "40percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 60:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "50percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 70:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "60percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 80:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "70percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 90:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "80percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 100:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "90percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] == 100:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "100percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(man_percent_result[i])+"%"

                if i == "dino_man":
                    if man_percent_result[i] == 0:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "0percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 20:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "10percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 30:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "20percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 40:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "30percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 50:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "40percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 60:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "50percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 70:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "60percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 80:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "70percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 90:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "80percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 100:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "90percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] == 100:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "100percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(man_percent_result[i])+"%"

                if i == "monkey_man":
                    if man_percent_result[i] == 0:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "0percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 20:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "10percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 30:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "20percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 40:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "30percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 50:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "40percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 60:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "50percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 70:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "60percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 80:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "70percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 90:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "80percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 100:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "90percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] == 100:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "100percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(man_percent_result[i])+"%"

                if i == "snake_man":
                    if man_percent_result[i] == 0:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "0percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 20:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "10percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 30:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "20percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 40:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "30percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 50:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "40percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 60:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "50percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 70:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "60percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 80:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "70percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 90:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "80percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] < 100:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "90percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(man_percent_result[i])+"%"
                    elif man_percent_result[i] == 100:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "100percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(man_percent_result[i])+"%"
            if man_animal == ['dog_man']:
                self.root.get_screen('resultScreen').ids.resultAnimal.text = "당신은 강아지상입니다!"
                self.root.get_screen('detailScreen').ids.resultAnimall.text = "당신은 강아지상입니다!"
                self.root.get_screen('detailScreen').ids.detailComent1.text = "다정다감하고 귀여운 당신은"
                self.root.get_screen('detailScreen').ids.detailComent2.text = "모든 사람들에게 즐거움을 주는 호감형이다!"
                self.root.get_screen('detailScreen').ids.detailComent3.text = "상냥하고 활발한 성격으로 인기폭발로"
                self.root.get_screen('detailScreen').ids.detailComent4.text = "애교와 웃음이 많아 사랑스럽다."
                self.root.get_screen('detailScreen').ids.detailComent5.text = "강아지상 남자 연예인"
                self.root.get_screen('detailScreen').ids.detailComent6.text = "강다니엘, 김혜성, 남주혁, 박보검, 백현,"
                self.root.get_screen('detailScreen').ids.detailComent7.text = "송중기, 이제훈, 임시완, 정해인, 지성"
                self.root.get_screen('detailScreen').ids.Danimal.source = "Ddog.png"




            elif man_animal == ['cat_man']:
                self.root.get_screen('resultScreen').ids.resultAnimal.text = "당신은 고양이상입니다!"
                self.root.get_screen('detailScreen').ids.resultAnimall.text = "당신은 고양이상입니다!"
                self.root.get_screen('detailScreen').ids.detailComent1.text = "도도하고 섹시한 당신의 첫인상은"
                self.root.get_screen('detailScreen').ids.detailComent2.text = "차가워 보이지만 묘한 매력을 풍겨 인기만점!"
                self.root.get_screen('detailScreen').ids.detailComent3.text = "시크한 츤데레로 연인에게 끊임없이"
                self.root.get_screen('detailScreen').ids.detailComent4.text = "설렘을 안겨주는 당신은 고양이와 닮았다!"
                self.root.get_screen('detailScreen').ids.detailComent5.text = "고양이상 남자 연예인"
                self.root.get_screen('detailScreen').ids.detailComent6.text = "강다니엘, 서강준, 슈가, 시우민, 안재현,"
                self.root.get_screen('detailScreen').ids.detailComent7.text = "우지, 인성, 장기용, 지코, 황민현"
                self.root.get_screen('detailScreen').ids.Danimal.source = "Dcat.png"
            elif man_animal == ['snake_man']:
                self.root.get_screen('resultScreen').ids.resultAnimal.text = "당신은 뱀상입니다!"
                self.root.get_screen('detailScreen').ids.resultAnimall.text = "당신은 뱀상입니다!"
                self.root.get_screen('detailScreen').ids.detailComent1.text = "시크하고 날카로운 이미지를 소유한 당신은"
                self.root.get_screen('detailScreen').ids.detailComent2.text = "첫인상은 다가가기 어려울 수 있지만"
                self.root.get_screen('detailScreen').ids.detailComent3.text = "가까워지면 한없이 따뜻하다. "
                self.root.get_screen('detailScreen').ids.detailComent4.text = "상대를 홀리는 힘이 강력하다!"
                self.root.get_screen('detailScreen').ids.detailComent5.text = "뱀상 남자 연예인"
                self.root.get_screen('detailScreen').ids.detailComent6.text = "진영, 권현빈, 뉴, 박시후, 서인국,"
                self.root.get_screen('detailScreen').ids.detailComent7.text = "우도환, 이준기, 케이, 홍종현"
                self.root.get_screen('detailScreen').ids.Danimal.source = "Dsnake.png"
            elif man_animal == ['monkey_man']:
                self.root.get_screen('resultScreen').ids.resultAnimal.text = "당신은 원숭이상입니다!"
                self.root.get_screen('detailScreen').ids.resultAnimall.text = "당신은 원숭이상입니다!"
                self.root.get_screen('detailScreen').ids.detailComent1.text = "재주가 많고 머리가 비상한 당신은"
                self.root.get_screen('detailScreen').ids.detailComent2.text = "리더성향이 강하여 어딜가든 몰입을 잘하고"
                self.root.get_screen('detailScreen').ids.detailComent3.text = "책임감을 가지고 살아간다."
                self.root.get_screen('detailScreen').ids.detailComent4.text = "결단력이 뛰어나 후회를 잘 하지 않는 편!"
                self.root.get_screen('detailScreen').ids.detailComent5.text = "원숭이상 남자 연예인"
                self.root.get_screen('detailScreen').ids.detailComent6.text = "빈지노, 신하균, 옹성우, 유아인, 이승기,"
                self.root.get_screen('detailScreen').ids.detailComent7.text = "임영웅, 지드래곤, 찬열, 크러쉬, 토니안"
                self.root.get_screen('detailScreen').ids.Danimal.source = "Dmonkey.png"
            elif man_animal == ['dino_man']:
                self.root.get_screen('resultScreen').ids.resultAnimal.text = "당신은 공룡상입니다!"
                self.root.get_screen('detailScreen').ids.resultAnimall.text = "당신은 공룡상입니다!"
                self.root.get_screen('detailScreen').ids.detailComent1.text = "무심한 성격에 첫인상은 나쁜 남자 같지만,"
                self.root.get_screen('detailScreen').ids.detailComent2.text = "알고 보면 따뜻함이 묻어나는 당신!"
                self.root.get_screen('detailScreen').ids.detailComent3.text = "선뜻 다가가지 못하지만 한번 다가가면"
                self.root.get_screen('detailScreen').ids.detailComent4.text = "헤어나올 수 없는 매력을 가진 카리스마"
                self.root.get_screen('detailScreen').ids.detailComent5.text = "공룡상 남자 연예인"
                self.root.get_screen('detailScreen').ids.detailComent6.text = "공유, 김경남, 김성오, 김우빈, 동해,"
                self.root.get_screen('detailScreen').ids.detailComent7.text = "류준열, 종현, 윤두준, 이민기, 탑"
                self.root.get_screen('detailScreen').ids.Danimal.source = "Ddino.png"
        elif self.root.get_screen('main').ids.genderClick.text == "womanClick":
            woman = get_animal(e)
            woman_animal, girl_percent_result = woman.woman()
            self.root.get_screen('detailScreen').ids.animalName1.text = "강아지상"
            self.root.get_screen('detailScreen').ids.animalName2.text = "고양이상"
            self.root.get_screen('detailScreen').ids.animalName3.text = "여우상"
            self.root.get_screen('detailScreen').ids.animalName4.text = "물고기상"
            self.root.get_screen('detailScreen').ids.animalName5.text = "뱀상"
            for i in girl_percent_result.keys():
                if i == "dog_girl":
                    if girl_percent_result[i] == 0:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "0percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 20:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "10percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 30:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "20percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 40:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "30percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 50:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "40percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 60:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "50percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 70:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "60percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 80:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "70percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 90:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "80percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 100:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "90percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] == 100:
                        self.root.get_screen('detailScreen').ids.animalGraph1.source = "100percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent1.text = str(girl_percent_result[i]) + "%"

                if i == "cat_girl":
                    if girl_percent_result[i] == 0:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "0percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 20:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "10percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 30:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "20percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 40:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "30percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 50:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "40percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 60:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "50percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 70:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "60percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 80:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "70percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 90:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "80percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 100:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "90percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] == 100:
                        self.root.get_screen('detailScreen').ids.animalGraph2.source = "100percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent2.text = str(girl_percent_result[i]) + "%"

                if i == "fox_girl":
                    if girl_percent_result[i] == 0:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "0percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 20:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "10percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 30:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "20percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 40:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "30percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 50:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "40percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 60:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "50percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 70:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "60percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 80:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "70percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 90:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "80percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 100:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "90percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] == 100:
                        self.root.get_screen('detailScreen').ids.animalGraph3.source = "100percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent3.text = str(girl_percent_result[i]) + "%"

                if i == "fish_girl":
                    if girl_percent_result[i] == 0:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "0percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 20:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "10percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 30:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "20percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 40:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "30percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 50:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "40percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 60:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "50percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 70:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "60percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 80:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "70percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 90:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "80percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 100:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "90percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] == 100:
                        self.root.get_screen('detailScreen').ids.animalGraph4.source = "100percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent4.text = str(girl_percent_result[i]) + "%"

                if i == "snake_girl":
                    if girl_percent_result[i] == 0:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "0percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 20:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "10percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 30:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "20percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 40:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "30percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 50:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "40percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 60:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "50percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 70:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "60percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 80:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "70percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 90:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "80percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] < 100:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "90percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(girl_percent_result[i]) + "%"
                    elif girl_percent_result[i] == 100:
                        self.root.get_screen('detailScreen').ids.animalGraph5.source = "100percent.png"
                        self.root.get_screen('detailScreen').ids.animalPercent5.text = str(girl_percent_result[i]) + "%"
            if woman_animal == ['dog_girl']:
                self.root.get_screen('resultScreen').ids.resultAnimal.text = "당신은 강아지상입니다!"
                self.root.get_screen('detailScreen').ids.resultAnimall.text = "당신은 강아지상입니다!"
                self.root.get_screen('detailScreen').ids.detailComent1.text = "다정다감하고 귀여운 당신은"
                self.root.get_screen('detailScreen').ids.detailComent2.text = "모든 사람들에게 즐거움을 주는 호감형이다!"
                self.root.get_screen('detailScreen').ids.detailComent3.text = "상냥하고 활발한 성격으로 인기폭발로"
                self.root.get_screen('detailScreen').ids.detailComent4.text = "애교와 웃음이 많아 사랑스럽다."
                self.root.get_screen('detailScreen').ids.detailComent5.text = "강아지상 여자 연예인"
                self.root.get_screen('detailScreen').ids.detailComent6.text = "박보영, 박신혜, 손예진, 수지, 안유진,"
                self.root.get_screen('detailScreen').ids.detailComent7.text = "우기, 윤승아, 츄, 한가인, 한효주"
                self.root.get_screen('detailScreen').ids.Danimal.source = "Ddog.png"
            elif woman_animal == ['cat_girl']:
                self.root.get_screen('resultScreen').ids.resultAnimal.text = "당신은 고양이상입니다!"
                self.root.get_screen('detailScreen').ids.resultAnimall.text = "당신은 고양이상입니다!"
                self.root.get_screen('detailScreen').ids.detailComent1.text = "도도하고 섹시한 당신의 첫인상은"
                self.root.get_screen('detailScreen').ids.detailComent2.text = "차가워 보이지만 묘한 매력을 풍겨 인기만점!"
                self.root.get_screen('detailScreen').ids.detailComent3.text = "시크한 츤데레로 연인에게 끊임없이"
                self.root.get_screen('detailScreen').ids.detailComent4.text = "설렘을 안겨주는 당신은 고양이와 닮았다!"
                self.root.get_screen('detailScreen').ids.detailComent5.text = "고양이상 여자 연예인"
                self.root.get_screen('detailScreen').ids.detailComent6.text = "경리, 김민희, 노제, 안소희, 오연서,"
                self.root.get_screen('detailScreen').ids.detailComent7.text = "이성경, 제니, 청하, 한소희, 현아"
                self.root.get_screen('detailScreen').ids.Danimal.source = "Dcat.png"
            elif woman_animal == ['snake_girl']:
                self.root.get_screen('resultScreen').ids.resultAnimal.text = "당신은 뱀상입니다!"
                self.root.get_screen('detailScreen').ids.resultAnimall.text = "당신은 뱀상입니다!"
                self.root.get_screen('detailScreen').ids.detailComent1.text = "시크하고 날카로운 이미지를 소유한 당신은"
                self.root.get_screen('detailScreen').ids.detailComent2.text = "첫인상은 다가가기 어려울 수 있지만"
                self.root.get_screen('detailScreen').ids.detailComent3.text = "가까워지면 한없이 따뜻하다. "
                self.root.get_screen('detailScreen').ids.detailComent4.text = "상대를 홀리는 힘이 강력하다!"
                self.root.get_screen('detailScreen').ids.detailComent5.text = "뱀상 여자 연예인"
                self.root.get_screen('detailScreen').ids.detailComent6.text = "강승현, 규리, 김완선, 신보라, 씨엘,"
                self.root.get_screen('detailScreen').ids.detailComent7.text = "장윤주, 차예련, 카리나, 한예슬, 헤이즈"
                self.root.get_screen('detailScreen').ids.Danimal.source = "Dsnake.png"
            elif woman_animal == ['fox_girl']:
                self.root.get_screen('resultScreen').ids.resultAnimal.text = "당신은 여우상입니다!"
                self.root.get_screen('detailScreen').ids.resultAnimall.text = "당신은 여우상입니다!"
                self.root.get_screen('detailScreen').ids.detailComent1.text = "사람을 홀리는 매력을 가진 당신은"
                self.root.get_screen('detailScreen').ids.detailComent2.text = "선뜻 다가가기 힘든 섹시한 매력을 가졌다."
                self.root.get_screen('detailScreen').ids.detailComent3.text = "우아한 외모에 더해 뛰어난 센스의 성격을"
                self.root.get_screen('detailScreen').ids.detailComent4.text = "가진 당신은 어딜가도 주목받는 주인공이다!"
                self.root.get_screen('detailScreen').ids.detailComent5.text = "여우상 여자 연예인"
                self.root.get_screen('detailScreen').ids.detailComent6.text = "김민주, 사나, 서효림, 션, 예지,"
                self.root.get_screen('detailScreen').ids.detailComent7.text = "이주빈, 조보아, 지연, 지호, 쯔위"
                self.root.get_screen('detailScreen').ids.Danimal.source = "Dfox.png"
            elif woman_animal == ['fish_girl']:
                self.root.get_screen('resultScreen').ids.resultAnimal.text = "당신은 물고기상입니다!"
                self.root.get_screen('detailScreen').ids.resultAnimall.text = "당신은 물고기상입니다!"
                self.root.get_screen('detailScreen').ids.detailComent1.text = "맹한 이미지를 가져 누구나"
                self.root.get_screen('detailScreen').ids.detailComent2.text = "보호본능을 자극하는 당신은"
                self.root.get_screen('detailScreen').ids.detailComent3.text = "변에 엄마같은 따뜻한 사람이 많다."
                self.root.get_screen('detailScreen').ids.detailComent4.text = "여유가 넘쳐 스트레스가 덜하고 건강한 사람!"
                self.root.get_screen('detailScreen').ids.detailComent5.text = "물고기상 여자 연예인"
                self.root.get_screen('detailScreen').ids.detailComent6.text = "구하라, 김민정, 레이디제인, 바다, 서우,"
                self.root.get_screen('detailScreen').ids.detailComent7.text = "설리, 은정, 이은형, 조이, 황보라"
                self.root.get_screen('detailScreen').ids.Danimal.source = "Dfish.png"

    def build(self):
        global screen_manager
        screen_manager = ScreenManager()
        screen_manager.add_widget(Builder.load_file("splash.kv"))
        screen_manager.add_widget(Builder.load_file("main.kv"))
        screen_manager.add_widget(GallaryScreen(name="gallary"))
        screen_manager.add_widget(LoadingScreen(name="loadingScreen"))
        screen_manager.add_widget(ResultScreen(name="resultScreen"))
        screen_manager.add_widget(DetailScreen(name="detailScreen"))
        return screen_manager

    def on_start(self):
        Clock.schedule_once(self.main, 8)

    def main(self, *args):
        screen_manager.current = "main"


MainApp().run()
