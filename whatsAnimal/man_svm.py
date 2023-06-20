import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import numpy as np
girl_embedding = pd.read_csv('girl_embedding.csv')
vector = range(1, 129)
girlEmbeddingVector = girl_embedding.iloc[:, vector]
girlEmbeddingResult = girl_embedding[['result']]
girl_feature_train, girl_feature_test, girl_target_train, girl_target_test = train_test_split(girlEmbeddingVector,
                                                                                                  girlEmbeddingResult,
                                                                                                  test_size=0.34)
girl_svm_model = SVC(kernel="rbf", C=80, gamma=0.1)
girl_svm_model.fit(girl_feature_train, girl_target_train)  # 학습모델
girl_target_pred = girl_svm_model.predict(girl_feature_test)  # 테스트
girl_np_target_pred = np.array([])
girl_np_target_pred = np.append(girl_np_target_pred, girl_target_pred)

girl_np_target_test = np.array([])
girl_np_target_test = np.append(girl_np_target_test, girl_target_test)
    # print("예측된 라벨:", np_target_pred)
    # print("ground-truth 라벨:", np_target_test)
print("girl_  prediction accuracy: {:.2f}".format(np.mean(girl_np_target_pred == girl_np_target_test)))  # 예측 정확도

man_embedding = pd.read_csv('man_embedding.csv')
vector = range(1, 129)
manEmbeddingVector = man_embedding.iloc[:, vector]
manEmbeddingResult = man_embedding[['result']]
man_feature_train, man_feature_test, man_target_train, man_target_test = train_test_split(manEmbeddingVector,
                                                                                              manEmbeddingResult,
                                                                                              test_size=0.34)
man_svm_model = SVC(kernel="rbf", C=80, gamma=0.1)
man_svm_model.fit(man_feature_train, man_target_train)  # 학습모델
man_target_pred = man_svm_model.predict(man_feature_test)  # 테스트

man_np_target_pred = np.array([])
man_np_target_pred = np.append(man_np_target_pred, man_target_pred)

man_np_target_test = np.array([])
man_np_target_test = np.append(man_np_target_test, man_target_test)
# print("예측된 라벨:", np_target_pred)
# print("ground-truth 라벨:", np_target_test)
print("man  prediction accuracy: {:.2f}".format(np.mean(man_np_target_pred == man_np_target_test)))  # 예측 정확도


# print("훈련 세트 정확도: {:.2f}".format(girl_svm_model.score(girl_feature_train, girl_target_train)))
# print("테스트 세트 정확도: {:.2f}".format(girl_svm_model.score(girl_feature_test, girl_target_test)))
#
#
# #정확도 점수 테스트
# scores = []
# for thisC in [*range(1,100)]:
#     svc=SVC(kernel='rbf',C=thisC)
#     model=svc.fit(girl_feature_train,girl_target_train)
#     scoreTrain=model.score(girl_feature_train,girl_target_train)
#     scoreTest=model.score(girl_feature_test,girl_target_test)
#     print("선형 SVM : C:{}, training score:{:2f}, test score:{:2f}".format
#           (thisC,scoreTrain, scoreTest))
#     scores.append([scoreTrain, scoreTest])








# joblib.dump(svm_model,'svmModel.tflite')


# print("훈련 세트 정확도: {:.2f}".format(svm_model.score(feature_train, target_train)))
# print("테스트 세트 정확도: {:.2f}".format(svm_model.score(feature_test, target_test)))
# print(confusion_matrix(target_test, target_pred))

#정확도 점수 테스트
# scores = []
# for thisC in [*range(1,100)]:
#     svc=SVC(kernel='rbf',C=thisC)
#     model=svc.fit(feature_train,target_train)
#     scoreTrain=model.score(feature_train,target_train)
#     scoreTest=model.score(feature_test,target_test)
#     print("선형 SVM : C:{}, training score:{:2f}, test score:{:2f}".format
#           (thisC,scoreTrain, scoreTest))
#     scores.append([scoreTrain, scoreTest])