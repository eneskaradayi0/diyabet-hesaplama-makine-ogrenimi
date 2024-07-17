//knn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Veri setinin içe aktarılması
veri_seti = pd.read_csv('../Dataset/diyabet.csv')
X = veri_seti.iloc[:, :-1].values
y = veri_seti.iloc[:, 8].values


X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Özellik Ölçekleme
sc = StandardScaler()
X_egitim = sc.fit_transform(X_egitim)
X_test = sc.transform(X_test)

# Parametre değerlendirme
knnclf = KNeighborsClassifier()
parametreler = {'n_neighbors': range(1, 20)}
gridsearch = GridSearchCV(knnclf, parametreler, cv=10, scoring='roc_auc')
gridsearch.fit(X, y)
print("En iyi parametreler: ", gridsearch.best_params_)
print("En iyi ROC AUC skoru: ", gridsearch.best_score_)

# K-NN modelinin Eğitim setine uygulanması
knnSınıflandırıcı = KNeighborsClassifier(n_neighbors=gridsearch.best_params_['n_neighbors'])
knnSınıflandırıcı.fit(X_egitim, y_egitim)
print('K-NN sınıflandırıcısının eğitim setindeki doğruluğu: {:.2f}'.format(knnSınıflandırıcı.score(X_egitim, y_egitim)))
print('K-NN sınıflandırıcısının test setindeki doğruluğu: {:.2f}'.format(knnSınıflandırıcı.score(X_test, y_test)))

# Test seti sonuçlarının tahmin edilmesi
y_tahmin = knnSınıflandırıcı.predict(X_test)

# Karışıklık Matrisi oluşturulması
cm = confusion_matrix(y_test, y_tahmin)

print('TP - Doğru Negatif: {}'.format(cm[0,0]))
print('FP - Yanlış Pozitif: {}'.format(cm[0,1]))
print('FN - Yanlış Negatif: {}'.format(cm[1,0]))
print('TP - Doğru Pozitif: {}'.format(cm[1,1]))
print('Doğruluk Oranı: {:.2f}'.format((cm[0,0] + cm[1,1]) / np.sum(cm)))
print('Yanlış Sınıflandırma Oranı: {:.2f}'.format((cm[0,1] + cm[1,0]) / np.sum(cm)))

# ROC AUC skorunun hesaplanması
roc_auc = roc_auc_score(y_test, y_tahmin)
print('ROC AUC skoru: {:.5f}'.format(roc_auc))

