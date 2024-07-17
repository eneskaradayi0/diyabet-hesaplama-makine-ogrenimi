import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

# Veri setinin içe aktarılması
veri_seti = pd.read_csv('../Dataset/diyabet.csv')
X = veri_seti.iloc[:, :-1].values
y = veri_seti.iloc[:, 8].values

# Veri setinin Eğitim seti ve Test setine bölünmesi
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Özellik Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_egitim = sc.fit_transform(X_egitim)
X_test = sc.transform(X_test)

# Parametre değerlendirme
logclf = LogisticRegression(random_state=42, solver='liblinear')  # 'liblinear' solver'ı l1 ve l2 cezalarını destekler
parametreler = {'C': [1, 4, 10], 'penalty': ['l1', 'l2']}
gridsearch = GridSearchCV(logclf, parametreler, cv=10, scoring='roc_auc')
gridsearch.fit(X_egitim, y_egitim)
print("En iyi parametreler: ", gridsearch.best_params_)
print("En iyi ROC AUC skoru: ", gridsearch.best_score_)

# Modelin en iyi parametrelerle ayarlanması
logreg_classifier = LogisticRegression(C=gridsearch.best_params_['C'], 
                                       penalty=gridsearch.best_params_['penalty'], 
                                       solver='liblinear', 
                                       random_state=42)
logreg_classifier.fit(X_egitim, y_egitim)
print("Eğitim setinde doğruluk: {:.3f}".format(logreg_classifier.score(X_egitim, y_egitim)))
print("Test setinde doğruluk: {:.3f}".format(logreg_classifier.score(X_test, y_test)))

# Test seti sonuçlarının tahmin edilmesi
y_tahmin = logreg_classifier.predict(X_test)

# Karışıklık Matrisi oluşturulması
cm = confusion_matrix(y_test, y_tahmin)

print('Doğru Negatif: {}'.format(cm[0, 0]))
print('Yanlış Pozitif: {}'.format(cm[0, 1]))
print('Yanlış Negatif: {}'.format(cm[1, 0]))
print('Doğru Pozitif: {}'.format(cm[1, 1]))
print('Doğruluk Oranı: {:.2f}'.format((cm[0, 0] + cm[1, 1]) / np.sum(cm)))
print('Yanlış Sınıflandırma Oranı: {:.2f}'.format((cm[0, 1] + cm[1, 0]) / np.sum(cm)))

# ROC AUC skorunun hesaplanması
roc_auc = roc_auc_score(y_test, y_tahmin)
print('ROC AUC skoru: {:.5f}'.format(roc_auc))

# Tahminlerin grafiğinin çizilmesi
plt.hist(y_tahmin, bins=10)
plt.xlim(0, 1)
plt.xlabel("Tahmin Edilen Olasılıklar")
plt.ylabel("Frekans")
plt.show()

