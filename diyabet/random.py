import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Veri setinin içe aktarılması
veri_seti = pd.read_csv('../Dataset/diyabet.csv')
X = veri_seti.iloc[:, :-1].values
y = veri_seti.iloc[:, 8].values

# Veri setinin Eğitim seti ve Test setine bölünmesi
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Özellik Ölçekleme
sc = StandardScaler()
X_egitim = sc.fit_transform(X_egitim)
X_test = sc.transform(X_test)

# Parametre değerlendirme
rfclf = RandomForestClassifier(random_state=42)
parametreler = {
    'n_estimators': [50, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7],
    'criterion': ['gini', 'entropy']
}

gridsearch = GridSearchCV(rfclf, parametreler, cv=10, scoring='roc_auc', n_jobs=-1)
gridsearch.fit(X_egitim, y_egitim)
print("En iyi parametreler: ", gridsearch.best_params_)
print("En iyi ROC AUC skoru: ", gridsearch.best_score_)

# En iyi parametrelerle Random Forest modeli eğitme
rf = RandomForestClassifier(n_estimators=gridsearch.best_params_['n_estimators'],
                            criterion=gridsearch.best_params_['criterion'],
                            max_depth=gridsearch.best_params_['max_depth'],
                            max_features=gridsearch.best_params_['max_features'],
                            random_state=42)
rf.fit(X_egitim, y_egitim)
print("Eğitim setinde doğruluk: {:.3f}".format(rf.score(X_egitim, y_egitim)))
print("Test setinde doğruluk: {:.3f}".format(rf.score(X_test, y_test)))

# Test seti sonuçlarının tahmin edilmesi
y_tahmin = rf.predict(X_test)

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

# Sınıflandırma raporu
print(classification_report(y_test, y_tahmin))

# Tahminlerin grafiğinin çizilmesi
plt.hist(y_tahmin, bins=10)
plt.xlim(0, 1)
plt.xlabel("Tahmin Edilen Olasılıklar")
plt.ylabel("Frekans")
plt.show()

