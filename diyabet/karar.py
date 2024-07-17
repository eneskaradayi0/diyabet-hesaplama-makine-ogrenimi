import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Veri setinin içe aktarılması
veri_seti = pd.read_csv('../Dataset/diabetes.csv')
X = veri_seti.iloc[:, :-1].values
y = veri_seti.iloc[:, 8].values

# Veri setinin Eğitim seti ve Test setine bölünmesi
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Özellik Ölçekleme
sc = StandardScaler()
X_egitim = sc.fit_transform(X_egitim)
X_test = sc.transform(X_test)

# Parametre değerlendirme
agacclf = DecisionTreeClassifier(random_state=42)
parametreler = {
    'max_depth': [6, 7, 8, 9],
    'min_samples_split': [2, 3, 4, 5],
    'max_features': [1, 2, 3, 4]
}

gridsearch = GridSearchCV(agacclf, parametreler, cv=10, scoring='roc_auc', n_jobs=-1)
gridsearch.fit(X_egitim, y_egitim)
print("En iyi parametreler: ", gridsearch.best_params_)
print("En iyi ROC AUC skoru: ", gridsearch.best_score_)

# En iyi parametrelerle Karar Ağacı modeli eğitme
agac = DecisionTreeClassifier(max_depth=gridsearch.best_params_['max_depth'],
                              max_features=gridsearch.best_params_['max_features'],
                              min_samples_split=gridsearch.best_params_['min_samples_split'],
                              random_state=42)
agac.fit(X_egitim, y_egitim)
print("Eğitim setinde doğruluk: {:.3f}".format(agac.score(X_egitim, y_egitim)))
print("Test setinde doğruluk: {:.3f}".format(agac.score(X_test, y_test)))

# Test seti sonuçlarının tahmin edilmesi
y_tahmin = agac.predict(X_test)

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

# ROC eğrisinin çizilmesi
fpr, tpr, _ = roc_curve(y_test, y_tahmin)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
