import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# SVM modeli için hiperparametre ayarı ve grid search
svm = SVC(random_state=42)
parametreler = {'kernel': ('linear', 'rbf'), 'C': [1, 0.25, 0.5, 0.75],
                'gamma': [1, 2, 3, 'auto'], 'decision_function_shape': ['ovo', 'ovr'],
                'shrinking': [True, False]}

skorlar = ['precision', 'recall']

for skor in skorlar:
    print("# %s için hiperparametre ayarları yapılıyor" % skor)
    print()

    grid_search = GridSearchCV(SVC(), parametreler, cv=5, scoring='%s_macro' % skor)
    grid_search.fit(X_egitim, y_egitim)

    print("Geliştirme setinde bulunan en iyi parametreler:")
    print(grid_search.best_params_)
    print()

    print("Geliştirme setinde grid skorları:")
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) için %r" % (mean, std * 2, params))
    print()

    print("Detaylı sınıflandırma raporu:")
    y_tahmin = grid_search.predict(X_test)
    print(classification_report(y_test, y_tahmin))
    print()

# En iyi parametrelerle SVM modeli eğitme
svm_model = SVC(kernel='rbf', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'], random_state=42)
svm_model.fit(X_egitim, y_egitim)
y_tahmin = svm_model.predict(X_test)
print('SVM ile Doğruluk: {:.2f}%'.format(accuracy_score(y_test, y_tahmin) * 100))

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

