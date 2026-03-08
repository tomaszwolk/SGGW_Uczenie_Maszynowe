import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)         #Ustawienie ziarna, do generacji losowej - stały punkt startowy dla generatora liczb losowych, zapewnia taki sam zbiór danych przy kolejnych uruchomieniach programu
x = np.random.normal(5.0, 1.0, 500)
y = np.random.normal(10.0, 1.0, 500)

#plt.scatter(x, y)  #Prosty scatterplot przygotowany na podstawie wygenerowanych danych
#plt.show()         #Wyświetlenie wykresu

x1 = np.random.normal(5.0, 1.0, 500)
y1 = np.random.normal(10.0, 1.0, 500)
"""
plt.subplot(1, 2, 1)    #przygotowanie podwykresów na jednym, głównym wykresie
plt.scatter(x, y)
plt.title("Plot 1")     #tytuł i opisy do osi
plt.xlabel("Label X")
plt.ylabel("Label Y")
plt.grid()

plt.subplot(1, 2, 2)
plt.scatter(x1, y1)

plt.title("Plot 2")
plt.xlabel("Label X")
plt.ylabel("Label Y")
plt.grid()

plt.show()"""

xBarPlot = np.array(["A", "B", "C", "D", "E"])
yBarPlot = np.array([33, 11, 34, 88, 99])

#plt.bar(xBarPlot, yBarPlot)    #Wykres słupkowy
#plt.show()

import sklearn.datasets as skd
import pandas as pd
#Przykładowe metody służace do generacji danych o konkretnym kształcie, np. w celu testowania metod
#x, y = skd.make_circles(n_samples=300, shuffle=True, noise=0.05, random_state=432)
#x, y = skd.make_moons(n_samples=300, shuffle=True, noise=0.05, random_state=432)
x, y = skd.make_blobs(n_samples=600, centers=5, n_features=2, random_state=432)

#plt.scatter(x[:, 0], x[:, 1], c=y)
#plt.show()

x, y = skd.load_iris(return_X_y=True)       #Ładowanie zbioru danych Iris jako macierzy w bibliotece numpy

frameData = skd.load_iris(as_frame=True)    #Ładowanie tej samej biblioteki jako ramki danych w pandas (nie wszystkie biblioteki z scikit-learn mają taką możliwość)
#print(frameData.keys())    #wypisanie kluczy ramki danych

df = frameData['frame']
df_X = frameData['data']
df_Y = frameData['target']
feature = frameData['feature_names']
#print(frameData['target_names'])

#print(feature)
#Skalowanie danych - należy je wykonać, gdy np. wartości w różnych kolumnach znacznie różnią się rzędem wielkości.
#Czasami warto zrobić skalowanie niezależnie, w celu sprawdzenia, czy poprawi to jakość modelu.
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()    #tworzenie modelu do skalowania danych

scaledDFIris = scale.fit_transform(df_X)
#print(scaledDFIris)

"""
Podział na zbiór train/test
Nie chcemy weryfikować zbioru danych na tych samych przykładach,
które model widział w trakcie treningu. Standardem jest podział zbioru danych
na co najmniej dwa podzbiory: trenujący i testowy. """
np.random.seed(3343)
newX = np.random.normal(3, 1, 100)
newY = np.random.normal(150, 40, 100) /newX

train_x = newX[:80]
train_y = newY[:80]

test_x = newX[80:]
test_y = newY[80:]

myModel = np.poly1d(np.polyfit(train_x, train_y, 4))

myModelLinie = np.linspace(0, 6, 100)

"""
plt.scatter(train_x, train_y)
plt.plot(myModelLinie, myModel(myModelLinie))
plt.show()"""

from sklearn.metrics import r2_score
#Zbiór testowy wykorzystujemy później, do zweryfikowania zachowania modelu po tym, jak zobaczy nowe, wcześniej nieznane dane.
r2 = r2_score(train_y, myModel(train_x))
r2Test = r2_score(test_y, myModel(test_x))

#print("The score model achieved was {} for training data and {} for test data".format(r2, r2Test))

np.random.seed(1234)
actualValues = np.random.binomial(1, 0.8, size=3000)    #generacja wartości 0/1 w celu stworzenia zbioru pod macierz konfuzji
predictedValues = np.random.binomial(1, 0.8, size=3000)

from sklearn import metrics
"""
Macierz konfuzji, służy do zweryfikowania tego jak często nasz model myli się w przypisywaniu
klas/etykiet. Macierz pokazuje wynki liczbowe dla każdej klasy pokazując ilość poprawnych
klasyfikacji, oraz ilość pomyłek gdy obecna klasa została pomylona z dowolną inną obecną
w zbiorze danych. Macierz zwykle jest też kolorowana zgodnie z wielkością poszczególnych
wartości, co ułatwia analizę zależności pomiędzy klasyfikacją poszczególnych kategorii.
"""
confusionMatrix = metrics.confusion_matrix(actualValues, predictedValues)
cmDisplay = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=[0, 1])
#cmDisplay.plot()   #Macierz konfuzji może być wyświetlona korzystając ze standardowego wykresu w matplotlib
#plt.show()

#Podbiblioteka metrics w scikit-learn posiada również implementację różnych miar, umożliwiających diagnozę modelu
accuracy = metrics.accuracy_score(actualValues, predictedValues)
precision = metrics.precision_score(actualValues, predictedValues)
sensitivity = metrics.recall_score(actualValues, predictedValues)
specificity = metrics.recall_score(actualValues, predictedValues, pos_label=0)
f1Score = metrics.f1_score(actualValues, predictedValues)

print("The results for our model are: Accuracy: {}, Precision: {}, Sensitivity: {}, Specificity: {} and F1: {}".format(accuracy, precision, sensitivity, specificity, f1Score))

#Prosta implementacja algorytmu grupującego K-Means
from sklearn.cluster import KMeans
x = [5, 6, 11, 5, 3, 12, 13, 6, 10, 12]
y = [21, 18, 23, 16, 17, 24, 23, 22, 21, 20]

#plt.scatter(x, y)
#plt.show()

dataForKM = list(zip(x, y))
inertials = []

#Przy tworzeniu tego modelu, zwykle nie znamy najlepszej liczby klastrów - najprościej
#sprawdzić to iteracyjnie
for i in range (1, 7):
    modelKM = KMeans(n_clusters=i)
    modelKM.fit(dataForKM)
    inertials.append(modelKM.inertia_)

plt.plot(range(1, 7), inertials, marker='o')
plt.title('Elbow method verification for K-Means classifier')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()