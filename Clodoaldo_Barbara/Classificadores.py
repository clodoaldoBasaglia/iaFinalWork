from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

treino = pd.read_csv('treinofinal.csv', delimiter=';')
teste = pd.read_csv('testefinal.csv', delimiter=';')


#treino e teste vão ser diminuidos em 40% para aumentar a velocidade

treino, NoTreino, = train_test_split(treino, test_size=0.4, random_state=0)
teste, NoTeste = train_test_split(teste, test_size=0.4, random_state=0)

infoDeTreino = treino.drop(columns='classe')
infoDeTeste = teste.drop(columns='classe')

classesTreino = treino['classe'].values
classesTeste = teste['classe'].values

dtree = DecisionTreeClassifier(max_depth=11)
print("Iniciando o Fit da Decision Tree")
dtree.fit(infoDeTreino, classesTreino)
print("Resultado DT: ", dtree.score(infoDeTeste, classesTeste))
ks = [3,5,7]
for i in ks:
    print("K = ",i)
    knn = KNeighborsClassifier(n_neighbors=i)
    print("Iniciando o Fit do KNN")
    knn.fit(infoDeTreino, classesTreino)
    print("Resultado KNN com K= " + str(i), knn.score(infoDeTeste, classesTeste))

svm = SVC(kernel='poly',C=1 ,gamma='auto')
print("Iniciando o Fit da SVM")
svm.fit(infoDeTreino, classesTreino)
print("Resultado SVM: ", svm.score(infoDeTeste, classesTeste))

