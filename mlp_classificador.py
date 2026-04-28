import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import time

def carregar_dados(dataset, nome):
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target)
    print(f"\nDataset {nome} carregado: {X.shape[0]} amostras, {X.shape[1]} features")
    return X, y, dataset.target_names

def exibir_metricas(nome_modelo, nome_dataset, y_real, y_pred):
    acc = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_real, y_pred, average='weighted', zero_division=0)
    print(f"  [{nome_modelo} - {nome_dataset}] Acurácia: {acc:.4f} | Precisão: {prec:.4f} | Revocação: {rec:.4f}")
    return acc, prec, rec

def salvar_matriz(y_real, y_pred, classes, nome_modelo, nome_dataset):
    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Matriz de Confusão - {nome_modelo} ({nome_dataset})")
    plt.ylabel("Real")
    plt.xlabel("Previsto")
    nome_arquivo = f"matriz_{nome_modelo.lower()}_{nome_dataset.lower()}.png"
    plt.savefig(nome_arquivo)
    plt.close()
    print(f"  Matriz salva: {nome_arquivo}")

def rodar_mlp(X, y, classes, nome_dataset):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_scaled, y, test_size=0.2, random_state=21
    )

    modelo = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=21)
    modelo.fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)

    acc, prec, rec = exibir_metricas("MLP", nome_dataset, y_teste, previsoes)
    salvar_matriz(y_teste, previsoes, classes, "MLP", nome_dataset)

    return acc, prec, rec

def rodar_knn(X, y, classes, nome_dataset, k=5):
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=21
    )

    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)

    acc, prec, rec = exibir_metricas("KNN", nome_dataset, y_teste, previsoes)
    salvar_matriz(y_teste, previsoes, classes, "KNN", nome_dataset)

    return acc, prec, rec

if __name__ == "__main__":
    inicio = time.time()

    iris = load_iris()
    X_iris, y_iris, classes_iris = carregar_dados(iris, "Iris")

    print("\n--- Rodando no Iris ---")
    acc_mlp_iris, prec_mlp_iris, rec_mlp_iris = rodar_mlp(X_iris, y_iris, classes_iris, "Iris")
    acc_knn_iris, prec_knn_iris, rec_knn_iris = rodar_knn(X_iris, y_iris, classes_iris, "Iris")

    wine = load_wine()
    X_wine, y_wine, classes_wine = carregar_dados(wine, "Wine")

    print("\n--- Rodando no Wine ---")
    acc_mlp_wine, prec_mlp_wine, rec_mlp_wine = rodar_mlp(X_wine, y_wine, classes_wine, "Wine")
    acc_knn_wine, prec_knn_wine, rec_knn_wine = rodar_knn(X_wine, y_wine, classes_wine, "Wine")

    print(f"\nTempo total de execução: {time.time() - inicio:.2f}s")