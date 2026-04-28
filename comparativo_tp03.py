import time
from mlp_classificador import rodar_mlp, rodar_knn, carregar_dados
from sklearn.datasets import load_iris, load_wine

def gerar_relatorio_final(resultados):
    with open("analise_comparativa_tp03.txt", "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("      RELATÓRIO COMPARATIVO: MLP VS KNN\n")
        f.write("==================================================\n\n")
        
        for dado in resultados:
            f.write(f"DATASET: {dado['nome']}\n")
            f.write("-" * 30 + "\n")
            
            f.write(f"MODELO: MLP (Rede Neural Multicamadas)\n")
            f.write(f"  Acurácia:  {dado['mlp'][0]:.4f}\n")
            f.write(f"  Precisão:  {dado['mlp'][1]:.4f}\n")
            f.write(f"  Revocação: {dado['mlp'][2]:.4f}\n\n")
            
            f.write(f"MODELO: KNN (K-Vizinhos Mais Próximos)\n")
            f.write(f"  Acurácia:  {dado['knn'][0]:.4f}\n")
            f.write(f"  Precisão:  {dado['knn'][1]:.4f}\n")
            f.write(f"  Revocação: {dado['knn'][2]:.4f}\n")
            f.write("\n" + "=" * 50 + "\n\n")
            
    print("\n[OK] Relatório 'analise_comparativa_tp03.txt' gerado com sucesso!")

def executar():
    print("Iniciando bateria de testes comparativos...")
    bateria_resultados = []

    ds_iris = load_iris()
    X_i, y_i, nomes_i = carregar_dados(ds_iris, "Iris")
    res_mlp_i = rodar_mlp(X_i, y_i, nomes_i, "Iris")
    res_knn_i = rodar_knn(X_i, y_i, nomes_i, "Iris")
    
    bateria_resultados.append({
        'nome': 'Iris',
        'mlp': res_mlp_i,
        'knn': res_knn_i
    })

    ds_wine = load_wine()
    X_w, y_w, nomes_w = carregar_dados(ds_wine, "Wine")
    res_mlp_w = rodar_mlp(X_w, y_w, nomes_w, "Wine")
    res_knn_w = rodar_knn(X_w, y_w, nomes_w, "Wine")

    bateria_resultados.append({
        'nome': 'Wine',
        'mlp': res_mlp_w,
        'knn': res_knn_w
    })

    gerar_relatorio_final(bateria_resultados)

if __name__ == "__main__":
    executar()