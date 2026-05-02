# Projeto 3 - MLPClassifier (Perceptron Multicamadas)

Este projeto tem como objetivo aplicar o classificador MLP (Multi-Layer Perceptron) utilizando a biblioteca Scikit-learn e comparar seu desempenho com o algoritmo KNN. A análise foi realizada utilizando dois datasets clássicos: Iris e Wine.

## Estrutura do Projeto

- mlp_classificador.py: Executa os classificadores MLP e KNN nos datasets Iris e Wine, calcula métricas e gera matrizes de confusão.
- comparativo_tp03.py: Gera um arquivo .txt com os resultados comparativos entre os modelos.

## Funcionalidades

### Classificação com MLP
- Utiliza o MLPClassifier da biblioteca Scikit-learn.
- Aplicação de normalização com StandardScaler.
- Configuração com duas camadas ocultas.
- Avaliação com métricas de desempenho.

### Classificação com KNN
- Utiliza o KNeighborsClassifier da biblioteca Scikit-learn.
- Executado com K = 5.
- Serve como base de comparação com o MLP.

### Métricas e Avaliação
- Acurácia
- Precisão
- Revocação
- Matrizes de confusão (geradas como imagens .png)

## Como Executar

1. Instalar dependências:

pip install pandas numpy matplotlib seaborn scikit-learn

2. Executar o projeto:

python comparativo_tp03.py

## Resultados

Após a execução, serão gerados:

Imagens:
- matriz_mlp_iris.png
- matriz_mlp_wine.png
- matriz_knn_iris.png
- matriz_knn_wine.png

Arquivo de saída:
- analise_comparativa_tp03.txt

Este arquivo contém:
- Métricas de desempenho (acurácia, precisão e revocação)
- Comparação entre os modelos MLP e KNN

## Conjuntos de Dados

### Iris
- 150 amostras
- 3 classes:
  - Setosa
  - Versicolor
  - Virginica
- 4 atributos:
  - SepalLength
  - SepalWidth
  - PetalLength
  - PetalWidth

### Wine
- 178 amostras
- 3 classes
- 13 atributos químicos

## Observações

- O MLP apresentou melhor desempenho geral, especialmente no dataset Wine.
- O KNN teve desempenho inferior no Wine devido à sensibilidade à escala dos dados.
- A normalização foi essencial para o bom funcionamento do MLP.

## Conclusão

O projeto demonstra que modelos mais complexos, como redes neurais, tendem a apresentar melhor desempenho em datasets mais desafiadores. No entanto, algoritmos simples como o KNN ainda são eficazes em problemas mais básicos.

## Autores

- Lucas de Oliveira Pereira  
- Renan Augusto Da Silva

## Link para o vídeo de explicação

https://youtu.be/dStOARu-r6A