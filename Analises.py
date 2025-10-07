# %% IMPORTAÇÕES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
sns.set_style("darkgrid")

# %% CARREGAR DADOS
dados = pd.read_csv(r'C:\Users\phill\projetos\TECH4\Obesity.csv')

# %% RENOMEAR COLUNAS PRINCIPAIS
dados = dados.rename(columns={
    "Age": "Idade",
    "Weight": "Peso",
    "Height": "Altura",
    "Gender": "Genero",
    "Obesity": "Obesidade",
    "SMOKE": "Fumar",
    "family_history": "Historico_Familiar",
    "CALC": "frequencia_consumo_alcool",
    "FAVC": "consumo_alimentos_altas_calorias",
    "FCVC": "consumo_vegetais_diario",
    "NCP": "numero_refeicoes_diario",
    "SCC": "monitoramento_calorias",
    "CH2O": "consumo_agua_diario",
    "FAF": "atividade_fisica_diaria",
    "TUE": "uso_eletronicos_diario",
    "CAEC": "comida_entre_refeicoes",
    "MTRANS": "meio_transporte"
})

# %% VISUALIZAR DADOS
dados.head()
dados.info()
dados.shape

# %% VERIFICAR DUPLICADOS
dados.duplicated().sum()
dados.loc[dados.duplicated(keep=False), :]

# %% CONVERTENDO COLUNAS NUMÉRICAS
for col in ['Idade', 'Peso', 'consumo_vegetais_diario','numero_refeicoes_diario', 
            'consumo_agua_diario','atividade_fisica_diaria','uso_eletronicos_diario']:
    dados[col] = dados.loc[:,col].round().astype(int)

# %% ESTATÍSTICAS DESCRITIVAS
dados.describe()

# %% ANÁLISE UNIVARIADA - VARIÁVEIS CATEGÓRICAS
plt.figure(figsize=(18,15))
for i,col in enumerate(dados.select_dtypes(include="object").columns[:-1]):
    plt.subplot(4,2,i+1)
    sns.countplot(data=dados, x=col, palette=sns.color_palette("Set2"))

# %% DISTRIBUIÇÃO DE OBESIDADE
dados["Obesidade"].value_counts().sort_values(ascending=False).plot(kind="bar", color="red")

# %% ANÁLISE UNIVARIADA - VARIÁVEIS NUMÉRICAS
plt.figure(figsize=(18,15))
for i,col in enumerate(dados.select_dtypes(include="number").columns[:3]):
    plt.subplot(4,2,i+1)
    sns.boxplot(data=dados, x=col, palette=sns.color_palette("Set2"))

# %% REMOÇÃO DE OUTLIERS (Idade)
dados = dados[np.abs(stats.zscore(dados["Idade"])) < 2].reset_index(drop=True)
sns.boxplot(data=dados, x="Idade")
dados.shape

# %% DISTRIBUIÇÃO VARIÁVEIS NUMÉRICAS RESTANTES
plt.figure(figsize=(18,15))
for i,col in enumerate(dados.select_dtypes(include="number").columns[3:]):
    plt.subplot(4,2,i+1)
    sns.countplot(data=dados, x=col)

# %% ANÁLISE MULTIVARIADA
dados.groupby(['Obesidade', 'consumo_alimentos_altas_calorias'])["consumo_alimentos_altas_calorias"].count()

# %% VISUALIZAÇÃO MULTIVARIADA
plt.figure(figsize=(10,7))
sns.countplot(data=dados, x="Obesidade", hue="consumo_alimentos_altas_calorias", 
              palette=sns.color_palette("Dark2"))
plt.xticks(rotation=-20)
plt.show()

# %% MEDIANA DE IDADE POR TIPO DE OBESIDADE
dados.groupby("Obesidade")["Idade"].median()

# %% GRÁFICO - IDADE MÉDIA POR OBESIDADE
dados.groupby("Obesidade")["Idade"].median().sort_values(ascending=False).plot(kind="bar", color = sns.color_palette("Set2"))
plt.title("Idade média por tipo de obesidade")

# %% MÉDIA DE PESO POR OBESIDADE
dados.groupby("Obesidade")["Peso"].mean()

# %% GRÁFICO - PESO MÉDIO POR OBESIDADE
dados.groupby("Obesidade")["Peso"].mean().sort_values(ascending=False).plot(kind="bar", color=sns.color_palette("Set2"))

# %% INFLUÊNCIA DO GÊNERO NA OBESIDADE
plt.figure(figsize=(10,7))
sns.countplot(data=dados, x="Obesidade", hue="Genero", palette=sns.color_palette("Dark2"))
plt.xticks(rotation=-20)
plt.show()

# %% INFLUÊNCIA DE COMER ENTRE REFEIÇÕES NA OBESIDADE
plt.figure(figsize=(10,7))
sns.countplot(data=dados, x="Obesidade", hue="comida_entre_refeicoes", palette=sns.color_palette("Dark2"))
plt.xticks(rotation=-20)
plt.show()

# %% INFLUÊNCIA DE HISTÓRICO FAMILIAR NA OBESIDADE
plt.figure(figsize=(10,7))
sns.countplot(data=dados, x="Obesidade", hue="Historico_Familiar", palette=sns.color_palette("Dark2"))
plt.xticks(rotation=-20)
plt.show()

# %% RELAÇÃO ENTRE CONSUMO DE ÁLCOOL E FUMO
sns.countplot(data=dados, x="frequencia_consumo_alcool", hue=dados.Fumar)

# %% PREPARAÇÃO PARA MODELAGEM
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from joblib import dump

# %% ENCODING - VARIÁVEIS ORDINAIS
encoder = LabelEncoder()
dados_modelo = dados.copy()
for col in ['frequencia_consumo_alcool','comida_entre_refeicoes','Obesidade']:
    dados_modelo[col] = encoder.fit_transform(dados_modelo[col])

# %% ENCODING - VARIÁVEIS NOMINAIS
colunas_objeto = dados_modelo.select_dtypes(include="object").columns
dummies = pd.get_dummies(dados_modelo[colunas_objeto], dtype=int)
dados_modelo = pd.concat([dados_modelo, dummies], axis=1).drop(columns=colunas_objeto)

# %% NORMALIZAR DADOS NUMÉRICOS
x = dados_modelo.drop(columns="Obesidade")
y = dados_modelo["Obesidade"]
scaler = MaxAbsScaler()
for col in x.columns:
    scaler.fit(x[[col]])
    x[col] = scaler.transform(x[[col]])

# %% DIVIDIR CONJUNTO TREINO-TESTE
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# %% DICIONÁRIO DE MODELOS
modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Máquina de Vetores de Suporte": svm.SVC(kernel='linear'),
    "Árvore de Decisão": DecisionTreeClassifier(
        criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5
    )
}

# Binarizar as classes
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Loop pelos modelos
for nome, modelo in modelos.items():
    print(f"\n{'='*30}\nTreinando modelo: {nome}\n{'='*30}")

    pipeline = Pipeline([
        ('normalizador', StandardScaler()),
        ('classificador', modelo)
    ])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

    # Obter scores para ROC
    if hasattr(pipeline.named_steps['classificador'], "predict_proba"):
        y_score = pipeline.predict_proba(x_test)
    else:
        y_score = pipeline.decision_function(x_test)

    auc_ovr = roc_auc_score(y_test_bin, y_score, multi_class='ovr')
    print(f"ROC-AUC (One-vs-Rest): {auc_ovr:.4f}")

    # Plot ROC para cada classe
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f'Classe {i} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0,1], [0,1], 'k--', label='Aleatório')
    plt.title(f'Curva ROC - {nome}')
    plt.xlabel('Falso Positivo')
    plt.ylabel('Verdadeiro Positivo')
    plt.legend()
    plt.show()

    # SALVAR O K-NEAREST NEIGHBORS
    if nome == "K-Nearest Neighbors":
        dump(pipeline, 'knn_pipeline.pkl')
        print("Pipeline KNN salvo em 'knn_pipeline.pkl'")

#%% 