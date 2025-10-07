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

# %%

valores_unicos = {
    coluna: dados[coluna].unique().tolist()
    for coluna in dados.select_dtypes(include=['object']).columns
}

valores_unicos

# %% VERIFICAR DUPLICADOS
dados.duplicated().sum()
dados.loc[dados.duplicated(keep=False), :]

# %% CONVERTENDO COLUNAS NUMÉRICAS
for col in ['Idade', 'Peso', 'consumo_vegetais_diario','numero_refeicoes_diario', 
            'consumo_agua_diario','atividade_fisica_diaria','uso_eletronicos_diario']:
    dados[col] = dados.loc[:,col].round().astype(int)

# %%
# Dicionário de tradução
traducao = {
    'Female': 'Feminino',
    'Male': 'Masculino',
    'yes': 'Sim',
    'no': 'Não',
    'Sometimes': 'Às vezes',
    'Frequently': 'Frequentemente',
    'Always': 'Sempre',
    'Public_Transportation': 'Transporte Público',
    'Walking': 'Caminhada',
    'Automobile': 'Automóvel',
    'Motorbike': 'Motocicleta',
    'Bike': 'Bicicleta',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Nível I',
    'Overweight_Level_II': 'Sobrepeso Nível II',
    'Obesity_Type_I': 'Obesidade Tipo I',
    'Obesity_Type_II': 'Obesidade Tipo II',
    'Obesity_Type_III': 'Obesidade Tipo III',
    'Insufficient_Weight': 'Abaixo do Peso'
}

# Substitui os valores diretamente no DataFrame
dados = dados.replace(traducao)

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

# %% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from joblib import dump

# %% CARREGAR DADOS
dados_modelo = dados.copy()

# %% ENCODING - VARIÁVEIS ORDINAIS E TARGET
le_freq = LabelEncoder()
dados_modelo['frequencia_consumo_alcool'] = le_freq.fit_transform(dados_modelo['frequencia_consumo_alcool'])
dump(le_freq, "encoder_frequencia.joblib")

le_comida = LabelEncoder()
dados_modelo['comida_entre_refeicoes'] = le_comida.fit_transform(dados_modelo['comida_entre_refeicoes'])
dump(le_comida, "encoder_comida.joblib")

le_obesidade = LabelEncoder()
dados_modelo['Obesidade'] = le_obesidade.fit_transform(dados_modelo['Obesidade'])
dump(le_obesidade, "encoder_obesidade.joblib")

# %% SEPARAR FEATURES E TARGET
X = dados_modelo.drop(columns="Obesidade")
y = dados_modelo["Obesidade"]

# %% DEFINIR COLUNAS POR TIPO
colunas_numericas = ['Idade', 'Peso', 'Altura']
colunas_categoricas = ['Genero', 'Fumar', 'Historico_Familiar']
colunas_ordinais = ['frequencia_consumo_alcool', 'comida_entre_refeicoes']

# %% DIVIDIR CONJUNTO TREINO/TESTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# %% CRIAR PIPELINE KNN
preprocessador = ColumnTransformer(transformers=[
    ('num', StandardScaler(), colunas_numericas),
    ('cat', OneHotEncoder(handle_unknown='ignore'), colunas_categoricas),
    ('ord', 'passthrough', colunas_ordinais)
])

pipeline_knn = Pipeline([
    ('preprocessador', preprocessador),
    ('classificador', KNeighborsClassifier(n_neighbors=3))
])

# Treinar
pipeline_knn.fit(X_train, y_train)

# Avaliar
y_pred = pipeline_knn.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# ROC-AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]
y_score = pipeline_knn.predict_proba(X_test)
auc_ovr = roc_auc_score(y_test_bin, y_score, multi_class='ovr')
print(f"ROC-AUC (One-vs-Rest): {auc_ovr:.4f}")

plt.figure(figsize=(8,6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f'Classe {i} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0,1], [0,1], 'k--', label='Aleatório')
plt.title('Curva ROC - KNN')
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.legend()
plt.show()

# Salvar pipeline KNN
dump(pipeline_knn, 'knn_pipeline.pkl')
print("✅ Pipeline KNN salvo em 'knn_pipeline.pkl'")

#%% 