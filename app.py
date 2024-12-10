import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from tabulate import tabulate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split

# Carregar os datasets
portugal_data = pd.read_csv('student-por.csv', sep=';')
mathematics_data = pd.read_csv('student-mat.csv', sep=';')

portugal_data['Course'] = 'P'
mathematics_data['Course'] = 'M'

combined_data = pd.concat([portugal_data, mathematics_data], ignore_index=True)

# Calcular a média das colunas G1, G2 e G3
combined_data['result'] = (combined_data[['G1', 'G2', 'G3']].mean(axis=1).round(2) > 10).astype(int)

# Filtrar os alunos que passaram (média maior que 10)
passed_students = combined_data[combined_data['result'] > 10]

# Filtrar os alunos que não passaram (média menor ou igual a 10)
failed_students = combined_data[combined_data['result'] <= 10]


def converter_nota(nota):
    if nota > 10:
        return 1
    else:
        return 0

# Verificar a nova distribuição dos alunos que passaram após o SMOTE
print("Distribuição após oversampling dos alunos que passaram:")

combined_data[['G1', 'G2', 'G3']] = combined_data[['G1', 'G2', 'G3']].applymap(converter_nota)

label_encoder = LabelEncoder()

for col in combined_data:
    combined_data[col] = label_encoder.fit_transform(combined_data[col])

X = combined_data.drop(columns=['G3','result','Course'])
y = combined_data['G3']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


oversample = SMOTE(sampling_strategy = 1)

X_over, y_over = oversample.fit_resample(X_train, y_train)

print(pd.Series(y_over).value_counts())


selector = SelectKBest(score_func=mutual_info_classif, k='all')
X1 = combined_data.drop(columns=['G3','result','Course'])

y = combined_data['G3']
X_new = selector.fit_transform(X1, y)

new_features = ['G2', 'G1', 'failures', 'Medu', 'paid', 'Fedu', 'Fjob', 'Mjob', 'guardian', 'age', 'schoolsup', 'higher']

newTest = X_over[new_features]

X_train, X_test, y_train, y_test = train_test_split(newTest, y_over, test_size=0.1, random_state=42)

#Melhores hiperparametros encontrados para o adaBoost
ada_boost = AdaBoostClassifier(algorithm='SAMME.R', learning_rate=0.5, n_estimators=50, random_state=42)

ada_boost.fit(X_train, y_train)

y_pred_ada = ada_boost.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_ada)
print(f"Acurácia do modelo AdaBoost: {accuracy * 100:.2f}%")


# Título da página
st.title("Previsão de Nota Final - Estudantes do Ensino Médio")

# Descrição
st.write("Insira as informações do estudante para prever se ele irá **passar** ou **reprovar**.")

# Formulário para entrada de dados
with st.form("student_data"):
    # Campos de entrada
    G2 = st.selectbox("Segundo ano",[0,1], format_func=lambda x: ["REPROVADO","APROVADO"][x])
    G1 = st.selectbox("Primeiro ano",[0,1], format_func=lambda x: ["REPROVADO","APROVADO"][x] )
    failures = st.selectbox("Número de reprovações:", [0, 1, 2, 4])
    Medu = st.selectbox("Educação da mãe:", [0, 1, 2, 3, 4], format_func=lambda x: ["Nenhuma","Educação primária","5º ano ao 9º ano","educação secundária","educação superior"][x])
    paid = st.selectbox("Aulas extras pagas:", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
    Fedu = st.selectbox("Educação do pai:", [0, 1, 2, 3, 4], format_func=lambda x: ["Nenhuma","Educação primária","5º ano ao 9º ano","educação secundária","educação superior"][x])
    Fjob = st.selectbox("Trabalho do pai:", [0, 1, 2, 3, 4,],
                        format_func=lambda x: ["Dono de casa", "Saúde", "Outros", "Serviços civis", "Professor"][x])
    Mjob = st.selectbox("Trabalho da mãe:", [0, 1, 3, 4, 2],
                        format_func=lambda x: ["Dona de casa", "Saúde", "Outros", "Serviços civis", "Professora"][x])
    guardian = st.selectbox("Guardião legal:", [0, 1, 2], format_func=lambda x: ["Pai", "Mãe", "Outros"][x])
    age = st.selectbox("Idade:", [0, 1, 2, 3, 4, 5, 6, 7],
                       format_func=lambda x: ["15", "16", "17", "18", "19", "20", "21", "22"][x])
    schoolsup = st.selectbox("Apoio educacional extra:", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
    higher = st.selectbox("Deseja fazer ensino superior:", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")

    # Botão de envio
    submit = st.form_submit_button(label='Fazer Previsão')

# Lógica de previsão, que será executada quando o botão for pressionado
if submit:
    # Criar um dataframe com os dados do formulário
    input_data = pd.DataFrame({
        'G2': [G2],
        'G1': [G1],
        'failures': [failures],
        'Medu': [Medu],
        'paid': [paid],
        'Fedu': [Fedu],
        'Fjob': [Fjob],
        'Mjob': [Mjob],
        'guardian': [guardian],
        'age': [age],
        'schoolsup': [schoolsup],
        'higher': [higher]
    })

    final_grade = ada_boost.predict(input_data)

    if final_grade == 1:
        st.success(f"Com base nos dados informados, a análise feita indica que o estudante irá **passar**! 🎉")
    else:
        st.error(f"Com base nos dados informados, a análise feita indica que o estudante irá **reprovar**. 😢")
