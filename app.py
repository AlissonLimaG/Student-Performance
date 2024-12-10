import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo treinado
model = joblib.load('modelo_adaBoost.pkl')

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

    final_grade = model.predict(input_data)

    if final_grade == 1:
        st.success(f"Com base nos dados informados, a análise feita indica que o estudante irá **passar**! 🎉")
    else:
        st.error(f"Com base nos dados informados, a análise feita indica que o estudante irá **reprovar**. 😢")
