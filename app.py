# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:54:30 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import plotly.express as px

# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Predicción de Riesgo de Clientes")

# Carga automática del archivo datos_clientes.csv
def load_data():
    try:
        data = pd.read_csv("datos_clientes_balanceado.csv")  # Asegúrate de usar el archivo balanceado
        return data
    except FileNotFoundError:
        st.error("El archivo 'datos_clientes_balanceado.csv' no se encontró.")
        return None

data = load_data()
if data is not None:
    st.write("### Datos cargados")
    st.dataframe(data)

    # Configuración del modelo
    st.sidebar.header("Configuración del Modelo")
    # Eliminar 'Ingresos' del selectbox
    target = st.sidebar.selectbox(
        "Selecciona la variable objetivo", 
        [col for col in data.columns if col != "Ingresos"]  # Excluir 'Ingresos' de las opciones
    )
    features = st.sidebar.multiselect("Selecciona las variables predictoras", [col for col in data.columns if col != target])

    if target and features:
        # Filtrar clases con solo una instancia
        class_counts = data[target].value_counts()
        valid_classes = class_counts[class_counts > 1].index
        filtered_data = data[data[target].isin(valid_classes)]

        X = filtered_data[features]
        y = filtered_data[target]
        
        # Ajustar el tamaño de la prueba (test_size) para que sea suficiente para cada clase
        # El tamaño de la prueba debe ser al menos el número de clases, o un 20% de los datos
        test_size = max(0.2, 1 / len(np.unique(y)))  # Un tamaño de prueba mínimo del 20% o suficiente para cubrir todas las clases

        # Si hay muchas clases, asegúrate de que haya al menos un ejemplo de cada clase en el conjunto de prueba
        if len(np.unique(y)) > len(y) * test_size:
            test_size = len(np.unique(y)) / len(y)  # Asegurar que el test_size sea suficientemente grande

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        # Entrenamiento del modelo con balanceo de clases
        model = RandomForestClassifier(random_state=42, class_weight='balanced')  # Añadir class_weight='balanced'
        model.fit(X_train, y_train)

        # Predicción y métricas
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Cálculo del AUC-ROC
        try:
            unique_classes = np.unique(y_test)  # Obtener las clases únicas presentes en y_test
            n_classes = len(unique_classes)  # Número de clases únicas presentes

            if n_classes > 2:  # Caso de clasificación multiclase
                # Calcular el AUC-ROC con las probabilidades para todas las clases presentes
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', labels=unique_classes)
            else:  # Caso binario
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        except ValueError as e:
            st.error(f"Error al calcular el AUC-ROC: {e}")
            auc_score = None

        conf_matrix = confusion_matrix(y_test, y_pred)

        # Resultados
        st.write("### Resultados del Modelo")
        if auc_score is not None:
            st.write(f"**AUC-ROC:** {auc_score:.2f}")
        st.write("**Matriz de Confusión:**")
        st.write(conf_matrix)
        st.write("**Reporte de Clasificación:**")
        st.text(classification_report(y_test, y_pred))

        # Visualización de la curva ROC para clasificación binaria
        if auc_score is not None and n_classes == 2:
            fig = px.area(
                x=[0, 1],
                y=[0, auc_score],
                labels={'x': "Falsos Positivos", 'y': "Verdaderos Positivos"},
                title="Curva ROC",
                width=700,
                height=400
            )
            st.plotly_chart(fig)

        # Predicción de nuevos clientes
        st.sidebar.header("Predicción de Nuevos Clientes")
        user_input = {}
        for feature in features:
            user_input[feature] = st.sidebar.number_input(f"{feature}", value=float(X[feature].mean()))
        user_input_df = pd.DataFrame([user_input])

        if st.sidebar.button("Predecir Riesgo"):
            prediction = model.predict(user_input_df)[0]
            prob = model.predict_proba(user_input_df)[0]
            st.sidebar.write(f"**Probabilidades:** {prob}")
            st.sidebar.write("**Resultado:**", "Alto Riesgo" if prediction == 1 else "Bajo Riesgo")
        
        # Sección de ayuda
        st.sidebar.write("### Ayuda")
        st.sidebar.write("""
            Este modelo utiliza un Random Forest para predecir el riesgo de clientes basado en sus características. 
            Las predicciones están basadas en las variables seleccionadas y la variable objetivo que se elija.
            El modelo calcula la probabilidad de que un cliente pertenezca a una clase de riesgo, proporcionando una 
            clasificación de 'Alto Riesgo' o 'Bajo Riesgo'.
            
            **Desarrollado por:**
            Javier Horacio Pérez Ricárdez
        """)
else:
    st.write("Por favor, verifica que el archivo 'datos_clientes_balanceado.csv' esté disponible en el directorio.")
