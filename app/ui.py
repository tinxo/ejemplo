import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Subscription Predictor",
    page_icon="🔮",
    layout="wide"
)

# Configuración de la API
API_URL = "http://localhost:8000"

# Función para verificar conexión con la API
@st.cache_data(ttl=60)
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.json() if response.status_code == 200 else None
    except:
        return None

# Función para obtener información del modelo
@st.cache_data(ttl=300)
def get_model_info():
    try:
        response = requests.get(f"{API_URL}/model/info")
        return response.json() if response.status_code == 200 else None
    except:
        return None

# Header
st.title("🔮 Subscription Predictor")
st.markdown("Predice la probabilidad de suscripción basado en edad e ingresos")

# Verificar estado de la API
health = check_api_health()
if not health:
    st.error("❌ No se puede conectar con la API. Asegúrate de que esté ejecutándose.")
    st.stop()

# Sidebar con información del modelo
st.sidebar.header("📊 Información del Modelo")
model_info = get_model_info()
if model_info:
    st.sidebar.success("✅ Modelo cargado")
    st.sidebar.json(model_info)
else:
    st.sidebar.error("❌ Modelo no disponible")

# Layout principal
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎯 Realizar Predicción")
    
    # Inputs con validación
    age = st.number_input(
        "Edad", 
        min_value=18, 
        max_value=100, 
        value=35,
        help="Edad en años (18-100)"
    )
    
    income = st.number_input(
        "Ingreso Anual", 
        min_value=0, 
        max_value=500000, 
        value=50000,
        step=1000,
        help="Ingreso anual en USD"
    )
    
    if st.button("🚀 Predecir", type="primary"):
        with st.spinner("Realizando predicción..."):
            try:
                payload = {"age": int(age), "income": int(income)}
                response = requests.post(f"{API_URL}/predict", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Mostrar resultado principal
                    prediction = result["prediction"]
                    probabilities = result["prediction_proba"]
                    
                    if prediction == "yes":
                        st.success(f"✅ **Probable Suscripción** (Confianza: {probabilities['yes']:.1%})")
                    else:
                        st.info(f"ℹ️ **No Suscripción** (Confianza: {probabilities['no']:.1%})")
                    
                    # Mostrar probabilidades
                    st.subheader("📈 Probabilidades")
                    prob_df = pd.DataFrame(
                        list(probabilities.items()), 
                        columns=['Resultado', 'Probabilidad']
                    )
                    prob_df['Probabilidad'] = prob_df['Probabilidad'].apply(lambda x: f"{x:.1%}")
                    st.table(prob_df)
                    
                    # Gráfico de probabilidades
                    fig = px.bar(
                        x=list(probabilities.keys()),
                        y=list(probabilities.values()),
                        title="Distribución de Probabilidades",
                        color=list(probabilities.values()),
                        color_continuous_scale="viridis"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Información del modelo usado
                    with st.expander("🔍 Detalles del Modelo"):
                        st.json(result["model_info"])
                    
                else:
                    st.error(f"Error en la predicción: {response.text}")
                    
            except Exception as e:
                st.error(f"Error de conexión: {str(e)}")

with col2:
    st.header("📊 Explorar Datos")
    
    # Simulador de rangos
    st.subheader("🎛️ Explorador de Rangos")
    
    age_range = st.slider("Rango de Edad", 18, 100, (25, 65))
    income_range = st.slider("Rango de Ingreso", 20000, 200000, (30000, 100000), step=5000)
    
    if st.button("🔍 Explorar Combinaciones"):
        with st.spinner("Generando predicciones..."):
            # Crear grid de predicciones
            ages = list(range(age_range[0], age_range[1]+1, 5))
            incomes = list(range(income_range[0], income_range[1]+1, 5000))
            
            results = []
            for a in ages[:10]:  # Limitar para evitar demasiadas requests
                for i in incomes[:10]:
                    try:
                        payload = {"age": a, "income": i}
                        response = requests.post(f"{API_URL}/predict", json=payload)
                        if response.status_code == 200:
                            result = response.json()
                            results.append({
                                'Edad': a,
                                'Ingreso': i,
                                'Predicción': result['prediction'],
                                'Prob_Yes': result['prediction_proba'].get('yes', 0)
                            })
                    except:
                        continue
            
            if results:
                df_results = pd.DataFrame(results)
                
                # Heatmap de probabilidades
                pivot_table = df_results.pivot_table(
                    values='Prob_Yes', 
                    index='Edad', 
                    columns='Ingreso', 
                    aggfunc='mean'
                )
                
                fig = px.imshow(
                    pivot_table,
                    title="Mapa de Calor: Probabilidad de Suscripción",
                    color_continuous_scale="RdYlGn",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de resultados
                st.subheader("📋 Resultados Detallados")
                st.dataframe(df_results, use_container_width=True)

# Footer con información adicional
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🏥 Estado API", "Activa" if health else "Inactiva")

with col2:
    if model_info:
        st.metric("🤖 Tipo Modelo", model_info.get("model_type", "N/A"))

with col3:
    st.metric("⏰ Última Actualización", datetime.now().strftime("%H:%M:%S"))

# Opción de feedback (si implementaste el endpoint)
st.markdown("---")
st.header("💬 Feedback del Modelo")

with st.expander("📝 Enviar Feedback"):
    st.info("Si conoces el resultado real, ayúdanos a mejorar el modelo")
    
    feedback_age = st.number_input("Edad (Feedback)", min_value=18, max_value=100, value=35, key="fb_age")
    feedback_income = st.number_input("Ingreso (Feedback)", min_value=0, value=50000, key="fb_income")
    actual_result = st.selectbox("Resultado Real", ["yes", "no"])
    
    if st.button("📤 Enviar Feedback"):
        try:
            feedback_payload = {
                "age": int(feedback_age),
                "income": int(feedback_income),
                "actual_subscription": actual_result
            }
            # Solo si implementaste el endpoint de feedback
            # response = requests.post(f"{API_URL}/feedback", json=feedback_payload)
            # if response.status_code == 200:
            #     st.success("✅ Feedback enviado correctamente")
            # else:
            #     st.error("❌ Error enviando feedback")
            st.info("💡 Endpoint de feedback no implementado aún")
        except Exception as e:
            st.error(f"Error: {str(e)}")