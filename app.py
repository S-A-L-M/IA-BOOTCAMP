import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import os
from logic import process_data, apply_segmentation, get_explanation_tree
from sklearn.tree import plot_tree

# Configuración de página (Tab del navegador)
st.set_page_config(page_title="Retail Intelligence AI", layout="wide", initial_sidebar_state="expanded")

# Estilo CSS para limpieza visual
st.markdown("""
    <style>
    .stDeployButton {display:none;} 
    [data-testid="stMetricValue"] { font-size: 28px; }
    </style>
""", unsafe_allow_html=True)

#  Carga de Datos con Manejo de Errores y Caché Silencioso. Validar comportamiento hot-reload
@st.cache_data(show_spinner=False)
def load_and_validate_data(file_path):
    if not os.path.exists(file_path):
        return None, f"No se encontró el archivo '{file_path}'. Por favor, verifica que esté en la carpeta del proyecto."
    try:
        data = process_data(file_path)
        return data, None
    except Exception as e:
        return None, f"Error al procesar los datos: {str(e)}"

# --- HEADER PRINCIPAL ---
st.title("🛍️ Sistema de Inteligencia de Clientes")
st.caption("MVP: Segmentación Automática basada en Comportamiento de Compra (RFM)")

# --- FLUJO DE EJECUCIÓN ---
DATA_FILE = "Data_clean_model.xlsx"

# Mostramos un spinner solo durante la carga inicial
with st.spinner('Actualizando base de datos...'):
    rfm, error_msg = load_and_validate_data(DATA_FILE)

if error_msg:
    st.error(error_msg)
    st.info("Tip: El archivo debe ser el Excel limpio generado por el equipo de automatizaciónn")
    st.stop()

if rfm is not None:
    # Def Estratégicas (Lenguaje de Negocio)
    segment_info = {
        "3": {"label": " Cliente VIP ", "color": "blue", "desc": "Tus mejores clientes. Compran seguido y gastan mucho.", "accion": "Programa de lealtad y regalos."},
        "2": {"label": "Clientes Leales", "color": "green", "desc": "Clientes constantes que responden bien a promociones.", "accion": "Venta cruzada (upselling)."},
        "1": {"label": "Clientes Potenciales", "color": "orange", "desc": "Clientes nuevos o con compras esporádicas de alto valor.", "accion": "Incentivo para segunda compra."},
        "0": {"label": "Clientes En Riesgo", "color": "red", "desc": "No han comprado hace tiempo. ¡Peligro de abandono!", "accion": "Descuentos de reactivación."}
    }

    # --- SIDEBAR (Panel de Control) ---
    st.sidebar.header("Configuración de IA")
    k = st.sidebar.select_slider(
        "Nivel de detalle de grupos", 
        options=[2, 3, 4], 
        value=4,
        help="Ajusta cuántos grupos de clientes quieres identificar."
    )
    
    # Ejecución del Modelo ML
    df_res = apply_segmentation(rfm, n_clusters=k)
    df_res['Segmento'] = df_res['Cluster'].astype(str).map(lambda x: segment_info[x]['label'])

    # --- SECCIÓN 1: KPIs ---
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clientes Analizados", f"{len(df_res):,}")
    m2.metric("Ventas Totales", f"$ {df_res['Monetary'].sum():,.0f}")
    m3.metric("Ticket Promedio", f"$ {df_res['Monetary'].mean():,.2f}")
    m4.metric("N° de VIPs", len(df_res[df_res['Cluster'] == k-1]))
    # --- SECCIÓN 2: AYUDA AL USUARIO ---
    with st.expander("Guía: Qué significan estos grupos???"):
        cols = st.columns(k)
        for i in range(k):
            info = segment_info[str(i)]
            cols[i].markdown(f"**{info['label']}**")
            cols[i].caption(info['desc'])
            cols[i].info(f" **Acción:** {info['accion']}")

    # --- SECCIÓN 3: VISUALIZACIÓN ---
    st.subheader("🌐 Mapa de Comportamiento 3D")
    st.write("Visualiza dónde se ubica cada cliente según su Recencia, Frecuencia y Gasto.")
    
    color_map = {segment_info[str(i)]['label']: segment_info[str(i)]['color'] for i in range(k)}
    fig = px.scatter_3d(
        df_res, x='Recency', y='Frequency', z='Monetary', 
        color='Segmento',
        color_discrete_map=color_map,
        opacity=0.7, height=650,
        labels={'Recency': 'Días sin comprar', 'Frequency': 'N° de Compras', 'Monetary': 'Total Gastado ($)'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- SECCIÓN 4: EXPLICABILIDAD (Árbol de Decisión) ---
    st.divider()
    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        st.subheader("Cómo segmenta la IA?")
    
    # Llamamos a la lógica y extraemos los 3 valores (incluyendo 'acc')
    tree_model, features, acc = get_explanation_tree(df_res)
    
    # --- MÉTRICA DE VALIDACIÓN ---
    # Mostramos qué tan "bueno" es el modelo para clasificar
    st.markdown(f"**Nivel de Confianza (Accuracy):** `{acc*100:.1f}%`")
    st.progress(acc) # Una barra visual que se llena según la precisión
    st.caption("Validación técnica: El modelo aprendió con el 70% de los datos y se verificó con el 30% restante.")

    # --- GRÁFICO DEL ÁRBOL ---
    fig_tree, ax = plt.subplots(figsize=(12, 8))
    plot_tree(
        tree_model, 
        feature_names=['Días Inactivo', 'N° Compras', 'Gasto Total'], 
        class_names=[segment_info[str(i)]['label'] for i in range(k)],
        filled=True, rounded=True, fontsize=10, ax=ax
    )
    st.pyplot(fig_tree)

    with col_right:
        st.subheader("Perfil Promedio")
        st.write("Valores típicos por cada grupo:")
        resumen = df_res.groupby('Segmento').agg({
            'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'
        }).rename(columns={'Recency': 'Días Inactivo', 'Frequency': 'Compras', 'Monetary': 'Gasto ($)'})
        st.dataframe(resumen.style.format("{:,.1f}").background_gradient(cmap='Greens'), use_container_width=True)

    st.sidebar.success(" Modelo entrenado en tiempo real")