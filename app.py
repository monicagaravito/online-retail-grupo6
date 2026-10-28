# =====================================================================
# Caso de Estudio UCI Online Retail - Grupo 6
# PASO 5: Despliegue en Streamlit
# Ejecutar:  streamlit run app.py
# =====================================================================
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Online Retail - Grupo 6", page_icon="🛒", layout="wide")

COLORES = {
    "Clientes VIP": "#2b9348",
    "Clientes leales": "#2b5c8f",
    "Clientes en riesgo": "#f4a261",
    "Clientes perdidos": "#d9534f",
}


# ---------- Carga de datos y modelo (se guarda en caché) ----------
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_clientes.joblib")


@st.cache_data
def cargar_datos():
    with open("kpis.json") as f:
        kpis = json.load(f)
    return {
        "kpis": kpis,
        "rfm": pd.read_csv("rfm_segmentos.csv"),
        "mensual": pd.read_csv("resumen_mensual.csv"),
        "productos": pd.read_csv("top_productos.csv"),
        "paises": pd.read_csv("top_paises.csv"),
        "metricas": pd.read_csv("metricas_clustering.csv"),
    }


modelo = cargar_modelo()
datos = cargar_datos()
rfm = datos["rfm"]

# ---------- Encabezado ----------
st.title("🛒 Caso de Estudio UCI Online Retail")
st.caption(
    "Grupo 6 · Segmentación de clientes con RFM + K-Means · "
    "Transacciones entre 01/12/2010 y 09/12/2011"
)

k = datos["kpis"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ventas totales", f"£{k['ventas_total']:,.0f}")
c2.metric("Facturas", f"{k['facturas']:,}")
c3.metric("Clientes", f"{k['clientes']:,}")
c4.metric("Productos", f"{k['productos']:,}")

tab_eda, tab_seg, tab_pred, tab_met = st.tabs(
    ["📊 Exploración", "👥 Segmentos", "🔮 Clasificar cliente", "📏 Métricas del modelo"]
)

# ---------- 1. Exploración ----------
with tab_eda:
    mensual = datos["mensual"].melt(
        id_vars="Mes", value_vars=["Ventas", "Devoluciones"],
        var_name="Tipo", value_name="Unidades",
    )
    fig = px.line(
        mensual, x="Mes", y="Unidades", color="Tipo", markers=True,
        title="Evolución mensual: ventas vs devoluciones",
        color_discrete_map={"Ventas": "#2b5c8f", "Devoluciones": "#d9534f"},
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(
            datos["productos"].sort_values("Quantity"), x="Quantity", y="Description",
            orientation="h", title="Top 10 productos más vendidos",
            labels={"Quantity": "Unidades", "Description": ""},
            color_discrete_sequence=["#2ec4b6"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.bar(
            datos["paises"].sort_values("Quantity"), x="Quantity", y="Country",
            orientation="h", title="Top 10 países por unidades vendidas",
            labels={"Quantity": "Unidades", "Country": ""},
            color_discrete_sequence=["#6a4c93"],
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- 2. Segmentos ----------
with tab_seg:
    st.subheader("Perfil de cada segmento")
    st.write(
        "**Recency**: días desde la última compra · **Frequency**: número de facturas · "
        "**Monetary**: gasto total (£)"
    )
    perfil = (
        rfm.groupby("Segmento")
        .agg(
            Clientes=("CustomerID", "count"),
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
        )
        .round(1)
        .sort_values("Monetary", ascending=False)
    )
    perfil["% clientes"] = (perfil["Clientes"] / perfil["Clientes"].sum() * 100).round(1)
    st.dataframe(perfil, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(
            perfil.reset_index(), names="Segmento", values="Clientes",
            title="Distribución de clientes", color="Segmento",
            color_discrete_map=COLORES, hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.scatter(
            rfm, x="Recency", y="Monetary", color="Segmento", log_y=True,
            opacity=0.5, title="Recency vs Monetary por segmento",
            color_discrete_map=COLORES, hover_data=["CustomerID", "Frequency"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Buscar un cliente")
    cliente = st.selectbox("CustomerID", sorted(rfm["CustomerID"].unique()))
    st.dataframe(rfm[rfm["CustomerID"] == cliente], use_container_width=True, hide_index=True)

# ---------- 3. Clasificar un cliente nuevo ----------
with tab_pred:
    st.subheader("¿A qué segmento pertenece un cliente?")
    st.write("Ingresa el comportamiento de compra y el modelo K-Means lo asigna a un segmento.")

    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Días desde la última compra", min_value=1, max_value=400, value=30)
    frequency = c2.number_input("Número de facturas", min_value=1, max_value=300, value=5)
    monetary = c3.number_input("Gasto total (£)", min_value=1.0, max_value=300000.0, value=1500.0, step=100.0)

    if st.button("Clasificar", type="primary"):
        entrada = pd.DataFrame([[recency, frequency, monetary]], columns=modelo["columnas"])
        x = modelo["scaler"].transform(np.log1p(entrada))
        cluster = int(modelo["kmeans"].predict(x)[0])
        segmento = modelo["nombres"][cluster]

        st.success(f"Segmento asignado: **{segmento}**")
        promedio = perfil.loc[segmento, ["Recency", "Frequency", "Monetary"]]
        st.write("Comparación con el promedio del segmento:")
        st.dataframe(
            pd.DataFrame({"Este cliente": [recency, frequency, monetary], "Promedio segmento": promedio.values},
                         index=["Recency", "Frequency", "Monetary"]),
            use_container_width=True,
        )

# ---------- 4. Métricas ----------
with tab_met:
    st.subheader("Comparación de algoritmos de clustering")
    st.write(
        "**Silhouette** ↑ mejor · **Davies-Bouldin** ↓ mejor · **Calinski-Harabasz** ↑ mejor. "
        "Se eligió **K-Means con k=4** por tener los mejores valores en las tres métricas."
    )
    st.dataframe(datos["metricas"].round(3), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.metric("Estabilidad ARI (media)", f"{k['ari_media']:.3f}")
    c2.metric("Desviación estándar ARI", f"{k['ari_sd']:.3f}")
    st.caption("ARI cercano a 1 = el modelo da los mismos grupos aunque se re-entrene con otra muestra.")
