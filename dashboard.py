# dashboard_final.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Titre ---
st.set_page_config(page_title="Dashboard Adoption IA", layout="wide")
st.title("Analyse mondiale de l'adoption de l'IA")

# --- Lecture des données depuis GitHub ---
url = "https://raw.githubusercontent.com/sorel2030/Projet-final-SCI1402/main/tes_donnees.csv"
df = pd.read_csv(url)

# --- Normalisation et clustering ---
features = ['GDP', 'Tool Adoption Rate', 'Avg Productivity Change (%)']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

if 'Cluster' not in df.columns:
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Barre latérale : sélection des pays ---
st.sidebar.header("Filtrer les pays")
pays_selection = st.sidebar.multiselect("Choisissez les pays", df['country'].unique(), default=df['country'].tolist())

df_filtered = df[df['country'].isin(pays_selection)]

# --- KPIs ---
st.subheader("Indicateurs globaux des pays sélectionnés")
col1, col2, col3 = st.columns(3)
col1.metric("Moyenne GDP", f"{df_filtered['GDP'].mean():,.0f}")
col2.metric("Moyenne Taux adoption IA", f"{df_filtered['Tool Adoption Rate'].mean():.2f}%")
col3.metric("Moyenne Productivité (%)", f"{df_filtered['Avg Productivity Change (%)'].mean():.2f}%")

# --- Graphique : GDP vs Adoption IA avec clusters ---
st.subheader("Clusters : PIB vs Adoption IA")
fig_scatter = px.scatter(
    df_filtered,
    x="GDP",
    y="Tool Adoption Rate",
    color="Cluster",
    hover_name="country",
    size="Avg Productivity Change (%)",
    title="Clusters des pays selon GDP et adoption IA",
    labels={"GDP": "PIB par habitant", "Tool Adoption Rate": "Taux adoption IA"}
)
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Graphique : Productivité vs Adoption IA ---
st.subheader("Productivité vs Adoption IA")
fig_productivity = px.scatter(
    df_filtered,
    x="Tool Adoption Rate",
    y="Avg Productivity Change (%)",
    color="Cluster",
    hover_name="country",
    title="Adoption IA vs Productivité",
    labels={"Tool Adoption Rate": "Taux adoption IA", "Avg Productivity Change (%)": "Variation Productivité (%)"}
)
st.plotly_chart(fig_productivity, use_container_width=True)

# --- Carte mondiale des clusters ---
st.subheader("Carte mondiale des clusters")
fig_map = px.choropleth(
    df_filtered,
    locations="country",
    locationmode="country names",
    color="Cluster",
    hover_name="country",
    title="Clusters mondiaux d'adoption IA",
    color_continuous_scale=px.colors.sequential.Viridis
)
st.plotly_chart(fig_map, use_container_width=True)

# --- Comparaison multi-pays : bar chart ---
st.subheader("Comparaison des indicateurs par pays")
fig_bar = px.bar(
    df_filtered.melt(id_vars='country', value_vars=['GDP', 'Tool Adoption Rate', 'Avg Productivity Change (%)']),
    x='country',
    y='value',
    color='variable',
    barmode='group',
    title="Comparaison GDP, Adoption IA et Productivité par pays",
    labels={"value": "Valeur", "variable": "Indicateur"}
)
st.plotly_chart(fig_bar, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("Dashboard interactif développé pour le projet SCI1402. Source des données : tes_donnees.csv")
