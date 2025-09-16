import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Titre du dashboard
st.title("Analyse de l'adoption de l'IA dans le monde")

# --- Lecture des données depuis GitHub ---
url = "https://raw.githubusercontent.com/sorel2030/Projet-final-SCI1402/main/tes_donnees.csv"
df = pd.read_csv(url)

# --- Normalisation pour clustering ---
features = ['GDP', 'Tool Adoption Rate', 'Avg Productivity Change (%)']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# --- Clustering (si pas déjà présent dans le CSV) ---
if 'Cluster' not in df.columns:
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Sélection d'un pays pour afficher ses indicateurs ---
pays = st.selectbox("Sélectionnez un pays", df["country"].unique())
pays_data = df[df["country"] == pays].iloc[0]

st.subheader(f"Indicateurs pour {pays}")
st.write(f"- **GDP** : {pays_data['GDP']}")
st.write(f"- **Taux d’adoption IA** : {pays_data['Tool Adoption Rate']}%")
st.write(f"- **Productivité moyenne (%)** : {pays_data['Avg Productivity Change (%)']}%")
st.write(f"- **Cluster** : {pays_data['Cluster']}")

# --- Scatter plot GDP vs Tool Adoption Rate ---
fig1 = px.scatter(
    df,
    x="GDP",
    y="Tool Adoption Rate",
    color="Cluster",
    hover_name="country",
    title="Clusters : PIB vs Adoption de l'IA"
)
st.plotly_chart(fig1)

# --- Carte mondiale des clusters ---
fig2 = px.choropleth(
    df,
    locations="country",
    locationmode="country names",
    color="Cluster",
    title="Carte mondiale des clusters IA"
)
st.plotly_chart(fig2)

# --- Importance des variables (optionnel si disponible) ---
# Exemple pour montrer un graphique si tu as déjà calculé importances RF
if {'GDP', 'Tool Adoption Rate', 'Avg Productivity Change (%)'}.issubset(df.columns):
    st.subheader("Importance des variables (Random Forest simulée)")
    rf_importances = df[features].mean()  # placeholder, adapter si tu as les vraies importances
    fig3, ax = plt.subplots()
    sns.barplot(x=rf_importances.index, y=rf_importances.values, ax=ax, palette="Blues_d")
    ax.set_ylabel("Importance")
    ax.set_title("Importance des variables")
    st.pyplot(fig3)
