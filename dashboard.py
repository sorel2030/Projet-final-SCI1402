import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Titre du dashboard
st.title("Analyse de l'adoption de l'IA dans le monde")

# Charger les données
df = pd.read_csv("tes_donnees.csv")

# Normalisation des variables pour clustering
features = ['GDP', 'Tool Adoption Rate', 'Avg Productivity Change (%)']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Ajouter les clusters si pas déjà fait
if 'Cluster' not in df.columns:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

# Sélection d’un pays
pays = st.selectbox("Sélectionnez un pays", df["country"].unique())
pays_data = df[df["country"] == pays].iloc[0]

st.subheader(f"Indicateurs pour {pays}")
st.write(f"- **GDP** : {pays_data['GDP']}")
st.write(f"- **Taux d’adoption IA** : {pays_data['Tool Adoption Rate']}%")
st.write(f"- **Productivité moyenne (%)** : {pays_data['Avg Productivity Change (%)']}%")
st.write(f"- **Cluster** : {pays_data['Cluster']}")

# Scatter plot GDP vs Tool Adoption Rate
fig1 = px.scatter(
    df,
    x="GDP",
    y="Tool Adoption Rate",
    color="Cluster",
    hover_name="country",
    title="Clusters : PIB vs Adoption de l'IA"
)
st.plotly_chart(fig1)

# Carte mondiale des clusters
fig2 = px.choropleth(
    df,
    locations="country",
    locationmode="country names",
    color="Cluster",
    title="Carte mondiale des clusters IA"
)
st.plotly_chart(fig2)

# Importance des variables si disponible
if 'RF_Importance_GDP' in df.columns:  # optionnel, si tu as calculé l'importance RF dans le notebook
    st.subheader("Importance des variables (Random Forest)")
    rf_importances = df[['GDP', 'Tool Adoption Rate', 'Avg Productivity Change (%)']].mean()
    fig3, ax = plt.subplots()
    sns.barplot(x=rf_importances.index, y=rf_importances.values, ax=ax, palette="Blues_d")
    ax.set_ylabel("Importance")
    ax.set_title("Importance des variables")
    st.pyplot(fig3)
