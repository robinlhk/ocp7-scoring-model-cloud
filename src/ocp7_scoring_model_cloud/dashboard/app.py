import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import shap
from streamlit_shap import st_shap
from dashboard_funcs import histo_chart, request_prediction, read_parquet_from_azure

AZURE_STORAGE_CONNECTION_STRING = st.secrets["AZURE_STORAGE_CONNECTION_STRING"]
AZURE_CONTAINER_NAME="ocp7-datasets"
viz_test_blob_path = "data/08_reporting/viz_df_test.parquet"
viz_train_blob_path = "data/08_reporting/viz_df_train.parquet"
full_train_blob_path = "data/05_model_input/full_df_train.parquet"
full_test_blob_path = "data/05_model_input/full_df_test.parquet"
model_api_url = "https://ocp7-rlhk-modelapi.azurewebsites.net/predict"
shap_values_api_url = "https://ocp7-rlhk-modelapi.azurewebsites.net/explain_local" #"http://127.0.0.1:8000/explain_local"

st.set_page_config(layout="wide", page_title="Credit Scoring Dashboard", page_icon="📈")


@st.cache_data
def read_data(blob_path, select_id=None, sample_size=.4, columns="All"):
    # Read the dataframe from Azure
    df = read_parquet_from_azure(AZURE_CONTAINER_NAME, blob_path, AZURE_STORAGE_CONNECTION_STRING)

    # Select the necessary rows
    if select_id is not None:
        df = df[df["SK_ID_CURR"] == select_id]
    elif select_id is None:
        if columns != "All":
            df = df[columns]
        else:
            # Sample the dataframe
            df = df.sample(frac = sample_size, random_state=42)
    return df

st.sidebar.header("Prêt à Dépenser - Credit Scoring Dashboard")
st.title("Credit Scoring Dashboard 📈")
st.markdown(
    "L'objectif de ce dashboard est de visualiser les données des clients et de déterminer une probabilité de défaut de paiement."
)
select_df_type = st.sidebar.selectbox(
    "Selectionnez le jeu de données", ["Train"]
)

if select_df_type == "Train":
    df = read_data(viz_train_blob_path, sample_size=.3)
    id_list = df["SK_ID_CURR"].unique()
    # id_list = read_data(viz_train_blob_path, columns=["SK_ID_CURR"])["SK_ID_CURR"].unique()
else:
    df = read_data(viz_test_blob_path, sample_size=.1)
    # id_list = read_data(viz_test_blob_path, columns=["SK_ID_CURR"])["SK_ID_CURR"].unique()
    id_list = df["SK_ID_CURR"].unique()

selected_id = st.sidebar.selectbox(
    "Selectionnez un identifiant-client", id_list
)

available_columns = df.columns.tolist()
available_columns.remove("SK_ID_CURR")
default_columns = [
    x for x in available_columns if x not in ["DAYS_BIRTH", "DAYS_EMPLOYED"]
]

selected_row = df.loc[df["SK_ID_CURR"]==selected_id][default_columns]

if select_df_type == "Train":
    selected_full_df = read_data(full_train_blob_path, select_id=selected_id)
else:
    selected_full_df = read_data(full_test_blob_path, select_id=selected_id)

st.sidebar.write("Informations sur le client")
st.sidebar.dataframe(
    selected_row.T.reset_index().rename(
        columns={"index": "Information", "0": "Valeur"}
    ),
    hide_index=True,
)

st.header("Visualisation des données du client", divider='rainbow')
st.write('Cette section permet de visualiser des statistiques descriptives sur le client sélectionné et de voir son positionnement dans la population globale sur différents critères. On a également représenté les distributions des variables pour les clients en défaut et non en défaut.')
st.write("Note : si une donnée client n'apparaît pas comme verticale dans les graphique c'est qu'elle est manquante.")

# if st.button("📈 Afficher les statistiques descriptives"):
# Define the charts
selected_age = selected_row["AGE"].values[0]
selected_amt_credit = selected_row["AMT_CREDIT"].values[0]
selected_amt_income_total = selected_row["AMT_INCOME_TOTAL"].values[0]
selected_amt_annuity = selected_row["AMT_ANNUITY"].values[0]
selected_amt_goods_price = selected_row["AMT_GOODS_PRICE"].values[0]
selected_ext_source_1 = selected_row["EXT_SOURCE_1"].values[0]
selected_ext_source_2 = selected_row["EXT_SOURCE_2"].values[0]
selected_ext_source_3 = selected_row["EXT_SOURCE_3"].values[0]
selected_time_current_job_years = selected_row["TIME_CURRENT_JOB_YEARS"].values[0]

fig1 = histo_chart(df, "AGE", "Distribution de l'âge des clients", True, selected_age, nbins=10)
fig9 = histo_chart(df, "TIME_CURRENT_JOB_YEARS", "Distribution de l'ancienneté dans l'emploi", True,
                   selected_time_current_job_years, nbins=30)
fig2 = histo_chart(df, "AMT_CREDIT", "Distribution du montant du crédit", True, selected_amt_credit, nbins=50)
fig3 = histo_chart(df, "AMT_INCOME_TOTAL", "Distribution du revenu total", True, selected_amt_income_total, nbins=100)
fig4 = histo_chart(df, "AMT_ANNUITY", "Distribution de l'annuité", True, selected_amt_annuity, nbins=30)
fig5 = histo_chart(df, "AMT_GOODS_PRICE", "Distribution du prix des biens", True, selected_amt_goods_price, nbins=50)
fig6 = histo_chart(df, "EXT_SOURCE_1", "Distribution de l'EXT_SOURCE_1", True, selected_ext_source_1, nbins=50)
fig7 = histo_chart(df, "EXT_SOURCE_2", "Distribution de l'EXT_SOURCE_2", True, selected_ext_source_2, nbins=50)
fig8 = histo_chart(df, "EXT_SOURCE_3", "Distribution de l'EXT_SOURCE_3", True, selected_ext_source_3, nbins=50)

# st.write(selected_ext_source_1, selected_ext_source_2, selected_ext_source_3)
expander1 = st.expander(
    "📈 Statistiques descriptives : le client dans l'ensemble de la population",
    expanded=False,
)
col1, col2 = expander1.columns(2, gap="small")
# Display the selected row

col1.plotly_chart(fig1)
col2.plotly_chart(fig2)
#col1.plotly_chart(fig3)
col2.plotly_chart(fig4)
col1.plotly_chart(fig5)
col2.plotly_chart(fig6)
col1.plotly_chart(fig7)
col2.plotly_chart(fig8)
col1.plotly_chart(fig9)

st.header("Sélection d'un taux d'intérêt pour le crédit", divider='rainbow')
st.write(
    "Cette section permet de calculer la probabilité de défaut du client acceptable en fonction du taux d'intérêt proposé."
)
interest_rate = (
        st.slider(
            "Sélectionnez le taux d'intérêt moyen (en %)",
            min_value=0.0,
            max_value=20.0,
            value=10.0,
            step=0.5,
        )
        / 100
)

acceptable_proba = interest_rate / (1 + interest_rate)

st.markdown(
    r"La probabilité de défaut telle que l'espérance de profit de la banque est positive : $$\mathbb{P}(D)\leq\frac{i}{1+i}$$ ")
st.write(f"Pour un taux d'intérêt de **{interest_rate * 100}%**, la probabilité de défaut acceptable est de **{round(acceptable_proba * 100, 2)}%**.")

st.header("Prédiction de la probabilité de défaut du client et interprétabilité", divider='rainbow')
st.write("Cette section permet de prédire la probabilité de défaut du client et d'interpréter les résultats. Le graphique de SHAP (SHapley Additive exPlanations) permet de visualiser l'importance des variables dans la prédiction. ")
st.write("Lecture graphique : les variables en bleu diminuent la probabilité de défaut, tandis que celles en rouge l'augmentent. Les variables les plus importantes sont en haut du graphique.")

if st.button("Prédire la probabilité de défaut du client et visualiser les SHAP values"):
    features = [
        f for f in selected_full_df.columns if f not in ["SK_ID_CURR", "TARGET"]
    ]
    df_query = selected_full_df[features]

    prediction = request_prediction(
        df_query, model_url=model_api_url
    )
    proba_non_default = round(prediction["prediction"][0][0], 2)
    proba_default = round(prediction["prediction"][0][1], 3)
    st.write(
        f"""D'après le modèle, la probabilité de défaut du client est de {round(proba_default * 100, 1)}%."""
    )
    if proba_default > acceptable_proba:
        st.markdown(
            """
            Etant donné le <span style="color:blue;">taux d'intérêt moyen</span>, il est recommandé de
            <span style="background-color:red; color:white;">ne pas accorder le prêt</span>.
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            Etant donné le taux d'intérêt moyen, il est recommandé d'<span style="background-color:green; color:white;">accorder le prêt</span>.
            """,
            unsafe_allow_html=True
        )
    shap_values_response = request_prediction(df_query, model_url=shap_values_api_url)
    shap_values = shap.Explanation(
        values=np.array(shap_values_response['values']),
        base_values=np.array(shap_values_response['base_values']),
        data=np.array(shap_values_response['data']),
        feature_names=np.array(shap_values_response['feature_names'])
    )
    st_shap(shap.waterfall_plot(shap_values[0]))

