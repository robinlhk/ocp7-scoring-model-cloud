import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os
from dashboard_funcs import histo_chart, request_prediction, read_parquet_from_azure
from dotenv import load_dotenv

#
# Function to read Parquet file from Azure Blob Storage
# load_dotenv()
#
# if os.getenv("ENV") == "prod":

AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=ocp7;AccountKey=lxYlW5w50DkmrWxrxopFj9TL9qrBQnjf2NtXTjeoElzRU2GaOV6hcznhASKB3+SJPSeshrlLo/JP+AStm1sHOQ==;EndpointSuffix=core.windows.net"
AZURE_CONTAINER_NAME="ocp7-datasets"
viz_test_blob_path = "data/08_reporting/viz_df_test.parquet"
viz_train_blob_path = "data/08_reporting/viz_df_train.parquet"
full_train_blob_path = "data/05_model_input/full_df_train.parquet"
model_api_url = "https://ocp7-rlhk-modelapi.azurewebsites.net/predict"
# else:
#     viz_test_df_path = "data/08_reporting/viz_df_test.parquet"
#     viz_train_df_path = "data/08_reporting/viz_df_train.parquet"
#     full_train_df_path = "data/05_model_input/full_train_df.parquet"

st.set_page_config(layout="wide", page_title="Credit Scoring Dashboard", page_icon="📈")

st.sidebar.header("Prêt à Dépenser - Credit Scoring Dashboard")
st.markdown(
    "L'objectif de ce dashboard est de visualiser les données des clients et de déterminer un score de solvabilité (credit score)."
)
select_df_type = st.sidebar.selectbox(
    "Selectionnez le jeu de données", ["Train", "Test"]
)
#st.write(read_parquet_from_azure(os.getenv("AZURE_CONTAINER_NAME"), viz_train_blob_path, os.getenv("AZURE_STORAGE_CONNECTION_STRING")))
# if os.getenv("ENV") == "prod":
if select_df_type == "Train":
    df = read_parquet_from_azure(AZURE_CONTAINER_NAME, viz_train_blob_path, AZURE_STORAGE_CONNECTION_STRING)
else:
    df = read_parquet_from_azure(AZURE_CONTAINER_NAME, viz_test_blob_path, AZURE_STORAGE_CONNECTION_STRING)
# else:
#     if select_df_type == "Train":
#         df = pd.read_parquet(viz_train_df_path)
#     else:
#         df = pd.read_parquet(viz_test_df_path)

selected_id = st.sidebar.selectbox(
    "Selectionnez un identifiant-client", df["SK_ID_CURR"].unique()
)

available_columns = df.columns.tolist()
available_columns.remove("SK_ID_CURR")
default_columns = [
    x for x in available_columns if x not in ["DAYS_BIRTH", "DAYS_EMPLOYED"]
]

selected_row = df.loc[df["SK_ID_CURR"] == selected_id, default_columns]

st.sidebar.write("Informations sur le client")
st.sidebar.dataframe(
    selected_row.T.reset_index().rename(
        columns={"index": "Information", "0": "Valeur"}
    ),
    hide_index=True,
)

# Define the charts
selected_age = df.loc[df["SK_ID_CURR"] == selected_id, "AGE"].values[0]
selected_amt_credit = df.loc[df["SK_ID_CURR"] == selected_id, "AMT_CREDIT"].values[0]
selected_amt_income_total = df.loc[df["SK_ID_CURR"] == selected_id, "AMT_INCOME_TOTAL"].values[0]
selected_amt_annuity = df.loc[df["SK_ID_CURR"] == selected_id, "AMT_ANNUITY"].values[0]
selected_amt_goods_price = df.loc[df["SK_ID_CURR"] == selected_id, "AMT_GOODS_PRICE"].values[0]
selected_ext_source_1 = df.loc[df["SK_ID_CURR"] == selected_id, "EXT_SOURCE_1"].values[0]
selected_ext_source_2 = df.loc[df["SK_ID_CURR"] == selected_id, "EXT_SOURCE_2"].values[0]
selected_ext_source_3 = df.loc[df["SK_ID_CURR"] == selected_id, "EXT_SOURCE_3"].values[0]
selected_time_current_job_years = df.loc[df["SK_ID_CURR"] == selected_id, "TIME_CURRENT_JOB_YEARS"].values[0]

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

expander1 = st.expander(
    "📈 Statistiques descriptives : le client dans l'ensemble de la population",
    expanded=False,
)
col1, col2 = expander1.columns(2, gap="small")
# Display the selected row

col1.plotly_chart(fig1)
col2.plotly_chart(fig2)
col1.plotly_chart(fig3)
col2.plotly_chart(fig4)
col1.plotly_chart(fig5)
col2.plotly_chart(fig6)
col1.plotly_chart(fig7)
col2.plotly_chart(fig8)
col1.plotly_chart(fig9)

interest_rate = (
        st.slider(
            "Sélectionnez le taux d'intérêt moyen (en %)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
        )
        / 100
)
acceptable_proba = interest_rate / (1 + interest_rate)

st.markdown(
    r"La probabilité de défaut telle que l'espérance de gain de la banque est positive : $$\mathbb{P}(D)\geq\frac{i}{1+i}$$ ")

if st.button("Prédire la probabilité de défaut du client"):
    # if os.getenv("ENV") == "prod":
    if select_df_type == "Train":
        full_df = read_parquet_from_azure(AZURE_CONTAINER_NAME, full_train_blob_path, AZURE_STORAGE_CONNECTION_STRING)
        selected_full_df = full_df.loc[full_df["SK_ID_CURR"] == selected_id]
        #TODO: implement test data
# else:
#     if select_df_type == "Train":
#         full_df = pd.read_parquet(full_train_df_path)
#         selected_full_df = full_df.loc[full_df["SK_ID_CURR"] == selected_id]
    features = [
        f for f in selected_full_df.columns if f not in ["SK_ID_CURR", "TARGET"]
    ]
    df_query = selected_full_df[features]
    # st.write(df_query)
    # st.write(selected_full_df)

    prediction = request_prediction(
        df_query, model_url=model_api_url
    )
    proba_non_default = round(prediction["prediction"][0][0], 2)
    proba_default = round(prediction["prediction"][0][1], 3)
    st.write(
        f"""D'après le modèle, la probabilité de défaut du client est de {proba_default * 100}%."""
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
    # st.write("Retour du modèle", prediction)
    # st.write("Elements envoyés au modèle pour la prédiction", selected_full_df)


# def main():
#     MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
#
#     st.title('Test de l\'API de prédiction')
#
#     # Load your data into a DataFrame (assuming data.csv is your data file)
#     df = pd.read_csv('data.csv')
#
#     # Display the DataFrame in Streamlit
#     st.dataframe(df)
#
#     predict_btn = st.button('Prédire')
#     if predict_btn:
#         data = [[revenu_med, age_med, nb_piece_med, nb_chambre_moy,
#                  taille_pop, occupation_moy, latitude, longitude]]
#         pred = None
#
#         if api_choice == 'MLflow':
#             pred = request_prediction(MLFLOW_URI, data)[0] * 100000
#         elif api_choice == 'Cortex':
#             pred = request_prediction(CORTEX_URI, data)[0] * 100000
#         elif api_choice == 'Ray Serve':
#             pred = request_prediction(RAY_SERVE_URI, data)[0] * 100000
#         st.write(
#             'Le prix médian d\'une habitation est de {:.2f}'.format(pred))

# # app.py  import streamlit as st  # Your Streamlit app code here
# if __name__ == '__main__':
#     st.set_option('server.enableCORS', True)