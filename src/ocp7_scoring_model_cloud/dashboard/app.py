import streamlit as st
import requests
import pandas as pd


from kedro.framework.context import KedroSession
from kedro.framework.startup import bootstrap_project

# Set the project path and bootstrap Kedro
project_path = "C:/Users/9509298u/Documents/GitHub/OC_Projects/ocp7-scoring-model-cloud"
bootstrap_project(project_path)

# Load the Kedro session
with KedroSession.create(project_path=project_path) as session:
    context = session.load_context()

    # Access a dataset from the Kedro catalog
    dataset_name = "example_dataset"
    df = context.catalog.load(dataset_name)

    # Display the DataFrame in Streamlit
    st.dataframe(df)

def request_prediction(model_uri, data):
    # Convert DataFrame to a list of dictionaries (records format)
    data_records = data.to_dict(orient='records')

    # Create the input payload
    data = {'dataframe_records': data_records}
    # Set up the request headers and URL
    headers = {'Content-Type': 'application/json'}
    url = "http://127.0.0.1:5000/invocations"

    # Send the POST request
    response = requests.post(url, headers=headers, json=data)

    # Check the response
    if response.status_code == 200:
        predictions = response.json()
        return predictions
    else:
        return print("Error:", response.status_code, response.text)
#
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
#
#
# if __name__ == '__main__':
#     main()
