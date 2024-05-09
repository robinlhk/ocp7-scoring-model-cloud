import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

def histo_chart(df: pd.DataFrame, column:str, title:str, line_chart:bool, selected_value:float, nbins:int=50):
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df[column],
            histnorm="probability",
            nbinsx=nbins,
            #xbins=dict(start=0.0, end=df["AGE"].max(), size=5),
            opacity=0.8,
        )
    )
    if line_chart:
        fig.add_vline(
            x=selected_value,
            line_dash="dash",
            line_color="green",
            annotation_text="Client sélectionné",
        )
    fig.update_layout(
        title_text=title, width=500, height=500
    )
    return fig

def request_prediction(data, model_url:str = "http://0.0.0.0:5000/invocations"):
    # Convert DataFrame to a list of dictionaries (records format)
    data_records = data.to_dict(orient='records')

    # Create the input payload
    data = {'dataframe_records': data_records}
    # Set up the request headers and URL
    headers = {'Content-Type': 'application/json'}

    # Send the POST request
    response = requests.post(model_url, headers=headers, json=data)

    # Check the response
    if response.status_code == 200:
        predictions = response.json()
        return predictions
    else:
        return print("Error:", response.status_code, response.text)
