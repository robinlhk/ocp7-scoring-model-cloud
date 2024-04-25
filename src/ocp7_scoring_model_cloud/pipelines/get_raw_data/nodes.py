"""
This is a boilerplate pipeline 'get_raw_data'
generated using Kedro 0.19.5
"""

# https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip

import os
import requests
from zipfile import ZipFile
from io import BytesIO

def check_raw_data_folder_empty():
    raw_data_path = './data/01_raw'
    # Ensure the directory exists and is empty
    return os.path.exists(raw_data_path) and not os.listdir(raw_data_path)

def download_and_extract_data(url, extract_to='data/01_raw'):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Assuming the file is a zip for this example
    with ZipFile(BytesIO(response.content)) as the_zip:
        the_zip.extractall(path=extract_to)
    print(f"Data extracted to {extract_to}")
