"""
This is a boilerplate pipeline 'get_raw_data'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import check_raw_data_folder_empty, download_and_extract_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=check_raw_data_folder_empty, inputs=None, outputs="is_empty"),
            node(
                func=download_and_extract_data,
                inputs=["params:url", "is_empty"],
                outputs=None,
            ),
        ]
    )
