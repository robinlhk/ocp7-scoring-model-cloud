"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    preprocess_application_train,
    preprocess_bureau_and_balance,
    preprocess_previous_applications,
    preprocess_pos_cash,
    preprocess_installments_payments,
    preprocess_credit_card_balance,
    join_datasets,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_application_train,
                inputs="base_df",
                outputs="preprocess_train_df",
                name="preprocess_train",
            ),
            node(
                func=node_preprocess_bureau_and_balance,
                inputs=["bureau_df", "bureau_balance_df"],
                outputs="bureau_agg",
                name="preprocess_bureau",
            ),
            node(
                func=node_preprocess_previous_applications,
                inputs="previous_application_df",
                outputs="previous_application_agg",
                name="preprocess_previous_applications",
            ),
            node(
                func=node_preprocess_pos_cash,
                inputs="pos_cash_df",
                outputs="pos_agg",
                name="preprocess_pos_cash",
            ),
            node(
                func=node_preprocess_installments_payments,
                inputs="installments_payments_df",
                outputs="ins_agg",
                name="preprocess_installments_payments",
            ),
            node(
                func=node_preprocess_credit_card_balance,
                inputs="credit_card_balance_df",
                outputs="cc_agg",
                name="preprocess_credit_card_balance",
            ),
            node(
                func=node_join_datasets,
                inputs=[
                    "preprocess_train_df",
                    "bureau_agg",
                    "previous_application_agg",
                    "pos_agg",
                    "ins_agg",
                    "cc_agg",
                ],
                outputs="preprocessed_df",
                name="join_datasets",
            ),
        ]
    )
