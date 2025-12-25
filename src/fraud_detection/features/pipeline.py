from sklearn.pipeline import Pipeline
from fraud_detection.features.custom_features import CustomFeatureTransformer
from fraud_detection.features.preprocessing import build_preprocessing_pipeline


def build_feature_pipeline(num_features, cat_features):
    return Pipeline(
        steps=[
            ("custom_features", CustomFeatureTransformer()),
            ("preprocessing", build_preprocessing_pipeline(
                numeric_features=num_features,
                categorical_features=cat_features,
            )),
        ]
    )
