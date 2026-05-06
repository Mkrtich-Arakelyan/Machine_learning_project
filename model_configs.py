from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import FunctionTransformer

RANDOM_STATE = 42

def clip_extreme_values(X):
    X = np.asarray(X)
    return np.clip(X, -100000, 100000)

def get_model_configs():

    return {
        "KNN": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clipper", FunctionTransformer(clip_extreme_values)),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier())
            ]),
            "params": {
                "model__n_neighbors": [3, 5, 7, 11, 13, 15],
                "model__metric": ["euclidean", "manhattan"]
            }
        },
        "Naive Bayes": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GaussianNB())
            ]),
            "params": {
                "model__var_smoothing": [1e-9, 1e-8, 1e-6]
            }
        },

        "Logistic Regression": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clipper", FunctionTransformer(clip_extreme_values)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=100,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                ))
            ]),
            "params": {
                "model__C": [0.01, 0.1,  1],
                "model__penalty": ["l1", "l2"]
            }
        },

        "LDA": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clipper", FunctionTransformer(clip_extreme_values)),
                ("scaler", StandardScaler()),
                ("model", LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto"
                ))
            ]),
            "params": {}
        },

        "QDA": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clipper", FunctionTransformer(clip_extreme_values)),
                ("scaler", StandardScaler()),
                ("model", QuadraticDiscriminantAnalysis(
                    reg_param=0.1
                ))
            ]),
            "params": {}
        },



        "Decision Tree": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced"
                ))
            ]),
            "params": {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [ 5, 10,  15, None],
                "model__min_samples_split": [ 0, 25],
                "model__min_samples_leaf": [5, 10],
                "model__max_features": [None, "sqrt"]
            }
        },

        "SVM": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clipper", FunctionTransformer(clip_extreme_values)),
                ("scaler", StandardScaler()),
                ("model", SVC(
                    class_weight="balanced",
                    probability=False,
                    random_state=RANDOM_STATE
                ))
            ]),
            "params": [
                {
                    "model__kernel": ["linear"],
                    "model__C": [0.1, 1]
                },
                {
                    "model__kernel": ["rbf"],
                    "model__C": [1],
                    "model__gamma": ["scale"]
                }
            ]
        },

        "Random Forest": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1
                ))
            ]),
            "params": {
                "model__n_estimators": [100],
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [10, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": ["sqrt", "log2"]
            }
        }
    }
