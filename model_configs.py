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

RANDOM_STATE = 42

def get_model_configs():

    return {
        "KNN": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier())
            ]),
            "params": {
                "model__n_neighbors": [3, 5, 7, 11,13, 15],
                "model__metric": ["euclidean", "manhattan"]
            }
        },
        "Naive Bayes": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GaussianNB())
            ]),
            "params": {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
            }
        },

        "Logistic Regression": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                ))
            ]),
            "params": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l1", "l2"]
            }
        },

        "LDA": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearDiscriminantAnalysis())
            ]),
            "params": [
                {
                    "model__solver": ["svd"]
                },
                {
                    "model__solver": ["lsqr"],
                    "model__shrinkage": [None, "auto", 0.1, 0.5, 0.9]
                },
                {
                    "model__solver": ["eigen"],
                    "model__shrinkage": [None, "auto", 0.1, 0.5, 0.9]
                }
            ]
        },

        "QDA": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", QuadraticDiscriminantAnalysis())
            ]),
            "params": {
                "model__reg_param": [0.0, 0.01, 0.05, 0.1, 0.5, 0.9]
            }
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
                "model__max_depth": [3, 5, 10, 15, None],
                "model__min_samples_split": [2, 10, 25],
                "model__min_samples_leaf": [1, 5, 10],
                "model__max_features": [None, "sqrt", "log2"]
            }
        },

        "SVM": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE
                ))
            ]),
            "params": [
                {
                    "model__kernel": ["linear"],
                    "model__C": [0.1, 1, 10]
                },
                {
                    "model__kernel": ["rbf"],
                    "model__C": [0.1, 1, 10],
                    "model__gamma": ["scale", "auto"]
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
                "model__n_estimators": [100, 200],
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_split": [2, 10],
                "model__min_samples_leaf": [1, 5],
                "model__max_features": ["sqrt", "log2"]
            }
        }
    }
