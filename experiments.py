from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from Dataset import PolishBankruptcyDataset
from model_configs import get_model_configs
from evaluations import evaluate_model


class BankruptcyExperiment:

    def __init__(self, year, scoring="f1"):
        self.year = year
        self.scoring = scoring

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.results = None
        self.grid_results = None

        self.best_models = {}
        self.best_params = {}

        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def load_data(self):
        dataset = PolishBankruptcyDataset(self.year)
        self.X, self.y = dataset.get_data()
        self.y = self.y.squeeze()
        return self

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )
        return self

    def train_models(self):
        configs = get_model_configs()

        final_results = []
        all_grid_results = []

        cv = StratifiedKFold(
            n_splits=2,
            shuffle=True,
            random_state=42
        )

        for model_name, config in configs.items():
            print(f"Training {model_name} using {self.scoring} scoring...")

            grid = GridSearchCV(
                estimator=config["pipeline"],
                param_grid=config["params"],
                cv=cv,
                scoring=self.scoring,
                n_jobs=-1,
                return_train_score=True
            )

            grid.fit(self.X_train, self.y_train)

            best_model = grid.best_estimator_

            self.best_models[model_name] = best_model
            self.best_params[model_name] = grid.best_params_

            result = evaluate_model(
                model_name=model_name,
                year=self.year,
                estimator=best_model,
                X_test=self.X_test,
                y_test=self.y_test
            )

            result["best_params"] = grid.best_params_
            result[f"best_cv_{self.scoring}"] = grid.best_score_
            result["scoring_used"] = self.scoring

            final_results.append(result)
            print(f"Finished {model_name}", flush=True)
            print(result, flush=True)

            grid_df = pd.DataFrame(grid.cv_results_)
            grid_df["model"] = model_name
            grid_df["year"] = self.year
            grid_df["scoring_used"] = self.scoring

            all_grid_results.append(grid_df)

        self.results = pd.DataFrame(final_results)
        self.grid_results = pd.concat(all_grid_results, ignore_index=True)

        return self

    def run(self):
        self.load_data()
        self.split()
        self.train_models()
        return self

    def show_results(self, sort_by=None):
        if sort_by is None:
            sort_by = "f1"

        print(self.results.sort_values(sort_by, ascending=False))
        return self

    def show_grid_results(self):
        columns = [
            "year",
            "model",
            "scoring_used",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "params"
        ]

        print(
            self.grid_results[columns]
            .sort_values(["model", "rank_test_score"])
        )

        return self

    def save_results(self):
        safe_scoring_name = self.scoring.lower().replace(" ", "_")

        self.results.to_csv(
            self.results_dir / f"final_model_results_year_{self.year}_{safe_scoring_name}.csv",
            index=False
        )

        self.grid_results.to_csv(
            self.results_dir / f"grid_search_results_year_{self.year}_{safe_scoring_name}.csv",
            index=False
        )

        return self