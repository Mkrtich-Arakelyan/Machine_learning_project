from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

from Dataset import PolishBankruptcyDataset
from evaluations import get_probability_scores


class BankruptcyVisualizer:
    """
    Creates visualizations for one trained BankruptcyExperiment object.
    These plots use the live experiment object, not only saved CSVs.

    Used for:
    - metric bar charts per year
    - ROC curves per year
    - confusion matrix for best model
    - threshold analysis plots
    """

    def __init__(self, output_dir="figures", show_plots=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show_plots = show_plots

    def _year_dir(self, year):
        year_dir = self.output_dir / f"year_{year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        return year_dir

    def _safe_filename(self, name):
        return (
            name.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

    def _finish_plot(self, save_path):
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if self.show_plots:
            plt.show()

        plt.close()

    def plot_metric_bar(self, exp, metric="f1"):
        """
        Bar chart comparing models by one metric for one year.
        Example metrics: accuracy, precision, recall, f1, roc_auc.
        """

        if exp.results is None or exp.results.empty:
            print("No results available.")
            return

        if metric not in exp.results.columns:
            print(f"Metric '{metric}' not found in results.")
            return

        df = exp.results.copy()
        df = df.dropna(subset=[metric])
        df = df.sort_values(metric, ascending=False)

        plt.figure(figsize=(10, 6))
        plt.bar(df["model"], df[metric])

        plt.title(f"{metric.upper()} Comparison - Year {exp.year}")
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45, ha="right")

        for i, value in enumerate(df[metric]):
            plt.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

        save_path = self._year_dir(exp.year) / f"{metric}_comparison.png"
        self._finish_plot(save_path)

    def plot_confusion_matrix(self, exp, model_name=None):
        """
        Plots confusion matrix for a selected model.
        If model_name is None, uses the best model by F1-score.
        """

        if exp.results is None or exp.results.empty:
            print("No results available.")
            return

        if model_name is None:
            best_row = exp.results.sort_values("f1", ascending=False).iloc[0]
            model_name = best_row["model"]

        selected = exp.results[exp.results["model"] == model_name]

        if selected.empty:
            print(f"Model '{model_name}' not found in results.")
            return

        row = selected.iloc[0]

        cm = np.array([
            [row["tn"], row["fp"]],
            [row["fn"], row["tp"]]
        ])

        plt.figure(figsize=(6, 5))
        plt.imshow(cm)

        plt.title(f"Confusion Matrix - {model_name} (Year {exp.year})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.xticks([0, 1], ["Non-Bankrupt", "Bankrupt"])
        plt.yticks([0, 1], ["Non-Bankrupt", "Bankrupt"])

        for i in range(2):
            for j in range(2):
                plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=12)

        filename_model = self._safe_filename(model_name)
        save_path = self._year_dir(exp.year) / f"confusion_matrix_{filename_model}.png"
        self._finish_plot(save_path)

    def plot_roc_curves(self, exp):
        """
        Plots ROC curves for all models that provide either:
        - predict_proba
        - decision_function

        This uses the live fitted models, because ROC curves require prediction scores.
        """

        if not exp.best_models:
            print("No trained models available.")
            return

        plt.figure(figsize=(9, 7))

        plotted_any = False

        for model_name, model in exp.best_models.items():
            y_score = get_probability_scores(model, exp.X_test)

            if y_score is None:
                continue

            fpr, tpr, _ = roc_curve(exp.y_test, y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
            plotted_any = True

        if not plotted_any:
            print("No ROC-compatible models found.")
            plt.close()
            return

        plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

        plt.title(f"ROC Curves - Year {exp.year}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(fontsize=8)

        save_path = self._year_dir(exp.year) / "roc_curves.png"
        self._finish_plot(save_path)

    def plot_threshold_curve(self, exp, model_name, metric="f1"):
        """
        Plots one metric versus classification threshold for a selected model.
        Works only for models included in exp.threshold_results.
        """

        if exp.threshold_results is None or exp.threshold_results.empty:
            print("No threshold results available.")
            return

        if metric not in exp.threshold_results.columns:
            print(f"Metric '{metric}' not found in threshold results.")
            return

        df_model = (
            exp.threshold_results[exp.threshold_results["model"] == model_name]
            .sort_values("threshold")
        )

        if df_model.empty:
            print(f"No threshold results available for {model_name}.")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(df_model["threshold"], df_model[metric], marker="o")

        plt.title(f"{metric.upper()} vs Threshold - {model_name} (Year {exp.year})")
        plt.xlabel("Threshold")
        plt.ylabel(metric.upper())
        plt.xticks(df_model["threshold"])

        filename_model = self._safe_filename(model_name)
        save_path = self._year_dir(exp.year) / f"{filename_model}_{metric}_threshold.png"
        self._finish_plot(save_path)

    def plot_precision_recall_threshold(self, exp, model_name):
        """
        Plots precision, recall, and F1 against threshold for one model.
        Useful for explaining the precision-recall tradeoff.
        """

        if exp.threshold_results is None or exp.threshold_results.empty:
            print("No threshold results available.")
            return

        df_model = (
            exp.threshold_results[exp.threshold_results["model"] == model_name]
            .sort_values("threshold")
        )

        if df_model.empty:
            print(f"No threshold results available for {model_name}.")
            return

        plt.figure(figsize=(8, 5))

        plt.plot(df_model["threshold"], df_model["precision"], marker="o", label="Precision")
        plt.plot(df_model["threshold"], df_model["recall"], marker="o", label="Recall")
        plt.plot(df_model["threshold"], df_model["f1"], marker="o", label="F1")

        plt.title(f"Threshold Analysis - {model_name} (Year {exp.year})")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.xticks(df_model["threshold"])
        plt.legend()

        filename_model = self._safe_filename(model_name)
        save_path = self._year_dir(exp.year) / f"{filename_model}_threshold_analysis.png"
        self._finish_plot(save_path)


class AcrossYearVisualizer:
    """
    Creates plots comparing models across all five forecasting years.
    These plots use saved CSV files from the results folder.
    """

    def __init__(
        self,
        results_dir="results",
        output_dir="figures/across_years",
        scoring="f1",
        show_plots=False
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scoring = scoring
        self.show_plots = show_plots

    def _finish_plot(self, save_path):
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if self.show_plots:
            plt.show()

        plt.close()

    def load_all_results(self):
        """
        Loads final_model_results_year_1_f1.csv ... final_model_results_year_5_f1.csv
        and combines them into one DataFrame.
        """

        all_results = []

        for year in [1, 2, 3, 4, 5]:
            file_path = self.results_dir / f"final_model_results_year_{year}_{self.scoring}.csv"

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing file: {file_path}. Run main.py first to generate results."
                )

            df = pd.read_csv(file_path)
            all_results.append(df)

        return pd.concat(all_results, ignore_index=True)

    def plot_metric_across_years(self, metric):
        """
        Line plot of a metric across forecasting years for every model.
        Example: F1 across years, ROC-AUC across years.
        """

        df = self.load_all_results()

        if metric not in df.columns:
            print(f"Metric '{metric}' not found.")
            return

        df = df.dropna(subset=[metric])

        pivot_df = df.pivot(index="year", columns="model", values=metric)

        plt.figure(figsize=(11, 6))

        for model in pivot_df.columns:
            plt.plot(
                pivot_df.index,
                pivot_df[model],
                marker="o",
                label=model
            )

        plt.title(f"{metric.upper()} Across Forecasting Years")
        plt.xlabel("Forecasting Year")
        plt.ylabel(metric.upper())
        plt.xticks([1, 2, 3, 4, 5])
        plt.legend(fontsize=8)

        save_path = self.output_dir / f"{metric}_across_years.png"
        self._finish_plot(save_path)

    def plot_best_model_by_metric(self, metric="f1"):
        """
        Bar chart showing the best model for each year according to a selected metric.
        """

        df = self.load_all_results()

        if metric not in df.columns:
            print(f"Metric '{metric}' not found.")
            return

        df = df.dropna(subset=[metric])
        best_rows = df.loc[df.groupby("year")[metric].idxmax()]

        plt.figure(figsize=(9, 5))
        plt.bar(best_rows["year"].astype(str), best_rows[metric])

        for i, row in best_rows.iterrows():
            plt.text(
                str(row["year"]),
                row[metric],
                row["model"],
                ha="center",
                va="bottom",
                rotation=20,
                fontsize=8
            )

        plt.title(f"Best Model per Year by {metric.upper()}")
        plt.xlabel("Forecasting Year")
        plt.ylabel(metric.upper())

        save_path = self.output_dir / f"best_model_by_{metric}.png"
        self._finish_plot(save_path)

    def plot_metric_heatmap(self, metric="f1"):
        """
        Heatmap-style plot of model performance across years.
        Rows = models, columns = years.
        """

        df = self.load_all_results()

        if metric not in df.columns:
            print(f"Metric '{metric}' not found.")
            return

        df = df.dropna(subset=[metric])
        pivot_df = df.pivot(index="model", columns="year", values=metric)

        plt.figure(figsize=(10, 6))
        plt.imshow(pivot_df)

        plt.title(f"{metric.upper()} Heatmap: Models Across Years")
        plt.xlabel("Forecasting Year")
        plt.ylabel("Model")

        plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)

        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                value = pivot_df.iloc[i, j]
                plt.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)

        plt.colorbar(label=metric.upper())

        save_path = self.output_dir / f"{metric}_heatmap.png"
        self._finish_plot(save_path)

    def plot_class_distribution_across_years(self):
        """
        Plots actual bankrupt company percentage for all five years.
        Uses the original datasets, not the results CSV.
        """

        rows = []

        for year in [1, 2, 3, 4, 5]:
            dataset = PolishBankruptcyDataset(year)
            X, y = dataset.get_data()

            counts = y.value_counts()
            total = counts.sum()

            bankrupt_count = counts.get(1, 0)
            bankrupt_percent = bankrupt_count / total * 100

            rows.append({
                "year": year,
                "bankrupt_percent": bankrupt_percent
            })

        class_df = pd.DataFrame(rows)

        plt.figure(figsize=(8, 5))
        plt.plot(
            class_df["year"],
            class_df["bankrupt_percent"],
            marker="o"
        )

        plt.title("Bankrupt Class Percentage Across Years")
        plt.xlabel("Forecasting Year")
        plt.ylabel("Bankrupt Companies (%)")
        plt.xticks([1, 2, 3, 4, 5])

        for _, row in class_df.iterrows():
            plt.text(
                row["year"],
                row["bankrupt_percent"],
                f"{row['bankrupt_percent']:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8
            )

        save_path = self.output_dir / "bankrupt_percentage_across_years.png"
        self._finish_plot(save_path)

    def make_all_across_year_plots(self):
        """
        Generates all main across-year plots.
        """

        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            self.plot_metric_across_years(metric)

        self.plot_best_model_by_metric(metric="f1")
        self.plot_best_model_by_metric(metric="roc_auc")

        self.plot_metric_heatmap(metric="f1")
        self.plot_metric_heatmap(metric="recall")
        self.plot_metric_heatmap(metric="precision")
        self.plot_metric_heatmap(metric="roc_auc")

        self.plot_class_distribution_across_years()