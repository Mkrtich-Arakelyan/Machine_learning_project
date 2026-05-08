import pandas as pd

from experiments import BankruptcyExperiment
from visualization import BankruptcyVisualizer, AcrossYearVisualizer


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)


def main():
    years = [1, 2, 3, 4, 5]

    visualizer = BankruptcyVisualizer(
        output_dir="figures",
        show_plots=False
    )

    print("\nF1 SCORING")

    for year in years:
        print("\n" + "=" * 60)
        print(f"Year {year}")
        print("=" * 60)

        exp = BankruptcyExperiment(year, scoring="f1")

        # Step 1: Load data
        exp.load_data()

        # Show class distribution
        print("\nClass distribution:")
        print(exp.y.value_counts())

        print("\nClass distribution (%):")
        print((exp.y.value_counts(normalize=True) * 100).round(2), "%")

        # Step 2: Split and train
        exp.split()
        exp.train_models()

        # Show best hyperparameters
        print("\nBest params:")
        for model, params in exp.best_params.items():
            print(model, params)

        # Show final model results
        print("\nResults sorted by F1:")
        exp.show_results(sort_by="f1")

        # Save CSV results
        exp.save_results()

        # ---------------------------
        # Visualizations for this year
        # ---------------------------

        visualizer.plot_metric_bar(exp, metric="accuracy")
        visualizer.plot_metric_bar(exp, metric="precision")
        visualizer.plot_metric_bar(exp, metric="recall")
        visualizer.plot_metric_bar(exp, metric="f1")
        visualizer.plot_metric_bar(exp, metric="roc_auc")

        visualizer.plot_roc_curves(exp)

        # Confusion matrix for best model by F1
        visualizer.plot_confusion_matrix(exp)

        # Threshold analysis for probability-based models
        # Random Forest is especially useful because it usually has strong ROC-AUC.
        visualizer.plot_threshold_curve(exp, model_name="Random Forest", metric="f1")
        visualizer.plot_precision_recall_threshold(exp, model_name="Random Forest")

    # ---------------------------
    # Across-year visualizations
    # ---------------------------

    across_visualizer = AcrossYearVisualizer(
        results_dir="results",
        output_dir="figures/across_years",
        scoring="f1",
        show_plots=False
    )

    across_visualizer.make_all_across_year_plots()


if __name__ == "__main__":
    main()