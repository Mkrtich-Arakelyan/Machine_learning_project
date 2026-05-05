from experiments import BankruptcyExperiment


def main():
    years = [1, 2, 3, 4, 5]

    print("\nF1 SCORING")
    for year in years:
        print(f"\nYear {year}")

        exp = BankruptcyExperiment(year, scoring="f1")

        # Step 1: load data
        exp.load_data()

        # 🔥 ADD THIS HERE
        print("\nClass distribution:")
        print(exp.y.value_counts())
        print((exp.y.value_counts(normalize=True) * 100).round(2), "%")

        # Step 2: split + train
        exp.split()
        exp.train_models()

        print("\nBest params:")
        for model, params in exp.best_params.items():
            print(model, params)

        print("\nResults:")
        exp.show_results(sort_by="f1")

        exp.save_results()


if __name__ == "__main__":
    main()