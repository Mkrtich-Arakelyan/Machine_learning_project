import numpy as np
import pandas as pd

from Dataset import PolishBankruptcyDataset


def clipping_report(X, lower=-100, upper=100):
    values = X.to_numpy()

    total_values = values.size
    below = np.sum(values < lower)
    above = np.sum(values > upper)
    clipped = below + above

    print(f"Total values: {total_values}")
    print(f"Values below {lower}: {below}")
    print(f"Values above {upper}: {above}")
    print(f"Total clipped values: {clipped}")
    print(f"Percentage clipped: {clipped / total_values * 100:.4f}%")


def clipping_report_by_column(X, lower=-10000, upper=10000):
    report = pd.DataFrame({
        "below_lower": (X < lower).sum(),
        "above_upper": (X > upper).sum(),
        "total_clipped": ((X < lower) | (X > upper)).sum(),
        "percent_clipped": ((X < lower) | (X > upper)).mean() * 100
    })

    return report.sort_values("percent_clipped", ascending=False)


def main():
    for year in [1, 2, 3, 4, 5]:
        dataset = PolishBankruptcyDataset(year)
        X, y = dataset.get_data()

        print("\n" + "=" * 40)
        print(f"Year {year}")
        print("=" * 40)

        clipping_report(X, lower=-10000, upper=10000)

        print("\nTop clipped columns:")
        report = clipping_report_by_column(X, lower=-10000, upper=10000)
        print(report.head(15))


if __name__ == "__main__":
    main()