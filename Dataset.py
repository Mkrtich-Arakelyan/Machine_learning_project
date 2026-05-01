from ucimlrepo import fetch_ucirepo
import pandas as pd

class PolishBankruptcyDataset:
    DEFAULT_RENAME_MAP ={
    "A1": "net_profit_to_total_assets",
    "A2": "total_liabilities_to_total_assets",
    "A3": "working_capital_to_total_assets",
    "A4": "current_assets_to_short_term_liabilities",
    "A5": "cash_securities_receivables_minus_st_liabilities_to_operating_expenses_minus_depreciation_days",
    "A6": "retained_earnings_to_total_assets",
    "A7": "ebit_to_total_assets",
    "A8": "book_value_equity_to_total_liabilities",
    "A9": "sales_to_total_assets",
    "A10": "equity_to_total_assets",
    "A11": "gross_profit_extra_items_financial_expenses_to_total_assets",
    "A12": "gross_profit_to_short_term_liabilities",
    "A13": "gross_profit_plus_depreciation_to_sales",
    "A14": "gross_profit_plus_interest_to_total_assets",
    "A15": "total_liabilities_to_gross_profit_plus_depreciation_days",
    "A16": "gross_profit_plus_depreciation_to_total_liabilities",
    "A17": "total_assets_to_total_liabilities",
    "A18": "gross_profit_to_total_assets",
    "A19": "gross_profit_to_sales",
    "A20": "inventory_to_sales_days",
    "A21": "sales_growth_ratio",
    "A22": "operating_profit_to_total_assets",
    "A23": "net_profit_to_sales",
    "A24": "three_year_gross_profit_to_total_assets",
    "A25": "equity_minus_share_capital_to_total_assets",
    "A26": "net_profit_plus_depreciation_to_total_liabilities",
    "A27": "operating_profit_to_financial_expenses",
    "A28": "working_capital_to_fixed_assets",
    "A29": "log_total_assets",
    "A30": "total_liabilities_minus_cash_to_sales",
    "A31": "gross_profit_plus_interest_to_sales",
    "A32": "current_liabilities_to_cost_of_goods_sold_days",
    "A33": "operating_expenses_to_short_term_liabilities",
    "A34": "operating_expenses_to_total_liabilities",
    "A35": "profit_on_sales_to_total_assets",
    "A36": "total_sales_to_total_assets",
    "A37": "current_assets_minus_inventories_to_long_term_liabilities",
    "A38": "constant_capital_to_total_assets",
    "A39": "profit_on_sales_to_sales",
    "A40": "quick_assets_minus_receivables_to_short_term_liabilities",
    "A41": "total_liabilities_to_monthly_operating_profit_plus_depreciation",
    "A42": "operating_profit_to_sales",
    "A43": "receivables_and_inventory_turnover_days",
    "A44": "receivables_to_sales_days",
    "A45": "net_profit_to_inventory",
    "A46": "current_assets_minus_inventory_to_short_term_liabilities",
    "A47": "inventory_to_cost_of_goods_sold_days",
    "A48": "ebitda_to_total_assets",
    "A49": "ebitda_to_sales",
    "A50": "current_assets_to_total_liabilities",
    "A51": "short_term_liabilities_to_total_assets",
    "A52": "short_term_liabilities_to_cost_of_goods_sold_days",
    "A53": "equity_to_fixed_assets",
    "A54": "constant_capital_to_fixed_assets",
    "A55": "working_capital",
    "A56": "gross_margin",
    "A57": "liquid_working_capital_to_adjusted_sales_cost",
    "A58": "total_costs_to_total_sales",
    "A59": "long_term_liabilities_to_equity",
    "A60": "sales_to_inventory",
    "A61": "sales_to_receivables",
    "A62": "short_term_liabilities_to_sales_days",
    "A63": "sales_to_short_term_liabilities",
    "A64": "sales_to_fixed_assets",
}

    def __init__(self, year:int):
        valid_years = [1, 2, 3, 4, 5]
        if year not in valid_years:
            raise ValueError("year must be an integer from 1 to 5")

        self.year = year
        self.X = None
        self.y = None

    def load_data(self):
        polish_companies_bankruptcy = fetch_ucirepo(id=365)

        # data (as pandas dataframes)
        self.X = polish_companies_bankruptcy.data.features
        self.y = polish_companies_bankruptcy.data.targets
        return self

    def rename_columns(self, rename_map=None):
        if rename_map is None:
            rename_map = self.DEFAULT_RENAME_MAP

        self.X = self.X.rename(columns=rename_map)
        return self

    def _get_year_dataset(self):
        # Ensure y is a Series

        self.y = self.y.squeeze()

        # Filter rows for the given year
        mask = self.X['year'] == self.year
        X_year = self.X.loc[mask].copy()
        y_year = self.y.loc[mask].copy()

        # Drop 'year' column (VERY IMPORTANT)
        X_year = X_year.drop(columns=['year'])

        return X_year, y_year

    def get_data(self):
        self.load_data()
        self.rename_columns()
        return self._get_year_dataset(self.X, self.y, self.year)



