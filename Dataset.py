import pandas as pd
from scipy.io import arff


class PolishBankruptcyDataset:
    DEFAULT_RENAME_MAP = {
        "Attr1": "net_profit_to_total_assets",
        "Attr2": "total_liabilities_to_total_assets",
        "Attr3": "working_capital_to_total_assets",
        "Attr4": "current_assets_to_short_term_liabilities",
        "Attr5": "cash_securities_receivables_minus_st_liabilities_to_operating_expenses_minus_depreciation_days",
        "Attr6": "retained_earnings_to_total_assets",
        "Attr7": "ebit_to_total_assets",
        "Attr8": "book_value_equity_to_total_liabilities",
        "Attr9": "sales_to_total_assets",
        "Attr10": "equity_to_total_assets",
        "Attr11": "gross_profit_extra_items_financial_expenses_to_total_assets",
        "Attr12": "gross_profit_to_short_term_liabilities",
        "Attr13": "gross_profit_plus_depreciation_to_sales",
        "Attr14": "gross_profit_plus_interest_to_total_assets",
        "Attr15": "total_liabilities_to_gross_profit_plus_depreciation_days",
        "Attr16": "gross_profit_plus_depreciation_to_total_liabilities",
        "Attr17": "total_assets_to_total_liabilities",
        "Attr18": "gross_profit_to_total_assets",
        "Attr19": "gross_profit_to_sales",
        "Attr20": "inventory_to_sales_days",
        "Attr21": "sales_growth_ratio",
        "Attr22": "operating_profit_to_total_assets",
        "Attr23": "net_profit_to_sales",
        "Attr24": "three_year_gross_profit_to_total_assets",
        "Attr25": "equity_minus_share_capital_to_total_assets",
        "Attr26": "net_profit_plus_depreciation_to_total_liabilities",
        "Attr27": "operating_profit_to_financial_expenses",
        "Attr28": "working_capital_to_fixed_assets",
        "Attr29": "log_total_assets",
        "Attr30": "total_liabilities_minus_cash_to_sales",
        "Attr31": "gross_profit_plus_interest_to_sales",
        "Attr32": "current_liabilities_to_cost_of_goods_sold_days",
        "Attr33": "operating_expenses_to_short_term_liabilities",
        "Attr34": "operating_expenses_to_total_liabilities",
        "Attr35": "profit_on_sales_to_total_assets",
        "Attr36": "total_sales_to_total_assets",
        "Attr37": "current_assets_minus_inventories_to_long_term_liabilities",
        "Attr38": "constant_capital_to_total_assets",
        "Attr39": "profit_on_sales_to_sales",
        "Attr40": "quick_assets_minus_receivables_to_short_term_liabilities",
        "Attr41": "total_liabilities_to_monthly_operating_profit_plus_depreciation",
        "Attr42": "operating_profit_to_sales",
        "Attr43": "receivables_and_inventory_turnover_days",
        "Attr44": "receivables_to_sales_days",
        "Attr45": "net_profit_to_inventory",
        "Attr46": "current_assets_minus_inventory_to_short_term_liabilities",
        "Attr47": "inventory_to_cost_of_goods_sold_days",
        "Attr48": "ebitda_to_total_assets",
        "Attr49": "ebitda_to_sales",
        "Attr50": "current_assets_to_total_liabilities",
        "Attr51": "short_term_liabilities_to_total_assets",
        "Attr52": "short_term_liabilities_to_cost_of_goods_sold_days",
        "Attr53": "equity_to_fixed_assets",
        "Attr54": "constant_capital_to_fixed_assets",
        "Attr55": "working_capital",
        "Attr56": "gross_margin",
        "Attr57": "liquid_working_capital_to_adjusted_sales_cost",
        "Attr58": "total_costs_to_total_sales",
        "Attr59": "long_term_liabilities_to_equity",
        "Attr60": "sales_to_inventory",
        "Attr61": "sales_to_receivables",
        "Attr62": "short_term_liabilities_to_sales_days",
        "Attr63": "sales_to_short_term_liabilities",
        "Attr64": "sales_to_fixed_assets",
    }

    def __init__(self, year: int):
        if year not in [1, 2, 3, 4, 5]:
            raise ValueError("year must be an integer from 1 to 5")

        self.year = year
        self.X = None
        self.y = None

    def load_data(self):
        file_path = f"{self.year}year.arff"

        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)

        # Convert class from bytes to integer if needed
        if df["class"].dtype == object:
            df["class"] = df["class"].str.decode("utf-8").astype(int)
        else:
            df["class"] = df["class"].astype(int)

        self.X = df.drop(columns=["class"])
        self.y = df["class"]

        return self

    def rename_columns(self, rename_map=None):
        if rename_map is None:
            rename_map = self.DEFAULT_RENAME_MAP

        self.X = self.X.rename(columns=rename_map)
        return self

    def get_data(self):
        self.load_data()
        self.rename_columns()
        return self.X, self.y