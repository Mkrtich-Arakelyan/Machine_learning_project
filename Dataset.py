from ucimlrepo import fetch_ucirepo
import pandas as pd
pd.set_option('display.max_columns', None)
# fetch dataset 
default_of_credit_card_clients = fetch_ucirepo(id=350)
'''
Description of variables
X1: Amount of the given credit (NT dollar): it includes both the individual
consumer credit and his/her family (supplementary) credit.
X2: Gender (1 = male; 2 = female).
X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
X4: Marital status (1 = married; 2 = single; 3 = others).
X5: Age (year).
X6 - X11: History of past payment. We tracked the past monthly payment records
(from April to September, 2005) as follows: X6 = the repayment status in September,
2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status
in April, 2005. The measurement scale for the repayment status is: -1 = pay duly;
1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment
delay for eight months; 9 = payment delay for nine months and above.
X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in
September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount
of bill statement in April, 2005. 
X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 
2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.
'''
# data (as pandas dataframes) 
df = default_of_credit_card_clients.data.features
y = default_of_credit_card_clients.data.targets

rename_dict = {
    'X1': 'LIMIT_BAL',
    'X2': 'GENDER',
    'X3': 'EDUCATION',
    'X4': 'MARRIAGE',
    'X5': 'AGE',

    'X6': 'PAY_0',
    'X7': 'PAY_2',
    'X8': 'PAY_3',
    'X9': 'PAY_4',
    'X10': 'PAY_5',
    'X11': 'PAY_6',

    'X12': 'BILL_AMT1',
    'X13': 'BILL_AMT2',
    'X14': 'BILL_AMT3',
    'X15': 'BILL_AMT4',
    'X16': 'BILL_AMT5',
    'X17': 'BILL_AMT6',

    'X18': 'PAY_AMT1',
    'X19': 'PAY_AMT2',
    'X20': 'PAY_AMT3',
    'X21': 'PAY_AMT4',
    'X22': 'PAY_AMT5',
    'X23': 'PAY_AMT6'
}
df = df.rename(columns=rename_dict)

# metadata 
print(default_of_credit_card_clients.metadata)

print(df.shape)
print(df.head())
print(df.columns)