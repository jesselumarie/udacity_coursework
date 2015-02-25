import pandas as pd

df = pd.DataFrame(pd.read_pickle("final_project_dataset.pkl"))
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
financial_df = df.transpose()[['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']]




print financial_df.sort('total_payments', ascending=0)

