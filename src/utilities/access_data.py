import pandas as pd

### data ###
data = pd.read_csv('data/external/potential-talents.csv').set_index('id')
data.drop_duplicates(inplace = True)
data.drop('fit', axis=1, inplace=True)
df_v1 = data.copy()
