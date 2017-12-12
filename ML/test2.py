import pandas as pd
data = pd.read_table('data_for_attribute', sep='\s+')
columns = list(data.columns)
print(columns)
for name, group in data.groupby(['a','b']):
    print(group)
    if
    print(name)