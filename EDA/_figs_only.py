import sys, pandas as pd
sys.path.insert(0, '.')
import eda_femoral as E
df = pd.read_csv('per_image_stats.csv')
E.figures(df, '../Femoral Fractures', '.')
print('figs done')
