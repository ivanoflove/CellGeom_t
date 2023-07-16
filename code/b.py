import pandas as pd
from itertools import product

# 定义参数值
width_pitch_values = [1.6, 1.8, 2, 2.2, 2.4]
width_rib_a_values = [1]
width_rib_c_values = [1]
height_anode_values = [0.41]
height_cathode_values = [0.025]
height_electrolyte_values = [0.01]

# 生成排列组合
combinations = list(product(width_pitch_values, width_rib_a_values, width_rib_c_values, height_anode_values, height_cathode_values, height_electrolyte_values))

# 创建 DataFrame
df = pd.DataFrame(combinations, columns=['width_pitch', 'width_rib_a', 'width_rib_c', 'height_anode', 'height_cathode', 'height_electrolyte'])

# 写入 CSV 文件
df.to_csv('output.csv', index=False)