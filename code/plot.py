import pandas as pd
import matplotlib.pyplot as plt

file_resault = r"../data/resault.csv"
df = pd.read_csv(file_resault, delimiter=',')
df['ΔT'] = df['max_temperature'] - df['min_temperature']

x = df['height_electrolyte'][20:25]
y1 = df['current'][20:25]
y2 = df['ΔT'][20:25]
print(y1)
fig, ax1 = plt.subplots()

# plot current as y
ax1.plot(x, y1, color='red', marker='o')
ax1.set_xlabel('height_electrolyte/(mm)')
ax1.set_ylabel('current/(A/m^2)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# plot ΔT as y
ax2 = ax1.twinx()
ax2.plot(x, y2, color='blue', marker='s')
ax2.set_ylabel('$\Delta$ T/(℃)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

ax1.legend(['Current'], loc='upper left')
ax2.legend(['$\Delta$ T'], loc='upper right')
# plt.title('Width Pitch vs Current and $\Delta$ T')

# plt.savefig(r'../data/picture/height_electrolyte.png', dpi=300, bbox_inches='tight')
plt.show()







