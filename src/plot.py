import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
file_path = './models/sdcn-flickr-fix/results.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Filtering out the "Starting_point" control_net from the data for the lines
lines_data = data[data['control_net'] != 'Starting_point']

# Filtering out mediapipeface
# ~ is for not contains
lines_data = lines_data[~data['run_name'].str.contains('face')]


l = 250
b = 500

# Extracting the 'map' value for 'Starting_point' control_net at data_size=250
real_low = data[(data['control_net'] == 'Starting_point') & (data['data_size'] == l)]['map'].iloc[0]
real_best = data[(data['control_net'] == 'Starting_point') & (data['data_size'] == b)]['map'].iloc[0]


# Creating the plot in a square format
fig, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(data=lines_data, x="data_size", y="map", hue="control_net", marker="o", ci=None)


# Adding a horizontal line for 'Starting_point' map value at data_size=250
line_low = ax.axhline(y=real_low, color='grey', linestyle='--')
ax.text(l, real_low, f'  MAP: {real_low:.2f}', verticalalignment='bottom', horizontalalignment='right')

# Adding a horizontal line for 'Starting_point' map value at data_size=625
line_best = ax.axhline(y=real_best, color='grey', linestyle='--')
ax.text(b, real_best, f'  MAP: {real_best:.2f}', verticalalignment='bottom', horizontalalignment='right')


# Adding title and labels
ax.set_title('MAP vs Data Size (Excluding "Starting_point" Line)')
ax.set_xlabel('Data Size')
ax.set_ylabel('MAP')

# Updating the legend to include the horizontal lines
handles, labels = ax.get_legend_handles_labels()
handles.extend([line_low, line_best])
labels.extend([f'real{l}', f'real{b}'])
ax.legend(handles=handles, labels=labels, title='Control Net', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()