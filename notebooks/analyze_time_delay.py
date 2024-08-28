import pandas as pd

import matplotlib.pyplot as plt


# Dowling lab figure settings
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=3)


# Plot specified value function value versus time delay value
def plot_time_delay_stats(df, key, filename, log=False):
    colors = [plt.cm.Dark2(i) for i in range(7)]
    linestyles = ['-', '--', ':']
    runs = [1, 2, 3]
    periods = [0.5, 2, 5, 0]
    labels = {0.5: 'Sine, 30s period, run {}', 
              2: 'Sine, 2min period, run {}', 
              5: 'Sine, 5min period, run {}', 
              0: 'Step, run {}'}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for run in runs:
        for ind, period in enumerate(periods):
            # Choose the color
            color = colors[ind]
            
            # Choose linestyle
            linestyle = linestyles[run - 1]
            
            # Assign label from mapping
            label = labels[period].format(run)
            
            # Grab data
            temp_df = df.loc[(df['run'] == run) & (df['period'] == period)]
            
            plt.plot(temp_df['delay'], temp_df[key], color=color, ls=linestyle, label=label)

    plt.title(key, fontsize=BIGGER_SIZE)
    plt.legend(bbox_to_anchor=(1.0, 1.25))
    if log:
        ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(filename)

# Read in the data
df = pd.read_csv('Surveying_time_delay_0_through_10_u_to_0.csv')
df['obj'] = df['obj']

# Plot objective value
plot_time_delay_stats(df, 'obj', 'time_delay_obj_0.png', log=True)

# Plot E-opt
plot_time_delay_stats(df, 'E', 'time_delay_E_0.png')

# Plot D-opt
plot_time_delay_stats(df, 'D', 'time_delay_D_0.png')

# Plot ME-opt
plot_time_delay_stats(df, 'ME', 'time_delay_ME_0.png')
