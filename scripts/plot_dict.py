import matplotlib.pyplot as plt
import numpy as np

def plot_dict(dict_arrays, fname, use_xlabel='Epochs', use_ylabel='Value', use_title=None):
    # Font size select custom or adjusted on `magnify` value.
    font_size = np.interp(0.1, [0.1,1], [10.5,50])

    # Font variables dictionary. Keep it in this format for future updates.
    font_dict = dict(
        family='DejaVu Sans',
        color='black',
        weight='normal',
        size=font_size,
        )

    # Single plot figure.
    plt.subplot(1, 2, 1)

    # Use maximum length of steps. In case each arrya has different lengths.
    max_steps = []

    # Plot each array.
    for index, (use_label, array) in enumerate(dict_arrays.items()):
        # Set steps plotted on x-axis - we can use step if 1 unit has different value.
        if 0 > 0:
            # Offset all steps by start_step.
            steps = np.array(range(0, len(array))) * 1 + 0
            max_steps = steps if len(max_steps) < len(steps) else max_steps
        else:
            steps = np.array(range(1, len(array) + 1)) * 1
            max_steps = steps if len(max_steps) < len(steps) else max_steps

        # Plot array as a single line.
        plt.plot(steps, array, linestyle=(['-'] * len(dict_arrays))[index], label=use_label)

        # Plots points values.
        if ([False] * len(dict_arrays))[index]:
            # Loop through each point and plot the label.
            for x, y in zip(steps, array):
                # Add text label to plot.
                plt.text(x, y, str(round(y, 3)), fontdict=font_dict)

    # Set horizontal axis name.
    plt.xlabel(use_xlabel, fontdict=font_dict)

    # Use x ticks with steps or labels.
    plt.xticks(max_steps, None, rotation=0)

    # Set vertical axis name.
    plt.ylabel(use_ylabel, fontdict=font_dict)

    # Adjust both axis labels font size at same time.
    plt.tick_params(labelsize=font_dict['size'])

    # Place legend best position.
    plt.legend(loc='best', fontsize=font_dict['size'])

    # Adjust font for title.
    font_dict['size'] *= 1.8

    # Set title of figure.
    plt.title(use_title, fontdict=font_dict)

    # Rescale `magnify` to be used on inches.
    magnify = 0.1
    magnify *= 15

    # Display grid depending on `use_grid`.
    plt.grid(True)

    # Make figure nice.
    plt.tight_layout()

    # Get figure object from plot.
    fig = plt.gcf()

    # Get size of figure.
    figsize = fig.get_size_inches()

    # Change size depending on height and width variables.
    figsize = [figsize[0] * 3 * magnify, figsize[1] * 1 * magnify]

    # Set the new figure size with magnify.
    fig.set_size_inches(figsize)

    plt.savefig(fname)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    return 