# import ?
import numpy as np
from matplotlib import pyplot as plt
from fn_utils import SoftMax


def show_samples(dataset, sensors, classes, title):
    """Show samples of dataset with label.
    
    Args:
        dataset (ImageFolder): ImageFolder dataset.
        title (str): Title of images.
    """
    print(title)
    fig, ax = plt.subplots(nrows=len(classes), ncols=len(sensors), figsize = (20, 14))
    
    for i in range(len(classes)):
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        for ind in indices:
            if dataset[ind][1] == i:
                break
        for j in range(len(sensors)):
            ax[i, j].plot(dataset[ind][0][:, j], c=['r', 'g', 'b', 'c', 'k', 'magenta'][j])
            ax[i, j].set_title(f'class: {classes[i]}, sensor: {sensors[j]}')
            ax[i, j].grid(True)
    
    plt.show()


# def plot_cm(targets, predictions, title, classes):
#     ?


# def plot_history(history):
#     ?
