import os

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shelve
import torch
import numpy as np

parser = argparse.ArgumentParser(description="Plot results")
#parser.add_argument("--test_results_shelve", type=str, required=True, nargs='+',
#                    help="File that results were saved to")
parser.add_argument("--test_results", type=str, required=True, nargs='+',
                    help="Directories that results were saved to")
parser.add_argument("--labels", type=str, required=False, nargs='+',
                    help="Labels for each curve. Must be the same length as --test_results")
parser.add_argument("--title", type=str, required=False,
                    help="Title for the plot")
parser.add_argument("--output", type=str, required=True,
                    help="Output file")
parser.add_argument("--scatter", action="store_true",
                    help="Plot scatter plot")

def get_data_from_shelve(shelve_file):
    scatter_data = []
    plot_data = []
    with shelve.open(shelve_file, flag='r') as db:
        for k,v in db.items():
            x = int(k.split('/')[-1].split('.')[1])
            for y in v:
                scatter_data.append((x,y))
            plot_data.append((x,np.mean(v)))
    plot_data = np.array(sorted(plot_data, key=lambda x: x[0]))
    scatter_data = np.array(scatter_data)

    return {
        'scatter': [scatter_data[:,0], scatter_data[:,1]],
        'plot': [plot_data[:,0], plot_data[:,1]]
    }

def get_data_from_pickles(directory):
    scatter_data = []
    plot_data = []
    for f in os.listdir(directory):
        if f.endswith('.pt'):
            x = int(f.split('.')[0])
            try:
                with open(os.path.join(directory, f), 'rb') as f:
                    vals = torch.load(f)
            except:
                print(f"Could not load {f}")
                continue
            for y in vals:
                scatter_data.append((x,y))
            plot_data.append((x,np.mean(vals)))
    plot_data = np.array(sorted(plot_data, key=lambda x: x[0]))
    scatter_data = np.array(scatter_data)

    return {
        'scatter': [scatter_data[:,0], scatter_data[:,1]],
        'plot': [plot_data[:,0], plot_data[:,1]]
    }

def ema_smooth(data, alpha=0.9):
    smoothed = []
    for i in range(len(data)):
        if i == 0:
            smoothed.append(data[i])
        else:
            if np.isfinite(smoothed[-1]):
                smoothed.append(alpha * smoothed[-1] + (1 - alpha) * data[i])
            else:
                smoothed.append(data[i])
    return smoothed

def main(flags):
    #for shelve_file in flags.test_results_shelve:
    #    data = get_data(shelve_file)
    #    label = shelve_file.split('/')[-1].split('.')[0]
    #    plt.scatter(*data['scatter'], alpha=0.1)
    #    plt.plot(*data['plot'], label=label)
    for i,directory in enumerate(flags.test_results):
        data = get_data_from_pickles(directory)
        if flags.labels is not None:
            label = flags.labels[i]
        else:
            label = directory.split('/')[-1]
        if flags.scatter:
            plt.scatter(*data['scatter'], alpha=0.1)
        plt.plot(data['plot'][0], ema_smooth(data['plot'][1]), label=label)

    plt.xlabel('Steps')
    plt.ylabel('Return')
    if flags.title is not None:
        plt.title(flags.title)
    plt.legend()
    plt.grid()

    plt.savefig(flags.output)
    print(f"Saved to {os.path.abspath(flags.output)}")

if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
