import os

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shelve
import numpy as np

parser = argparse.ArgumentParser(description="Plot results")
parser.add_argument("--test_results_shelve", type=str, required=True, nargs='+',
                    help="File that results were saved to")
parser.add_argument("--output", type=str, required=True,
                    help="Output file")

def get_data(shelve_file):
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

def main(flags):
    for shelve_file in flags.test_results_shelve:
        data = get_data(shelve_file)
        label = shelve_file.split('/')[-1].split('.')[0]
        plt.scatter(*data['scatter'], alpha=0.1)
        plt.plot(*data['plot'], label=label)

    plt.xlabel('Steps')
    plt.ylabel('Return')
    plt.legend()
    plt.grid()

    plt.savefig(flags.output)
    print(f"Saved to {os.path.abspath(flags.output)}")

if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
