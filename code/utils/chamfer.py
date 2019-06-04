"""
we want to compare this one to (1) Lars' scipy code and (2) our brute-force implementation
"""

import torch
from scipy.spatial import KDTree
import numpy as np
import MYTH
import time
from itertools import combinations
from utils.timer import Timer
from utils.octree import generate_octree, chamfer as octree_chamfer
from utils.visualization import plot

def scipy_kdtree_distance_cpu(points_from, points_to):
    kdtree = KDTree(points_to.transpose())
    dist, _ = kdtree.query(points_from.transpose(), k=1)
    return dist

def bruteforce_distance_gpu(points_from, points_to):
    N = points_from.shape[1]
    M = points_to.shape[1]
    dists = np.zeros((N,))
    if N == 0:
        return dists
    if M == 0:
        dists.fill(np.inf)
        return dists

    points_from = torch.Tensor(points_from).cuda()
    points_to = torch.Tensor(points_to).cuda()
    dists = torch.Tensor(dists).cuda()

    MYTH.bruteforce_distance_gpu(points_from, points_to, dists, N, M)

    torch.cuda.synchronize()

    dists = np.sqrt(dists.cpu().numpy())

    return dists

def octree_distance_gpu(points_from, points_to):
    points_from = torch.Tensor(points_from).cuda()
    points_to = torch.Tensor(points_to).cuda()
    tree = generate_octree(points_to)
    dists = points_from.new_zeros(points_from.shape)
    dists = octree_chamfer(points_from, tree, own_tree=False)
    return dists.cpu().numpy()

if __name__ == "__main__":
    functions = {
        # 'scipy': scipy_kdtree_distance_cpu,
        # 'brute': bruteforce_distance_gpu,
        'octree': octree_distance_gpu,
    }
    limits = {
        'scipy': 10000,
        'brute': np.inf,
        'octree': np.inf,
    }

    # bases = np.array([10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000,2000000,5000000])
    bases = np.array([100000, ])
    timings = {
        'scipy': np.zeros(len(bases)),
        'brute': np.zeros(len(bases)),
        'octree': np.zeros(len(bases)),
    }

    for base_idx, nr_points_base in enumerate(bases):
        print("Base number of points: %d" % nr_points_base)
        points_1 = np.random.rand(3, nr_points_base)
        points_2 = np.random.rand(3, nr_points_base)

        results = {}

        for algorithm in functions:
            if limits[algorithm] >= nr_points_base:
                print("  %s" % algorithm)
                # warm up
                functions[algorithm](points_2, points_1)
                # actual measurements
                start = Timer.current_time_millis()
                dists_12 = functions[algorithm](points_1, points_2)
                timings[algorithm][base_idx] = max(1, Timer.current_time_millis() - start)
                # cooldown
                functions[algorithm](points_1, points_2)
                # store the results
                results[algorithm] = dists_12
            else:
                timings[algorithm][base_idx] = np.nan
        
        if len(results.keys()) > 1:
            for first_algorithm, second_algorithm in combinations(results.keys(), 2):
                first_result = results[first_algorithm]
                second_result = results[second_algorithm]
                diff_12 = (np.abs(first_result - second_result) / first_result).max()
                if diff_12 > 1e-5:
                    print("[CHAMFER|ERROR] '%s' and '%s' do not agree" % (first_algorithm, second_algorithm))

    xlabels = []
    datas = []
    legend = []
    for algorithm in functions:
        datas.append(timings[algorithm])
        xlabels.append(bases)
        legend.append("%s: X -> X" % algorithm)

    if(len(bases) > 1):
        plot(
            datas,
            "Timing comparison for chamfer distance calculations",
            xlabels=xlabels,
            xaxis="Number of base points",
            yaxis="Processing time (ms)",
            legends=legend,
            logscale_x=True,
            logscale_y=True,
            plot_to_screen=False
        ).savefig("/tmp/sdonne/debug_images/00000.png")
