import argparse

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(
    prog="roll",
    description=(
        "Simulates dice rolling to estimate the expected number of rolls "
        "needed to encounter a sequence of interest."
    )
)
parser.add_argument(
    "sequence",
    metavar="S",
    type=int,
    nargs="+",
    help="list of integers representing sides of a die",
)
parser.add_argument(
    "-n",
    "--num_sides",
    type=int,
    nargs="?",
    default=6,
    help="the number of sides (faces) on the die.",
)
parser.add_argument(
    "-d",
    "--distribution",
    type=float,
    nargs="+",
    default=None,
    help="probability distribution of the die (must sum to 1)",
)
parser.add_argument(
    "-r",
    "--repeat",
    type=int,
    nargs=1,
    default=10_000,
    help="number of times to repeat the experiment",
)


rng = np.random.RandomState(0)


def simulate_rolls(target_seq, num_sides, repeat, dist=None):
    target_len = len(target_seq)
    sides = np.arange(1, num_sides + 1)

    def roll():
        n_iters = 0
        cur_target = 0
        while cur_target < target_len:
            n_iters += 1
            roll = rng.choice(sides, p=dist)
            if roll == target_seq[cur_target]:
                cur_target += 1
            else:
                cur_target = 0
        return n_iters

    return np.fromiter((roll() for _ in range(repeat)), int)


def print_stats(rolls):
    ps = [0, 50, 99, 100]
    vals = np.percentile(rolls, ps)
    n_digits = 3
    p_str = ", ".join(f"p{p}: {round(val, n_digits)}" for p, val in zip(ps, vals))
    uniq_rolls, counts = np.unique(rolls, return_counts=True)
    mode = uniq_rolls[np.argmax(counts)]
    print(f"avg: {round(np.mean(rolls), n_digits)}, {p_str}, mode: {mode}")


def plot_prob(rolls):
    hist, bin_edges = np.histogram(rolls, bins=np.max(rolls), density=True)
    plt.hist(bin_edges[:-1], bin_edges, weights=hist, cumulative=True)
    x_lbl = "Number of Rolls"
    y_lbl = "Probability of Success"
    plt.title(f"{y_lbl} vs. {x_lbl}")
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    sequence = args.sequence  # what is the sequence of interest?
    num_sides = args.num_sides  # how many sides on this die?
    dist = args.distribution  # probability distribution (None = uniform)
    repeat = args.repeat  # how many times to repeat the experiment?
    print(f"sequence: {sequence}, num_sides: {num_sides}, dist: {dist}, repeat: {repeat}")

    rolls = simulate_rolls(sequence, num_sides=num_sides, repeat=repeat, dist=dist)
    print_stats(rolls)
    plot_prob(rolls)
