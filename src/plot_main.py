import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt

from advisors.mc_advisor import MonteCarloAdvisor

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def plot_speedup(advisor: MonteCarloAdvisor, name: str):
    speedup = advisor.max_speedup_after_n_iterations
    assert speedup[0] == 1.0
    iterations = list(range(len(speedup)))
    step = max(1, len(iterations) // 10)
    ticks = iterations[::step]

    plt.figure()
    plt.step(iterations, speedup, where="post")
    plt.fill_between(iterations, speedup, 1.0, alpha=0.1, step="post")

    plt.xlabel("Number of Iterations")
    plt.ylabel("Max Speedup")
    plt.ylim(bottom=1.0)
    plt.title(f"Max Speedup over Iterations for {name}")
    plt.xticks(ticks)  # show one tick per iteration
    plt.tight_layout()

    os.makedirs(f"plots/export/{name}", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"plots/export/{name}/{name}_{ts}.png"
    plt.savefig(fname)
