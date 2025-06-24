import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt

from advisors.mc_advisor import MonteCarloAdvisor

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def plot_speedup(advisor: MonteCarloAdvisor, name: str, plot_dir: str):
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

    os.makedirs(f"{plot_dir}/{name}", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{plot_dir}/{name}/{name}_{ts}.png"
    plt.savefig(fname)


def log_results(advisor: MonteCarloAdvisor, args, start_time, name: str, plot_dir: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_dir = f"{plot_dir}/{name}/"
    os.makedirs(path_dir, exist_ok=True)
    path = f"{path_dir}/{name}_{ts}.log"
    with open(path, "w+") as f:
        f.write(f"RESULTS FOR {name}\n")
        f.write(f"Arguments: {args}\n")
        f.write(f"Program start at: {start_time} and ended at {ts}\n")
        f.write(f"{'Iteration':<10} {'Max':<10} {'Run':<100} {'Score':<10}\n")
        for i, r in enumerate(advisor.all_runs):
            f.write(
                f"{i:<10} {advisor.max_speedup_after_n_iterations[i]:<10.5f} {str(r[0]):<100} {r[1]:<10.5f}\n"
            )
        f.write(str(advisor) + "\n")
        f.write(f"Best run: {advisor.get_max_run()}\n")
        f.write(f"Best state: {advisor.get_max_state()}\n")
