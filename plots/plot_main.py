import matplotlib.pyplot as plt
from datetime import datetime

from advisors.mc_advisor import MonteCarloAdvisor


def plot_speedup(advisor: MonteCarloAdvisor):
    speedup = advisor.max_speedup_after_n_iterations
    iterations = list(range(len(speedup)))
    step = max(1, len(iterations) // 10)
    ticks = iterations[::step]

    plt.figure()
    plt.plot(iterations, speedup)
    plt.fill_between(iterations, speedup, 1.0, alpha=0.3)

    plt.xlabel("Number of Iterations")
    plt.ylabel("Max Speedup")
    plt.ylim(bottom=1.0)
    plt.title(f"Max Speedup over Iterations ({type(advisor).__name__})")
    plt.xticks(ticks)  # show one tick per iteration
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = (
        f"/scr/sophia.herrmann/src/monte-carlo-advisor/"
        f"plots/{type(advisor).__name__}_{ts}.png"
    )
    plt.savefig(fname)
