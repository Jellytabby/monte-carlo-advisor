import matplotlib.pyplot as plt
from datetime import datetime

from advisors.mc_advisor import MonteCarloAdvisor


def plot_speedup(advisor: MonteCarloAdvisor):
    print("HEREEE")
    max_state = advisor.get_max_leaf_state()

    mc_speedup = []
    curr = advisor.root
    for decision in max_state.decisions:
        curr = curr[decision]
        mc_speedup.append(curr.score)

    default_speedup = [1.0 for _ in advisor.default_path]
    x = list(range(1, len(default_speedup) + 1))

    plt.figure()
    plt.plot(x, default_speedup, color="blue", label="Default")
    plt.plot(x, mc_speedup, color="red", label="Monte Carlo")

    plt.xlabel("Decisions")
    plt.ylabel("Speedup")
    plt.xticks(x)  # show a tick at each decision index
    plt.title("Speedup per Decision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"/scr/sophia.herrmann/src/monte-carlo-advisor/plots/{type(advisor).__name__}{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    )
