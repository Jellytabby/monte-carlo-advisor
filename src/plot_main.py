import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from advisors.mc_advisor import MonteCarloAdvisor

logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Plotter:
    def __init__(self, name: str, args, advisor: MonteCarloAdvisor, start_time) -> None:
        self.name: str = name
        self.args = args
        self.plot_dir: str = args.plot_directory
        self.path_dir = f"{self.plot_dir}/{name}"
        self.advisor: MonteCarloAdvisor = advisor
        self.start_time = start_time

        os.makedirs(self.path_dir + "/benchmark_histograms", exist_ok=True)
        self.pdf = PdfPages(
            f"{self.path_dir}/benchmark_histograms/{self.name}_{self.start_time}.pdf"
        )
        self.all_runtimes = []

    def plot_speedup(self):
        speedup = self.advisor.max_speedup_after_n_iterations
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
        plt.title(f"Max Speedup over Iterations for {self.name}")
        plt.xticks(ticks)  # show one tick per iteration
        plt.tight_layout()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.path_dir}/{self.name}_{ts}.png"
        plt.savefig(fname)

    def log_results(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{self.path_dir}/{self.name}_{ts}.log"
        with open(path, "w+") as f:
            f.write(f"RESULTS FOR {self.name}\n")
            f.write(f"Arguments: {self.args}\n")
            f.write(f"Program start at: {self.start_time} and ended at {ts}\n")
            f.write(f"{'Iteration':<10} {'Max':<10} {'Score':<10} {'Run':<100}\n")
            for i, r in enumerate(self.advisor.all_runs):
                f.write(
                    f"{i:<10} {self.advisor.max_speedup_after_n_iterations[i]:<10.5f} {r[1]:<10.5f} {str(r[0]):<100}\n"
                )
            f.write(str(self.advisor) + "\n")
            f.write(f"Best run: {self.advisor.get_max_run()}\n")
            self.plot_all_runtimes()
            self.pdf.close()
            # f.write(f"Best state: {self.advisor.get_max_state()}\n")

    def runtime_histogram(self, runtimes: list[int] | None = None):
        if runtimes:
            self.all_runtimes += runtimes
        else:
            runtimes = self.all_runtimes
        fig, ax = plt.subplots()

        # 1) draw the histogram and grab counts/bins
        counts, bins, patches = ax.hist(runtimes, bins=len(runtimes))

        # 3) add total runs as a text box in the upper‐right corner
        total = len(runtimes)
        ax.text(
            0.98,
            0.98,
            f"Total runs: {total}\nDiff between min and max: {max(runtimes) - min(runtimes)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
        )

        ax.set_xlabel("Runtime")
        ax.set_ylabel("Runs")
        ax.set_xlim(min(runtimes), max(runtimes))
        ax.set_title("Runtime Distribution")

        self.pdf.savefig(fig)
        plt.close(fig)

    def plot_all_runtimes(self):
        self.runtime_histogram()
