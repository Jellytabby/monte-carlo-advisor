import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from advisors.mc_advisor import MonteCarloAdvisor

logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Plotter:
    def __init__(
        self, name: str, plot_dir: str, advisor: MonteCarloAdvisor, start_time
    ) -> None:
        self.name: str = name
        self.plot_dir: str = plot_dir
        self.path_dir = f"{plot_dir}/{name}"
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

    def log_results(self, args):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{self.path_dir}/{self.name}_{ts}.log"
        with open(path, "w+") as f:
            f.write(f"RESULTS FOR {self.name}\n")
            f.write(f"Arguments: {args}\n")
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

    def runtime_histogram(self, runtimes: list[int]):
        self.all_runtimes += runtimes
        plt.figure()
        plt.ylabel("Runs")
        plt.xlabel("Runtime")
        plt.xlim(0.75e9, 1.2e9)
        plt.hist(runtimes, bins=20)
        self.pdf.savefig()
        plt.close()

    def plot_all_runtimes(self):
        self.runtime_histogram(self.all_runtimes)
