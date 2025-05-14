import tempfile
from sys import argv
import interactive_host


class State:
    def __init__(self):
        self.decisions = []

class InlineMonteCarloAdvisor(object):
    def __init__(self):
        # Takes an instance of a Board and optionally some keyword
        # arguments.  Initializes the list of game states and the
        # statistics tables.
        self.decisions:list[bool] = []

    def advice_false(self, _) -> bool:
        return False
    def advice_true(self, _) -> bool:
        return True

    def advice(self, _) -> bool:
        if len(self.decisions) == 0:
            last_decision = True
        else:
            last_decision = not self.decisions[-1]
        self.decisions.append(last_decision)
        # return last_decision
        return False


    def update(self, state):
        # Takes a game state, and appends it to the history.
        pass

    def get_play(self):
        # Causes the AI to calculate the best move from the
        # current game state and return it.
        pass

    def run_simulation(self):
        # Plays out a "random" game from the current position,
        # then updates the statistics tables with the result.
        pass


def optimize_module(mod, scoring_function):
    mc_advisor = InlineMonteCarloAdvisor()

    filename = "mod_temp"
    with tempfile.NamedTemporaryFile(suffix=".ll") as f1, \
        tempfile.NamedTemporaryFile(suffix=".bc") as f2:
        f1.write(mod)
        f1.flush()

        interactive_host.run_interactive(
            f"{filename}.channel-basename", 
            mc_advisor.advice_false,
            ['opt',
            '-passes=scc-oz-module-inliner',
            '-interactive-model-runner-echo-reply',
            '-enable-ml-inliner=release',
            f"-inliner-interactive-channel-base={filename}.channel-basename",
            '-o', f2.name,
            f1.name
            ])
        mod1 = f2.read()
        score1 = scoring_function(mod1)

    with tempfile.NamedTemporaryFile(suffix=".ll") as f1, \
        tempfile.NamedTemporaryFile(suffix=".bc") as f2:
        f1.write(mod)
        f1.flush()

        interactive_host.run_interactive(
            f"{filename}.channel-basename", 
            mc_advisor.advice_true,
            ['opt',
            '-passes=scc-oz-module-inliner',
            '-interactive-model-runner-echo-reply',
            '-enable-ml-inliner=release',
            f"-inliner-interactive-channel-base={filename}.channel-basename",
            '-o', f2.name,
            f1.name
            ])
        mod2 = f2.read()
        score2 = scoring_function(mod2)

    print(score1, score2)
    if score1 < score2:
        return mod1
    else:
        return mod2


if __name__ == "__main__":
    mc_advisor = InlineMonteCarloAdvisor()
    interactive_host.run_interactive(argv[1], mc_advisor.advice, argv[2:])
    print(mc_advisor.decisions)
