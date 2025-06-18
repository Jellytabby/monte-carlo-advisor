# LULESH Modifications

## before running:
1. add `__attribute__((inputgen_entry))` to function of interest
2. `make clean && INPUTGEN_RECORD_DUMP_PATH=./recorded_inputs/ make`
3. `INPUTGEN_RECORD_DUMP_FIRST_N=1 ./lulesh2.0`

## to run Monte Carlo Advisor:
`python3 src/monte_carlo_main.py -r 20 -c 17 -lua -ia  ../LULESH/recorded_inputs/_ZL16LagrangeLeapFrogR6Domain/`
