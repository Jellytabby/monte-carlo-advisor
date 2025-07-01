# — Configurable inputs —
INPUT_DIR    ?= /scr/$(USER)/src/LULESH/recorded_inputs/_ZL16LagrangeLeapFrogR6Domain/
RECORDED_INP := $(INPUT_DIR)input-0.inp


# — Detect source extension & compiler —
# ifeq ($(wildcard $(INPUT)_main.cpp),)
#   SRC_EXT    := c
#   CC         := clang
# else
#   SRC_EXT    := cpp
#   CC         := clang++
# endif
#
# # — Source files —
# MAIN_SRC     := $(INPUT)_main.$(SRC_EXT)
MODULE_SRC   := $(INPUT_DIR)replay_module.bc
# PROF_SRC     := profiler/mc_profiler.$(SRC_EXT)
CC			 := clang++
OUT          ?= $(INPUT_DIR)replay.out
#
# # — Object & intermediate files —
# MAIN_OBJ     := $(MAIN_SRC:.$(SRC_EXT)=.o)
# PROF_OBJ     := $(PROF_SRC:.$(SRC_EXT)=.o)
MODULE_PRE_BC  := $(INPUT_DIR)mod-pre-mc.bc
MODULE_POST_BC := $(INPUT_DIR)mod-post-mc.bc
MODULE_OBJ    := $(INPUT_DIR)mod-post-mc.o

# — Project-specific flags (comment out entire line to disable) —
# INC          := -I/scr/sophia.herrmann/src/PolyBenchC-4.2.1/utilities/
# EXTRA_OBJS   := /scr/sophia.herrmann/src/PolyBenchC-4.2.1/utilities/polybench.o
EXTRA_FLAGS  := -linputgen.replay_recorded-x86_64 -fopenmp -fPIE

# — Compiler flags —
CFLAGS       := -O3 $(INC) $(EXTRA_FLAGS)

# $@ -- the target name of the current rule
# $< -- the first prerequisite of the current rule
# $^ -- all prerequisites of the current rule

# — Phony targets —
.PHONY: all clean run run_baseline readable

print: 
	@echo $(INPUT_DIR)

# — Default target —
all: $(OUT) $(MODULE_POST_BC)

# — Link final executable —
$(OUT): $(MODULE_OBJ)
	$(CC) $^ $(EXTRA_FLAGS) -o $@

# — Baseline build & run —
$(INPUT_DIR)baseline.out: $(MODULE_SRC)
	$(CC) $(CFLAGS) $^ -o $@

run_baseline: $(INPUT_DIR)baseline.out
	$(INPUT_DIR)baseline.out $(RECORDED_INP)

# — Compile main source —
$(MAIN_OBJ): $(MAIN_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

# — Compile profiler —
$(PROF_OBJ): $(PROF_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

# — Emit LLVM bitcode from loop source —
mod-pre-mc.bc: $(MODULE_PRE_BC)

# need to remove dead code or else we get compiler errors for missing functions
$(MODULE_PRE_BC): $(MODULE_SRC)
	opt -passes=globaldce $< -o $@

# — Dump readable LLVM IR —
readable: $(MODULE_SRC)
	clang++ $(CFLAGS) \
	  -Xclang -disable-llvm-passes -emit-llvm -S \
	  $< -o readable.ll

# — Run optimization passes on bitcode —
$(MODULE_POST_BC): $(MODULE_PRE_BC)
	opt -O3 $< -o $@

# — Compile optimized bitcode to object —
$(MODULE_OBJ): $(MODULE_POST_BC)
	llc -O3 -relocation-model=pic -filetype=obj $< -o $@

module_obj: $(MODULE_OBJ)


# — Clean up artifacts —
clean:
	rm -f $(INPUT_DIR)*.o $(INPUT_DIR)mod-pre-mc.bc $(INPUT_DIR)mod-post-mc.bc $(INPUT_DIR)*.out $(INPUT_DIR)*.in *.ll

# — Run the built executable —
run: all
	$(OUT) $(RECORDED_INP)

