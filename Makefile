# — Configurable inputs —
INPUT        ?= loop
DIR          := $(dir $(INPUT))

# — Detect source extension & compiler —
ifeq ($(wildcard $(INPUT)_main.cpp),)
  SRC_EXT    := c
  CC         := clang
else
  SRC_EXT    := cpp
  CC         := clang++
endif

# — Source files —
MAIN_SRC     := $(INPUT)_main.$(SRC_EXT)
MODULE_SRC   := $(INPUT)_module.$(SRC_EXT)
PROF_SRC     := profiler/mc_profiler.$(SRC_EXT)
OUT          ?= $(INPUT).out

# — Object & intermediate files —
MAIN_OBJ     := $(MAIN_SRC:.$(SRC_EXT)=.o)
PROF_OBJ     := $(PROF_SRC:.$(SRC_EXT)=.o)
MODULE_PRE_BC  := $(DIR)mod-pre-mc.bc
MODULE_POST_BC := $(DIR)mod-post-mc.bc
MODULE_OBJ    := $(DIR)mod-post-mc.o

# — Project-specific flags (comment out entire line to disable) —
INC          := -I/scr/sophia.herrmann/src/PolyBenchC-4.2.1/utilities/
EXTRA_OBJS   := /scr/sophia.herrmann/src/PolyBenchC-4.2.1/utilities/polybench.o
EXTRA_FLAGS  := -DPOLYBENCH_USE_C99_PROTO -lm

# — Compiler flags —
CFLAGS       := -O3 $(INC) $(EXTRA_FLAGS)

# $@ -- the target name of the current rule
# $< -- the first prerequisite of the current rule
# $^ -- all prerequisites of the current rule

# — Phony targets —
.PHONY: all clean run run_baseline readable

# — Default target —
all: $(OUT) $(MODULE_POST_BC)

# — Link final executable —
$(OUT): $(MAIN_OBJ) $(PROF_OBJ) $(MODULE_OBJ) $(EXTRA_OBJS)
	$(CC) $(EXTRA_FLAGS) $^ -o $@

# — Baseline build & run —
$(DIR)baseline.out: $(MAIN_SRC) $(MODULE_SRC) $(PROF_SRC) $(EXTRA_OBJS)
	$(CC) $(CFLAGS) $^ -o $@

run_baseline: $(DIR)baseline.out
	$(DIR)baseline.out

# — Compile main source —
$(MAIN_OBJ): $(MAIN_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

# — Compile profiler —
$(PROF_OBJ): $(PROF_SRC)
	$(CC) $(CFLAGS) -c $< -o $@

# — Emit LLVM bitcode from loop source —
mod-pre-mc.bc: $(MODULE_PRE_BC)

$(MODULE_PRE_BC): $(MODULE_SRC)
	$(CC) $(CFLAGS) \
	  -Xclang -disable-llvm-passes -emit-llvm \
	  -c $< -o $@

# — Dump readable LLVM IR —
readable: $(MODULE_SRC)
	clang++ $(CFLAGS) \
	  -Xclang -disable-llvm-passes -emit-llvm -S \
	  $< -o readable.ll

mod-post-mc.bc: $(MODULE_POST_BC)

# — Run optimization passes on bitcode —
$(MODULE_POST_BC): $(MODULE_PRE_BC)
	opt -O3 $< -o $@

# — Compile optimized bitcode to object —
$(MODULE_OBJ): $(MODULE_POST_BC)
	llc -O3 -filetype=obj $< -o $@

module_obj: $(MODULE_OBJ)


# — Clean up artifacts —
clean:
	rm -f $(DIR)*.o $(DIR)*.bc $(DIR)*.out $(DIR)*.in *.ll

# — Run the built executable —
run: all
	$(OUT)

