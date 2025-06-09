CC 			?= clang++
INPUT       ?= loop
DIR         := $(dir $(INPUT))
BASE        := $(notdir $(INPUT))

ifeq ($(findstring ++,$(CC)),++)
  SRC_EXT := cpp
else
  SRC_EXT := c
endif

# Source filenames (auto‐derived from INPUT)
MAIN_SRC    := $(INPUT)_main.$(SRC_EXT)
MODULE_SRC  := $(INPUT)_module.$(SRC_EXT)
PROF_SRC    := inline_profiler.cpp
OUT         ?= $(INPUT).out

MAIN_OBJ    := $(MAIN_SRC:.cpp=.o)
MODULE_PRE_BC   := $(DIR)mod-pre-mc.bc
MODULE_POST_BC   := $(DIR)mod-post-mc.bc
MODULE_OBJ  := $(DIR)mod-post-mc.o
PROF_OBJ    := $(PROF_SRC:.cpp=.o)


# adapt to specific project
INC := -I /scr/sophia.herrmann/src/PolyBenchC-4.2.1/utilities/
EXTRA_OBJS := /scr/sophia.herrmann/src/PolyBenchC-4.2.1/utilities/polybench.o
EXTRA_FLAGS := -DPOLYBENCH_USE_C99_PROTO


# $@ -- the target name of the current rule
# $< -- the first prerequisite of the current rule
# $^ -- all prerequisites of the current rule

.PHONY: all clean run

all: $(OUT) $(MODULE_POST_BC)

# link final executable
$(OUT): $(MAIN_OBJ) $(PROF_OBJ) $(MODULE_OBJ) $(EXTRA_OBJS)
	$(CC)  $^ -o $@

# compile main
$(MAIN_OBJ): $(MAIN_SRC) 
	$(CC) -O3 $(INC) $(EXTRA_FLAGS) -c $< -o $@

# compile profiler
$(PROF_OBJ): $(PROF_SRC)
	clang++ -O3 -c $< -o $@

mod-pre-mc.bc: $(MODULE_PRE_BC)

# emit LLVM bitcode from your loop source
$(MODULE_PRE_BC): $(MODULE_SRC)
	clang++ -O3 $(INC) -Xclang -disable-llvm-passes -emit-llvm \
	         -c $< -o $@

readable: $(MODULE_SRC)
	clang++ -O3 -Xclang -disable-llvm-passes -emit-llvm \
	         -S $< -o readable.ll

$(MODULE_POST_BC): $(MODULE_PRE_BC)
	opt -O3 $< -o $@

# compile post-pass object from bitcode
# $(MODULE_OBJ): $(MODULE_POST_BC)
# 	clang++ -O3 --debug=inline -c $< -o $@

$(MODULE_OBJ): $(MODULE_POST_BC)
	llc -O3 -filetype=obj $< -o $@

$(DIR)baseline.out: $(MAIN_SRC) $(MODULE_SRC) $(PROF_SRC) $(EXTRA_OBJS)
	$(CC) -O3 $(INC) $(EXTRA_FLAGS) $^ -o $@

run_baseline: $(DIR)baseline.out
	$(DIR)baseline.out

clean:
	rm -f $(DIR)*.o $(DIR)*.bc $(DIR)*.out $(DIR)*.in

run: all
	./$(OUT)

