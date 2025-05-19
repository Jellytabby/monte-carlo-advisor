MAIN_SRC    := loop_main.cpp
MODULE_SRC  := loop_module.cpp
PROF_SRC    := inline_profiler.cpp
OUT         := a.out

MAIN_OBJ    := $(MAIN_SRC:.cpp=.o)
MODULE_PRE_BC   := mod-pre-mc.bc
MODULE_POST_BC   := mod-post-mc.bc
MODULE_OBJ  := mod-post-mc.o
PROF_OBJ    := $(PROF_SRC:.cpp=.o)

# $@ -- the target name of the current rule
# $< -- the first prerequisite of the current rule
# $^ -- all prerequisites of the current rule

.PHONY: all clean run

all: $(OUT) $(MODULE_POST_BC)

# link final executable
$(OUT): $(MAIN_OBJ) $(PROF_OBJ) $(MODULE_OBJ)
	clang++ $^ -o $@

# compile main
$(MAIN_OBJ): $(MAIN_SRC)
	clang++ -O3 -c $< -o $@

# compile profiler
$(PROF_OBJ): $(PROF_SRC)
	clang++ -O3 -c $< -o $@

# emit LLVM bitcode from your loop source
$(MODULE_PRE_BC): $(MODULE_SRC)
	clang++ -O3 -Xclang -disable-llvm-passes -emit-llvm \
	         -c $< -o $@

$(MODULE_POST_BC): $(MODULE_PRE_BC)
	opt -O3 $< -o $@

# compile post-pass object from bitcode
$(MODULE_OBJ): $(MODULE_POST_BC)
	clang++ -O3 -c $< -o $@

baseline.out: $(MAIN_SRC) $(MODULE_SRC) $(PROF_SRC)
	clang++ -O3 $^ -o $@

run_baseline: baseline.out
	./baseline.out

clean:
	rm -f *.o *.bc *.out

run: all
	./$(OUT)

