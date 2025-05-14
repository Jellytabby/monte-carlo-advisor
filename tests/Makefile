MAIN_SRC    := loop_main.cpp
MODULE_SRC  := loop_module.cpp
PROF_SRC    := inline_profiler.cpp
OUT         := a.out

MAIN_OBJ    := $(MAIN_SRC:.cpp=.o)
MODULE_BC   := mod-pre-mc.bc
MODULE_OBJ  := mod-post-mc.o
PROF_OBJ    := $(PROF_SRC:.cpp=.o)

# $@ -- the target name of the current rule
# $< -- the first prerequisite of the current rule
# $^ -- all prerequisites of the current rule

.PHONY: all clean run

all: $(OUT)

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
$(MODULE_BC): $(MODULE_SRC)
	clang++ -O1 -Xclang -disable-llvm-passes -emit-llvm \
	         -c $< -o $@

# compile post-pass object from bitcode
$(MODULE_OBJ): $(MODULE_BC)
	clang++ -O0 -c $< -o $@

clean:
	rm -f *.o *.bc $(OUT)

run: all
	./$(OUT)

