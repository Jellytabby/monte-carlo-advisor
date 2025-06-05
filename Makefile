INPUT        ?= loop
DIR         := $(dir $(INPUT))
BASE        := $(notdir $(INPUT))

# Source filenames (auto‐derived from INPUT)
MAIN_SRC    := $(INPUT)_main.cpp
MODULE_SRC  := $(INPUT)_module.cpp
PROF_SRC    := tests/inline_profiler.cpp
OUT         ?= $(INPUT).out


MAIN_OBJ := $(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(MAIN_SRC)))
# PROF_OBJ := $(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(PROF_SRC)))

MAIN_OBJ    := $(MAIN_SRC:.cpp=.o)
MODULE_PRE_BC   := $(DIR)mod-pre-mc.bc
MODULE_POST_BC   := $(DIR)mod-post-mc.bc
MODULE_OBJ  := $(DIR)mod-post-mc.o
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

mod-pre-mc.bc: $(MODULE_PRE_BC)

# emit LLVM bitcode from your loop source
$(MODULE_PRE_BC): $(MODULE_SRC)
	clang++ -O3 -Xclang -disable-llvm-passes -emit-llvm \
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

$(DIR)baseline.out: $(MAIN_SRC) $(MODULE_SRC) $(PROF_SRC)
	clang++ -O3 $^ -o $@

run_baseline: $(DIR)baseline.out
	$(DIR)baseline.out

clean:
	rm -f $(DIR)*.o $(DIR)*.bc $(DIR)*.out $(DIR)*.in

run: all
	./$(OUT)

