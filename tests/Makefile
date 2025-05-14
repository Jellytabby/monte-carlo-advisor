inline_profiler.o: inline_profiler.cpp
	clang++ -O3 -o inline_profiler.o -c inline_profiler.cpp

main.o: main.cpp
	clang++ -O3 -o main.o -c main.cpp

mod-post-mc.o: mod-post-mc.bc
	clang++ -O0 -c mod-post-mc.bc -o mod-post-mc.o

a.out: main.o inline_profiler.o  mod-post-mc.o
	clang++ main.o inline_profiler.o mod-post-mc.o -o a.out

mod-pre-mc.bc:
	clang++ -O1 -Xclang -disable-llvm-passes -emit-llvm -c loop_module.cpp -o mod-pre-mc.bc


run: a.out
	./a.out


