#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// --- Internal state ---
static char        *mc_file   = NULL;
static uint64_t     mc_duration = 0;
static int          mc_timing   = 0;
static int          mc_valid    = 1;
static struct timespec mc_start;

// Called before main()
static void __attribute__((constructor)) mc_timer_init(void) {
    mc_file = getenv("MC_INLINE_PROFILING_FILE");
}

// Called after main() (or exit)
static void __attribute__((destructor)) mc_timer_fini(void) {
    FILE *out = stdout;
    if (mc_file) {
        FILE *f = fopen(mc_file, "w");
        if (f) out = f;
    }
    if (mc_valid) {
        fprintf(out, "MC_INLINE_TIMER %llu\n",
                (unsigned long long)mc_duration);
    } else {
        fprintf(out, "MC_INLINE_TIMER_INVALID\n");
    }
    if (out != stdout) fclose(out);
}

// Exposed hooks (no C++ name‐mangling)
void __mc_inline_begin(void) {
    if (mc_timing) mc_valid = 0;  // nested begin→invalid
    mc_timing = 1;
    clock_gettime(CLOCK_MONOTONIC, &mc_start);
}

void __mc_inline_end(void) {
    struct timespec end;
    if (!mc_timing) mc_valid = 0; // unmatched end→invalid
    mc_timing = 0;
    clock_gettime(CLOCK_MONOTONIC, &end);

    uint64_t this_ns =
        (uint64_t)(end.tv_sec  - mc_start.tv_sec)  * 1000000000ULL +
        (uint64_t)(end.tv_nsec - mc_start.tv_nsec);
    mc_duration += this_ns;
}

