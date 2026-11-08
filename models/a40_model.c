/*
 * a40_accurate_emulator.c
 *
 * A40-inspired executable teaching model (not validated as cycle accurate)
 *
 * Build: gcc -O3 -std=c11 -march=native a40_accurate_emulator.c -o a40_sim -lm
 * Run: ./a40_sim [workload_type]
 *
 * ARCHITECTURAL FEATURES MODELED:
 * ================================
 * - 84 SMs (10,752 CUDA cores @ 128 cores/SM)
 * - 4 warp schedulers per SM with dual-issue capability
 * - 48 resident warps per SM (1536 threads/SM)
 * - 65,536 x 32-bit registers per SM (256 KB register file)
 * - 164 KB configurable L1/shared memory per SM
 * - 6 MB L2 cache with 16-way set associativity
 * - 12 memory channels, 384-bit GDDR6 interface
 * - 48 GB GDDR6 @ 696 GB/s bandwidth
 *
 * PIPELINE MODELING:
 * ==================
 * - 6-stage instruction pipeline per warp:
 *   I-Fetch -> I-Buffer -> Decode -> Issue -> Execute -> Writeback
 * - Scoreboard-based dependency tracking
 * - Separate execution units: FP32, FP64, INT32, SFU, LD/ST, Tensor
 * - SIMT divergence stack for branch handling
 * - Operand collector with 4-cycle collection time
 *
 * MEMORY HIERARCHY:
 * =================
 * - Per-SM L1 cache: 128 KB, 4-way set-associative, 128B lines
 * - Shared L2: 6 MB, 16-way set-associative, 128B lines
 * - Realistic DRAM timing: tRCD, tRP, tCAS, row buffer hits
 * - Memory coalescing unit with transaction combining
 * - Crossbar interconnect model (SM <-> L2 <-> Memory)
 *
 * ACCURACY NOTES:
 * ===============
 * No real A40 measurements or timing-correlation validation were supplied.
 * Results are deterministic predictions of the simplified model ONLY.
 * Sector counts and selected constants have public documentation;
 * scheduler, cache, execution and DRAM timing are not calibrated.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ============================================================================
// ARCHITECTURAL CONSTANTS (A40 / Ampere GA102)
// ============================================================================

#define SM_COUNT 84
#define WARP_SIZE 32
#define MAX_WARPS_PER_SM 48
#define MAX_THREADS_PER_SM (MAX_WARPS_PER_SM * WARP_SIZE)
#define WARP_SCHEDULERS_PER_SM 4
#define DISPATCH_UNITS_PER_SCHEDULER 2

// Execution units per SM (Ampere GA102 configuration)
#define FP32_UNITS_PER_SM 128
#define INT32_UNITS_PER_SM 128
#define FP64_UNITS_PER_SM 2
#define SFU_UNITS_PER_SM 32
#define TENSOR_CORES_PER_SM 4
#define LD_ST_UNITS_PER_SM 32

// Register file
#define REGS_PER_SM 65536
#define REG_BANKS 32
#define REG_BANK_WIDTH 4 // 4 registers per bank per cycle
// Shared memory / L1
#define SHMEM_SIZE_PER_SM (164 * 1024)
#define SHMEM_BANKS 32
#define L1_SIZE_PER_SM (128 * 1024)
#define L1_LINE_SIZE 128
#define L1_WAYS 4

// L2 Cache
#define L2_SIZE (6 * 1024 * 1024)
#define L2_LINE_SIZE 128
#define L2_WAYS 16
#define L2_BANKS 32
#define L2_LATENCY_BASE 193 // cycles (measured average)

// Memory system
#define MEMORY_CHANNELS 12
#define DRAM_BANKS_PER_CHANNEL 16
#define DRAM_ROW_SIZE 8192
#define DRAM_BURST_SIZE 128
#define MEMORY_TOTAL_GB 48

// GDDR6 timing (in GPU cycles @ ~1.56 GHz boost)
#define DRAM_tRCD 24        // RAS to CAS delay
#define DRAM_tRP 24         // Row precharge time
#define DRAM_tCAS 20        // CAS latency
#define DRAM_tRAS 52        // Row active time
#define DRAM_tRTP 8         // Read to precharge
#define DRAM_tWR 16         // Write recovery time
#define DRAM_tCCDL 8        // CAS to CAS (same bank group)
#define DRAM_tCCDS 4        // CAS to CAS (different bank group)
#define DRAM_tRRDL 8        // RAS to RAS (same bank group)
#define DRAM_tRRDS 4        // RAS to RAS (different bank group)
#define DRAM_BURST_CYCLES 4
#define DRAM_REFRESH_INTERVAL 4800 // ~7.8µs at 1.56 GHz

// Pipeline depths
#define PIPE_FETCH 1
#define PIPE_IBUFFER 4
#define PIPE_DECODE 2
#define PIPE_ISSUE 1
#define PIPE_OPERAND_COLLECT 4

// Execution latencies (in cycles)
#define LATENCY_INT32 4
#define LATENCY_FP32 4
#define LATENCY_FP64 8
#define LATENCY_SFU_FAST 8      // rsqrt, rcp
#define LATENCY_SFU_SLOW 16     // sin, cos, log
#define LATENCY_TENSOR 8        // Tensor core MMA
#define LATENCY_MEMORY_MIN 28   // L1 hit
#define LATENCY_BRANCH 1

// Memory coalescing
#define COALESCE_WINDOW_SIZE 32 // analyze 32 threads at once
#define MAX_TRANSACTIONS_PER_WARP 32

// Simulation limits
#define MAX_INSTRUCTIONS_PER_WARP 512
#define SIM_CYCLE_LIMIT 50000000
#define STAT_PRINT_INTERVAL 100000

// User-observable, bounded scenarios make the full model inexpensive to test.
static int g_sm_count = 1;
static int g_pattern = -1; // -1: vary pattern by SM/warp
static int g_workload = 0; // 0: mixed, 1: memory, 2: compute
static int g_max_instructions = 128;
static uint64_t g_cycle_limit = 1000000;
static bool g_quiet = false;


#define MAX_ISSUES_PER_SM (WARP_SCHEDULERS_PER_SM * DISPATCH_UNITS_PER_SCHEDULER)
#define MEM_BANDWIDTH_GBPS 696.0f // Effective bandwidth after overheads
// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;

// Instruction opcodes
typedef enum {
    OP_NOP,
    OP_IADD, OP_IMUL, OP_ISUB,
    OP_FADD, OP_FMUL, OP_FFMA,
    OP_DADD, OP_DMUL,
    OP_SFU_RCP, OP_SFU_SQRT, OP_SFU_SIN, OP_SFU_COS,
    OP_TENSOR_MMA,
    OP_LD_GLOBAL, OP_LD_SHARED, OP_LD_CONST,
    OP_ST_GLOBAL, OP_ST_SHARED,
    OP_BRA, OP_BRA_COND, OP_EXIT,
    OP_SSY, OP_SYNC // NVIDIA's stack synchronization ops
} Opcode;

// Branch predicate evaluation
typedef enum {
    PRED_TRUE,      // All threads take branch
    PRED_FALSE,     // No threads take branch
    PRED_UNIFORM,   // All active threads agree
    PRED_DIVERGE    // Threads disagree - divergence!
} PredicateResult;

// Instruction pipe classification for dual-issue
typedef enum {
    PIPE_COMPUTE,    // FP/INT/SFU/Tensor
    PIPE_MEMORY,     // LD/ST
    PIPE_CONTROL     // BRA/EXIT/SSY/SYNC
} InstructionPipe;

static inline InstructionPipe get_instruction_pipe(Opcode op) {
    switch (op) {
        case OP_LD_GLOBAL:
        case OP_LD_SHARED:
        case OP_LD_CONST:
        case OP_ST_GLOBAL:
        case OP_ST_SHARED:
            return PIPE_MEMORY;
        case OP_BRA:
        case OP_BRA_COND:
        case OP_EXIT:
        case OP_SSY:
        case OP_SYNC:
            return PIPE_CONTROL;
        default:
            return PIPE_COMPUTE;
    }
}

// Instruction representation
typedef struct {
    Opcode op;
    u8 dst_reg;      // destination register
    u8 src1_reg;     // source operands
    u8 src2_reg;
    u8 src3_reg;
    u8 pred_reg;     // predicate register for conditional branches
    u32 imm;         // immediate value / address offset / branch target
    u32 active_mask; // which threads in warp are active (SIMT)
    u64 thread_addrs[WARP_SIZE]; // Per-thread addresses for memory ops
    u32 thread_preds[WARP_SIZE]; // Per-thread predicates for branches
} Instruction;

// Divergence stack entry (NVIDIA-style)
typedef struct {
    u32 active_mask;       // Which threads are active in this path
    u32 pc;                // Program counter for this path
    bool is_convergence;   // True if this is a reconvergence point
} DivergenceEntry;

#define MAX_DIVERGENCE_DEPTH 32

// Full SIMT divergence stack per warp
typedef struct {
    DivergenceEntry stack[MAX_DIVERGENCE_DEPTH];
    int top;               // Stack pointer (-1 = empty)
    u32 reconverge_pc;     // Next reconvergence point
    bool in_divergence;    // Currently executing diverged path
} DivergenceStack;

// Memory coalescing analysis result
typedef struct {
    int num_transactions;      // Unique 32-byte sectors, up to 64 for crossing 4-byte accesses
    u64 transaction_addrs[WARP_SIZE * 2]; // Start and end sectors of 4-byte accesses
    int transaction_sizes[WARP_SIZE * 2]; // Size of each transaction in bytes
    bool fully_coalesced;
} CoalesceResult;

// Warp state
typedef enum {
    WARP_IDLE,
    WARP_FETCH,
    WARP_IBUFFER,
    WARP_DECODE,
    WARP_ISSUE_WAIT,
    WARP_OPERAND_WAIT,
    WARP_EXEC_WAIT,
    WARP_WRITEBACK,
    WARP_BARRIER_WAIT,
    WARP_COMPLETED
} WarpState;

// Scoreboard entry
typedef struct {
    bool valid;
    u64 ready_cycle;
    u8 reg_id;
} ScoreboardEntry;

// Warp control block
typedef struct {
    int warp_id;
    WarpState state;
    u32 pc;
    u32 active_mask; // 32-bit mask for active threads

    // Pipeline state
    Instruction fetch_instr;
    Instruction ibuffer[PIPE_IBUFFER];
    int ibuffer_head, ibuffer_tail;
    Instruction decode_instr;
    Instruction issue_instr;

    // Dual-issue tracking
    bool issued_compute_this_cycle;
    bool issued_memory_this_cycle;

    // Execution tracking
    u64 exec_ready_cycle;
    u64 next_fetch_cycle;

    // Register scoreboard (per warp tracking)
    ScoreboardEntry scoreboard[256];

    // Full SIMT divergence stack
    DivergenceStack div_stack;

    // Statistics
    u64 instructions_issued;
    u64 cycles_stalled;
    u64 memory_accesses;
    u64 coalesced_accesses;
    u64 uncoalesced_accesses;
    u64 dual_issue_cycles;
    u64 branches_taken;
    u64 branches_diverged;
    u64 divergence_cycles;
} Warp;

// Execution unit
typedef struct {
    int count;                      // number of this type of unit
    int latency;                    // execution latency
    bool pipelined;                 // can accept new op every cycle
    u64 *busy_until;                // array[count] of cycle when free
    int throughput_per_cycle;       // max ops per cycle
} ExecUnit;

// Memory transaction
typedef struct {
    u64 address;
    int size;
    bool is_write;
    int warp_id;
    u64 issue_cycle;
    u64 ready_cycle;
    bool completed;
} MemTransaction;

// L1 Cache (per SM)
typedef struct {
    int sets;
    int ways;
    int line_size;
    u64 **tags;          // [sets][ways]
    u8 **lru;            // [sets][ways]
    bool **valid;        // [sets][ways]
    u64 hits;
    u64 misses;
} L1Cache;

// L2 Cache (shared)
typedef struct {
    int sets;
    int ways;
    int line_size;
    int banks;
    u64 **tags;          // [sets][ways]
    u8 **lru;            // [sets][ways]
    bool **valid;        // [sets][ways]
    u64 *bank_busy;      // [banks] - cycle when bank free
    u64 hits;
    u64 misses;
} L2Cache;

// DRAM Request
typedef struct {
    u64 addr;
    u64 arrival_cycle;
    int sm_id;
    int warp_id;
    bool is_write;
    bool completed;
} DRAMRequest;

// DRAM Bank (with bank groups)
typedef struct {
    bool row_open;
    u64 open_row;
    u64 next_activate_cycle;
    u64 next_precharge_cycle;
    u64 next_column_cycle;
    int bank_group; // 0-3 for GDDR6
} DRAMBank;

// Memory Channel with request queue
#define DRAM_QUEUE_SIZE 64
typedef struct {
    DRAMBank banks[DRAM_BANKS_PER_CHANNEL];
    DRAMRequest req_queue[DRAM_QUEUE_SIZE];
    int queue_head, queue_tail, queue_count;
    u64 next_command_cycle;
    u64 bytes_transferred_this_window;
    u64 bandwidth_window_start;
    double max_bytes_per_cycle;
} MemoryChannel;

// SM (Streaming Multiprocessor)
typedef struct {
    int sm_id;

    // Warp pool
    Warp warps[MAX_WARPS_PER_SM];
    int active_warp_count;

    // Execution units
    ExecUnit fp32_unit;
    ExecUnit int32_unit;
    ExecUnit fp64_unit;
    ExecUnit sfu_unit;
    ExecUnit tensor_unit;
    ExecUnit ldst_unit;

    // Register file banks
    u64 reg_bank_busy[REG_BANKS];

    // Shared memory banks
    u64 shmem_bank_busy[SHMEM_BANKS];

    // L1 Cache
    L1Cache l1;

    // Pending memory transactions
    MemTransaction mem_queue[256];
    int mem_queue_head, mem_queue_tail;

    // Dual-issue tracking per scheduler
    int dual_issues_this_cycle;

    // Statistics
    u64 cycles_active;
    u64 instructions_issued;
    u64 reg_bank_conflicts;
    u64 shmem_bank_conflicts;
    u64 warp_stalls;
    u64 total_memory_transactions;
    u64 coalesced_transactions;
    u64 dual_issue_cycles;
    u64 total_branches;
    u64 total_divergent_branches;
    u64 total_divergence_cycles;
} SM;

// GPU State
typedef struct {
    u64 cycle;
    SM *sms; // g_sm_count dynamically allocated elements
    L2Cache l2;
    MemoryChannel mem_channels[MEMORY_CHANNELS];

    // Global statistics
    u64 total_instructions;
    u64 total_cycles;
    u64 total_l1_hits;
    u64 total_l1_misses;
    u64 total_l2_hits;
    u64 total_l2_misses;
    u64 total_dram_accesses;
    u64 total_reg_conflicts;
    u64 total_shmem_conflicts;
    u64 total_warp_stalls;
} GPU;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static inline u64 max_u64(u64 a, u64 b) { return a > b ? a : b; }
static inline u32 min_u32(u32 a, u32 b) { return a < b ? a : b; }

// LCG random number generator
static u32 rng_state = 0x12345678;
static inline u32 fast_rand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

// ============================================================================
// CACHE FUNCTIONS
// ============================================================================

void l1_init(L1Cache *cache) {
    cache->line_size = L1_LINE_SIZE;
    cache->ways = L1_WAYS;
    cache->sets = L1_SIZE_PER_SM / (cache->line_size * cache->ways);

    cache->tags = malloc(cache->sets * sizeof(u64*));
    cache->lru = malloc(cache->sets * sizeof(u8*));
    cache->valid = malloc(cache->sets * sizeof(bool*));

    for (int i = 0; i < cache->sets; i++) {
        cache->tags[i] = calloc(cache->ways, sizeof(u64));
        cache->lru[i] = calloc(cache->ways, sizeof(u8));
        cache->valid[i] = calloc(cache->ways, sizeof(bool));
        for (int j = 0; j < cache->ways; j++) {
            cache->lru[i][j] = j;
        }
    }
    cache->hits = 0;
    cache->misses = 0;
}

static inline u32 l1_get_set(L1Cache *cache, u64 addr) {
    return (addr / cache->line_size) % cache->sets;
}

static inline u64 l1_get_tag(L1Cache *cache, u64 addr) {
    return addr / cache->line_size;
}

bool l1_access(L1Cache *cache, u64 addr) {
    u32 set = l1_get_set(cache, addr);
    u64 tag = l1_get_tag(cache, addr);

    // Check for hit
    for (int way = 0; way < cache->ways; way++) {
        if (cache->valid[set][way] && cache->tags[set][way] == tag) {
            // Update LRU
            u8 old_lru = cache->lru[set][way];
            for (int w = 0; w < cache->ways; w++) {
                if (cache->lru[set][w] < old_lru) {
                     cache->lru[set][w]++;
                }
            }
            cache->lru[set][way] = 0;
            cache->hits++;
            return true;
        }
    }

    // Miss - find LRU way
    int victim = 0;
    u8 max_lru = 0;
    for (int way = 0; way < cache->ways; way++) {
        if (!cache->valid[set][way]) {
            victim = way;
            break;
        }
        if (cache->lru[set][way] > max_lru) {
            max_lru = cache->lru[set][way];
            victim = way;
        }
    }

    // Install new line
    cache->tags[set][victim] = tag;
    cache->valid[set][victim] = true;
    for (int w = 0; w < cache->ways; w++) {
        cache->lru[set][w]++;
    }
    cache->lru[set][victim] = 0;
    cache->misses++;
    return false;
}

void l2_init(L2Cache *cache) {
    cache->line_size = L2_LINE_SIZE;
    cache->ways = L2_WAYS;
    cache->banks = L2_BANKS;
    cache->sets = L2_SIZE / (cache->line_size * cache->ways);

    cache->tags = malloc(cache->sets * sizeof(u64*));
    cache->lru = malloc(cache->sets * sizeof(u8*));
    cache->valid = malloc(cache->sets * sizeof(bool*));
    cache->bank_busy = calloc(cache->banks, sizeof(u64));

    for (int i = 0; i < cache->sets; i++) {
        cache->tags[i] = calloc(cache->ways, sizeof(u64));
        cache->lru[i] = calloc(cache->ways, sizeof(u8));
        cache->valid[i] = calloc(cache->ways, sizeof(bool));
        for (int j = 0; j < cache->ways; j++) {
            cache->lru[i][j] = j;
        }
    }
    cache->hits = 0;
    cache->misses = 0;
}

static inline u32 l2_get_set(L2Cache *cache, u64 addr) {
    return (addr / cache->line_size) % cache->sets;
}

static inline u64 l2_get_tag(L2Cache *cache, u64 addr) {
    return addr / cache->line_size;
}

static inline int l2_get_bank(L2Cache *cache, u64 addr) {
    return (addr / cache->line_size) % cache->banks;
}

bool l2_access(L2Cache *cache, u64 addr, u64 cycle, u64 *ready_cycle) {
    u32 set = l2_get_set(cache, addr);
    u64 tag = l2_get_tag(cache, addr);
    int bank = l2_get_bank(cache, addr);

    // Bank conflict check
    if (cache->bank_busy[bank] > cycle) {
        *ready_cycle = cache->bank_busy[bank] + L2_LATENCY_BASE;
        return false; // bank busy, retry later
    }

    cache->bank_busy[bank] = cycle + 1;

    // Check for hit
    for (int way = 0; way < cache->ways; way++) {
        if (cache->valid[set][way] && cache->tags[set][way] == tag) {
            u8 old_lru = cache->lru[set][way];
            for (int w = 0; w < cache->ways; w++) {
                if (cache->lru[set][w] < old_lru) {
                     cache->lru[set][w]++;
                }
            }
            cache->lru[set][way] = 0;
            cache->hits++;
            *ready_cycle = cycle + L2_LATENCY_BASE;
            return true;
        }
    }

    // Miss - find victim
    int victim = 0;
    u8 max_lru = 0;
    for (int way = 0; way < cache->ways; way++) {
        if (!cache->valid[set][way]) {
            victim = way;
            break;
        }
        if (cache->lru[set][way] > max_lru) {
            max_lru = cache->lru[set][way];
            victim = way;
        }
    }

    cache->tags[set][victim] = tag;
    cache->valid[set][victim] = true;
    for (int w = 0; w < cache->ways; w++) {
        cache->lru[set][w]++;
    }
    cache->lru[set][victim] = 0;
    cache->misses++;
    return false; // miss
}

// ============================================================================
// DRAM FUNCTIONS
// ============================================================================

u64 dram_access(MemoryChannel *channels, u64 addr, u64 cycle) {
    int channel = (addr / DRAM_BURST_SIZE) % MEMORY_CHANNELS;
    int bank = (addr / DRAM_ROW_SIZE) % DRAM_BANKS_PER_CHANNEL;
    u64 row = addr / DRAM_ROW_SIZE;

    MemoryChannel *ch = &channels[channel];
    DRAMBank *b = &ch->banks[bank];

    // Enforce bandwidth limit (696 GB/s = ~446 bytes/cycle at 1.56 GHz)
    if (cycle - ch->bandwidth_window_start >= 1000) {
        // Reset bandwidth counter every 1000 cycles
        ch->bytes_transferred_this_window = 0;
        ch->bandwidth_window_start = cycle;
    }

    if (ch->bytes_transferred_this_window + DRAM_BURST_SIZE > ch->max_bytes_per_cycle * 1000) {
        // Bandwidth saturated - stall
        return cycle + 100; // Penalty for saturation
    }

    u64 ready_time = max_u64(cycle, ch->next_command_cycle);
    ready_time = max_u64(ready_time, b->next_column_cycle);

    // Check bank group timing
    int bank_group = bank / 4;
    for (int i = 0; i < DRAM_BANKS_PER_CHANNEL; i++) {
        if (i != bank && (i / 4) == bank_group) {
            // Same bank group - enforce tCCDL
            ready_time = max_u64(ready_time, ch->banks[i].next_column_cycle);
        }
    }

    if (b->row_open && b->open_row == row) {
        // Row buffer hit
        ready_time += DRAM_tCAS + DRAM_BURST_CYCLES;
        b->next_column_cycle = ready_time;
    } else {
        // Row buffer miss
        if (b->row_open) {
             // Precharge old row
             ready_time = max_u64(ready_time, b->next_precharge_cycle);
             ready_time += DRAM_tRP;
        }
        // Activate new row
        ready_time += DRAM_tRCD;
        ready_time += DRAM_tCAS + DRAM_BURST_CYCLES;

        b->row_open = true;
        b->open_row = row;
        b->next_activate_cycle = ready_time;
        b->next_precharge_cycle = ready_time + DRAM_tRAS;
        b->next_column_cycle = ready_time;
    }

    ch->next_command_cycle = ready_time + 1;
    ch->bytes_transferred_this_window += DRAM_BURST_SIZE;

    return ready_time;
}

// ============================================================================
// EXECUTION UNIT FUNCTIONS
// ============================================================================

void exec_unit_init(ExecUnit *unit, int count, int latency, bool pipelined, int throughput) {
    unit->count = count;
    unit->latency = latency;
    unit->pipelined = pipelined;
    unit->throughput_per_cycle = throughput;
    unit->busy_until = calloc(count, sizeof(u64));
}

bool exec_unit_try_issue(ExecUnit *unit, u64 cycle, u64 *ready_cycle) {
    if (unit->pipelined) {
        // Pipelined unit - just check if we have throughput
        *ready_cycle = cycle + unit->latency;
        return true;
    } else {
        // Non-pipelined - find free unit
        for (int i = 0; i < unit->count; i++) {
             if (unit->busy_until[i] <= cycle) {
                 unit->busy_until[i] = cycle + unit->latency;
                 *ready_cycle = unit->busy_until[i];
                 return true;
             }
        }
        return false;
    }
}

// ============================================================================
// DIVERGENCE STACK FUNCTIONS
// ============================================================================

void div_stack_init(DivergenceStack *stack) {
    stack->top = -1;
    stack->reconverge_pc = 0;
    stack->in_divergence = false;
}

void div_stack_push(DivergenceStack *stack, u32 active_mask, u32 pc, bool is_convergence) {
    if (stack->top >= MAX_DIVERGENCE_DEPTH - 1) {
        // Stack overflow - shouldn't happen with reasonable code
        return;
    }
    stack->top++;
    stack->stack[stack->top].active_mask = active_mask;
    stack->stack[stack->top].pc = pc;
    stack->stack[stack->top].is_convergence = is_convergence;
}

DivergenceEntry div_stack_pop(DivergenceStack *stack) {
    DivergenceEntry entry = {0};
    if (stack->top >= 0) {
        entry = stack->stack[stack->top];
        stack->top--;
    }
    return entry;
}

DivergenceEntry div_stack_peek(DivergenceStack *stack) {
    DivergenceEntry entry = {0};
    if (stack->top >= 0) {
        entry = stack->stack[stack->top];
    }
    return entry;
}

bool div_stack_empty(DivergenceStack *stack) {
    return stack->top < 0;
}

// Evaluate predicate for conditional branch
PredicateResult evaluate_predicate(Instruction *instr, u32 active_mask) {
    // Generate per-thread predicates based on thread ID and instruction
    // This simulates conditional evaluation: if (threadIdx.x < N) ...

    int true_count = 0;
    int false_count = 0;

    for (int tid = 0; tid < WARP_SIZE; tid++) {
        if (!(active_mask & (1u << tid))) continue;

        // Simulate predicate evaluation based on thread ID and immediate value
        // Example: "if (threadIdx.x < imm)" pattern
        bool pred = false;

        switch (instr->imm % 8) {
            case 0: // if (tid < 16)
                pred = (tid < 16);
                break;
            case 1: // if (tid >= 16)
                pred = (tid >= 16);
                break;
            case 2: // if (tid % 2 == 0) - even threads
                pred = (tid % 2 == 0);
                break;
            case 3: // if (tid % 4 < 2) - quarter divergence
                pred = (tid % 4 < 2);
                break;
            case 4: // if (tid < 8 || tid >= 24) - two groups
                pred = (tid < 8) || (tid >= 24);
                break;
            case 5: // All true (uniform)
                pred = true;
                break;
            case 6: // All false (uniform)
                pred = false;
                break;
            case 7: // Complex pattern
                pred = ((tid * 7 + 13) % 3) == 0;
                break;
        }

        instr->thread_preds[tid] = pred ? 1 : 0;
        if (pred) true_count++; else false_count++;
    }

    int active_count = true_count + false_count;

    if (active_count == 0) return PRED_UNIFORM;
    if (true_count == active_count) return PRED_TRUE;
    if (false_count == active_count) return PRED_FALSE;
    return PRED_DIVERGE;
}

// Handle divergent branch using NVIDIA's SSY/SYNC model
void handle_divergent_branch(Warp *w, Instruction *instr, u32 branch_target, u32 fallthrough_pc) {
    // Calculate which threads take branch vs fallthrough
    u32 taken_mask = 0;
    u32 not_taken_mask = 0;

    for (int tid = 0; tid < WARP_SIZE; tid++) {
        if (!(w->active_mask & (1u << tid))) continue;

        if (instr->thread_preds[tid]) {
            taken_mask |= (1u << tid);
        } else {
            not_taken_mask |= (1u << tid);
        }
    }
    // NVIDIA's divergence handling:
    // 1. Push reconvergence point (fallthrough + offset) with full mask
    // 2. Push not-taken path
    // 3. Execute taken path immediately

    u32 reconverge_pc = fallthrough_pc + 32; // Simplified: reconverge after both paths

    // Push reconvergence point with full original mask
    div_stack_push(&w->div_stack, w->active_mask, reconverge_pc, true);

    // Push not-taken path (if any threads didn't take branch)
    if (not_taken_mask != 0) {
        div_stack_push(&w->div_stack, not_taken_mask, fallthrough_pc, false);
    }

    // Execute taken path with reduced mask
    w->active_mask = taken_mask;
    w->pc = branch_target;
    w->div_stack.in_divergence = true;

    w->branches_diverged++;
}

// Check for reconvergence
bool check_reconvergence(Warp *w) {
    if (div_stack_empty(&w->div_stack)) {
        w->div_stack.in_divergence = false;
        return false;
    }

    DivergenceEntry top = div_stack_peek(&w->div_stack);

    // Reached a reconvergence point
    if (top.is_convergence && w->pc >= top.pc) {
        // Pop and restore full mask
        DivergenceEntry entry = div_stack_pop(&w->div_stack);
        w->active_mask = entry.active_mask;
        w->pc = entry.pc;

        // Check if we're still in divergence (nested branches)
        w->div_stack.in_divergence = !div_stack_empty(&w->div_stack);
        return true;
    }

    // Finished one path, start next
    if (!top.is_convergence && w->pc >= top.pc + 16) {
        DivergenceEntry entry = div_stack_pop(&w->div_stack);
        w->active_mask = entry.active_mask;
        w->pc = entry.pc;
        return true;
    }

    return false;
}

// ============================================================================
// MEMORY COALESCING FUNCTIONS
// ============================================================================

// Analyze memory access pattern and determine coalescing
CoalesceResult analyze_coalescing(Instruction *instr, int warp_id) {
    (void)warp_id;
    CoalesceResult result = {0};

    // GDDR6 cache line is 128 bytes - memory system works in 32-byte sectors
    const int SECTOR_SIZE = 32;
    const int WORD_SIZE = 4;

    // Count active threads
    int active_threads = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (instr->active_mask & (1u << i)) {
            active_threads++;
        }
    }
    if (active_threads == 0) {
        result.fully_coalesced = true;
        result.num_transactions = 0;
        return result;
    }

    // A 4-byte access can straddle two 32-byte sectors when unaligned.
    // Collect both sectors without double-counting any shared between lanes.
    u64 sectors[WARP_SIZE * 2];
    int num_sectors = 0;

    for (int tid = 0; tid < WARP_SIZE; tid++) {
        if (!(instr->active_mask & (1u << tid))) continue;

        u64 addr = instr->thread_addrs[tid];
        u64 first = (addr / SECTOR_SIZE) * SECTOR_SIZE;
        u64 last = ((addr + WORD_SIZE - 1) / SECTOR_SIZE) * SECTOR_SIZE;
        for (u64 sector = first; sector <= last; sector += SECTOR_SIZE) {
            bool found = false;
            for (int j = 0; j < num_sectors; j++) {
                if (sectors[j] == sector) { found = true; break; }
            }
            if (!found) sectors[num_sectors++] = sector;
        }
    }

    // Each unique sector requires a separate transaction
    result.num_transactions = num_sectors;
    for (int i = 0; i < num_sectors; i++) {
        result.transaction_addrs[i] = sectors[i];
        result.transaction_sizes[i] = SECTOR_SIZE;
    }

    // A "within 128B span" test is NOT a coalescing measurement on Ampere.
    // NVIDIA documents the count of distinct 32-byte sectors. Classifying
    // efficiency is task-dependent (duplicate/broadcast words are possible).
    // Keep the legacy boolean solely as a minimal-sector *toy* condition for
    // aligned 4-byte words; raw transaction count is the primary evidence.
    u64 unique_words[WARP_SIZE];
    int unique_count = 0;
    for (int tid = 0; tid < WARP_SIZE; ++tid) {
        if (!(instr->active_mask & (1u << tid))) continue;
        u64 addr = instr->thread_addrs[tid];
        bool seen = false;
        for (int j = 0; j < unique_count; ++j)
            if (unique_words[j] == addr) { seen = true; break; }
        if (!seen) unique_words[unique_count++] = addr;
    }
    int ideal_sectors = (unique_count * WORD_SIZE + SECTOR_SIZE - 1) / SECTOR_SIZE;
    result.fully_coalesced = num_sectors == ideal_sectors;

    return result;
}

// Generate per-thread addresses for a memory instruction
void generate_thread_addresses(Instruction *instr, int warp_id, int pattern_type) {
    (void)warp_id;
    u64 base = instr->imm;

    for (int tid = 0; tid < WARP_SIZE; tid++) {
        if (!(instr->active_mask & (1u << tid))) {
            instr->thread_addrs[tid] = 0;
            continue;
        }

        switch (pattern_type) {
            case 0: // Coalesced (stride-1, 4-byte elements)
                instr->thread_addrs[tid] = base + tid * 4;
                break;
            case 1: // Stride-2 (2x worse)
                instr->thread_addrs[tid] = base + tid * 8;
                break;
            case 2: // Stride-32 (worst case - 32 transactions)
                instr->thread_addrs[tid] = base + tid * 128;
                break;
            case 3: // Random (bad coalescing)
                instr->thread_addrs[tid] = base + ((tid * 997) % 4096) * 4;
                break;
            case 4: // Broadcast (all same address - perfect coalescing)
                instr->thread_addrs[tid] = base;
                break;
            default: // Mixed pattern
                if (tid < 16) {
                    instr->thread_addrs[tid] = base + tid * 4; // coalesced
                } else {
                    instr->thread_addrs[tid] = base + tid * 128; // scattered
                }
        }
    }
}

// ============================================================================
// WARP & INSTRUCTION FUNCTIONS
// ============================================================================

void warp_init(Warp *w, int id) {
    memset(w, 0, sizeof(Warp));
    w->warp_id = id;
    w->state = WARP_IDLE;
    w->active_mask = 0xFFFFFFFF;
    w->pc = 0;
    div_stack_init(&w->div_stack);
}

void scoreboard_set(Warp *w, u8 reg, u64 ready_cycle) {
    w->scoreboard[reg].valid = true;
    w->scoreboard[reg].ready_cycle = ready_cycle;
    w->scoreboard[reg].reg_id = reg;
}

bool scoreboard_check(Warp *w, u8 reg, u64 cycle) {
    if (!w->scoreboard[reg].valid) return true;
    return w->scoreboard[reg].ready_cycle <= cycle;
}

void scoreboard_clear(Warp *w, u8 reg) {
    w->scoreboard[reg].valid = false;
}

// ============================================================================
// WORKLOAD GENERATION
// Each warp's instructions are produced deterministically in WARP_FETCH.
// The original version wrote 128/64/32 sample instructions repeatedly into
// a single fetch_instr field; those were overwritten by WARP_FETCH and never
// executed. This revision uses explicit scenario selection instead.
// ============================================================================
// SM SIMULATION FUNCTIONS
// ============================================================================

void sm_init(SM *sm, int id, int warp_count) {
    memset(sm, 0, sizeof(SM));
    sm->sm_id = id;
    sm->active_warp_count = warp_count;

    // Initialize execution units (Ampere configuration)
    exec_unit_init(&sm->fp32_unit, FP32_UNITS_PER_SM, LATENCY_FP32, true, 128);
    exec_unit_init(&sm->int32_unit, INT32_UNITS_PER_SM, LATENCY_INT32, true, 128);
    exec_unit_init(&sm->fp64_unit, FP64_UNITS_PER_SM, LATENCY_FP64, true, 2);
    exec_unit_init(&sm->sfu_unit, SFU_UNITS_PER_SM, LATENCY_SFU_FAST, true, 32);
    exec_unit_init(&sm->tensor_unit, TENSOR_CORES_PER_SM, LATENCY_TENSOR, false, 4);
    exec_unit_init(&sm->ldst_unit, LD_ST_UNITS_PER_SM, 1, true, 32);

    // Initialize L1 cache
    l1_init(&sm->l1);

    // Initialize warps
    for (int i = 0; i < warp_count; i++) {
        warp_init(&sm->warps[i], i);
        sm->warps[i].state = WARP_FETCH;

        // WARP_FETCH builds the selected workload; nothing is preloaded here.
    }
}

bool sm_schedule_warp(SM *sm, GPU *gpu, Warp *w, u64 cycle) {
    if (w->state != WARP_ISSUE_WAIT && w->state != WARP_OPERAND_WAIT) {
        return false;
    }

    Instruction *instr = &w->issue_instr;
    InstructionPipe pipe = get_instruction_pipe(instr->op);

    // Dual-issue check: can't issue two from same pipe in same cycle
    if (pipe == PIPE_COMPUTE && w->issued_compute_this_cycle) {
        return false;
    }
    if (pipe == PIPE_MEMORY && w->issued_memory_this_cycle) {
        return false;
    }

    // Check register dependencies
if (instr->op != OP_NOP && instr->op != OP_EXIT) {
    if (!scoreboard_check(w, instr->src1_reg, cycle)) return false;
    if (!scoreboard_check(w, instr->src2_reg, cycle)) return false;
    if (!scoreboard_check(w, instr->src3_reg, cycle)) return false;
}

// Check register bank conflicts
int dst_bank = instr->dst_reg % REG_BANKS;
if (sm->reg_bank_busy[dst_bank] > cycle) {
    sm->reg_bank_conflicts++;
    return false;
}

// Try to issue to appropriate execution unit
u64 ready_cycle = 0;
bool issued = false;

switch (instr->op) {
    case OP_IADD:
    case OP_IMUL:
    case OP_ISUB:
        issued = exec_unit_try_issue(&sm->int32_unit, cycle, &ready_cycle);
        break;
    case OP_FADD:
    case OP_FMUL:
    case OP_FFMA:
        issued = exec_unit_try_issue(&sm->fp32_unit, cycle, &ready_cycle);
        break;
    case OP_DADD:
    case OP_DMUL:
        issued = exec_unit_try_issue(&sm->fp64_unit, cycle, &ready_cycle);
        break;
    case OP_SFU_RCP:
    case OP_SFU_SQRT:
    case OP_SFU_SIN:
    case OP_SFU_COS:
        issued = exec_unit_try_issue(&sm->sfu_unit, cycle, &ready_cycle);
        break;
    case OP_TENSOR_MMA:
        issued = exec_unit_try_issue(&sm->tensor_unit, cycle, &ready_cycle);
        break;
    case OP_LD_GLOBAL:
    case OP_ST_GLOBAL:
        // Memory operations - analyze coalescing first
        if (exec_unit_try_issue(&sm->ldst_unit, cycle, &ready_cycle)) {
            // Analyze coalescing pattern
            CoalesceResult coalesce = analyze_coalescing(instr, w->warp_id);

            // Track coalescing statistics
            sm->total_memory_transactions += coalesce.num_transactions;
            if (coalesce.fully_coalesced) {
                sm->coalesced_transactions++;
                w->coalesced_accesses++;
            } else {
                w->uncoalesced_accesses++;
            }

            // Process each transaction separately
            u64 max_ready = cycle;
            for (int i = 0; i < coalesce.num_transactions; i++) {
                u64 addr = coalesce.transaction_addrs[i];
                bool l1_hit = l1_access(&sm->l1, addr);

                if (l1_hit) {
                    u64 txn_ready = cycle + LATENCY_MEMORY_MIN;
                    if (txn_ready > max_ready) max_ready = txn_ready;
                } else {
                    // L1 miss - check L2
                    u64 l2_ready = 0;
                    bool l2_hit = l2_access(&gpu->l2, addr, cycle, &l2_ready);

                    if (l2_hit) {
                        if (l2_ready > max_ready) max_ready = l2_ready;
                    } else {
                        // L2 miss - go to DRAM
                        u64 dram_ready = dram_access(gpu->mem_channels, addr, cycle);
                        if (dram_ready > max_ready) max_ready = dram_ready;
                        gpu->total_dram_accesses++;
                    }
                }
            }

            // Uncoalesced accesses take longer (serialize transactions)
            // Each additional transaction adds interconnect + L2 bank conflict penalty
            if (coalesce.num_transactions > 4) {
                max_ready += (coalesce.num_transactions - 4) * 10; // penalty
            }

            ready_cycle = max_ready;
            w->memory_accesses++;
            issued = true;
        }
        break;
    case OP_LD_SHARED:
    case OP_ST_SHARED: {
        int shmem_bank = (instr->imm / 4) % SHMEM_BANKS;
        if (sm->shmem_bank_busy[shmem_bank] > cycle) {
            sm->shmem_bank_conflicts++;
            return false;
        }
        sm->shmem_bank_busy[shmem_bank] = cycle + 1;
        ready_cycle = cycle + 28; // shared memory latency
        issued = true;
        break;
    }
    case OP_BRA_COND: {
        // Conditional branch - evaluate predicate
        PredicateResult pred_result = evaluate_predicate(instr, w->active_mask);

        w->branches_taken++;
        sm->total_branches++;

        if (pred_result == PRED_DIVERGE) {
            // Divergent branch - push/pop stack
            u32 branch_target = instr->imm;
            u32 fallthrough = w->pc + 1;
            handle_divergent_branch(w, instr, branch_target, fallthrough);

            sm->total_divergent_branches++;

            // Branch takes 1 cycle + divergence penalty
            ready_cycle = cycle + LATENCY_BRANCH + 5; // divergence setup cost
        } else if (pred_result == PRED_TRUE) {
            // All threads take branch - uniform
            w->pc = instr->imm;
            ready_cycle = cycle + LATENCY_BRANCH;
        } else {
            // All threads skip branch - fall through
            w->pc++;
            ready_cycle = cycle + LATENCY_BRANCH;
        }

        w->state = WARP_EXEC_WAIT;
        w->exec_ready_cycle = ready_cycle;
        return true; // Branch handled specially
    }
    case OP_SYNC:
        // Reconvergence point - check divergence stack
        check_reconvergence(w);
        ready_cycle = cycle + 1;
        issued = true;
        break;
    case OP_EXIT:
        w->state = WARP_COMPLETED;
        return true;
    default:
        ready_cycle = cycle + 1;
        issued = true;
}
    if (!issued) {
        w->cycles_stalled++;
        return false;
    }

    // Mark register as busy in scoreboard
    if (instr->dst_reg != 0) {
        scoreboard_set(w, instr->dst_reg, ready_cycle);
        sm->reg_bank_busy[dst_bank] = cycle + 1;
    }

    // Update dual-issue tracking
    if (pipe == PIPE_COMPUTE) {
        w->issued_compute_this_cycle = true;
    } else if (pipe == PIPE_MEMORY) {
        w->issued_memory_this_cycle = true;
    }

    // Check if this is a dual-issue cycle
    if (w->issued_compute_this_cycle && w->issued_memory_this_cycle) {
        w->dual_issue_cycles++;
        sm->dual_issue_cycles++;
    }

    // Track divergence overhead
    if (w->div_stack.in_divergence) {
        w->divergence_cycles++;
        sm->total_divergence_cycles++;
    }

    // Update warp state
    w->exec_ready_cycle = ready_cycle;
    w->state = WARP_EXEC_WAIT;
    w->instructions_issued++;
    sm->instructions_issued++;
    gpu->total_instructions++;

    return true;
}
void sm_step(SM *sm, GPU *gpu) {
    u64 cycle = gpu->cycle;

    // Reset dual-issue flags at start of cycle
    for (int i = 0; i < sm->active_warp_count; i++) {
        sm->warps[i].issued_compute_this_cycle = false;
        sm->warps[i].issued_memory_this_cycle = false;
    }
    sm->dual_issues_this_cycle = 0;

    // Process each warp through pipeline stages
    for (int i = 0; i < sm->active_warp_count; i++) {
        Warp *w = &sm->warps[i];

        // Check for reconvergence at start of each cycle
        if (w->div_stack.in_divergence && w->state == WARP_FETCH) {
            check_reconvergence(w);
        }

        switch (w->state) {
            case WARP_FETCH:
                // Fetch instruction
                if (w->pc < (u32)g_max_instructions && cycle >= w->next_fetch_cycle) {
                    // Generate next instruction (simplified - in real GPU this comes from I-cache)
                    Instruction instr = {0};

                    // Determine memory access pattern for this warp
                    // Vary patterns to demonstrate coalescing effects
                    int pattern = g_pattern >= 0 ? g_pattern : (sm->sm_id + w->warp_id) % 6;

                    // Synthetic instruction generation based on PC
                    u32 inst_pattern = (w->pc * 7 + w->warp_id * 13) % 100;

                    // Inject branches periodically to demonstrate divergence
                    if (g_workload == 0 && w->pc > 0 && (w->pc % 32) == 0 && w->pc + 40 < (u32)g_max_instructions) {
                        // Insert conditional branch
            instr.op = OP_BRA_COND;
            instr.imm = w->pc + 16; // Branch target (skip 16 instructions)
            instr.active_mask = w->active_mask;
            // Predicate pattern varies to create different divergence scenarios
        } else if (g_workload == 0 && w->pc > 0 && (w->pc % 48) == 0) {
            // Insert sync/reconvergence point
            instr.op = OP_SYNC;
            instr.active_mask = w->active_mask;
        } else if (g_workload == 2 || (g_workload == 0 && inst_pattern < 35)) {
            instr.op = OP_FFMA;
            instr.dst_reg = (w->pc % 30) + 1;
            instr.src1_reg = ((w->pc * 3) % 30) + 1;
            instr.src2_reg = ((w->pc * 5) % 30) + 1;
            instr.src3_reg = ((w->pc * 7) % 30) + 1;
        } else if (g_workload == 1 || inst_pattern < 55) {
            instr.op = (g_workload == 1 && (w->pc & 1u)) ? OP_ST_GLOBAL : OP_LD_GLOBAL;
            instr.dst_reg = (w->pc % 30) + 1;
            instr.imm = (sm->sm_id * 0x100000) + (w->warp_id * 0x10000) + (w->pc * 1024);
            instr.active_mask = w->active_mask;
            // Generate per-thread addresses with specific pattern
            generate_thread_addresses(&instr, w->warp_id, pattern);
        } else if (inst_pattern < 65) {
            instr.op = OP_IADD;
            instr.dst_reg = (w->pc % 30) + 1;
            instr.src1_reg = ((w->pc * 2) % 30) + 1;
            instr.src2_reg = ((w->pc * 4) % 30) + 1;
        } else if (inst_pattern < 75) {
            instr.op = OP_SFU_RCP;
            instr.dst_reg = (w->pc % 30) + 1;
            instr.src1_reg = ((w->pc * 3) % 30) + 1;
        } else if (inst_pattern < 80) {
            instr.op = OP_TENSOR_MMA;
            instr.dst_reg = (w->pc % 8) + 1;
            instr.src1_reg = ((w->pc * 2) % 16) + 16;
            instr.src2_reg = ((w->pc * 3) % 16) + 32;
        } else if (inst_pattern < 90) {
            instr.op = OP_ST_GLOBAL;
            instr.src1_reg = (w->pc % 30) + 1;
            instr.imm = (sm->sm_id * 0x100000) + (w->warp_id * 0x10000) + (w->pc * 1024);
            instr.active_mask = w->active_mask;
            generate_thread_addresses(&instr, w->warp_id, pattern);
        } else {
            instr.op = OP_FMUL;
            instr.dst_reg = (w->pc % 30) + 1;
            instr.src1_reg = ((w->pc * 2) % 30) + 1;
            instr.src2_reg = ((w->pc * 3) % 30) + 1;
        }

        if (instr.active_mask == 0) instr.active_mask = w->active_mask;
        w->fetch_instr = instr;
        w->state = WARP_IBUFFER;
        w->next_fetch_cycle = cycle + PIPE_FETCH;
    } else if (w->pc >= (u32)g_max_instructions) {
        w->state = WARP_COMPLETED;
    }
    break;

case WARP_IBUFFER:
    // Instruction buffer stage
    w->decode_instr = w->fetch_instr;
    w->state = WARP_DECODE;
    break;

case WARP_DECODE:
    // Decode stage
    w->issue_instr = w->decode_instr;
    w->state = WARP_ISSUE_WAIT;
    w->pc++;
    break;

case WARP_ISSUE_WAIT:
case WARP_OPERAND_WAIT:
    // Try to issue instruction
    if (sm_schedule_warp(sm, gpu, w, cycle)) {
        // Successfully issued
                } else {
                    w->cycles_stalled++;
                    sm->warp_stalls++;
                }
                break;

            case WARP_EXEC_WAIT:
                // Wait for execution to complete
                if (cycle >= w->exec_ready_cycle) {
                    // Clear scoreboard for destination register
                    if (w->issue_instr.dst_reg != 0) {
                        scoreboard_clear(w, w->issue_instr.dst_reg);
                    }
                    w->state = WARP_FETCH;
                }
                break;

            case WARP_BARRIER_WAIT:
            case WARP_WRITEBACK:
            case WARP_COMPLETED:
            case WARP_IDLE:
                break;
        }
    }
}

bool sm_has_active_warps(SM *sm) {
    for (int i = 0; i < sm->active_warp_count; i++) {
        if (sm->warps[i].state != WARP_COMPLETED && sm->warps[i].state != WARP_IDLE) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// GPU SIMULATION
// ============================================================================

void gpu_init(GPU *gpu, int warps_per_sm) {
    memset(gpu, 0, sizeof(GPU));

    // Initialize L2 cache
    l2_init(&gpu->l2);

    // Initialize memory channels with bandwidth limits
    for (int i = 0; i < MEMORY_CHANNELS; i++) {
        memset(&gpu->mem_channels[i], 0, sizeof(MemoryChannel));
        // 696 GB/s / 12 channels = 58 GB/s per channel
        // At 1.56 GHz: 58e9 / 1.56e9 = ~37.2 bytes/cycle per channel
        gpu->mem_channels[i].max_bytes_per_cycle = 37.2;

        // Initialize bank groups (GDDR6 has 4 bank groups)
        for (int b = 0; b < DRAM_BANKS_PER_CHANNEL; b++) {
            gpu->mem_channels[i].banks[b].bank_group = b / 4;
        }
    }

    // Initialize SMs
    gpu->sms = calloc((size_t)g_sm_count, sizeof(SM));
    if (!gpu->sms) { perror("calloc SMs"); exit(3); }
    for (int i = 0; i < g_sm_count; i++) {
        sm_init(&gpu->sms[i], i, warps_per_sm);
    }

    printf("GPU Initialized:\n");
    printf(" SMs: %d\n", g_sm_count);
    printf(" Warps per SM: %d\n", warps_per_sm);
    printf(" Total warps: %d\n", g_sm_count * warps_per_sm);
    printf(" Total threads: %d\n", g_sm_count * warps_per_sm * WARP_SIZE);
    printf(" L2 Cache: %d MB (%d sets, %d ways)\n",
           L2_SIZE / (1024*1024), gpu->l2.sets, gpu->l2.ways);
    printf(" Memory Channels: %d\n", MEMORY_CHANNELS);
    printf("\n");
}

void gpu_step(GPU *gpu) {
    // Step all SMs in parallel
    for (int i = 0; i < g_sm_count; i++) {
        if (sm_has_active_warps(&gpu->sms[i])) {
            sm_step(&gpu->sms[i], gpu);
        }
    }

    gpu->cycle++;
    gpu->total_cycles++;
}

bool gpu_is_active(GPU *gpu) {
    for (int i = 0; i < g_sm_count; i++) {
        if (sm_has_active_warps(&gpu->sms[i])) {
            return true;
        }
    }
    return false;
}

void gpu_print_stats(GPU *gpu) {
    printf("\n========================================\n");
    printf("TOY SIMULATION STATISTICS (NOT HARDWARE TIMINGS)\n");
    printf("========================================\n");
    printf("Total Cycles:              %llu\n", (unsigned long long)gpu->total_cycles);
    printf("Total Instructions Issued: %llu\n", (unsigned long long)gpu->total_instructions);
    printf("IPC (overall):             %.3f\n",
           (double)gpu->total_instructions / gpu->total_cycles);
    printf("\n");

    // Aggregate SM statistics
    u64 total_sm_instructions = 0;
    u64 total_sm_stalls = 0;
    u64 total_reg_conflicts = 0;
    u64 total_shmem_conflicts = 0;
    u64 total_mem_transactions = 0;
    u64 total_coalesced_txns = 0;
    u64 total_dual_issues = 0;
    u64 total_branches = 0;
    u64 total_divergent_branches = 0;
    u64 total_divergence_cycles = 0;

    u64 total_warp_coalesced = 0;
    u64 total_warp_uncoalesced = 0;
    u64 total_warp_dual_issue = 0;
    u64 total_warp_branches = 0;
    u64 total_warp_diverged = 0;

    for (int i = 0; i < g_sm_count; i++) {
        total_sm_instructions += gpu->sms[i].instructions_issued;
        total_sm_stalls += gpu->sms[i].warp_stalls;
        total_reg_conflicts += gpu->sms[i].reg_bank_conflicts;
        total_shmem_conflicts += gpu->sms[i].shmem_bank_conflicts;
        total_mem_transactions += gpu->sms[i].total_memory_transactions;
        total_coalesced_txns += gpu->sms[i].coalesced_transactions;
        total_dual_issues += gpu->sms[i].dual_issue_cycles;
        total_branches += gpu->sms[i].total_branches;
        total_divergent_branches += gpu->sms[i].total_divergent_branches;
        total_divergence_cycles += gpu->sms[i].total_divergence_cycles;

        for (int w = 0; w < gpu->sms[i].active_warp_count; w++) {
            total_warp_coalesced += gpu->sms[i].warps[w].coalesced_accesses;
            total_warp_uncoalesced += gpu->sms[i].warps[w].uncoalesced_accesses;
            total_warp_dual_issue += gpu->sms[i].warps[w].dual_issue_cycles;
            total_warp_branches += gpu->sms[i].warps[w].branches_taken;
            total_warp_diverged += gpu->sms[i].warps[w].branches_diverged;
        }
    }

    printf("SM Statistics:\n");
    printf(" Avg instructions/SM:         %llu\n", (unsigned long long)(total_sm_instructions / g_sm_count));
    printf(" Avg warp stalls/SM:          %llu\n", (unsigned long long)(total_sm_stalls / g_sm_count));
    printf(" Total reg conflicts:         %llu\n", (unsigned long long)total_reg_conflicts);
    printf(" Total shmem conflicts:       %llu\n", (unsigned long long)total_shmem_conflicts);
    printf("\n");

    // Branch and divergence statistics
double divergence_rate = total_branches > 0 ?
    (double)total_divergent_branches / total_branches * 100.0 : 0.0;
double divergence_overhead = gpu->total_cycles > 0 ?
    (double)total_divergence_cycles / gpu->total_cycles * 100.0 : 0.0;

printf("Branch & Divergence:\n");
printf(" Total branches:           %llu\n", (unsigned long long)total_branches);
printf(" Divergent branches:       %llu (%.1f%%)\n",
       (unsigned long long)total_divergent_branches, divergence_rate);
printf(" Divergence cycles:        %llu\n", (unsigned long long)total_divergence_cycles);
printf(" Divergence overhead:      %.2f%% of total cycles\n", divergence_overhead);

if (total_divergent_branches > 0) {
    double avg_divergence_cost = (double)total_divergence_cycles / total_divergent_branches;
    printf(" Avg cycles/divergence:    %.1f cycles\n", avg_divergence_cost);
}
printf("\n");

// Memory coalescing statistics
u64 total_memory_ops = total_warp_coalesced + total_warp_uncoalesced;
double coalesce_rate = total_memory_ops > 0 ?
    (double)total_coalesced_txns / total_memory_ops * 100.0 : 0.0;

double memory_coalesce_quality = total_memory_ops > 0 ?
    (double)total_warp_coalesced / total_memory_ops * 100.0 : 0.0;

printf("Memory Coalescing:\n");
printf(" Total memory transactions: %llu\n", (unsigned long long)total_mem_transactions);
printf(" Toy minimal-sector accesses: %llu (%.1f%%)\n",
       (unsigned long long)total_coalesced_txns, coalesce_rate);
printf(" Avg transactions/access:    %.2f\n",
       total_memory_ops > 0 ? (double)total_mem_transactions / total_memory_ops : 0.0);
printf(" Memory coalesce quality:    %.1f%%\n", memory_coalesce_quality);
printf("\n");

// Dual-issue statistics
double dual_issue_rate = gpu->total_cycles > 0 ?
    (double)total_dual_issues / gpu->total_cycles * 100.0 : 0.0;

printf("Dual-Issue Performance (not modeled):\n");
printf(" Total dual-issue cycles:    %llu\n", (unsigned long long)total_dual_issues);
printf(" Dual-issue rate:            %.2f%%\n", dual_issue_rate);
printf(" Effective IPC with dual:    %.3f\n",
       (double)(gpu->total_instructions + total_dual_issues) / gpu->total_cycles);
printf("\n");

// Memory hierarchy statistics
u64 total_l1_hits = 0, total_l1_misses = 0;
for (int i = 0; i < g_sm_count; i++) {
    total_l1_hits += gpu->sms[i].l1.hits;
    total_l1_misses += gpu->sms[i].l1.misses;
}

u64 total_l1_accesses = total_l1_hits + total_l1_misses;
double l1_hit_rate = total_l1_accesses > 0 ?
                     (double)total_l1_hits / total_l1_accesses * 100.0 : 0.0;

u64 total_l2_accesses = gpu->l2.hits + gpu->l2.misses;
double l2_hit_rate = total_l2_accesses > 0 ?
                     (double)gpu->l2.hits / total_l2_accesses * 100.0 : 0.0;

printf("Memory Hierarchy:\n");
printf(" L1 Accesses:               %llu\n", (unsigned long long)total_l1_accesses);
printf(" L1 Hit Rate:               %.2f%%\n", l1_hit_rate);
printf(" L2 Accesses:               %llu\n", (unsigned long long)total_l2_accesses);
printf(" L2 Hit Rate:               %.2f%%\n", l2_hit_rate);
printf(" DRAM Accesses:             %llu\n", (unsigned long long)gpu->total_dram_accesses);
printf("\n");

// Performance metrics
double avg_memory_latency = total_l1_misses > 0 ?
    (total_l1_misses * L2_LATENCY_BASE + gpu->total_dram_accesses * DRAM_tRCD) /
    (double)total_l1_misses : 0.0;

printf("Performance Metrics:\n");
    printf(" Estimated memory latency: %.1f cycles\n", avg_memory_latency);
    printf(" Instructions per cycle:    %.3f\n",
           (double)gpu->total_instructions / gpu->total_cycles);
    printf(" Cycles per instruction:    %.3f\n",
           (double)gpu->total_cycles / gpu->total_instructions);

    // Compute utilization
    double compute_utilization = (double)total_sm_instructions /
                                 (gpu->total_cycles * g_sm_count * MAX_ISSUES_PER_SM) * 100.0;
    printf(" Compute utilization:       %.2f%%\n", compute_utilization);

    printf("Note: performance, cache latencies and IPC reflect this model only.\n");
    printf("========================================\n");
}

void gpu_print_detailed_stats(GPU *gpu) {
    printf("\n========================================\n");
    printf("DETAILED PER-SM STATISTICS\n");
    printf("========================================\n");

    for (int i = 0; i < g_sm_count && i < 8; i++) { // Print first 8 SMs
        SM *sm = &gpu->sms[i];
        printf("SM %d:\n", i);
        printf(" Instructions:      %llu\n", (unsigned long long)sm->instructions_issued);
        printf(" Warp stalls:       %llu\n", (unsigned long long)sm->warp_stalls);
        printf(" L1 hits:           %llu\n", (unsigned long long)sm->l1.hits);
        printf(" L1 misses:         %llu\n", (unsigned long long)sm->l1.misses);
        printf(" Reg conflicts:     %llu\n", (unsigned long long)sm->reg_bank_conflicts);
        printf(" Shmem conflicts: %llu\n", (unsigned long long)sm->shmem_bank_conflicts);
        printf(" Mem transactions: %llu\n", (unsigned long long)sm->total_memory_transactions);
        printf(" Coalesced txns:    %llu\n", (unsigned long long)sm->coalesced_transactions);
        printf(" Dual-issue cycles:%llu\n", (unsigned long long)sm->dual_issue_cycles);
        printf(" Branches:          %llu\n", (unsigned long long)sm->total_branches);
        printf(" Divergent branches:%llu\n", (unsigned long long)sm->total_divergent_branches);
        printf(" Divergence cycles:%llu\n", (unsigned long long)sm->total_divergence_cycles);

        // Show sample warp statistics
        if (sm->active_warp_count > 0) {
            Warp *w = &sm->warps[0];
            printf(" [Warp 0] Issued: %llu, Stalls: %llu, Mem: %llu\n",
                   (unsigned long long)w->instructions_issued,
                   (unsigned long long)w->cycles_stalled,
                   (unsigned long long)w->memory_accesses);
            printf("           Coalesced: %llu/%llu, Branches: %llu, Diverged: %llu\n",
                   (unsigned long long)w->coalesced_accesses,
                   (unsigned long long)(w->coalesced_accesses + w->uncoalesced_accesses),
                   (unsigned long long)w->branches_taken,
                   (unsigned long long)w->branches_diverged);
            printf("           Stack depth: %d, In divergence: %s\n",
                   w->div_stack.top + 1,
                   w->div_stack.in_divergence ? "YES" : "NO");
        }
        printf("\n");
    }

    if (g_sm_count > 8) {
        printf("... (%d more SMs not shown)\n", g_sm_count - 8);
    }

    printf("\n========================================\n");
    printf("ARCHITECTURE FEATURES DEMONSTRATED\n");
    printf("========================================\n");
    printf("\n1. MEMORY COALESCING:\n");
    printf("   This simulation uses 6 varied access patterns:\n");
    printf("   - Pattern 0: Stride-1 (32x4B aligned -> 4 distinct 32B sectors)\n");
    printf("   - Pattern 1: Stride-2 (32x4B with stride 8 -> 8 sectors)\n");
    printf("   - Pattern 2: Stride-32 (worst case, 32 transactions!)\n");
    printf("   - Pattern 3: Random (poor locality, ~20 transactions)\n");
    printf("   - Pattern 4: Broadcast (all same address, 1 transaction)\n");
    printf("   - Pattern 5: Mixed (demonstrates variance)\n");
    printf("\n2. INSTRUCTION ISSUE (LIMITATION):\n");
    printf("   - One instruction per warp can issue per simulator step.\n");
    printf("   - Dual-issue hardware behavior is NOT implemented faithfully.\n");
    printf("\n3. SIMT DIVERGENCE:\n");
    printf("   - Full divergence stack (32 levels deep)\n");
    printf("   - Models NVIDIA's SSY/SYNC reconvergence\n");
    printf("   - Branch patterns include:\n");
    printf("     * if (threadIdx.x < 16)       - half warp divergence\n");
    printf("     * if (threadIdx.x %% 2 == 0) - even/odd split\n");
    printf("     * if (threadIdx.x %% 4 < 2)    - quarter divergence\n");
    printf("     * Complex nested patterns\n");
    printf("   - Shows exact divergence overhead in cycles\n");
    printf("\n4. MEMORY HIERARCHY:\n");
    printf("   - L1 cache (128 KB, 4-way) per SM\n");
    printf("   - L2 cache (6 MB, 16-way) with 32 banks\n");
    printf("   - GDDR6 with row buffer modeling\n");
    printf("   - Bank group timing (tCCDL/tCCDS)\n");
    printf("   - Bandwidth enforcement (696 GB/s total)\n");
    printf("\n5. EXECUTION UNITS:\n");
    printf("   - 128 FP32 units/SM (pipelined, 4-cycle latency)\n");
    printf("   - 128 INT32 units/SM (pipelined, 4-cycle latency)\n");
    printf("   - 32 SFU units/SM (pipelined, 8-16 cycle latency)\n");
    printf("   - 4 Tensor cores/SM (non-pipelined, 8-cycle latency)\n");
    printf("   - 32 LD/ST units/SM (pipelined)\n");
    printf("\n6. BANKING & CONFLICTS:\n");
    printf("   - 32 register file banks\n");
    printf("   - 32 shared memory banks\n");
    printf("   - 32 L2 banks with contention tracking\n");
    printf("\n========================================\n");
    printf("EDUCATIONAL INSIGHTS\n");
    printf("========================================\n");
    printf("\nThis simulator demonstrates THE THREE DOMINANT\n");
    printf("performance factors in modern GPU programming:\n");
    printf("\n1. MEMORY COALESCING (biggest impact):\n");
    printf("   Bad: array[threadIdx.x * 32] -> up to 32 sectors\n");
    printf("   Good: array[threadIdx.x]        -> 4 sectors for 32x4B\n");
    printf("   Timing impact requires memory-system calibration!\n");
    printf("\n2. DIVERGENCE (second biggest):\n");
    printf("   if (threadIdx.x < 16) { ... } else { ... }\n");
    printf("   -> Simplified divergence model schedules paths\n");
    printf("   Impact not calibrated to NVIDIA hardware\n");
    printf("\n3. INSTRUCTION MIX (optimization):\n");
    printf("   Interleave compute (FFMA) with memory (LD/ST)\n");
    printf("   -> Real GPUs may overlap pipelines; not modeled here\n");
    printf("   No speedup claim without hardware comparison\n");
    printf("\n========================================\n");
}

// ============================================================================
// MAIN
// ============================================================================

static void release_exec_unit(ExecUnit *u) { free(u->busy_until); u->busy_until = NULL; }
static void release_cache(u64 **tags, u8 **lru, bool **valid, int sets) {
    for (int i = 0; i < sets; ++i) { free(tags[i]); free(lru[i]); free(valid[i]); }
    free(tags); free(lru); free(valid);
}
static void gpu_destroy(GPU *gpu) {
    if (!gpu) return;
    for (int i = 0; i < g_sm_count; ++i) {
        SM *sm = &gpu->sms[i];
        release_exec_unit(&sm->fp32_unit); release_exec_unit(&sm->int32_unit);
        release_exec_unit(&sm->fp64_unit); release_exec_unit(&sm->sfu_unit);
        release_exec_unit(&sm->tensor_unit); release_exec_unit(&sm->ldst_unit);
        release_cache(sm->l1.tags, sm->l1.lru, sm->l1.valid, sm->l1.sets);
    }
    free(gpu->sms);
    release_cache(gpu->l2.tags, gpu->l2.lru, gpu->l2.valid, gpu->l2.sets);
    free(gpu->l2.bank_busy);
    free(gpu);
}

static void memory_probe(uint32_t base, uint32_t stride, uint32_t mask) {
    Instruction inst = {0}; inst.active_mask = mask;
    for (int t = 0; t < WARP_SIZE; ++t) {
        if (mask & (1u << t)) inst.thread_addrs[t] = (uint64_t)base + (uint64_t)t * stride;
    }
    CoalesceResult r = analyze_coalescing(&inst, 0);
    printf("PROBE base=%u stride=%u mask=0x%08x\n", base, stride, mask);
    printf("lane,address,sector32,line128\n");
    for (int t = 0; t < WARP_SIZE; ++t)
        if (mask & (1u << t))
            printf("%d,%llu,%llu,%llu\n", t, (unsigned long long)inst.thread_addrs[t],
              (unsigned long long)(inst.thread_addrs[t]/32),
              (unsigned long long)(inst.thread_addrs[t]/128));
    printf("sectors=%d; toy_minimal_sector_flag=%s\n", r.num_transactions,
           r.fully_coalesced?"yes":"no");
}

static void branch_probe(int predicate, uint32_t mask) {
    Instruction inst = {0}; inst.imm = (u32)predicate; inst.active_mask = mask;
    PredicateResult p = evaluate_predicate(&inst, mask);
    int taken = 0, active = 0;
    printf("BRANCH predicate=%d mask=0x%08x\n",predicate,mask);
    printf("lane,taken\n");
    for (int t = 0; t < WARP_SIZE; ++t) if (mask & (1u << t)) {
        int yes = inst.thread_preds[t] != 0;
        printf("%d,%d\n",t,yes); active++; taken += yes;
    }
    printf("active=%d; taken=%d; divergent=%s; predicate_result=%d\n",active,taken,
           p==PRED_DIVERGE?"yes":"no",p);
}

static void cache_probe(const char *list) {
    L1Cache cache={0}; l1_init(&cache);
    char *copy = malloc(strlen(list)+1);
    if (!copy) { perror("cache_probe malloc"); exit(3); }
    strcpy(copy,list);
    printf("L1 MODEL: 128-byte lines, %d ways, %d sets\n",cache.ways,cache.sets);
    printf("access,address,outcome,hit_count,miss_count\n");
    int num = 0;
    for(char *tok=strtok(copy,",");tok;tok=strtok(NULL,",")) {
        char *end=NULL; unsigned long long addr=strtoull(tok,&end,0);
        if (!tok[0]||!end||*end||num>1024) {
            fprintf(stderr,"Bad cache address in sequence\n"); free(copy); exit(2);
        }
        bool hit=l1_access(&cache,(u64)addr);
        printf("%d,%llu,%s,%llu,%llu\n",num++,addr,hit?"hit":"miss",
               (unsigned long long)cache.hits,(unsigned long long)cache.misses);
    }
    free(copy);
    release_cache(cache.tags,cache.lru,cache.valid,cache.sets);
}

static int read_int(const char *s, int lo, int hi) {
    char *end = NULL;
    long x = strtol(s, &end, 10);
    if (!s[0] || !end || *end || x < lo || x > hi) {
        fprintf(stderr, "Invalid integer: %s (expected %d..%d)\n", s, lo, hi);
        exit(2);
    }
    return (int)x;
}

static void usage(const char *name) {
    printf("Usage: %s [light|medium|heavy] [--sms 1..84] [--warps 1..48]\n"
           "       [--instructions 1..512] [--cycles 1..50000000]\n"
           "       [--pattern 0..5] [--workload mixed|memory|compute] [--quiet]\n"
           "Example: %s --sms 1 --warps 4 --instructions 48 --pattern 2 --workload memory\n"
           "       %s --probe --base 96 --stride 4 [--mask 0xffffffff]\n"
           "       %s --branch-probe --predicate 2 [--mask 0xffffffff]\n"
           "       %s --cache-seq 0,16384,32768,49152,0,65536,16384\n",
           name, name, name, name, name);
}

int main(int argc, char **argv) {
    int warps_per_sm = 16;
    bool detailed = false;
    bool probe_mode = false;
    bool branch_mode = false;
    const char *cache_sequence = NULL;
    int predicate = 0;
    int probe_base = 0, probe_stride = 4;
    u32 probe_mask = 0xffffffffu;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "light")) warps_per_sm = 16;
        else if (!strcmp(argv[i], "medium")) warps_per_sm = 32;
        else if (!strcmp(argv[i], "heavy")) warps_per_sm = 48;
        else if (!strcmp(argv[i], "detailed")) detailed = true;
        else if (!strcmp(argv[i], "--quiet")) g_quiet = true;
        else if (!strcmp(argv[i], "--probe")) probe_mode = true;
        else if (!strcmp(argv[i], "--branch-probe")) branch_mode = true;
        else if (i+1 < argc && !strcmp(argv[i], "--predicate")) predicate = read_int(argv[++i],0,7);
        else if (i+1 < argc && !strcmp(argv[i], "--cache-seq")) cache_sequence = argv[++i];
        else if (i+1 < argc && !strcmp(argv[i], "--base")) probe_base = read_int(argv[++i],0,65536);
        else if (i+1 < argc && !strcmp(argv[i], "--stride")) probe_stride = read_int(argv[++i],0,65536);
        else if (i+1 < argc && !strcmp(argv[i], "--mask")) {
            char *end = NULL; unsigned long v = strtoul(argv[++i], &end, 0);
            if (!end || *end || v > UINT32_MAX) { fprintf(stderr,"Bad mask\n"); return 2; }
            probe_mask = (u32)v;
        }
        else if (i+1 < argc && !strcmp(argv[i], "--sms")) g_sm_count = read_int(argv[++i],1,SM_COUNT);
        else if (i+1 < argc && !strcmp(argv[i], "--warps")) warps_per_sm = read_int(argv[++i],1,MAX_WARPS_PER_SM);
        else if (i+1 < argc && !strcmp(argv[i], "--instructions")) g_max_instructions = read_int(argv[++i],1,MAX_INSTRUCTIONS_PER_WARP);
        else if (i+1 < argc && !strcmp(argv[i], "--cycles")) g_cycle_limit = (uint64_t)read_int(argv[++i],1,SIM_CYCLE_LIMIT);
        else if (i+1 < argc && !strcmp(argv[i], "--pattern")) g_pattern = read_int(argv[++i],0,5);
        else if (i+1 < argc && !strcmp(argv[i], "--workload")) {
            const char *kind = argv[++i];
            if (!strcmp(kind,"mixed")) g_workload = 0;
            else if (!strcmp(kind,"memory")) g_workload = 1;
            else if (!strcmp(kind,"compute")) g_workload = 2;
            else { fprintf(stderr,"Unknown workload: %s\n",kind); return 2; }
        }
        else { fprintf(stderr,"Unknown argument: %s\n",argv[i]); usage(argv[0]); return 2; }
    }
    if (probe_mode) { memory_probe((u32)probe_base,(u32)probe_stride,probe_mask); return 0; }
    if (branch_mode) { branch_probe(predicate,probe_mask); return 0; }
    if (cache_sequence) { cache_probe(cache_sequence); return 0; }
    printf("A40-inspired executable model (NOT an accurate hardware emulator)\n");
    printf("Configuration: SMs=%d, warps/SM=%d, instructions/warp=%d, cycles_limit=%llu, pattern=%d, workload=%d\n",
           g_sm_count,warps_per_sm,g_max_instructions,(unsigned long long)g_cycle_limit,g_pattern,g_workload);
    // GPU is tens of MB: never allocate it on the process's limited stack.
    GPU *gpu = calloc(1, sizeof(*gpu));
    if (!gpu) { perror("calloc GPU"); return 3; }
    gpu_init(gpu, warps_per_sm);
    uint64_t last_print = 0;
    while (gpu_is_active(gpu) && gpu->cycle < g_cycle_limit) {
        gpu_step(gpu);
        if (!g_quiet && gpu->cycle - last_print >= STAT_PRINT_INTERVAL) {
            printf("[Cycle %llu] Instructions: %llu, IPC: %.3f\n",
                   (unsigned long long)gpu->cycle,(unsigned long long)gpu->total_instructions,
                   gpu->cycle ? (double)gpu->total_instructions/gpu->cycle : 0.0);
            last_print = gpu->cycle;
        }
    }
    bool finished = !gpu_is_active(gpu);
    printf("Termination: %s at cycle %llu\n", finished?"complete":"cycle limit",
           (unsigned long long)gpu->cycle);
    gpu_print_stats(gpu);
    if (detailed) gpu_print_detailed_stats(gpu);
    gpu_destroy(gpu);
    return finished ? 0 : 4;
}
