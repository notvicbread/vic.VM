#pragma once

#include "platform.h"
#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <atomic>
#include <unordered_map>
#include <fstream>
#include <string>

// ------------------------------------------------------------------
// Memory – guest physical RAM with MMU (paging, disk backing)
// ------------------------------------------------------------------
class Memory {
public:
    explicit Memory(uint64_t size, const char* backing_file = nullptr);
    ~Memory();

    uint8_t* ptr(uint64_t addr);          // direct pointer (may trigger page fault)
    uint8_t  read8(uint64_t addr);
    uint16_t read16(uint64_t addr);
    uint32_t read32(uint64_t addr);
    uint64_t read64(uint64_t addr);
    void write8(uint64_t addr, uint8_t  val);
    void write16(uint64_t addr, uint16_t val);
    void write32(uint64_t addr, uint32_t val);
    void write64(uint64_t addr, uint64_t val);

    bool load_image(const char* path, uint64_t base_addr = 0);
    void register_mmio(uint64_t start, uint64_t end,
                       std::function<uint64_t(uint64_t)> read_handler,
                       std::function<void(uint64_t,uint64_t)> write_handler);

private:
    static constexpr uint64_t PAGE_SIZE = 4096;
    struct Page {
        uint8_t data[PAGE_SIZE];
        bool present;
        bool dirty;
    };
    std::unordered_map<uint64_t, Page*> page_table_;  // virtual page -> physical page
    std::vector<uint8_t> ram_;                         // for small, fast allocations
    std::string backing_file_;
    mutable std::fstream backing_stream_;              // opened on demand
    std::mutex mtx_;

    struct MMIOEntry {
        uint64_t start, end;
        std::function<uint64_t(uint64_t)> read;
        std::function<void(uint64_t,uint64_t)> write;
    };
    std::vector<MMIOEntry> mmio_;

    Page* get_page(uint64_t vaddr, bool allocate);
    void flush_page(uint64_t vaddr, Page* page);
};

// ------------------------------------------------------------------
// TPU – Titan Processing Unit (forward declaration for microcode)
// ------------------------------------------------------------------
class TPU;

// ------------------------------------------------------------------
// MicrocodeEngine – defines new instructions
// ------------------------------------------------------------------
class MicrocodeEngine {
public:
    MicrocodeEngine(Memory* mem);
    ~MicrocodeEngine();

    bool define_instruction(uint64_t opcode, uint64_t microcode_addr, size_t length);
    bool execute(uint64_t opcode, TPU* tpu, uint32_t core_idx); // returns true if handled

private:
    struct Microcode {
        uint64_t addr;
        size_t length;
        void* jit_code; // for compiled version
    };
    std::unordered_map<uint64_t, Microcode> microcodes_;
    std::mutex mtx_;
    Memory* mem_;
};

// ------------------------------------------------------------------
// TPU – Titan Processing Unit (AI-optimized, massively parallel)
// ------------------------------------------------------------------
class TPU {
public:
    explicit TPU(Memory* mem, uint32_t num_cores);
    ~TPU();

    void mmio_write(uint64_t offset, uint64_t value);
    uint64_t mmio_read(uint64_t offset);
    bool has_interrupt() const { return pending_; }
    uint8_t interrupt_vector() const { return 0x20; }
    void ack_interrupt() { pending_ = false; }

    // For microcode engine
    void execute_builtin(uint32_t core_idx, uint64_t opcode);

private:
    static constexpr uint32_t REGS_PER_CORE = 64;

    struct Core {
        uint64_t pc;
        uint64_t regs[REGS_PER_CORE];
        bool halted;
    };
    std::vector<Core> cores_;
    std::atomic<bool> stop_;
    std::vector<thread_t> workers_;

    Memory* mem_;
    uint64_t ring_base_, ring_size_, head_, tail_;
    bool pending_;
    mutable std::mutex mtx_;

    // JIT cache
    struct JITEntry {
        void* code;
        size_t size;
    };
    std::unordered_map<uint64_t, JITEntry> jit_cache_;
    std::mutex jit_mtx_;

#ifdef ENABLE_MICROCODE
    MicrocodeEngine microcode_engine_;
#endif

    void worker_loop(uint32_t core_idx);
    void decode_and_exec(uint32_t core_idx, uint64_t opcode);
    void jit_compile(uint64_t start_pc);
    void emit_prologue(uint8_t*& p);
    void emit_epilogue(uint8_t*& p);

    // Basic instructions
    void i_mov(Core& core, uint32_t dst, uint64_t imm);
    void i_add(Core& core, uint32_t dst, uint32_t src1, uint32_t src2);
    void i_sub(Core& core, uint32_t dst, uint32_t src1, uint32_t src2);
    void i_mul(Core& core, uint32_t dst, uint32_t src1, uint32_t src2);
    void i_div(Core& core, uint32_t dst, uint32_t src1, uint32_t src2);
    void i_load(Core& core, uint32_t dst, uint32_t base, uint32_t offset);
    void i_store(Core& core, uint32_t src, uint32_t base, uint32_t offset);
    void i_jmp(Core& core, uint64_t target);
    void i_jz(Core& core, uint64_t target, uint32_t cond_reg);
    void i_hlt(Core& core);

#ifdef ENABLE_AI
    // AI tensor instructions
    void i_tensor_matmul(Core& core, uint32_t dst, uint32_t srcA, uint32_t srcB, uint64_t dims);
    void i_tensor_add(Core& core, uint32_t dst, uint32_t srcA, uint32_t srcB);
    void i_tensor_relu(Core& core, uint32_t dst, uint32_t src);
    void i_softmax(Core& core, uint32_t dst, uint32_t src, uint64_t dims);
    void i_attention(Core& core, uint32_t dst, uint32_t q, uint32_t k, uint32_t v, uint64_t dims);
#endif
};

// ------------------------------------------------------------------
// CPU – simple boot core (just to launch TPU)
// ------------------------------------------------------------------
class CPU {
public:
    static std::unique_ptr<CPU> create(Memory* mem);
    virtual ~CPU() = default;
    virtual bool step() = 0;
    virtual void inject_interrupt(uint8_t vec) = 0;
};

class InterpreterCPU : public CPU {
public:
    explicit InterpreterCPU(Memory* mem);
    bool step() override;
    void inject_interrupt(uint8_t vec) override;
private:
    Memory* mem_;
    uint64_t regs_[16], rip_;
    bool irq_pending_;
    uint8_t irq_vector_;
};

// ------------------------------------------------------------------
// VM – main machine
// ------------------------------------------------------------------
class VM {
public:
    VM();
    ~VM();
    bool init(uint64_t ram_size, const char* image, uint32_t tpu_cores, const char* backing = nullptr);
    void run();
    void stop() { running_ = false; }

private:
    std::unique_ptr<CPU> cpu_;
    std::unique_ptr<Memory> mem_;
    std::unique_ptr<TPU> tpu_;
    uint64_t ticks_;
    bool running_;
    std::mutex irq_lock_;
    void check_interrupts();
};
