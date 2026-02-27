#include "vm.h"
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include <iostream>
#include <cmath>
#include <sys/mman.h>

// ------------------------------------------------------------------
// Memory implementation with paging and disk backing
// ------------------------------------------------------------------
Memory::Memory(uint64_t size, const char* backing_file) {
    if (backing_file) {
        backing_file_ = backing_file;
        // backing_stream_ opened lazily
    } else {
        ram_.resize(size);
    }
}

Memory::~Memory() {
    for (auto& p : page_table_) {
        if (p.second->dirty && !backing_file_.empty()) {
            flush_page(p.first, p.second);
        }
        delete p.second;
    }
}

Memory::Page* Memory::get_page(uint64_t vaddr, bool allocate) {
    uint64_t page_num = vaddr / PAGE_SIZE;
    auto it = page_table_.find(page_num);
    if (it != page_table_.end()) return it->second;

    if (!allocate) return nullptr;

    Page* page = new Page();
    page->present = true;
    page->dirty = false;

    if (!backing_file_.empty()) {
        // Lazy open backing stream
        if (!backing_stream_.is_open()) {
            backing_stream_.open(backing_file_, std::ios::binary | std::ios::in | std::ios::out);
            if (!backing_stream_) {
                // Create file
                backing_stream_.open(backing_file_, std::ios::binary | std::ios::out);
                backing_stream_.close();
                backing_stream_.open(backing_file_, std::ios::binary | std::ios::in | std::ios::out);
            }
        }
        backing_stream_.seekg(page_num * PAGE_SIZE);
        backing_stream_.read((char*)page->data, PAGE_SIZE);
        if (backing_stream_.gcount() < (long)PAGE_SIZE) {
            memset(page->data, 0, PAGE_SIZE);
        }
    } else {
        memset(page->data, 0, PAGE_SIZE);
    }
    page_table_[page_num] = page;
    return page;
}

void Memory::flush_page(uint64_t vaddr, Page* page) {
    if (!page->dirty || backing_file_.empty()) return;
    uint64_t page_num = vaddr / PAGE_SIZE;
    if (!backing_stream_.is_open()) return;
    backing_stream_.seekp(page_num * PAGE_SIZE);
    backing_stream_.write((char*)page->data, PAGE_SIZE);
    backing_stream_.flush();
    page->dirty = false;
}

uint8_t* Memory::ptr(uint64_t addr) {
    if (addr < ram_.size()) return &ram_[addr];
    std::lock_guard<std::mutex> lock(mtx_);
    Page* page = get_page(addr, true);
    return page->data + (addr % PAGE_SIZE);
}

uint8_t Memory::read8(uint64_t addr) {
    if (addr < ram_.size()) return ram_[addr];
    for (auto& m : mmio_)
        if (addr >= m.start && addr < m.end)
            return (uint8_t)m.read(addr - m.start);
    std::lock_guard<std::mutex> lock(mtx_);
    Page* page = get_page(addr, false);
    return page ? page->data[addr % PAGE_SIZE] : 0;
}

uint16_t Memory::read16(uint64_t addr) {
    if (addr+1 < ram_.size()) return *(uint16_t*)&ram_[addr];
    uint16_t val = 0;
    for (int i = 0; i < 2; i++) val |= (uint16_t)read8(addr+i) << (i*8);
    return val;
}
uint32_t Memory::read32(uint64_t addr) {
    if (addr+3 < ram_.size()) return *(uint32_t*)&ram_[addr];
    uint32_t val = 0;
    for (int i = 0; i < 4; i++) val |= (uint32_t)read8(addr+i) << (i*8);
    return val;
}
uint64_t Memory::read64(uint64_t addr) {
    if (addr+7 < ram_.size()) return *(uint64_t*)&ram_[addr];
    uint64_t val = 0;
    for (int i = 0; i < 8; i++) val |= (uint64_t)read8(addr+i) << (i*8);
    return val;
}

void Memory::write8(uint64_t addr, uint8_t val) {
    if (addr < ram_.size()) { ram_[addr] = val; return; }
    for (auto& m : mmio_)
        if (addr >= m.start && addr < m.end) {
            m.write(addr - m.start, val);
            return;
        }
    std::lock_guard<std::mutex> lock(mtx_);
    Page* page = get_page(addr, true);
    page->data[addr % PAGE_SIZE] = val;
    page->dirty = true;
}

void Memory::write16(uint64_t addr, uint16_t val) {
    if (addr+1 < ram_.size()) { *(uint16_t*)&ram_[addr] = val; return; }
    for (int i = 0; i < 2; i++) write8(addr+i, (val >> (i*8)) & 0xFF);
}
void Memory::write32(uint64_t addr, uint32_t val) {
    if (addr+3 < ram_.size()) { *(uint32_t*)&ram_[addr] = val; return; }
    for (int i = 0; i < 4; i++) write8(addr+i, (val >> (i*8)) & 0xFF);
}
void Memory::write64(uint64_t addr, uint64_t val) {
    if (addr+7 < ram_.size()) { *(uint64_t*)&ram_[addr] = val; return; }
    for (int i = 0; i < 8; i++) write8(addr+i, (val >> (i*8)) & 0xFF);
}

void Memory::register_mmio(uint64_t start, uint64_t end,
                           std::function<uint64_t(uint64_t)> read_handler,
                           std::function<void(uint64_t,uint64_t)> write_handler) {
    mmio_.push_back({start, end, read_handler, write_handler});
}

bool Memory::load_image(const char* path, uint64_t base_addr) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    size_t len = f.tellg();
    f.seekg(0);
    for (size_t i = 0; i < len; i++) {
        uint8_t byte;
        f.read((char*)&byte, 1);
        write8(base_addr + i, byte);
    }
    return true;
}

// ------------------------------------------------------------------
// MicrocodeEngine implementation
// ------------------------------------------------------------------
#ifdef ENABLE_MICROCODE

MicrocodeEngine::MicrocodeEngine(Memory* mem) : mem_(mem) {}
MicrocodeEngine::~MicrocodeEngine() {
    for (auto& mc : microcodes_) {
        if (mc.second.jit_code) munmap(mc.second.jit_code, 4096);
    }
}

bool MicrocodeEngine::define_instruction(uint64_t opcode, uint64_t microcode_addr, size_t length) {
    std::lock_guard<std::mutex> lock(mtx_);
    microcodes_[opcode] = {microcode_addr, length, nullptr};
    return true;
}

bool MicrocodeEngine::execute(uint64_t opcode, TPU* tpu, uint32_t core_idx) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = microcodes_.find(opcode);
    if (it == microcodes_.end()) return false;

    Microcode& mc = it->second;
    if (mc.jit_code) {
        // Call JITted code (simplified; would need to pass core context)
        // For now, fallback to interpretation.
        // In production, you'd have a proper JIT for microcode.
    }

    // Interpret microcode: a sequence of TPU instructions
    TPU::Core* core = nullptr; // We'd need to get the core from tpu; but that's private.
    // This requires a redesign or friendship. For brevity, we'll note that
    // in a full implementation, we'd have a way to execute instructions on a core.
    // Here we skip details.
    return true;
}

#endif

// ------------------------------------------------------------------
// TPU implementation
// ------------------------------------------------------------------
TPU::TPU(Memory* mem, uint32_t num_cores) : mem_(mem), pending_(false)
#ifdef ENABLE_MICROCODE
    , microcode_engine_(mem)
#endif
{
    cores_.resize(num_cores);
    for (auto& c : cores_) {
        c.pc = 0;
        memset(c.regs, 0, sizeof(c.regs));
        c.halted = false;
    }
    stop_ = false;
    for (uint32_t i = 0; i < num_cores; ++i) {
        thread_t t;
        thread_create(&t, [](void* arg) -> thread_return_t {
            TPU* tpu = (TPU*)arg;
            uint32_t idx = (uintptr_t)arg >> 32; // pass index in high bits
            tpu->worker_loop(idx);
            return 0;
        }, (void*)((uintptr_t)i << 32 | (uintptr_t)this));
        workers_.push_back(t);
    }

    mem_->register_mmio(0xF0000000, 0xF000FFFF,
        [this](uint64_t offset) { return this->mmio_read(offset); },
        [this](uint64_t offset, uint64_t val) { this->mmio_write(offset, val); });
}

TPU::~TPU() {
    stop_ = true;
    for (auto& t : workers_) thread_join(t);
    for (auto& e : jit_cache_) {
        munmap(e.second.code, e.second.size);
    }
}

void TPU::mmio_write(uint64_t offset, uint64_t value) {
    std::lock_guard<std::mutex> lock(mtx_);
    switch (offset) {
        case 0x00: ring_base_ = value; break;
        case 0x08: ring_size_ = value; break;
        case 0x10: head_ = value; break;
        case 0x18:
            tail_ = value;
            for (auto& core : cores_) core.halted = false;
            pending_ = true;
            break;
        case 0x30: { // DEFINE_INSTRUCTION (microcode)
            uint64_t new_op = mem_->read64(ring_base_ + head_); head_ += 8;
            uint64_t micro_addr = mem_->read64(ring_base_ + head_); head_ += 8;
            uint64_t micro_len = mem_->read64(ring_base_ + head_); head_ += 8;
#ifdef ENABLE_MICROCODE
            microcode_engine_.define_instruction(new_op, micro_addr, micro_len);
#endif
            break;
        }
    }
}

uint64_t TPU::mmio_read(uint64_t offset) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (offset == 0x20) return head_;
    return 0;
}

void TPU::worker_loop(uint32_t core_idx) {
    Core& core = cores_[core_idx];
    while (!stop_) {
        if (!core.halted) {
            uint64_t opcode = mem_->read64(core.pc);
            core.pc += 8;
            decode_and_exec(core_idx, opcode);
        } else {
            thread_yield();
        }
    }
}

void TPU::decode_and_exec(uint32_t core_idx, uint64_t opcode) {
    Core& core = cores_[core_idx];
    uint8_t family = (opcode >> 56) & 0xFF;
    uint32_t dst   = (opcode >> 48) & 0xFFFF;
    uint32_t src1  = (opcode >> 32) & 0xFFFF;
    uint32_t src2  = (opcode >> 16) & 0xFFFF;
    uint64_t imm   = opcode & 0xFFFF;

    switch (family) {
        case 0x00: i_mov(core, dst, imm); break;
        case 0x01: i_add(core, dst, src1, src2); break;
        case 0x02: i_sub(core, dst, src1, src2); break;
        case 0x03: i_mul(core, dst, src1, src2); break;
        case 0x04: i_div(core, dst, src1, src2); break;
        case 0x05: i_load(core, dst, src1, src2); break;
        case 0x06: i_store(core, dst, src1, src2); break;
        case 0x07: i_jmp(core, imm); break;
        case 0x08: i_jz(core, imm, dst); break;
        case 0xFF: i_hlt(core); break;
#ifdef ENABLE_AI
        case 0x10: i_tensor_matmul(core, dst, src1, src2, imm); break;
        case 0x11: i_tensor_add(core, dst, src1, src2); break;
        case 0x12: i_tensor_relu(core, dst, src1); break;
        case 0x13: i_softmax(core, dst, src1, imm); break;
        case 0x14: i_attention(core, dst, src1, src2, src1, imm); break; // simplified
#endif
        default:
#ifdef ENABLE_MICROCODE
            if (!microcode_engine_.execute(opcode, this, core_idx)) {
                core.halted = true;
            }
#else
            core.halted = true;
#endif
            break;
    }
}

// Basic instructions
void TPU::i_mov(Core& core, uint32_t dst, uint64_t imm) {
    if (dst < REGS_PER_CORE) core.regs[dst] = imm;
}
void TPU::i_add(Core& core, uint32_t dst, uint32_t src1, uint32_t src2) {
    if (dst < REGS_PER_CORE && src1 < REGS_PER_CORE && src2 < REGS_PER_CORE)
        core.regs[dst] = core.regs[src1] + core.regs[src2];
}
void TPU::i_sub(Core& core, uint32_t dst, uint32_t src1, uint32_t src2) {
    if (dst < REGS_PER_CORE && src1 < REGS_PER_CORE && src2 < REGS_PER_CORE)
        core.regs[dst] = core.regs[src1] - core.regs[src2];
}
void TPU::i_mul(Core& core, uint32_t dst, uint32_t src1, uint32_t src2) {
    if (dst < REGS_PER_CORE && src1 < REGS_PER_CORE && src2 < REGS_PER_CORE)
        core.regs[dst] = core.regs[src1] * core.regs[src2];
}
void TPU::i_div(Core& core, uint32_t dst, uint32_t src1, uint32_t src2) {
    if (dst < REGS_PER_CORE && src1 < REGS_PER_CORE && src2 < REGS_PER_CORE && core.regs[src2] != 0)
        core.regs[dst] = core.regs[src1] / core.regs[src2];
}
void TPU::i_load(Core& core, uint32_t dst, uint32_t base, uint32_t offset) {
    if (dst < REGS_PER_CORE && base < REGS_PER_CORE) {
        uint64_t addr = core.regs[base] + offset;
        core.regs[dst] = mem_->read64(addr);
    }
}
void TPU::i_store(Core& core, uint32_t src, uint32_t base, uint32_t offset) {
    if (src < REGS_PER_CORE && base < REGS_PER_CORE) {
        uint64_t addr = core.regs[base] + offset;
        mem_->write64(addr, core.regs[src]);
    }
}
void TPU::i_jmp(Core& core, uint64_t target) {
    core.pc = target;
}
void TPU::i_jz(Core& core, uint64_t target, uint32_t cond_reg) {
    if (cond_reg < REGS_PER_CORE && core.regs[cond_reg] == 0)
        core.pc = target;
}
void TPU::i_hlt(Core& core) {
    core.halted = true;
}

#ifdef ENABLE_AI
// AI tensor instructions (call assembly)
extern "C" {
    void tensor_matmul_avx512(float* A, float* B, float* C, uint32_t M, uint32_t N, uint32_t K);
    void tensor_add_avx512(float* A, float* B, float* C, uint32_t N);
    void tensor_relu_avx512(float* A, float* B, uint32_t N);
    void softmax_avx512(float* input, float* output, uint32_t N);
    void attention_avx512(float* Q, float* K, float* V, float* output, uint32_t seq_len, uint32_t head_dim);
}

void TPU::i_tensor_matmul(Core& core, uint32_t dst, uint32_t srcA, uint32_t srcB, uint64_t dims) {
    uint32_t M = (dims >> 48) & 0xFFFF;
    uint32_t N = (dims >> 32) & 0xFFFF;
    uint32_t K = (dims >> 16) & 0xFFFF;
    float* A = (float*)mem_->ptr(core.regs[srcA]);
    float* B = (float*)mem_->ptr(core.regs[srcB]);
    float* C = (float*)mem_->ptr(core.regs[dst]);
    if (A && B && C) tensor_matmul_avx512(A, B, C, M, N, K);
}
void TPU::i_tensor_add(Core& core, uint32_t dst, uint32_t srcA, uint32_t srcB) {
    float* A = (float*)mem_->ptr(core.regs[srcA]);
    float* B = (float*)mem_->ptr(core.regs[srcB]);
    float* C = (float*)mem_->ptr(core.regs[dst]);
    if (A && B && C) tensor_add_avx512(A, B, C, 1024); // length from somewhere?
}
void TPU::i_tensor_relu(Core& core, uint32_t dst, uint32_t src) {
    float* A = (float*)mem_->ptr(core.regs[src]);
    float* B = (float*)mem_->ptr(core.regs[dst]);
    if (A && B) tensor_relu_avx512(A, B, 1024);
}
void TPU::i_softmax(Core& core, uint32_t dst, uint32_t src, uint64_t dims) {
    uint32_t N = dims & 0xFFFF;
    float* input = (float*)mem_->ptr(core.regs[src]);
    float* output = (float*)mem_->ptr(core.regs[dst]);
    if (input && output) softmax_avx512(input, output, N);
}
void TPU::i_attention(Core& core, uint32_t dst, uint32_t q, uint32_t k, uint32_t v, uint64_t dims) {
    uint32_t seq_len = (dims >> 48) & 0xFFFF;
    uint32_t head_dim = (dims >> 32) & 0xFFFF;
    float* Q = (float*)mem_->ptr(core.regs[q]);
    float* K = (float*)mem_->ptr(core.regs[k]);
    float* V = (float*)mem_->ptr(core.regs[v]);
    float* out = (float*)mem_->ptr(core.regs[dst]);
    if (Q && K && V && out) attention_avx512(Q, K, V, out, seq_len, head_dim);
}
#endif

// JIT stubs
void TPU::emit_prologue(uint8_t*& p) {
    *p++ = 0x55;                     // push rbp
    *p++ = 0x48; *p++ = 0x89; *p++ = 0xE5; // mov rbp, rsp
    *p++ = 0x48; *p++ = 0x83; *p++ = 0xEC; *p++ = 0x20; // sub rsp, 32
}
void TPU::emit_epilogue(uint8_t*& p) {
    *p++ = 0x48; *p++ = 0x89; *p++ = 0xEC; // mov rsp, rbp
    *p++ = 0x5D;                            // pop rbp
    *p++ = 0xC3;                            // ret
}
void TPU::jit_compile(uint64_t start_pc) {
    std::lock_guard<std::mutex> lock(jit_mtx_);
    if (jit_cache_.find(start_pc) != jit_cache_.end()) return;
    size_t size = 4096;
    void* mem = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) return;
    uint8_t* p = (uint8_t*)mem;
    emit_prologue(p);
    *p++ = 0xB8; *p++ = 0x2A; *p++ = 0x00; *p++ = 0x00; *p++ = 0x00; // mov eax, 42
    emit_epilogue(p);
    jit_cache_[start_pc] = {mem, size};
}

// ------------------------------------------------------------------
// InterpreterCPU
// ------------------------------------------------------------------
InterpreterCPU::InterpreterCPU(Memory* mem) : mem_(mem), rip_(0), irq_pending_(false) {
    memset(regs_, 0, sizeof(regs_));
}

void InterpreterCPU::inject_interrupt(uint8_t vec) {
    irq_pending_ = true;
    irq_vector_ = vec;
}

bool InterpreterCPU::step() {
    if (irq_pending_) {
        irq_pending_ = false;
    }
    uint8_t op = mem_->read8(rip_++);
    switch (op) {
        case 0x00: { // MOV reg, imm64
            uint8_t r = mem_->read8(rip_++);
            uint64_t imm = mem_->read64(rip_); rip_ += 8;
            regs_[r] = imm;
            break;
        }
        case 0x01: { // ADD reg, reg
            uint8_t dst = mem_->read8(rip_++);
            uint8_t src = mem_->read8(rip_++);
            regs_[dst] += regs_[src];
            break;
        }
        case 0xFF: // HLT
            return false;
        default:
            return false;
    }
    return true;
}

std::unique_ptr<CPU> CPU::create(Memory* mem) {
    return std::make_unique<InterpreterCPU>(mem);
}

// ------------------------------------------------------------------
// VM
// ------------------------------------------------------------------
VM::VM() : ticks_(0), running_(false) {}
VM::~VM() = default;

bool VM::init(uint64_t ram_size, const char* image, uint32_t tpu_cores, const char* backing) {
    mem_ = std::make_unique<Memory>(ram_size, backing);
    if (image && !mem_->load_image(image)) return false;
    cpu_ = CPU::create(mem_.get());
    tpu_ = std::make_unique<TPU>(mem_.get(), tpu_cores);
    return true;
}

void VM::check_interrupts() {
    std::lock_guard<std::mutex> lock(irq_lock_);
    if (tpu_ && tpu_->has_interrupt()) {
        cpu_->inject_interrupt(tpu_->interrupt_vector());
        tpu_->ack_interrupt();
    }
}

void VM::run() {
    running_ = true;
    auto last_time = std::chrono::high_resolution_clock::now();

    while (running_) {
        if (!cpu_->step()) break; // CPU halted

        ticks_++;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_time).count();
        if (elapsed >= 1000) { // 1 ms tick
            last_time = now;
        }
        check_interrupts();
    }
}
