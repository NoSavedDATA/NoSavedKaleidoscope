#pragma once






// Gets the 12 rightmost bits from a 16 bits value inside a uint64_t vec
inline uint16_t get_16_r12(const uint64_t* base, int idx) {
    int per_word = 4; // 64 / 16;
    int w = idx / per_word;
    int o = (idx % per_word) * 16;

    // 0xFFF gets the first 12 bits
    return (base[w] >> o) & 0xFFF;
}

inline void set_16_r12(uint64_t* base, int idx, uint16_t value) {
    value &= 0xFFF;             // keep 12 bits

    int per_word = 4; // 64 / 16;
    int w = idx / per_word;
    int o = (idx % per_word) * 16;

    uint64_t mask = (uint64_t)0xFFFF << o;
    base[w] = (base[w] & ~mask) | ((uint64_t)value << o);
}

// Set rightmost 12 and also set as marked
inline void set_16_r12_mark(uint64_t* base, int idx, uint16_t value) {
    value &= 0xFFF;             // keep 12 bits
    value |= (1u << 15);        // mark leftmost

    int per_word = 4; // 64 / 16;
    int w = idx / per_word;
    int o = (idx % per_word) * 16;

    uint64_t mask = (uint64_t)0xFFFF << o;
    base[w] = (base[w] & ~mask) | ((uint64_t)value << o);
}


// get only 1st leftmost bit inside u16
inline uint16_t get_16_L1(const uint64_t* base, int idx) {
    int per_word = 4; // 64 / 16;
    int w = idx / per_word;
    int o = (idx % per_word) * 16;

    return (base[w] >> (o+15)) & 1;
}

inline void set_16_L1(uint64_t* base, int idx, uint16_t val) {
    int per_word = 4; // 64 / 16;
    int w = idx / per_word;
    int o = (idx % per_word) * 16;
 
    uint64_t mask = (uint64_t)1 << (o+15);
    base[w] = (base[w] & ~mask) | ((uint64_t)val << (o+15));
}


inline uint16_t get_16_L2(const uint64_t* base, int idx) {
    int per_word = 4; // 64 / 16;
    int w = idx / per_word;
    int o = (idx % per_word) * 16;

    return (base[w] >> (o+14)) & 1;
}

inline void set_16_L2(uint64_t* base, int idx, uint16_t val) {
    int per_word = 4; // 64 / 16;
    int w = idx / per_word;
    int o = (idx % per_word) * 16;

    uint64_t mask = 1 << (o+14);
    base[w] = (base[w] & ~mask) | ((uint64_t)val << (o+14));
}


inline uint16_t get_16_l2(const uint64_t* base, int idx) {
    int per_word = 4; // 64 / 16;
    int w = idx / per_word;
    int o = (idx % per_word) * 16;

    return (base[w] >> (o+14)) & 3;
}


// Find first uint16 that has the 2 leftmost bits set to 0
inline int find_free_16_l2(uint64_t* buf, int words) {
    const int per = 4; // 4 x 16 bits inside each int64;
    for (int w = 0; w < words; ++w) {
        if (~buf[w]) {
            for (int i = 0; i < per; ++i) {
                int off = i * 16;
                uint64_t slot = (buf[w] >> (off+14)) & 3;
                // std::cout << "Got slot: " << slot << " for w=" << w << ", b=" << i << ".\n";
                if (slot == 0)      // free = zero
                    return w * per + i;
            }
        }
    }
    return -1;
}






inline int find_free(uint64_t *mark_bits, const int words) {
    for (size_t w = 0; w < words; ++w) {
        if (~mark_bits[w]) { // if any bit is 0
            for (size_t b = 0; b < 64; ++b) {
                if (!(mark_bits[w] & (1ULL << b)))
                    return w * 64 + b;
            }
        }
    }
    return -1; // no free slot
}

inline void mark_bits_alloc(uint64_t *mark_bits, const int idx) {
    size_t w = idx / 64;
    size_t b = idx % 64;
    mark_bits[w] |= (1ULL << b);
}

// Unmark (free) a slot
inline void mark_bits_free(uint64_t *mark_bits, const int idx) {
    size_t w = idx / 64;
    size_t b = idx % 64;
    mark_bits[w] &= ~(1ULL << b);
}


