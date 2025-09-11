#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <random> 


extern std::mutex MAIN_PRNG_MUTEX;

unsigned long long get_int_seed();

unsigned int generate_custom_seed(); 


class MT19937 {
public:
    MT19937(uint32_t seed); 

    uint32_t extract(); 

private:
    static const int w = 32;
    static const int n = 624;
    static const int m = 397;
    static const int r = 31;
    static const uint32_t a = 0x9908B0DF;
    static const int u = 11;
    static const uint32_t d = 0xFFFFFFFF;
    static const int s = 7;
    static const uint32_t b = 0x9D2C5680;
    static const int t = 15;
    static const uint32_t c = 0xEFC60000;
    static const int l = 18;
    static const uint32_t f = 1812433253;

    uint32_t state[n];
    int index;

    void twist(); 
};


class LCG {
public:
    LCG(uint32_t seed);

    uint32_t next(); 

    void setSeed(uint32_t seed); 

private:
    uint32_t state;
    static constexpr uint32_t a = 1664525; // Multiplier
    static constexpr uint32_t c = 1013904223; // Increment
    static constexpr uint32_t m = 0xFFFFFFFF; // Modulus (2^32 - 1)
};


class PhiloxRNG {
public:
    using uint32 = uint32_t;
    using uint64 = uint64_t;

    PhiloxRNG(uint64 seed1, uint64 seed2);

    std::array<uint32, 4> operator()(); 

private:
    uint32 counter = 0;
    uint32 key[4];
    static constexpr int rounds = 10;

    std::array<uint32, 4> singleRound(const std::array<uint32, 4>& ctr) const; 
};


extern "C" float print_randoms(float N, float std); 

extern LCG rng;

extern std::random_device rd2; // it is already defined at cu_common.h
extern std::mt19937 MAIN_PRNG;



unsigned long long time_seed(); 