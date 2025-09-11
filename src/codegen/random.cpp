#include <climits>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdint.h>
#include <thread>
#include "../common/extension_functions.h"
#include "include.h"


std::random_device rd2; // it is already defined at cu_common.h
std::mt19937 MAIN_PRNG(rd2()^get_millisecond_time());
std::mutex MAIN_PRNG_MUTEX;


unsigned long long get_int_seed()
{
  std::uniform_int_distribution<unsigned long long> dist(0, ULLONG_MAX);
  return dist(MAIN_PRNG);
}

unsigned int generate_custom_seed() {
    // Combine time, process ID, and thread ID to generate a seed
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    unsigned int nanoseconds = get_millisecond_time();

    unsigned int tid = std::hash<std::thread::id>{}(std::this_thread::get_id());


    unsigned int seed = nanoseconds ^ tid;
    return seed;
}



MT19937::MT19937(uint32_t seed) {
    // Initialize the state vector
    state[0] = seed;
    for (int i = 1; i < n; i++) {
        state[i] = f * (state[i - 1] ^ (state[i - 1] >> (w - 2))) + i;
    }
    index = n;
}
    
uint32_t MT19937::extract() {
    if (index >= n) {
        twist();
    }

    uint32_t y = state[index++];
    // Tempering
    y ^= (y >> u);
    y ^= (y << s) & b;
    y ^= (y << t) & c;
    y ^= (y >> l);

    return y;
}

void MT19937::twist() {
    for (int i = 0; i < n; i++) {
        uint32_t x = (state[i] & 0x80000000) + (state[(i + 1) % n] & 0x7FFFFFFF);
        uint32_t xA = x >> 1;
        if (x % 2 != 0) {
            xA ^= a;
        }
        state[i] = state[(i + m) % n] ^ xA;
    }
    index = 0;
}
    
    
LCG::LCG(uint32_t seed) : state(seed) {}
    
uint32_t LCG::next() {
    state = (a * state + c) % m;
    return state;
}
    
void LCG::setSeed(uint32_t seed) {
    state = seed;
}



PhiloxRNG::PhiloxRNG(uint64 seed1, uint64 seed2) : key{ static_cast<uint32>(seed1), static_cast<uint32>(seed1 >> 32),
                                                static_cast<uint32>(seed2), static_cast<uint32>(seed2 >> 32) } {}

std::array<uint32, 4> PhiloxRNG::operator()() {
    std::array<uint32, 4> ctr = { counter++, 0, 0, 0 };
    for (int i = 0; i < rounds; i++) {
        ctr = singleRound(ctr);
    }
    return ctr;
}


std::array<uint32, 4> PhiloxRNG::singleRound(const std::array<uint32, 4>& ctr) const {
    std::array<uint32, 4> output;
    const uint64 mult = 0xD2511F53;
    const uint64 add = 0xCD9E8D57;

    uint64 x = static_cast<uint64>(ctr[0]) * mult;
    uint64 y = static_cast<uint64>(ctr[2]) * add;

    output[0] = static_cast<uint32>(y >> 32) ^ ctr[1] ^ key[0];
    output[1] = static_cast<uint32>(x >> 32) ^ ctr[3] ^ key[1];
    output[2] = static_cast<uint32>(y) ^ ctr[2];
    output[3] = static_cast<uint32>(x) ^ ctr[0];

    return output;
}



extern "C" float print_randoms(float N, float std) {
    //float std = sqrt(2/fan_in);
    
    int n = (int) N;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, std);

    float* arr = (float*)malloc(n * sizeof(float));
    std::cout << "[";
    for (size_t i = 0; i < n; i++)
      std::cout << dist(gen) << " ";
    std::cout << "]";

    return 0;
}


unsigned long long time_seed() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto seed = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return seed;
}




extern "C" int randint(Scope_Struct *scope_struct, int b, int f)
{

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  float rand_float = dist(MAIN_PRNG);

  int rand_int = static_cast<int>(rand_float * (f - b + 1)) + b;

  return rand_int;
}