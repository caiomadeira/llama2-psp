#include "common.h"

#ifndef SAMPLER_H
#define SAMPLER_H

/*
O sampler pega logits e retorna uma amostragem de token.
Sampling pode ser feito em pequenos passos: greedy argmax, sampling,
top-p sampling.
*/
// struct used when sorting probabilities during top-p sampling
typedef struct {
    float prob;
    uint16_t index;
} ProbIndex;

typedef struct {
    uint16_t vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    uint32_t rng_state;
} Sampler;

void build_sampler(Sampler* sampler, 
                  uint16_t vocab_size, 
                  float temperature, 
                  float topp, 
                  uint32_t rng_seed);
void free_sampler(Sampler* sampler);

// generate.c
// sample the token given the logits and some hyperparameters
uint16_t sample(Sampler* sampler, float* logits);

uint32_t random_u32(uint32_t *state);
float random_f32(uint32_t *state);

#endif