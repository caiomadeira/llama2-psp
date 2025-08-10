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
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);

// generate.c
// sample the token given the logits and some hyperparameters
int sample(Sampler *sampler, float *logits);
unsigned int random_u32(unsigned long long *state);
float random_f32(unsigned long long *state);

#endif