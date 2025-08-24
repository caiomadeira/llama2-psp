#ifndef GENERATE_H
#define GENERATE_H

#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include "nnet.h"

char* generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, int* out_token_count);

#endif