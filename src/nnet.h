#ifndef NNET_H
#define NNET_H

#include "transformer.h"

// Neural Net Blocks; the dynamics of the Transformer

//void nnet_init(Transformer* transformer);
void softmax(uint16_t size);
float *forward(Transformer* transformer, uint16_t token, uint16_t pos);

#endif