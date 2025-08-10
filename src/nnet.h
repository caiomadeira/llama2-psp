#ifndef NNET_H
#define NNET_H

#include "transformer.h"

// Neural Net Blocks; the dynamics of the Transformer

//void nnet_init(Transformer* transformer);
void softmax(float* x, int size);
float *forward(Transformer* transformer, int token, int pos);

#endif