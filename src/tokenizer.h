#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "common.h"

/*
Essa struct mapeia os dados do arquivo tokenizer.bin criado pelo
script Python. Esses dados são organizados na memória do programa
no PSP.

char** vocab ptr pra um array de strings. armazena o vocabulario
de principal onde cada entrada no array é uma string que representa
um token (ex: gato, ndo, the)

*/

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    unsigned int  max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

void load_tokenizer(Tokenizer* t);
void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size);
// generate.cpp
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
char *decode(Tokenizer *t, int prev_token, int token);
void free_tokenizer(Tokenizer *t);

#endif