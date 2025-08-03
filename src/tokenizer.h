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
    char** vocab;
    float* vocab_scores;
    char** sorted_vocab_str;
    uint8_t* vocab_len;
    uint16_t* sorted_vocab_id;
    uint16_t vocab_size;
    uint8_t max_token_length;
    unsigned char byte_pieces[512];
    char* mmap_ptr;
    size_t mmap_size;
    char *str_buffer;
} Tokenizer;

void load_tokenizer(Tokenizer* t);

// generate.cpp
void encode(Tokenizer* t, 
    char *text, 
    int8_t bos, 
    int8_t eos, 
    int16_t *tokens, 
    uint16_t *n_tokens);

char* decode(Tokenizer* t,
            int16_t prev_token,
            int16_t token);

#endif