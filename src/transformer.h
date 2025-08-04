#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "common.h"

/*
Defini-se as 3 estruturas principais que organizam o modelo.
*/

// typedef uint32_t REUPtr;

typedef struct {
    uint16_t dimension; // transformer dimension
    uint16_t hidden_dimension; // for ffn layers
    uint16_t number_of_layers; // number of layers
    uint16_t number_of_heads; // number of query heads
    uint16_t number_key_value_heads; // number of key/value heads (can be < query heads because of multiquery)
    uint16_t vocab_size; // vocabulary size, usually 256 (byte-level)
    uint16_t sequence_len; // max sequence length
    uint16_t shared_weights;
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;


/*
Nessa struct são definidos os buffers de trabalho para cálculos.
Os dados do tipo float* por padrão (que não eram REUPtr antes) já são
ponteiros para a RAM principal.
No C64 a RAM era de 64kb, no psp a RAM será a principal de 32MB/64MB.
Os campos que eram REUPtr, viram float*. No C64 eles foram movidos
pro REU pra econimizar a RAM principal. No PSP há RAM de sobra, de certa forma.
*/
typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *fcir; // buffer for sin/cos used in rope (dim/n_heads,)
    float* q; // query (dim,) in REU for single matmul function
    float* k; // key (dim,) points into key_cache
    float* v; // value (dim,) points into value_cache
//    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float* att; // buffer for scores/attention values (n_heads, seq_len)
    float* logits; // output logits
    // kv cache
//    float* key_cache;   // (layer, seq_len, dim)
//    float* value_cache; // (layer, seq_len, dim)
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
    uint16_t current_layer;
} RunState;

typedef struct {
    Config* config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
//    int fd; // file descriptor for memory mapping
//    float* data; // memory mapped data pointer
//    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;


void build_transformer(Transformer *t, char* checkpoint_path);
void free_transformer(Transformer* t);
bool load_transformer(Transformer* t);

// void REU_getf(REUPtr ptr, volatile float* out, uint16_t size);
// void REU_putf(REUPtr ptr, volatile float* in, uint16_t size);

#endif