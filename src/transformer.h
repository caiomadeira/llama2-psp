#ifndef TRANSFORMER_H
#define TRANSFORMER_H

// Headers padrão para tipos como uint16_t e ssize_t. É uma boa prática incluí-los.
#include <stdint.h>
#include <sys/types.h>

// Seu header comum
#include "common.h"

// =============================================================================
// Definições das Estruturas do Modelo
// =============================================================================
// Estas structs são compatíveis entre C e C++, então não precisam
// estar dentro do bloco 'extern "C"'.

typedef struct {
    int dimension;             // Dimensão do transformer
    int hidden_dimension;      // Dimensão das camadas FFN
    int number_of_layers;      // Número de camadas
    int number_of_heads;       // Número de cabeças de atenção (query)
    int number_key_value_heads;// Número de cabeças de atenção (key/value)
    int vocab_size;            // Tamanho do vocabulário
    int sequence_len;          // Comprimento máximo da sequência
} Config;

typedef struct {
    // Tabela de embedding de tokens
    float* token_embedding_table;
    // Pesos para RMSNorm
    float* rms_att_weight;
    float* rms_ffn_weight;
    // Pesos para multiplicações de matriz (MatMuls)
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    // Pesos para as camadas Feed-Forward Network (FFN)
    float* w1;
    float* w2;
    float* w3;
    // Peso para o RMSNorm final
    float* rms_final_weight;
    // Pesos do classificador (logits)
    float* wcls;
} TransformerWeights;

typedef struct {
    // Buffers de ativação para os cálculos
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    // Cache para Key/Value
    float* key_cache;
    float* value_cache;
} RunState;

typedef struct {
    Config config;              // Hiperparâmetros da arquitetura
    TransformerWeights weights; // Pesos do modelo
    RunState state;             // Buffers de estado para os cálculos
    int fd;                     // Descritor de arquivo para o checkpoint
    float* data;                // Ponteiro para os dados do modelo carregados na RAM
    ssize_t file_size;          // Tamanho do arquivo do checkpoint
} Transformer;


// =============================================================================
// API Pública do Módulo Transformer
// =============================================================================
// O bloco a seguir é a correção crucial para os erros de linker.
// Ele garante que, quando este header for incluído por um arquivo C++,
// as funções abaixo serão tratadas com "linkagem C", evitando que o
// compilador C++ altere seus nomes (name mangling).

#ifdef __cplusplus
extern "C" {
#endif

// Declaração de variáveis globais (se houver)
extern char* g_weights_memory_block;

// Declarações (protótipos) das funções públicas

/**
 * Carrega a configuração e os pesos do modelo de um arquivo para a memória
 * e aloca os buffers de estado necessários.
 *
 * @param t Ponteiro para a struct Transformer a ser preenchida.
 * @param checkpoint_path Caminho para o arquivo .bin do modelo. Note o uso de 'const'
 * para corrigir o warning do compilador.
 */
void build_transformer(Transformer *t, char *checkpoint_path);

/**
 * Libera toda a memória associada ao transformer.
 */
void free_transformer(Transformer* t);

/**
 * Libera a memória alocada para os buffers de estado (RunState).
 */
void free_run_state(RunState* s);


#ifdef __cplusplus
}
#endif // Fim do bloco extern "C"

#endif // FIM DO HEADER TRANSFORMER_H