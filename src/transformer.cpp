#define SIGNATURE 0x5350324C

#include "transformer.h"

/*
Contém a logica de baixo nivel quy interage o hardware do PSP.
Esse arquivo é quase todo readaptado.
Funções com o prefixo REU_ e a struct REU são removidas.
*/

const unsigned char config_bin[] = {
    #embed CONFIG_BIN_PATH
};

void malloc_run_state(Transformer* t) {
    RunState* runstate = &t->state;
    Config* config = t->config;
    // float* casting pro calloc p compatibilidade com c++ e size_t em key_cache p evitar overflow em multiplicacoes grandes
    // usando calloc ao inves de malloc p/ nao disparar warnings no valgrind
    uint32_t key_value_dim = (config->dimension * config->number_key_value_heads) / config->number_of_heads;
    runstate->x = (float*)calloc(config->dimension, sizeof(float));
    runstate->xb = (float*)calloc(config->dimension, sizeof(float));
    runstate->xb2 = (float*)calloc(config->dimension, sizeof(float));
    runstate->hb = (float*)calloc(config->hidden_dimension, sizeof(float));
    runstate->hb2 = (float*)calloc(config->hidden_dimension, sizeof(float));
    runstate->logits = (float*)calloc(config->vocab_size, sizeof(float));
    // cache for sin/cos used in rope()
    runstate->fcir = (float*)calloc(config->dimension / config->number_of_heads, sizeof(float));

    runstate->q = (float*)calloc(config->dimension, sizeof(float));
    runstate->k = (float*)calloc(config->dimension, sizeof(float));
    runstate->v = (float*)calloc(config->dimension, sizeof(float));

    runstate->att = (float*)calloc(config->number_of_heads * config->sequence_len, sizeof(float));
    runstate->key_cache = (float*)calloc((size_t)config->number_of_layers * config->sequence_len * key_value_dim, sizeof(float));
    runstate->value_cache =  (float*)calloc((size_t)config->number_of_layers * config->sequence_len * key_value_dim, sizeof(float));
    
    // adicionando verificacao de erro
    // TODO: Verificar todas as chabes
    if (!runstate->x || !runstate->xb || !runstate->key_cache) {
        printf("Erro> falha ao alocar runstate!\n");
    }
}

void memory_map_weights(Transformer* t, char* weights_ptr) {
    TransformerWeights* w = &t->weights;
    Config* p = t->config;
    uint16_t shared_weights = p->shared_weights;

    char* ptr = weights_ptr;
    uint32_t head_size = p->dimension / p->number_of_heads;

    uint32_t number_of_layers = p->number_of_layers;

    /*
    Mapeando os pesos
    seja w uma matriz de pesos.

    token embedding_table: é a tabela de embedding de tokens.
    wq: matriz de query
    wk: matriz de key
    wv: matriz de value
    wo: matriz de output

    rms_att_weight: pesos da normalização (RMSNorm) da attention layer. Camada de feed-foward.
    pesos da camada feed-foward:
    w1, w2 e w3

    */
    w->token_embedding_table = (float*)ptr;
    ptr += (size_t)p->vocab_size * p->dimension * sizeof(float);
    w->rms_att_weight = (float*)ptr;
    ptr += (size_t)p->number_of_layers * p->dimension * sizeof(float);
    w->wq = (float*)ptr;
    ptr += (size_t)p->number_of_layers * p->dimension * (p->number_of_heads * head_size) * sizeof(float);
    
    w->wk = (float*)ptr;
    ptr += (size_t)p->number_of_layers * p->dimension * (p->number_key_value_heads * head_size) * sizeof(float);

    w->wv = (float*)ptr;
    ptr += (size_t)p->number_of_layers * p->dimension * (p->number_key_value_heads * head_size) * sizeof(float);

    w->wo = (float*)ptr;
    ptr += (size_t)p->number_of_layers * (p->number_of_heads * head_size) * p->dimension * sizeof(float);
    
    w->rms_ffn_weight = (float*)ptr;
    ptr += (size_t)p->number_of_layers * p->dimension * sizeof(float);

    w->w1 = (float*)ptr;
    ptr += (size_t)p->number_of_layers * p->dimension * p->hidden_dimension * sizeof(float);

    w->w2 = (float*)ptr;
    ptr += (size_t)p->number_of_layers * p->hidden_dimension * p->dimension * sizeof(float);

    w->w3 = (float*)ptr;
    ptr += (size_t)p->number_of_layers * p->dimension * p->hidden_dimension * sizeof(float);
    
    w->rms_final_weight = (float*)ptr;
    ptr += p->dimension * sizeof(float);

    ptr += (size_t)p->sequence_len * head_size / 2 * sizeof(float);
    ptr += (size_t)p->sequence_len * head_size / 2 * sizeof(float);

    w->wcls = p->shared_weights ? w->token_embedding_table : (float*)ptr;
}

char* g_weights_memory_block = NULL;

void load_transformer(Transformer* t)
{
    t->config = (Config*)config_bin;

    FILE* file = fopen("build/weights.psp", "rb");
    if (file == NULL) {
        printf("Error: can't open weights.psp\n");
        return;
    }

    // --- calcular o tamanho do arquivo ---
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    // --- fim do calculo do tamanho do arquivo ---

    g_weights_memory_block = (char*)malloc(file_size);
    if (g_weights_memory_block == NULL) {
        printf("Error: Can't allocate memory for weights.\n");
        fclose(file);
        return;
    }
    fread(g_weights_memory_block, 1, file_size, file);
    fclose(file);

    // verificando a assinatura posta antes
    uint32_t magic = *(uint32_t*)g_weights_memory_block;
    if (magic != SIGNATURE) {
        printf("Error: SIGNATURE INVALID in weights.psp\n");
        return;
    }

    // mapeando os weigths com memory_map_weights. O ponteiro começa depois da signature de 4 bytes.
    char* weights_ptr = g_weights_memory_block + sizeof(uint32_t);
    memory_map_weights(t, weights_ptr);

    malloc_run_state(t); //alocando os buffers de estado (trabalho) de RunState
}