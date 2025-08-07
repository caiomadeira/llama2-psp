#include "common.h"
#include <cstring> // necessario p/ usar o memcpy

#include "nnet.h"

/*
Cache

wifbuf: buffer da matriz de peso para matmul
xobuf: buffer de output geral para matmul
h1buff: buffer para a attention heads
h2buff; buffer para a attention heads
*/

// float *wifbuf;
// float *xobuf;
// float *h1buff;
// float *h2buff;

// void nnet_init(Transformer* transformer) {
//     Config* p = transformer->config;
//     uint8_t max_dimesion = p->hidden_dimension;
//     uint8_t dimesion = p->dimension;
//     uint8_t head_size = dimesion / p->number_of_heads;

//     // checagens p/ garantir que o buffer wifbuf é grande
//     if (p->dimension > max_dimesion) {
//         max_dimesion = p->dimension;
//     }

//     if (((p->dimension * p->number_key_value_heads) / p->number_of_heads) > max_dimesion) {
//         max_dimesion = (p->dimension * p->number_key_value_heads) / p->number_of_heads;
//     }

//     wifbuf = (float*)malloc(max_dimesion*sizeof(float));
//     xobuf = (float*)malloc(dimesion*sizeof(float));

//     h1buff = (float*)malloc(head_size * sizeof(float));
//     h2buff = (float*)malloc(head_size * sizeof(float));

//     if (!wifbuf || !xobuf || !h1buff || !h2buff) {
//         printf("Error: can't allocate buffers.\n");
//         return;
//     }
// }

// o = x * w
void rmsnorm(float* o, float* x, float* weight, uint8_t size) {
    // calculando a soma dos quadrados
    float ss = 0.0f;
    for (uint16_t j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }   

    ss /= size; // tira a media
    ss += 1e-5; // adiciona esse valor p/ evitar divisao com zero
    ss = 1.0f / sqrtf(ss); // tira a raiz quadrada. calcula o inverso da raiz p usar multiplicacoes em vez de divisao no loop

    // normaliza e escala
    for(uint16_t j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

// x eh um ponteiro pra a RAM
void softmax(float* x, uint16_t size) {
    if (size == 0) return;

    // encontra o valor máximo (para estabilidade numérica)
    float max_val = x[0];
    for (uint16_t i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // aplica exp e calcula a soma
    float sum = 0.0f;
    for (uint16_t i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // normaliza
    for (uint16_t i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// xout e w são ponteiros para a RAM (ex: s->q e w->wq)
// x é um ponteiro local (ex: s->xb)
void matmul(float* xout, float* x, float* w, uint16_t n, uint16_t d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // De longe a maior parte do tempo é gasta dentro desta pequena função
    for (uint16_t i = 0; i < d; i++) {
        float val = 0.0f;
        float* w_row = w + i * n;
        for (uint16_t j = 0; j < n; j++) {
            val += w_row[j] * x[j];
        }
        xout[i] = val;
    }
}

// Esta função é idêntica à de cima, mas mantive os 3 nomes por consistência com o original.
// No PSP, a distinção "local" (l) vs "remoto" não existe mais, tudo é RAM.
void matmul_l(float* xout, float* x, float* w, uint16_t n, uint16_t d) {
    matmul(xout, x, w, n, d);
}
void matmul_ll(float* xout, float* x, float* w, uint16_t n, uint16_t d) {
    matmul(xout, x, w, n, d);
}

void rope(RunState *s, Config* p, uint16_t pos) {
    static uint16_t last_pos = -1; // Usando static para manter o valor entre chamadas
    uint16_t head_size = p->dimension / p->number_of_heads;

    // RoPE: codificação posicional relativa. Rotaciona q e k em cada cabeça.
    if (last_pos != pos) {
        last_pos = pos;
        // armazena em cache os valores de sin/cos
        float val = (float)pos;
        for (uint16_t h = 0; h < head_size; h += 2) {
            // Usando as funções padrão do PSP (mais rápidas e precisas)
            s->fcir[h] = cosf(val);
            s->fcir[h+1] = sinf(val);
            val /= 10000.0f; // O valor original (10.0) era muito agressivo, 10000.0f é o padrão
        }
    }

    uint16_t kv_dim = (p->dimension * p->number_key_value_heads) / p->number_of_heads;
    
    for (uint16_t h = 0; h < p->number_of_heads; h++) {
        float* q = s->q + h * head_size;
        float* k = s->k + (h / (p->number_of_heads / p->number_key_value_heads)) * head_size;
        
        for (uint16_t i = 0; i < head_size; i += 2) {
            float q0 = q[i];
            float q1 = q[i+1];
            float fcr = s->fcir[i];
            float fci = s->fcir[i+1];
            q[i]   = q0 * fcr - q1 * fci;
            q[i+1] = q0 * fci + q1 * fcr;
            
            if (i < kv_dim) {
                float k0 = k[i];
                float k1 = k[i+1];
                k[i]   = k0 * fcr - k1 * fci;
                k[i+1] = k0 * fci + k1 * fcr;
            }
        }
    }
}

void attn(RunState *s, Config* p, uint16_t pos) {
    uint16_t head_size = p->dimension / p->number_of_heads;
    uint16_t kv_dim = (p->dimension * p->number_key_value_heads) / p->number_of_heads;
    uint16_t kv_mul = p->number_of_heads / p->number_key_value_heads;
    uint32_t loff = (uint32_t)s->current_layer * p->sequence_len * kv_dim; // Layer offset

    // multihead attention. itera sobre todas as cabeças
    for (uint16_t h = 0; h < p->number_of_heads; h++) {
        float* q = s->q + h * head_size;
        float* att = s->att + h * p->sequence_len;
        
        // itera sobre todos os passos de tempo (timesteps)
        for (uint16_t t = 0; t <= pos; t++) {
            float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            
            // calcula a pontuação de atenção (dot product de q e k)
            float score = 0.0f;
            for (uint16_t i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
            score /= sqrtf((float)head_size);
            att[t] = score;
        }

        // aplica softmax nas pontuações para obter os pesos de atenção
        softmax(att, pos + 1);

        // soma ponderada dos valores (values), armazena de volta em xb
        float* xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));

        for (uint16_t t = 0; t <= pos; t++) {
            float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float a = att[t]; // peso da atenção
            for (uint16_t i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
            }
        }
    }
}

float* forward(Transformer* transformer, uint16_t token, uint16_t pos) {
    Config* p = transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    uint16_t dim = p->dimension;
    uint16_t hidden_dim = p->hidden_dimension;
    uint16_t head_size = dim / p->number_of_heads;
    uint16_t kv_dim = (p->dimension * p->number_key_value_heads) / p->number_of_heads;

    // copia o token embedding para x
    float* content_row = w->token_embedding_table + (size_t)token * dim;
    memcpy(x, content_row, dim * sizeof(float));

    // forward em todas as camadas
    for (uint16_t l = 0; l < p->number_of_layers; l++) {
        s->current_layer = l; // Guardar a camada atual para uso no attn
        
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + (size_t)l * dim, dim);

        // ponteiros para o kv cache da posição atual
        uint32_t loff = (size_t)l * p->sequence_len * kv_dim;
        s->k = s->key_cache + loff + (size_t)pos * kv_dim;
        s->v = s->value_cache + loff + (size_t)pos * kv_dim;

        // matmuls para q, k, v
        matmul(s->q, s->xb, w->wq + (size_t)l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + (size_t)l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + (size_t)l * dim * kv_dim, dim, kv_dim);

        // aplica RoPE
        rope(s, p, pos);

        // multihead attention
        attn(s, p, pos);

        // matmul final da atenção
        matmul_l(s->xb2, s->xb, w->wo + (size_t)l * dim * dim, dim, dim);

        // conexão residual de volta para x
        for (uint16_t i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + (size_t)l * dim, dim);

        // FFN: w2(silu(w1(x)) * w3(x))
        matmul_l(s->hb, s->xb, w->w1 + (size_t)l * dim * hidden_dim, dim, hidden_dim);
        matmul_l(s->hb2, s->xb, w->w3 + (size_t)l * dim * hidden_dim, dim, hidden_dim);

        // SwiGLU
        for (uint16_t i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val))); // silu
            val *= s->hb2[i]; // * w3(x)
            s->hb[i] = val;
        }

        // matmul final da ffn
        matmul_l(s->xb, s->hb, w->w2 + (size_t)l * dim * hidden_dim, hidden_dim, dim);

        // conexão residual
        for (uint16_t i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // rmsnorm final
    rmsnorm(x, x, w->rms_final_weight, dim);

    // camada classificadora para os logits
    matmul_ll(s->logits, x, w->wcls, dim, p->vocab_size);
    return s->logits;
}