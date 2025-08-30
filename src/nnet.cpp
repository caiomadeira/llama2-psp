#include "common.h"
#include <cstring> // necessario p/ usar o memcpy

#include "nnet.h"

// o = x * w
void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculando a soma dos quadrados
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }   

    ss /= size; // tira a media
    ss += 1e-5; // adiciona esse valor p/ evitar divisao com zero
    ss = 1.0f / sqrtf(ss); // tira a raiz quadrada. calcula o inverso da raiz p usar multiplicacoes em vez de divisao no loop

    // normaliza e escala
    for(int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

// x eh um ponteiro pra a RAM
void softmax(float* x, int size) {
    if (size == 0) return;

    // encontra o valor máximo (para estabilidade numérica)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // aplica exp e calcula a soma
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // normaliza
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// xout e w são ponteiros para a RAM (ex: s->q e w->wq)
// x é um ponteiro local (ex: s->xb)
void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // De longe a maior parte do tempo é gasta dentro desta pequena função
    int i;
    #pragma omp parallel for private(i)
        for (i = 0; i < d; i++) {
            float val = 0.0f;
            float* w_row = w + i * n;
            for (int j = 0; j < n; j++) {
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

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dimension;
    int kv_dim = (p->dimension * p->number_key_value_heads) / p->number_of_heads;
    int kv_mul = p->number_of_heads / p->number_key_value_heads;
    int hidden_dim = p->hidden_dimension;
    int head_size = dim / p->number_of_heads;

    // copia o token embedding para x
    float* content_row = w->token_embedding_table + (size_t)token * dim;
    memcpy(x, content_row, dim * sizeof(float));

    // forward em todas as camadas
    for (unsigned long long l = 0; l < p->number_of_layers; l++) {
        // s->current_layer = l; // Guardar a camada atual para uso no attn
        
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + (size_t)l * dim, dim);

        // ponteiros para o kv cache da posição atual
        int loff = l * p->sequence_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // matmuls para q, k, v
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);
		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		for (int i = 0; i < dim; i += 2)
		{
			int head_dim = i % head_size;
			float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
			float val = pos * freq;
			float fcr = cosf(val);
			float fci = sinf(val);
			int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
			for (int v = 0; v < rotn; v++)
			{
				float *vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
				float v0 = vec[i];
				float v1 = vec[i + 1];
				vec[i] = v0 * fcr - v1 * fci;
				vec[i + 1] = v0 * fci + v1 * fcr;
			}
		}

		// multihead attention. iterate over all heads
		int h;
#pragma omp parallel for private(h)
		for (h = 0; h < p->number_of_heads; h++)
		{
			// get the query vector for this head
			float *q = s->q + h * head_size;
			// attention scores for this head
			float *att = s->att + h * p->sequence_len;
			// iterate over all timesteps, including the current one
			for (int t = 0; t <= pos; t++)
			{
				// get the key vector for this head and at this timestep
				float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
				// calculate the attention score as the dot product of q and k
				float score = 0.0f;
				for (int i = 0; i < head_size; i++)
				{
					score += q[i] * k[i];
				}
				score /= sqrtf(head_size);
				// save the score to the attention buffer
				att[t] = score;
			}

			// softmax the scores to get attention weights, from 0..pos inclusively
			softmax(att, pos + 1);

			// weighted sum of the values, store back into xb
			float *xb = s->xb + h * head_size;
			memset(xb, 0, head_size * sizeof(float));
			for (int t = 0; t <= pos; t++)
			{
				// get the value vector for this head and at this timestep
				float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
				// get the attention weight for this timestep
				float a = att[t];
				// accumulate the weighted value into xb
				for (int i = 0; i < head_size; i++)
				{
					xb[i] += a * v[i];
				}
			}
		}

		// final matmul to get the output of the attention
		matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

		// residual connection back into x
		for (int i = 0; i < dim; i++)
		{
			x[i] += s->xb2[i];
		}

		// ffn rmsnorm
		rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
		matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

		// SwiGLU non-linearity
		for (int i = 0; i < hidden_dim; i++)
		{
			float val = s->hb[i];
			// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
			val *= (1.0f / (1.0f + expf(-val)));
			// elementwise multiply with w3(x)
			val *= s->hb2[i];
			s->hb[i] = val;
		}

		// final matmul to get the output of the ffn
		matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

		// residual connection
		for (int i = 0; i < dim; i++)
		{
			x[i] += s->xb[i];
		}
	}

	// final rmsnorm
	rmsnorm(x, x, w->rms_final_weight, dim);

	// classifier into logits
	matmul(s->logits, x, w->wcls, p->dimension, p->vocab_size);
	return s->logits;
}