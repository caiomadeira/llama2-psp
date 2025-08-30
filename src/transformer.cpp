#define SIGNATURE 0x5350324C

#include "transformer.h"

#define MALLOC_FAILED(var, name) if (!var) { print("malloc failed for %s!\n", name); delay(4000); exit(EXIT_FAILURE); }
/*
Contém a logica de baixo nivel quy interage o hardware do PSP.
Esse arquivo é quase todo readaptado.
Funções com o prefixo REU_ e a struct REU são removidas.
*/

// const unsigned char config_bin[] = {
//     #embed CONFIG_BIN_PATH
// };

char* g_weights_memory_block = NULL;

extern "C" {

void malloc_run_state(RunState* s, Config* p) {
    //RunState* runstate = &t->state;
    //Config* config = t->config;
    // float* casting pro calloc p compatibilidade com c++ e size_t em key_cache p evitar overflow em multiplicacoes grandes
    // usando calloc ao inves de malloc p/ nao disparar warnings no valgrind
	int kv_dim = (p->dimension * p->number_key_value_heads) / p->number_of_heads;
	s->x = (float*)calloc(p->dimension, sizeof(float));
	s->xb = (float*)calloc(p->dimension, sizeof(float));
	s->xb2 = (float*)calloc(p->dimension, sizeof(float));
	s->hb = (float*)calloc(p->hidden_dimension, sizeof(float));
	s->hb2 = (float*)calloc(p->hidden_dimension, sizeof(float));
	s->q = (float*)calloc(p->dimension, sizeof(float));
	s->key_cache = (float*)calloc(p->number_of_layers * p->sequence_len * kv_dim, sizeof(float));
	s->value_cache = (float*)calloc(p->number_of_layers * p->sequence_len * kv_dim, sizeof(float));
	s->att = (float*)calloc(p->number_of_heads * p->sequence_len, sizeof(float));
	s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    // cache for sin/cos used in rope()
    // runstate->fcir = (float*)calloc(config->dimension / config->number_of_heads, sizeof(float));

    // runstate->k = (float*)calloc(config->dimension, sizeof(float));
    // runstate->v = (float*)calloc(config->dimension, sizeof(float));

    // adicionando verificacao de erro
    // TODO: Verificar todas as chabes
    MALLOC_FAILED(s->x, "s->x");
    MALLOC_FAILED(s->xb, "s->xb");
    MALLOC_FAILED(s->xb2, "s->xb2");
    MALLOC_FAILED(s->hb, "s->hb");
    MALLOC_FAILED(s->hb2, "s->hb2");
    MALLOC_FAILED(s->key_cache, "s->key_cache");
    MALLOC_FAILED(s->value_cache, "s->value_cache");
    MALLOC_FAILED(s->att, "s->att");
    MALLOC_FAILED(s->logits, "s->logits");
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
    int head_size = p->dimension / p->number_of_heads;
    unsigned long long number_of_layers = p->number_of_layers; //  make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ model

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
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dimension;

    w->rms_att_weight = ptr;
    ptr += number_of_layers * p->dimension;

    w->wq = ptr;
    ptr += number_of_layers * p->dimension * (p->number_of_heads * head_size);
    
    w->wk = ptr;
    ptr += number_of_layers * p->dimension * (p->number_key_value_heads * head_size);

    w->wv = ptr;
    ptr += number_of_layers * p->dimension * (p->number_key_value_heads * head_size);

    w->wo = ptr;
    ptr += number_of_layers * (p->number_of_heads * head_size) * p->dimension;
    
    w->rms_ffn_weight = ptr;
    ptr += number_of_layers * p->dimension;

    w->w1 = ptr;
    ptr += number_of_layers * p->dimension * p->hidden_dimension;

    w->w2 = ptr;
    ptr += number_of_layers * p->hidden_dimension * p->dimension;

    w->w3 = ptr;
    ptr += number_of_layers * p->dimension * p->hidden_dimension;
    
    w->rms_final_weight = ptr;
    ptr += p->dimension;

    ptr += p->sequence_len * head_size / 2;
    ptr += p->sequence_len * head_size / 2;

    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

/*
read_checkpoint carrega os dados do modelo (os pesos e configs do arquivo) de um arquivo binario
para a memoria do PSP. A diferenca da outra implementação em gen_model_files.py é que eu posso
carregar o arquivo direto do motor de inferencia e não preciso rodar o script py pra
gerar um arquivo config.bin antes.
*/

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights,
					 int *fd, float **data, ssize_t *file_size) {

	FILE *file = fopen(checkpoint, "rb");
	if (!file)
	{
		print("Couldn't open file %s\n", checkpoint);
		delay(1000);
		exit(EXIT_FAILURE);
	}

	// read in the config header
	if (fread(config, sizeof(Config), 1, file) != 1)
	{
		exit(EXIT_FAILURE);
	}
	// negative vocab size is hacky way of signaling unshared weights. bit yikes.
	int shared_weights = config->vocab_size > 0 ? 1 : 0;
	config->vocab_size = abs(config->vocab_size);
	// figure out the file size
	fseek(file, 0, SEEK_END); // move file pointer to end of file
	*file_size = ftell(file); // get the file size, in bytes
	fclose(file);
	// memory map the Transformer weights into the data pointer
	*fd = open(checkpoint, O_RDONLY); // open in read only mode
	if (*fd == -1)
	{
		print("open failed!\n");
		delay(1000);
		exit(EXIT_FAILURE);
	}

	off_t size = lseek(*fd, 0, SEEK_END);
	lseek(*fd, 0, SEEK_SET);
	*file_size = size;

	*data = (float*)malloc(*file_size);
	if (*data == NULL)
	{
		print("malloc failed!\n");
		close(*fd);
		exit(EXIT_FAILURE);
	}

	ssize_t bytes_read = read(*fd, *data, *file_size);
	if (bytes_read != *file_size)
	{
		print("read failed!\n");
		free(*data);
		close(*fd);
		delay(1000);
		exit(EXIT_FAILURE);
	}
	close(*fd);

	// float *weights_ptr = *data + sizeof(Config) / sizeof(float);
	float *weights_ptr = (float *)((char *)*data + sizeof(Config));
	memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char *checkpoint_path)
{
	read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size); // le a config e os pesos do checkpoint
	malloc_run_state(&t->state, &t->config); // faz a alocação dos runstate buffers
}

void free_run_state(RunState *s)
{
	free(s->x);
	free(s->xb);
	free(s->xb2);
	free(s->hb);
	free(s->hb2);
	free(s->q);
	free(s->att);
	free(s->logits);
	free(s->key_cache);
	free(s->value_cache);
}

void free_transformer(Transformer *t)
{
	if (t->data != NULL)
	{
		free(t->data);
		t->data = NULL;
	}
	if (t->fd != -1)
	{
		close(t->fd);
		t->fd = -1;
	}
	// free the RunState buffers
	free_run_state(&t->state);
}

}