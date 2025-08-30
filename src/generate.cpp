#include "generate.h"
/*
Generation Loop
*/

long time_in_ms() {
    SceKernelSysClock systemTime;
    sceKernelGetSystemTime(&systemTime);
    SceUInt64 full_time_value = ((SceUInt64)systemTime.hi << 32) | systemTime.low;
    return (long)(full_time_value / 1000);
}

void safe_print(char* piece) {
    if (piece == NULL || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_v = piece[0];
        if (!(isprint(byte_v) || isspace(byte_v))) { return; }
    }
    print("%s", piece);
}

char* generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, int* out_token_count) 
{
    char* empty_prompt = (char*)"";
    if (prompt == NULL) {
        prompt = empty_prompt;
    }

    // Encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;

    pspDebugScreenSetXY(0, 0);
    print("Tokenizando...");

    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        pspDebugScreenSetXY(0, 0);
        print("ERROR: Not tokens in prompt");
        while(1);
    }

    pspDebugScreenSetXY(0, 0);
    print("\t\t\t\t\t\t\t\t\t");
    pspDebugScreenSetXY(0, 2);


    // prepare nnet buffers
    //nnet_init(transformer);

    /*
    Começa o loop principal

    next: guarda o proximo token na sequencia.
    token = prompt_tokens[0]: pontapé inicial com o primeiro token no prompt.
    pos: posição na sequencia. 
    */

    // alocando um buffer grande para a resposta
    int result_buffer_size = 4096;
    char* result_buffer = (char*)calloc(result_buffer_size, sizeof(char));
    if (!result_buffer) { return NULL; }
    result_buffer[0] = '\0';
    int tokens_generated = 0;
    long start = 0;
    int next; // vai guardar o proximo token na sequencia
    int token = prompt_tokens[0]; // começa com o primeiro token do prompt
    int pos = 0; // posição na sequencia
    while(pos < steps) {
        pspDebugScreenSetXY(0, 0);
        pspDebugScreenClear();
        print("Generating tokens %d of %d...\n", pos + 1, steps);        
        // encaminha o transformer pra obter logits pro proximo token
        float* logits = forward(transformer, token, pos);

        // avança a máquina de estados
        if (pos < num_prompt_tokens - 1) {
            // se ainda estiver processando o prompt de entrada
            // fornece o proximo token de prompt
            next = prompt_tokens[pos + 1];
        } else {
            // caso contrário, faça uma amostra do próximo token dos logits
            next = sample(sampler, logits);
            tokens_generated++;
            /*
                Condição de término dependente de dados: o token BOS (=1)
                delimita sequências.
            */
            //if (next == 1) break; // i comment to make more tokens generated
            // imprima o token como string, decodifica com o objeto Tokenizer
            if (next == 1) { break; }
            char* piece = decode(tokenizer, token, next);
            // pspDebugScreenPrintf("%s", piece);
            if (strlen(result_buffer) + strlen(piece) < result_buffer_size)
                strcat(result_buffer, piece);
            else break;
        }
        pos++;
        token = next;
        if (start == 0) { start = time_in_ms(); }
    }

    free(prompt_tokens);
    if (out_token_count != NULL) {
        *out_token_count = tokens_generated;
    }

    return result_buffer;
}