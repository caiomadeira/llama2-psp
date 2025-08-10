/*

Inference for Llama-2 Transformer Model
PSP port by Caio Madeira

*/
#include "src/tokenizer.h"
#include "src/generate.h"
#include "src/nnet.h"
#include "src/sampler.h"
#include "src/transformer.h"

extern "C" {
    PSP_MODULE_INFO("Llama2PSP", 0, 1, 0);
    PSP_MAIN_THREAD_ATTR(THREAD_ATTR_USER | THREAD_ATTR_VFPU);
}

volatile int done = 0;
/*
Sobre os parâmetros:
temperature = 0.0 fornece respostas determinísticas. ~0.9 aumenta as chances de alucinação.
topp: propabilidade acumulada para amostragem top-p
steps: numero maximo de tokens a serem gerados
*/

void print_mem_info(const char* stage) {
    pspDebugScreenSetXY(0, pspDebugScreenGetY() + 1); // Pula uma linha
    print("[%s] Memoria Livre: %d KB\n", stage, sceKernelMaxFreeMemSize() / 1024);
    sceKernelDelayThread(2000000); // Pausa de 2 segundos para podermos ler
}

int main(int argc, char* argv[])
{
    // scePowerSetClockFrequency(333, 333, 166);
    SceCtrlData pad;
	SetupCallbacks();
	pspDebugScreenInit();

    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_ANALOG);
    pspDebugScreenPrintf("carregando o modelo llama2 p/ psp...\n");

    char *checkpoint_path = MODEL_PATH;
    char *tokenizer_path = TOKENIZER_BIN_PATH;

    float temperature = 0.0f;
    float topp = 0.9f;
    int steps = 256;
    int seed = 0;
    unsigned long long rng_seed = 0;
    time_t currentTime;

    if (seed == 0) {
        sceKernelLibcTime(&currentTime);
        rng_seed = (unsigned long long)currentTime;
    } else {
        rng_seed = seed;
    }
    delay(10);
    print("Llama2 PSP\n");
    print("-------------------------\n");

    print("Carregando checkpoint...\n");
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.sequence_len)
        steps = transformer.config.sequence_len; 
    print_mem_info("Transformer Loaded.");

    print("Carregando tokenizer...\n");
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    print_mem_info("Tokenizer Loaded.");

    print("Carregando Sampler...\n");
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // print("modelo carregado.!\n\n");
    //sceKernelDelayThread(2000000); // 2s

    // prompt do usuario - aloco memoria
    char *prompt = (char*)malloc(256);
    if (!prompt) return 1;

	while(!done) {
        pspDebugScreenClear();
        pspDebugScreenSetXY(0, 1);
        strcpy(prompt, "Once upon a time");
        
        // Sampler sampler;
        // uint64_t tick;
        // sceRtcGetCurrentTick(&tick);
        // srand(tick);

        // build_sampler(&sampler, transformer.config->vocab_size, 
        // temperature, topp, rand());
        
        char* generated_text = generate(&transformer, &tokenizer, &sampler, prompt, steps);
        
        pspDebugScreenPrintf("----------------------------------\n");
        pspDebugScreenPrintf("Prompt: %s\n\n", prompt);

        if (generated_text != NULL) {
            pspDebugScreenPrintf("Generated text: >>>>>%s<<<<\n", generated_text);
            free(generated_text);
        } else {
            pspDebugScreenPrintf("Error: To alocar memoria p/ o resultado.\n");
        }

        free_sampler(&sampler);
        pspDebugScreenPrintf("----------------------------------\n");
        pspDebugScreenPrintf("Pressione X p/ gerar novamente ou HOME p/ sair.\n");

        while(!done) {
            sceCtrlReadBufferPositive(&pad, 1);
            if (pad.Buttons & PSP_CTRL_CROSS) {
                break;
            }

            sceKernelDelayThread(50000); // pausa p/ economizar a CPU
        }
    }   

    free(prompt);
    free_transformer(&transformer);
    free_tokenizer(&tokenizer);
    sceKernelExitGame();
    return 0;
}