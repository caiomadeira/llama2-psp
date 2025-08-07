/*

Inference for Llama-2 Transformer Model
PSP port by Caio Madeira

*/
#include <psprtc.h> // relogio do psp (real-time clock)
#include <cstdlib> // rand e srand sao daui
#include <psppower.h>

#include "src/tokenizer.h"
#include "src/generate.h"
#include "src/nnet.h"
#include "src/sampler.h"
#include "src/transformer.h"

PSP_MODULE_INFO("Llama2PSP", 0, 1, 0);
PSP_MAIN_THREAD_ATTR(THREAD_ATTR_USER | THREAD_ATTR_VFPU);
volatile int done = 0;
/*
Sobre os parâmetros:
temperature = 0.0 fornece respostas determinísticas. ~0.9 aumenta as chances de alucinação.
topp: propabilidade acumulada para amostragem top-p
steps: numero maximo de tokens a serem gerados
*/
float temperature = 0.0f;
float topp = 0.9f;
uint16_t steps = 256;

void print_mem_info(const char* stage) {
    pspDebugScreenSetXY(0, pspDebugScreenGetY() + 1); // Pula uma linha
    pspDebugScreenPrintf("[%s] Memoria Livre: %d KB\n", stage, sceKernelMaxFreeMemSize() / 1024);
    sceKernelDelayThread(2000000); // Pausa de 2 segundos para podermos ler
}

int main(int argc, char* argv[])
{
    scePowerSetClockFrequency(333, 333, 166);
    SceCtrlData pad;
	SetupCallbacks();
	pspDebugScreenInit();
    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_ANALOG);
    pspDebugScreenPrintf("carregando o modelo llama2 p/ psp...\n");

    Tokenizer tokenizer;
    load_tokenizer(&tokenizer);
    print_mem_info("Apos Tokenizer");

    Transformer transformer;
    load_transformer(&transformer);
    print_mem_info("Apos Transformer");

    pspDebugScreenPrintf("modelo carregado.!\n\n");
    sceKernelDelayThread(2000000); // 2s

    // prompt do usuario - aloco memoria
    char *prompt = (char*)malloc(256);
    if (!prompt) return 1;

	while(!done) {
        pspDebugScreenClear();
        pspDebugScreenSetXY(0, 1);
        pspDebugScreenPrintf("Llama2 PSP - gerando texto.\n");
        pspDebugScreenPrintf("-------------------------\n");
        strcpy(prompt, "Once upon a time");
        pspDebugScreenPrintf("Prompt: %s\n\n", prompt);
        pspDebugScreenPrintf("gerando texto...\n");
        pspDebugScreenPrintf("-------------------------\n");
        
        Sampler sampler;
        uint64_t tick;
        sceRtcGetCurrentTick(&tick);
        srand(tick);

        build_sampler(&sampler, transformer.config->vocab_size, 
        temperature, topp, rand());
        
        char* generated_text = generate(&transformer, &tokenizer, &sampler, prompt, steps);
        
        pspDebugScreenClear();
        pspDebugScreenSetXY(0, 1);
        pspDebugScreenPrintf("Llama2-PSP - Resultado\n");
        pspDebugScreenPrintf("----------------------------------\n");
        pspDebugScreenPrintf("Prompt: %s\n\n", prompt);
        pspDebugScreenPrintf("Texto Gerado:\n");
        pspDebugScreenPrintf("----------------------------------\n");

        if (generated_text != NULL) {
            pspDebugScreenPrintf("%s", generated_text);
            free(generated_text);
        } else {
            pspDebugScreenPrintf("erro ao alocar memoria p/ o resultado.\n");
        }

        free_sampler(&sampler);
        pspDebugScreenPrintf("-------------------------\n");
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
    free_transformer_data(&transformer);
    free_tokenizer_data(&tokenizer);
    sceKernelExitGame();
    return 0;
}