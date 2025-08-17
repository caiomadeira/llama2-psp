/*

Inference for Llama-2 Transformer Model
PSP port by Caio Madeira

*/
#include "src/tokenizer.h"
#include "src/generate.h"
#include "src/nnet.h"
#include "src/sampler.h"
#include "src/transformer.h"
#include "src/ui.h"

extern "C" {
    PSP_MODULE_INFO("Llama2PSP", 0, 1, 0);
    PSP_MAIN_THREAD_ATTR(THREAD_ATTR_USER | THREAD_ATTR_VFPU);
}

Transformer transformer;
Tokenizer tokenizer;
Sampler sampler;

SceCtrlData pad;

// volatile int done = 0;
/*
Sobre os parâmetros:
temperature = 0.0 fornece respostas determinísticas. ~0.9 aumenta as chances de alucinação.
topp: propabilidade acumulada para amostragem top-p
steps: numero maximo de tokens a serem gerados
*/

char* texto_gerado_para_desenhar = NULL; // Ponteiro para guardar o texto gerado
bool precisa_gerar_texto = true;      // Flag que diz quando gerar o texto

// Coloque isso antes da sua função setup_llama
void print_debug(const char* message) {
    pspDebugScreenPrintf("[%s] Memoria livre: %d KB\n", message, sceKernelMaxFreeMemSize() / 1024);
    sceKernelDelayThread(100000); // Pequena pausa para garantir que a mensagem seja impressa
}

void print_mem_info(const char* stage) {
    pspDebugScreenSetXY(0, pspDebugScreenGetY() + 1); // Pula uma linha
    print("[%s] Memoria Livre: %d KB\n", stage, sceKernelMaxFreeMemSize() / 1024);
    sceKernelDelayThread(2000000); // Pausa de 2 segundos para podermos ler
}


int main(int argc, char* argv[])
{
    scePowerSetClockFrequency(333, 333, 166);
	// SetupCallbacks();
	pspDebugScreenInit();
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, APP_NAME);
    Font menu_font = LoadFont("src/assets/font/pixelplay.png");
    //GameScreen currentScreen = MENU;
    char *checkpoint_path = MODEL_PATH;
    char *tokenizer_path = TOKENIZER_BIN_PATH;

    GameScreen currentScreen = CHAT;    
    SetTargetFPS(60);
    MenuData* menu = init_menu();
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

    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.sequence_len)
        steps = transformer.config.sequence_len; 
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    char *prompt = (char*)malloc(256);
    const char* status_message = "";

	while(!WindowShouldClose()) {
        sceCtrlReadBufferPositive(&pad, 1);
        switch(currentScreen) {
            case MENU:
            {
                menu_update(menu, &pad);
                if ((pad.Buttons & PSP_CTRL_CROSS)) {
                    if (menu->current_option == 0) {
                        currentScreen = CHAT;
                    }
                    else if (menu->current_option == 1) {
                        currentScreen = EXIT;
                    }
                }
            } break;
            
            case CHAT:
            {
                if ((pad.Buttons & PSP_CTRL_CIRCLE)) {
                    //currentScreen = MENU;
                    if ((pad.Buttons & PSP_CTRL_CROSS) && !precisa_gerar_texto && texto_gerado_para_desenhar == NULL) {
                        precisa_gerar_texto = true; // LIGA A BANDEIRA PARA GERAR
                        status_message = "Gerando texto, aguarde...";
                    }
                }
                // Se apertar Círculo, limpa o texto para poder gerar de novo
                if ((pad.Buttons & PSP_CTRL_CIRCLE)) {
                    if (texto_gerado_para_desenhar != NULL) {
                        free(texto_gerado_para_desenhar);
                        texto_gerado_para_desenhar = NULL;
                    }
                }
            } break;

            case EXIT: { goto exit_loop; } break;
            default: break;
        }

        // ======= EXECUTANDO GERAÇÃO DO PROMPT ========
        if (precisa_gerar_texto) {
            strcpy(prompt, "Once upon a time");
            
            // Se já tínhamos um texto, liberamos a memória dele primeiro
            if (texto_gerado_para_desenhar != NULL) {
                free(texto_gerado_para_desenhar);
            }

            texto_gerado_para_desenhar = generate(&transformer, &tokenizer, &sampler, prompt, steps);
            
            // Abaixamos a bandeira para não gerar de novo
            precisa_gerar_texto = false;
            status_message = ""; // Limpa a mensagem de status
        }
        // =============================================

        BeginDrawing();
        ClearBackground(BLACK);
        switch(currentScreen) {
            // case MENU:
            // {
            //     menu_draw(menu, menu_font);
            // } break;
            case CHAT:
            {
                // Se o texto ainda está sendo gerado, mostramos a mensagem de status
                if (precisa_gerar_texto || strcmp(status_message, "") != 0) {
                    DrawTextEx(menu_font, status_message, (Vector2){ 10, 120 }, 10, 1.0f, RAYWHITE);
                }
                // Se o texto já foi gerado, nós o desenhamos
                else if (texto_gerado_para_desenhar != NULL) {
                    DrawTextEx(menu_font, texto_gerado_para_desenhar, (Vector2){ 10, 10 }, 8, 0.8f, RAYWHITE);
                    DrawTextEx(menu_font, "Pressione O para voltar", (Vector2){ 120, 250 }, 20, 2.0f, YELLOW);
                } else {
                    // Caso tenha dado erro na geração
                    DrawTextEx(menu_font, "Erro ao gerar texto.", (Vector2){ 50, 120 }, 20, 2.0f, RED);
                }
            } break;
            default: break;
        }

        EndDrawing();
    }

exit_loop:
    sceKernelDelayThread(50000); // pausa p/ economizar a CPU
    free(prompt);
    free_transformer(&transformer);
    free_tokenizer(&tokenizer);
    free_sampler(&sampler);
    sceKernelExitGame();
    return 0;
}