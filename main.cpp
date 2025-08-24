/*

Inference for Llama-2 Transformer Model
PSP port by Caio Madeira

*/
#include <psputility.h>
#include <psptypes.h>
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

SceCtrlData pad, old_pad;
SceUtilityOskParams osk_params;

// volatile int done = 0;
/*
Sobre os parâmetros:
temperature = 0.0 fornece respostas determinísticas. ~0.9 aumenta as chances de alucinação.
topp: propabilidade acumulada para amostragem top-p
steps: numero maximo de tokens a serem gerados
*/

#define KEYBOARD_ACTIVE true

char* generated_text_draw = NULL;
bool can_generated = false;      

bool is_btn_pressed(SceCtrlData pad, SceCtrlData old_pad, int button) {
    return (pad.Buttons & button) && !(old_pad.Buttons & button);
}

void launch_osk(char* description_text, unsigned short* initial_text, unsigned short* output_buffer, int max_length) {
    memset(&osk_params, 0, sizeof(SceUtilityOskParams));

    osk_params.base.size = sizeof(osk_params);
    sceUtilityGetSystemParamInt(PSP_SYSTEMPARAM_ID_INT_LANGUAGE, &osk_params.base.language);
    sceUtilityGetSystemParamInt(PSP_SYSTEMPARAM_ID_INT_UNKNOWN, &osk_params.base.buttonSwap);

    // comentnado p/ deixar o psp fisico e o ppsspp gerenciar as prioridades de thread
    #ifdef __PSP__
        osk_params.base.graphicsThread = 17;
        osk_params.base.accessThread = 19;
        osk_params.base.fontThread = 18;
        osk_params.base.soundThread = 16;
    #endif
    
    osk_params.datacount = 1;
    osk_params.data = (SceUtilityOskData*)malloc(sizeof(SceUtilityOskData));
    memset(osk_params.data, 0, sizeof(SceUtilityOskData));
    
    unsigned short desc_ucs2[128];
    for (int i = 0; description_text[i] != '\0' && i < 127; i++) {
        desc_ucs2[i] = (unsigned short)description_text[i];
        desc_ucs2[i+1] = 0;
    }

    osk_params.data->language = PSP_UTILITY_OSK_LANGUAGE_DEFAULT;
    osk_params.data->lines = 1;
    osk_params.data->inputtype = PSP_UTILITY_OSK_INPUTTYPE_ALL;
    osk_params.data->desc = desc_ucs2;
    osk_params.data->intext = initial_text;
    osk_params.data->outtext = output_buffer;
    osk_params.data->outtextlimit = max_length;

    sceUtilityOskInitStart(&osk_params);
}

void print_debug(const char* message) {
    pspDebugScreenPrintf("[%s] Memoria livre: %d KB\n", message, sceKernelMaxFreeMemSize() / 1024);
    sceKernelDelayThread(100000);
}

void print_mem_info(const char* stage) {
    pspDebugScreenSetXY(0, pspDebugScreenGetY() + 1);
    print("[%s] Memoria Livre: %d KB\n", stage, sceKernelMaxFreeMemSize() / 1024);
    sceKernelDelayThread(2000000);
}

int main(int argc, char* argv[])
{
    scePowerSetClockFrequency(333, 333, 166);
	// SetupCallbacks();
	pspDebugScreenInit();
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, APP_NAME);
    Font menu_font = LoadFont("src/assets/font/pixelplay.png");
    //GameScreen currentScreen = MENU;
    const char *checkpoint_path = MODEL_PATH;
    const char *tokenizer_path = TOKENIZER_BIN_PATH;

    GameScreen currentScreen = PROMPT;    
    SetTargetFPS(60);
    //MenuData* menu = init_menu();
    float temperature = 0.0f;
    float topp = 0.9f;
    int steps = 256;
    int seed = 0;
    unsigned long long rng_seed = 0;
    time_t currentTime;
    sceCtrlReadBufferPositive(&old_pad, 1); 

    if (seed == 0) {
        sceKernelLibcTime(&currentTime);
        rng_seed = (unsigned long long)currentTime;
    } else { rng_seed = seed; }
    delay(10);

    build_transformer(&transformer, (char*)checkpoint_path);
    if (steps == 0 || steps > transformer.config.sequence_len)
        steps = transformer.config.sequence_len; 
    build_tokenizer(&tokenizer, (char*)tokenizer_path, transformer.config.vocab_size);
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    unsigned short osk_initial_text[256];
    unsigned short osk_output_buffer[256];
    bool is_ok_active = false;

    char *prompt = (char*)malloc(256);
    char status_message[256] = "";
    char prompt_text[256] = "Once upon a time";
    strcpy(prompt_text, "Once upon a time");
    strcpy(status_message, "");

	while(!WindowShouldClose()) {
        old_pad = pad;
        sceCtrlReadBufferPositive(&pad, 1);
        if (is_ok_active) {
            switch(sceUtilityOskGetStatus()) {
                case PSP_UTILITY_DIALOG_INIT: break;
                case PSP_UTILITY_DIALOG_VISIBLE: 
                    sceUtilityOskUpdate(1);
                    break;
                case PSP_UTILITY_DIALOG_QUIT:
                    sceUtilityOskShutdownStart();
                    break;
                case PSP_UTILITY_DIALOG_FINISHED: break;
                case PSP_UTILITY_DIALOG_NONE:
                    if (osk_params.data != NULL) {
                        if (osk_params.data->result == PSP_UTILITY_OSK_RESULT_CHANGED || osk_params.data->result == PSP_UTILITY_OSK_RESULT_UNCHANGED) {
                            for (int i = 0; i < 256; i++) {
                                prompt_text[i] = (char)osk_output_buffer[i];
                                if (osk_output_buffer[i] == 0) break;
                            }
                            prompt_text[255] = '\0';
                        }

                        free(osk_params.data); // cleaning the allocated memory for keyboard data
                        osk_params.data = NULL;
                    }
                    is_ok_active = false;
                    break;
                default: break;
            }
        } else {
            switch(currentScreen) {
                case PROMPT:
                {
                    if (is_btn_pressed(pad, old_pad, PSP_CTRL_CROSS)) {
                            can_generated = true;
                            strcpy(status_message, "Generating text...");
                            currentScreen = CHAT;
                    }

                    if (is_btn_pressed(pad, old_pad, PSP_CTRL_TRIANGLE) && !can_generated) {
                        memset(osk_initial_text, 0, sizeof(osk_initial_text));
                        memset(osk_initial_text, 0, sizeof(osk_initial_text));
                        for(int i = 0; i < 256; ++i) {
                            osk_initial_text[i] = (unsigned short)prompt_text[i];
                            if (prompt_text[i] == '\0') break;
                        }

                        launch_osk((char*)"Type your prompt:", osk_initial_text, osk_output_buffer, 256);
                        is_ok_active = true;
                    }
                } break;

                case CHAT:
                {
                    if (can_generated) {
                        if (generated_text_draw != NULL) {
                            free(generated_text_draw);
                            generated_text_draw = NULL;
                        }

                        strcpy(prompt, prompt_text);
                        generated_text_draw = generate(&transformer, &tokenizer, &sampler, prompt, steps);
                        
                        can_generated = false;
                        strcpy(status_message, "");
                    }

                    if (is_btn_pressed(pad, old_pad, PSP_CTRL_CIRCLE)) {
                        currentScreen = PROMPT;
                        if (generated_text_draw != NULL) {
                            free(generated_text_draw);
                            generated_text_draw = NULL;
                        }
                    }

                } break;

                case EXIT: { goto exit_loop; } break;
                default: break;
            }

        }
        // ================================================
        //  DRAW
        // ================================================
        if (!is_ok_active) { 
            BeginDrawing();
            ClearBackground(BLACK);
            switch(currentScreen) {
                case PROMPT:
                {
                    DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GREEN);
                    DrawText("Prompt:", 20, 5, 5, BLACK);
                    DrawRectangleLines(20, 20, SCREEN_WIDTH - 40, 100, BLACK);
                    DrawText(prompt_text, 24, 24, 20, BLACK);
                    DrawText((char*)"Press X to generate prompt.", 20, SCREEN_HEIGHT / 2, 10, BLACK);
                } break;
                case CHAT:
                {
                    DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, PURPLE);
                    if (can_generated || strcmp(status_message, "") != 0) {
                        DrawTextEx(menu_font, status_message, (Vector2){ 10, 120 }, 10, 1.0f, RAYWHITE);
                    }
                    else if (generated_text_draw != NULL) {
                        // float pos_x = 10;
                        // float pos_y = 10;
                        // int font_size = 18;
                        /* f(text, font_size) -> pixel width */
                        int output_max_width = SCREEN_WIDTH - (pos_x + pos_y); 
                        //char* wrapped_generated_text = wrap_text(menu_font, generated_text_draw, font_size, 1.0f, output_max_width);
                        if (generated_text_draw != NULL) {
                            DrawText(generated_text_draw, pos_x, pos_y, font_size, RAYWHITE);
                            //DrawTextEx(menu_font, wrapped_generated_text, (Vector2){pos_x, pos_y}, font_size, 1.0f, RAYWHITE);
                            //free(wrapped_generated_text);
                            free(generated_text_draw)
                            DrawTextEx(menu_font, "Pressione O para voltar", (Vector2){ 120, 250 }, 20, 2.0f, YELLOW);
                        }
                    } else {
                        DrawTextEx(menu_font, "Erro ao gerar texto.", (Vector2){ 50, 120 }, 20, 2.0f, RED);
                    }
                } break;
                default: break;
            }

            EndDrawing();
        }
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