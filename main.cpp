/*

Inference for Llama-2 Transformer Model
PSP port by Caio Madeira

*/
#include "src/tokenizer.h"
#include "src/generate.h"
#include "src/nnet.h"
#include "src/sampler.h"
#include "src/transformer.h"

#define CONFIG 0 // 0, 1, 2

extern "C" {
    PSP_MODULE_INFO("Llama2PSP", 0, 1, 0);
    PSP_MAIN_THREAD_ATTR(THREAD_ATTR_USER | THREAD_ATTR_VFPU);
}

volatile int done = 0;

typedef struct AppMetrics {
    float total_generation_time_s;
    float tokens_per_second;
    int generated_token_count;
    int free_memory_kb;
    int cpu_clock_freq;
    int bus_clock_freq;
} AppMetrics;

/*
Sobre os parâmetros:
temperature = 0.0 fornece respostas determinísticas. ~0.9 aumenta as chances de alucinação.
topp: propabilidade acumulada para amostragem top-p
steps: numero maximo de tokens a serem gerados
*/
SceCtrlData pad, old_pad;

void print_mem_info(const char* stage) {
    pspDebugScreenSetXY(0, pspDebugScreenGetY() + 1);
    print("[%s] Free memory: %d KB\n", stage, sceKernelMaxFreeMemSize() / 1024);
    sceKernelDelayThread(2000000);
}

bool is_btn_pressed(SceCtrlData pad, SceCtrlData old_pad, int button) {
    return (pad.Buttons & button) && !(old_pad.Buttons & button);
}

bool HandleTextInput(char* buffer, int buffer_size, const char* title) {
    static int cursor_x = 0;
    static int cursor_y = 0;

    // layout do teclado
    const char* keyboard_layout[] = {
        "1234567890",
        "qwertyuiop",
        "asdfghjkl",
        "zxcvbnm",
        " "
    };
    int num_rows = 5;
    
    // d-pad navigation (simplifiquei)
    if (is_btn_pressed(pad, old_pad, PSP_CTRL_UP)) cursor_y--;
    if (is_btn_pressed(pad, old_pad, PSP_CTRL_DOWN)) cursor_y++;
    if (is_btn_pressed(pad, old_pad, PSP_CTRL_LEFT)) cursor_x--;
    if (is_btn_pressed(pad, old_pad, PSP_CTRL_RIGHT)) cursor_x++;

    // logic wrap
    if (cursor_y < 0) cursor_y = num_rows - 1;
    if (cursor_y >= num_rows) cursor_y = 0;

    int current_row_len = strlen(keyboard_layout[cursor_y]);
    if (cursor_x < 0) cursor_x = current_row_len - 1;
    if (cursor_x >= current_row_len) cursor_x = 0;

    //button actions
    if (is_btn_pressed(pad, old_pad, PSP_CTRL_CROSS)) {
        int len = strlen(buffer);
        if (len < buffer_size - 1) {
            // add char here to buffer
            buffer[len] = keyboard_layout[cursor_y][cursor_x];
            buffer[len + 1] = '\0';
        }
    }

    if (is_btn_pressed(pad, old_pad, PSP_CTRL_CIRCLE)) {
        int len = strlen(buffer);
        if (len > 0) {
            // backspace last char
            buffer[len - 1] = '\0';
        }
    }

    if (is_btn_pressed(pad, old_pad, PSP_CTRL_START)) {
        return true; // finish editing
    }

    pspDebugScreenClear();
    pspDebugScreenSetXY(0, 1);
    print("%s\n\n", title);
    
    print(" > %s_\n\n", buffer);

    // drawing keyoard layout (maube move)
    for (int y = 0; y < num_rows; y++) {
        for (int x = 0; x < strlen(keyboard_layout[y]); x++) {
            if (y == cursor_y && x == cursor_x) {
                pspDebugScreenPrintf("[%c]", keyboard_layout[y][x]);
            } else {
                pspDebugScreenPrintf(" %c ", keyboard_layout[y][x]);
            }
        }
        pspDebugScreenPrintf("\n");
    }
    
    print("\n\n[X] Write | [O] Backspace | [START] Confirm\n");

    return false; // return false if not finish eddting yet
}

int main(int argc, char* argv[])
{
    scePowerSetClockFrequency(333, 333, 166);
	SetupCallbacks();
	pspDebugScreenInit();

    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;
    AppMetrics metrics = {0};

    char *checkpoint_path = MODEL_PATH;
    char *tokenizer_path = TOKENIZER_BIN_PATH;
    // STATE VARS AND BUFFERS
    char prompt_text[100] = "Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked";
    float temperature = 0.9f;
    float topp = 0.9f;
    int steps = 256;

    if (CONFIG == 0) {
        temperature = 0.9f;
        topp = 0.9f;
        steps = 256;

    } else if (CONFIG == 1) {
        temperature = 0.6f;
        topp = 0.6f;
        steps = 256;

    } else if (CONFIG == 2) {
        temperature = 0.2f;
        topp = 0.2f;
        steps = 256;
    }

    int seed = 0;
    unsigned long long rng_seed = 0;
    char* generated_text = NULL;
    time_t currentTime;

    print("Loading tinystories 260k llama2 on PSP...\n");

    if (seed == 0) {
        sceKernelLibcTime(&currentTime);
        rng_seed = (unsigned long long)currentTime;
    } else { rng_seed = seed; }

    delay(10);
    print("Building and loading transformer...\n");
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.sequence_len)
        steps = transformer.config.sequence_len; 
    print_mem_info("Transformer Loaded.");

    print("Building and loading tokenizer...\n");
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    print_mem_info("Tokenizer Loaded.");

    print("building Sampler...\n");
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
    print("Sampler loaded.\n");

    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_ANALOG);
    metrics.free_memory_kb = sceKernelMaxFreeMemSize() / 1024;
    metrics.cpu_clock_freq = scePowerGetCpuClockFrequencyInt();
    metrics.bus_clock_freq = scePowerGetBusClockFrequencyInt();
    pspDebugScreenClear();
    pspDebugScreenSetXY(0, 1);
    print("Llama 2 PSP \n");
    print("Free memory: %d KB | Clock: %d/%d MHz\n", metrics.free_memory_kb, metrics.cpu_clock_freq, metrics.bus_clock_freq);
    print("-------------------------------------\n");
    print("Default prompt: %s\n", prompt_text);
    print("[X] generate text.\n\n");            
	while(!done) {
        old_pad = pad;
        sceCtrlReadBufferPositive(&pad, 1);
        metrics.free_memory_kb = sceKernelMaxFreeMemSize() / 1024;
        metrics.cpu_clock_freq = scePowerGetCpuClockFrequencyInt();
        metrics.bus_clock_freq = scePowerGetBusClockFrequencyInt();
        /*
        ============================
                USER SECTION
        ============================
        */

        if (is_btn_pressed(pad, old_pad, PSP_CTRL_CROSS)) { 
            // print("-------------------------------------\n");
            if (generated_text != NULL) free(generated_text);
            // need to recriate sampler with the new params in case of editing
            build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
            u64 start_ticks;
            sceRtcGetCurrentTick(&start_ticks);
            
            generated_text = generate(&transformer, &tokenizer, &sampler, prompt_text, steps, &metrics.generated_token_count);
            u64 end_ticks;
            sceRtcGetCurrentTick(&end_ticks);

            u64 tick_diff = end_ticks - start_ticks;
            long tick_resolution = sceRtcGetTickResolution(); 
            metrics.total_generation_time_s = (double)tick_diff / (double)tick_resolution;

            if (metrics.total_generation_time_s > 0)
                metrics.tokens_per_second = (float)metrics.generated_token_count / metrics.total_generation_time_s;
            else
                metrics.tokens_per_second = 0.0f;

            if (generated_text != NULL) {
                print("Prompt: %s\n\n", prompt_text);
                print("Generated text: %s\n\n", generated_text);
            } else { print("Error: generated text is NUll.\n"); }
            print("----------------------------------\n");
            print("Temperature: %.2f\n", temperature);
            print("Top-p: %.2f\n", topp);
            print("Steps: %d\n", steps);
            print("----------------------------------\n");
            print("Time: %.2f sec | Tokens/s: %.2f\n\n", metrics.total_generation_time_s, metrics.tokens_per_second);
            print("Free memory: %d KB | Clock: %d/%d MHz\n", metrics.free_memory_kb, metrics.cpu_clock_freq, metrics.bus_clock_freq);
        }
        sceDisplayWaitVblankStart();
    }

    if (generated_text) free(generated_text);
    free_transformer(&transformer);
    free_tokenizer(&tokenizer);
    free_sampler(&sampler);
    sceKernelExitGame();
    return 0;
}