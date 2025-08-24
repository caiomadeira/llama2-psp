#ifndef COMMON_H
#define COMMON_H

extern "C" {
    #include <pspkernel.h>
    #include <pspdisplay.h>
    #include <pspdebug.h>
    #include <pspgu.h>
    #include <psputility.h>
    #include <psptypes.h>
    #include <pspctrl.h>
    #include <psprtc.h> // relogio do psp (real-time clock)
    #include <psppower.h>

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <stdint.h>
    #include <math.h>
    #include <stdio.h>
    #include <ctype.h>
    #include <time.h>
    #include <fcntl.h>
    #include <unistd.h>
}
#include <cstdlib> // rand e srand sao daui
#include "raylib.h"


#define APP_NAME "Llama 2 PSP"
#define SCREEN_WIDTH 480
#define SCREEN_HEIGHT 272
#define BUF_WIDTH	(512)

#define TOKENIZER_BIN_PATH "tok512.bin"
#define MODEL_PATH "stories260K.bin"
#define PIXELPLAY_PATH "src/assets/font/pixelplay.png"


// #define TOKENIZER_BIN_PATH "ms0:/PSP/GAME/llama2psp/tokenizer.bin"
// #define WEIGHTS_PSP_PATH "ms0:/PSP/GAME/llama2psp/weights.psp"
// #define CONFIG_BIN_PATH "ms0:/PSP/GAME/llama2psp/config.bin"

#define print pspDebugScreenPrintf
//volatile int done = 0;
extern volatile int done;

/* REFINING FUNCTIONS */
#define print pspDebugScreenPrintf

void delay(int ms);

#endif