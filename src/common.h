#ifndef COMMON_H
#define COMMON_H

#include <pspkernel.h>
#include <pspdisplay.h>
#include <pspdebug.h>
#include <pspgu.h>
#include <psputility.h>
#include <psptypes.h>
#include <pspctrl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define SCREEN_WIDTH 480
#define SCREEN_HEIGHT 272
#define BUF_WIDTH	(512)

#define TOKENIZER_BIN_PATH "tokenizer.bin"
#define WEIGHTS_PSP_PATH "weights.psp"
#define CONFIG_BIN_PATH "config.bin"

#define print pspDebugScreenPrintf
//volatile int done = 0;
extern volatile int done;

int exit_callback(int arg1, int arg2, void *common);
int CallbackThread(SceSize args, void *argp);
int SetupCallbacks(void);

#endif