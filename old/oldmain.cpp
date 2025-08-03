#include <pspkernel.h>
#include <pspdisplay.h>
#include <pspdebug.h>
#include <pspgu.h>
#include <string.h>
#include <psputility.h>
#include <psptypes.h>

#include <pspctrl.h>

PSP_MODULE_INFO("OSK Sample", 0, 1, 1);
PSP_MAIN_THREAD_ATTR(THREAD_ATTR_USER);

static int done = 0;

int exit_callback(int arg1, int arg2, void *common)
{
	done = 1;
	
	return 0;
}

int CallbackThread(SceSize args, void *argp)
{
	int cbid = sceKernelCreateCallback("Exit Callback", exit_callback, NULL);
	sceKernelRegisterExitCallback(cbid);
	sceKernelSleepThreadCB();
	
	return 0;
}

int SetupCallbacks(void)
{
	int thid = sceKernelCreateThread("update_thread", CallbackThread, 0x11, 0xFA0, PSP_THREAD_ATTR_USER, 0);
	
	if(thid >= 0)
		sceKernelStartThread(thid, 0, 0);

	return thid;
}

static unsigned int __attribute__((aligned(16))) list[262144];

#define BUF_WIDTH	(512)
#define SCR_WIDTH	(480)
#define SCR_HEIGHT	(272)
#define PIXEL_SIZE	(4)
#define FRAME_SIZE	(BUF_WIDTH * SCR_HEIGHT * PIXEL_SIZE)

#define NUM_INPUT_FIELDS	(1) // apenas 1 campo de texto
#define TEXT_LENGTH			(128)

int main(int argc, char* argv[])
{
	SetupCallbacks();

	sceGuInit();
	sceGuStart(GU_DIRECT,list);
	sceGuDrawBuffer(GU_PSM_8888,(void*)0,BUF_WIDTH);
	sceGuDispBuffer(SCR_WIDTH,SCR_HEIGHT,(void*)FRAME_SIZE,BUF_WIDTH);
	sceGuDepthBuffer((void*)(FRAME_SIZE*2),BUF_WIDTH);
	sceGuOffset(2048 - (SCR_WIDTH/2),2048 - (SCR_HEIGHT/2));
	sceGuViewport(2048,2048,SCR_WIDTH,SCR_HEIGHT);
	sceGuDepthRange(0xc350,0x2710);
	sceGuScissor(0,0,SCR_WIDTH,SCR_HEIGHT);
	sceGuEnable(GU_SCISSOR_TEST);
	sceGuDepthFunc(GU_GEQUAL);
	sceGuEnable(GU_DEPTH_TEST);
	sceGuFrontFace(GU_CW);
	sceGuShadeModel(GU_FLAT);
	sceGuEnable(GU_CULL_FACE);
	sceGuEnable(GU_TEXTURE_2D);
	sceGuEnable(GU_CLIP_PLANES);
	sceGuFinish();
	sceGuSync(GU_SYNC_FINISH, GU_SYNC_WHAT_DONE);
	sceDisplayWaitVblankStart();
	sceGuDisplay(GU_TRUE);
	
    unsigned short intext[TEXT_LENGTH];
    unsigned short outtext[TEXT_LENGTH];
    unsigned short desc[TEXT_LENGTH];
    
    memset(intext, 0, sizeof(intext));
    memset(outtext, 0, sizeof(outtext));
    memset(desc, 0, sizeof(desc));
	
    const char* desc_str = "Digite sua mensagem:";
	int i;
    for(i=0; desc_str[i];i++) {
        desc[i] = desc_str[i];
    }

    const char* intext_str = "Olá PSP";
    for(i = 0; intext_str[i]; i++) {
        intext[i] = intext_str[i];
    }

    SceUtilityOskData data;
    memset(&data, 0, sizeof(SceUtilityOskData));
    data.language = PSP_UTILITY_OSK_LANGUAGE_DEFAULT;
    data.lines = 1;
    data.unk_24 = 1;
    data.inputtype = PSP_UTILITY_OSK_INPUTTYPE_ALL;
    data.desc = desc;
    data.intext = intext;
    data.outtextlength = TEXT_LENGTH;
    data.outtextlimit = TEXT_LENGTH - 1;
    data.outtext = outtext;
	
	SceUtilityOskParams params;
	memset(&params, 0, sizeof(params));
	params.base.size = sizeof(params);
	sceUtilityGetSystemParamInt(PSP_SYSTEMPARAM_ID_INT_LANGUAGE, &params.base.language);
	sceUtilityGetSystemParamInt(PSP_SYSTEMPARAM_ID_INT_UNKNOWN, &params.base.buttonSwap);
	params.base.graphicsThread = 17;
	params.base.accessThread = 19;
	params.base.fontThread = 18;
	params.base.soundThread = 16;
	params.datacount = NUM_INPUT_FIELDS;
	params.data = &data;

	sceUtilityOskInitStart(&params);

	while(!done)
	{
		sceGuStart(GU_DIRECT,list);
		sceGuClearColor(0);
		sceGuClearDepth(0);
		sceGuClear(GU_COLOR_BUFFER_BIT|GU_DEPTH_BUFFER_BIT);
		sceGuFinish();
		sceGuSync(GU_SYNC_FINISH, GU_SYNC_WHAT_DONE);

		switch(sceUtilityOskGetStatus())
		{
			case PSP_UTILITY_DIALOG_INIT:
				break;
			
			case PSP_UTILITY_DIALOG_VISIBLE:
				sceUtilityOskUpdate(1);
				break;
			
			case PSP_UTILITY_DIALOG_QUIT:
				sceUtilityOskShutdownStart();
				break;
			
			case PSP_UTILITY_DIALOG_FINISHED:
				break;
				
			case PSP_UTILITY_DIALOG_NONE:
				done = 1;
				
			default :
				break;
		}

		sceDisplayWaitVblankStart();
		sceGuSwapBuffers();
	}
    sceGuTerm();
	pspDebugScreenInit();
    
	pspDebugScreenSetXY(0, 2);
	
	if (data.result == PSP_UTILITY_OSK_RESULT_CHANGED) {
        pspDebugScreenPrintf("Texto digitado com sucesso:\n\n");
        int j;
        for(j = 0; data.outtext[j];j++)
        {
            unsigned c = data.outtext[j];
            if (c < 128) 
                pspDebugScreenPrintf("%c", c);
        }
    } else {
        pspDebugScreenPrintf("A digitação foi cancelada.");
    }
    pspDebugScreenPrintf("\n\n\nPressione START para sair.");

    SceCtrlData pad;
	while(1)
    {
        sceCtrlReadBufferPositive(&pad, 1);
        if (pad.Buttons & PSP_CTRL_START) {
            break;
        }
        sceKernelDelayThread(50000);	
    }
    
	sceKernelExitGame();
	return 0;
}