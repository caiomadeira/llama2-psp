#ifndef UI_H
#define UI_H

#include "common.h"

typedef enum GameScreen {
    MENU = 0,
    PROMPT,
    CHAT,
    EXIT
} GameScreen;

typedef struct MenuData {
    int current_option;
    const char* options[3];
} MenuData;

MenuData* init_menu(void);
void menu_update(MenuData* data, SceCtrlData* pad);
void menu_draw(MenuData* data, Font menu_font);
void free_menu(MenuData* data);

#endif
