#include "ui.h"

MenuData* init_menu(void) {
    MenuData* data = (MenuData*)malloc(sizeof(MenuData));
    if (data == NULL) { return NULL; }
    data->current_option = 0;
    data->options[0] = "Init Chat";
    data->options[1] = "Options";
    data->options[2] = "Exit";
    return data;
}

// logic where
void menu_update(MenuData* data, SceCtrlData* pad) {
    if (data == NULL || pad->Buttons == 0) return; // Evita múltiplas ações com um só toque

    int num_options = 3;

    if ((pad->Buttons & PSP_CTRL_UP)) {
        data->current_option--;
        if (data->current_option < 0) {
            data->current_option = num_options - 1; // Volta para a última opção
        }
        sceKernelDelayThread(150000); // Pequeno delay para não pular opções rápido demais
    }

    if ((pad->Buttons & PSP_CTRL_DOWN)) {
        data->current_option++;
        if (data->current_option >= num_options) {
            data->current_option = 0; // Volta para a primeira opção
        }
        sceKernelDelayThread(150000); // Pequeno delay
    }
}

void menu_draw(MenuData* data, Font menu_font) {
    if (data == NULL) return;
    DrawRectangleGradientH(0, 0, GetScreenWidth(), GetScreenHeight(), MAROON, GOLD);

    float text_size = 24;
    float initial_y = GetScreenHeight()/2.0f - 50;

    for (int i = 0; i < 3; i++) {
        Vector2 position = { 40, initial_y + (i * 35.0f) };
        Color color = RAYWHITE;

        if (i == data->current_option) {
            color = YELLOW;
            DrawTextEx(menu_font, ">", (Vector2){position.x - 20, position.y}, text_size, 2.0f, color);
        }
        
        DrawTextEx(menu_font, data->options[i], position, text_size, 2.0f, color);
    }
}

void free_menu(MenuData* data) {
    if (data != NULL) {
        free(data);
    }
}