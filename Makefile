TARGET = Llama2PSP

OBJ_DIR = .objects
BUILD_DIR = build
VPATH = .:src

SOURCES = main.cpp $(wildcard src/*.cpp)
OBJS = $(addprefix $(OBJ_DIR)/,$(notdir $(SOURCES:.cpp=.o)))

INCDIR =
CFLAGS = -O2 -Wall
CXXFLAGS = $(CFLAGS) -fno-exceptions -fno-rtti
ASFLAGS = $(CFLAGS)

BUILD_PRX = 1
LIBDIR =
LDFLAGS =
LIBS = -lpspgu -lpsppower

EXTRA_TARGETS = EBOOT.PBP
PSP_EBOOT_TITLE = Llama 2 PSP

PSP_EBOOT_ICON = assets/ICON0.png
PSP_EBOOT_PIC1 = assets/PIC1.png

.PHONY: build all clean post-build

build: all post-build

post-build:
	@echo "post-Build: organizing files in folder $(BUILD_DIR)/"
	@mkdir -p $(BUILD_DIR)
	@mv EBOOT.PBP $(BUILD_DIR)/
	@echo "compiling files..."
	@cp resources/stories260K.bin $(BUILD_DIR)/
	@cp resources/tok512.bin $(BUILD_DIR)/
	@cp $(PSP_EBOOT_ICON) $(BUILD_DIR)/
	@cp $(PSP_EBOOT_PIC1) $(BUILD_DIR)/
	@echo "finished build: $(BUILD_DIR)"

PSPSDK=$(shell psp-config --pspsdk-path)
include $(PSPSDK)/lib/build.mak

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	@echo "compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

clean:
	@echo "cleaning files"
	@rm -rf EBOOT.PBP $(TARGET).elf $(TARGET).prx PARAM.SFO
	@rm -rf $(OBJ_DIR) $(BUILD_DIR)