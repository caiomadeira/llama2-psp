TARGET = Llama22
OBJS = main.o src/utils.o src/generate.o src/nnet.o src/sampler.o src/tokenizer.o src/transformer.o src/ui.o

INCDIR =
CFLAGS = -O2 -Wall
CXXFLAGS = $(CFLAGS) -fno-exceptions -fno-rtti
ASFLAGS = $(CFLAGS)

BUILD_PRX = 1

LIBDIR =
LDFLAGS =
LIBS = -lpspgu -lpspnet_apctl -lpsphttp -lraylib -lphysfs -lcjson -lz -lglut -lGLU -lGL -lpspfpu -lpspvfpu -lpsppower -lpspaudio -lpspaudiolib -lmad -lpspmp3 -lpspjpeg

EXTRA_TARGETS = EBOOT.PBP
PSP_EBOOT_TITLE = New Llama2

PSPSDK=$(shell psp-config --pspsdk-path)
include $(PSPSDK)/lib/build.mak