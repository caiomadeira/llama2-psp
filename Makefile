TARGET = Llama2
OBJS = main.o src/utils.o src/generate.o src/nnet.o src/sampler.o src/tokenizer.o src/transformer.o

INCDIR =
CFLAGS = -O2 -Wall
CXXFLAGS = $(CFLAGS) -fno-exceptions -fno-rtti
ASFLAGS = $(CFLAGS)

BUILD_PRX = 1

LIBDIR =
LDFLAGS =
LIBS = -lpspgu

EXTRA_TARGETS = EBOOT.PBP
PSP_EBOOT_TITLE = Llama2

PSPSDK=$(shell psp-config --pspsdk-path)
include $(PSPSDK)/lib/build.mak