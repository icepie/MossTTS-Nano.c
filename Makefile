CC = cc
CXX = c++
UNAME_S := $(shell uname -s)

CFLAGS = -O2 -Wall -Wextra -std=c11 -DUSE_OPENBLAS -DEMBED_WEIGHTS
CXXFLAGS = -O2 -Wall -Wextra -std=c++17 -DEMBED_WEIGHTS

SP_DIR ?= /data/Projects/sensevoice-rs/target/debug/build/sentencepiece-sys-8091fd696127e19e/out

ifeq ($(UNAME_S),Linux)
TARGET = libnanotts.so
OPENBLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
INCLUDES = $(OPENBLAS_CFLAGS) -I$(SP_DIR)/include
CFLAGS += -D_POSIX_C_SOURCE=200809L
LDFLAGS = -lm -pthread -L$(SP_DIR)/lib -Wl,-rpath,$(SP_DIR)/lib -lopenblas -lsentencepiece
else
TARGET = libnanotts.dylib
INCLUDES = -Ilib/mac -Ilib/include
LDFLAGS = -lm -Llib/mac -lopenblas -lsentencepiece -Wl,-rpath,@executable_path/lib/mac
endif

C_SRCS = nanotts.c tensor.c ops.c tts.c codec.c prompt.c audio.c
CXX_SRCS = sentencepiece.cpp
C_OBJS = $(C_SRCS:.c=.o)
CXX_OBJS = $(CXX_SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): embed_weights.o $(C_OBJS) $(CXX_OBJS)
	$(CXX) -shared -o $@ $(C_OBJS) $(CXX_OBJS) embed_weights.o $(LDFLAGS)
	@echo "Built $(TARGET)"

%.o: %.c
	$(CC) $(CFLAGS) -fPIC $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fPIC $(INCLUDES) -c $< -o $@

embed_weights.o: model.bin
ifeq ($(UNAME_S),Linux)
	@printf '.global weights_data\n.global weights_size\n.section .rodata\n.balign 8\nweights_data:\n.incbin "model.bin"\nweights_size:\n.quad . - weights_data\n' > _embed_w.s
else
	@printf '.global _weights_data\n.global _weights_size\n.section __DATA,__weights\n.align 4\n_weights_data: .incbin "model.bin"\n_weights_size: .quad . - _weights_data\n' > _embed_w.s
endif
	as -o $@ _embed_w.s && rm _embed_w.s

clean:
	rm -f $(C_OBJS) $(CXX_OBJS) libnanotts.dylib libnanotts.so embed_weights.o

.PHONY: all clean
