CC = cc
CXX = c++
CFLAGS = -O2 -Wall -Wextra -std=c11 -DUSE_OPENBLAS -DEMBED_WEIGHTS
CXXFLAGS = -O2 -Wall -Wextra -std=c++17 -DEMBED_WEIGHTS
INCLUDES = -Ilib/mac -Ilib/include
LDFLAGS = -lm -Llib/mac -lopenblas -lsentencepiece -Wl,-rpath,@executable_path/lib/mac

C_SRCS = nanotts.c tensor.c ops.c tts.c codec.c prompt.c audio.c
CXX_SRCS = sentencepiece.cpp
C_OBJS = $(C_SRCS:.c=.o)
CXX_OBJS = $(CXX_SRCS:.cpp=.o)

all: libnanotts.dylib

libnanotts.dylib: embed_weights.o $(C_OBJS) $(CXX_OBJS)
	$(CXX) -shared -o $@ $(C_OBJS) $(CXX_OBJS) embed_weights.o $(LDFLAGS)
	@echo "Built libnanotts.dylib"

%.o: %.c
	$(CC) $(CFLAGS) -fPIC $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fPIC $(INCLUDES) -c $< -o $@

embed_weights.o: model.bin
	@printf '.global _weights_data\n.global _weights_size\n.section __DATA,__weights\n.align 4\n_weights_data: .incbin "model.bin"\n_weights_size: .quad . - _weights_data\n' > _embed_w.s
	as -o $@ _embed_w.s && rm _embed_w.s

clean:
	rm -f $(C_OBJS) $(CXX_OBJS) libnanotts.dylib embed_weights.o

.PHONY: all clean
