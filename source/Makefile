CC=g++

TARGET=test.exe

CFLAGS=-std=c++14
LFLAGS=

ifdef NOAVX
   CFLAGS += -msse4.2
else ifndef SIMD
   CFLAGS += -mavx2
else ifneq ($(SIMD), AVX2)
   CFLAGS += -msse4.2
else
   CFLAGS += -mavx2
endif

ifdef MKLROOT
   CFLAGS += -DUSE_MKL -m64 -I${MKLROOT}/include
   #LFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_cdft_core.a ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -lpthread -lm -ldl
   #LFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
   #LFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
   LFLAGS += -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
endif

ifdef DEBUG
   CFLAGS += -g -DDEBUG
else
   CFLAGS += -O3
endif

SRC:=$(wildcard *.cpp)
HEADERS=$(wildcard *.h)
OBJS := $(patsubst %.cpp, %.o,$(SRC))

%.o : %.cpp $(HEADERS) Makefile
	$(CC) -c $(CFLAGS) $< -o $@


%.asm : %.cpp $(HEADERS) Makefile
	$(CC) -c -g -Wa,-adhln -masm=intel $(CFLAGS) $< > $@

all : $(TARGET)


%.exe : $(OBJS)
	$(CC) $(LFLAGS) -o $@ $(OBJS)


# clean
.PHONY: clean
clean:
	rm -f *.o *.i *.ii *.exe *.asm *.stackdump
