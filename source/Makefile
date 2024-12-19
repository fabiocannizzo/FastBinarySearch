#
# Possible options
# SIMDLEVEL=[ FMA | AVX2 | AVX | SSE44 | SSE41 | SSE2 | NOSIMD ]  # default is FMA
# OPENMP=1                                         # default is OPENMP=0
# DEBUG=1                                          # default is DEBUG=0
# MKLROOT={path to MKL}
#

# compiler
CC=g++

# platform
PLATFORM := $(shell uname -s)

$(info platform=$(PLATFORM))
$(info compiler=$(CC))

# directories
BINDIR=bin
VPATH=demo:paper

# file extensions
ifeq ($(PLATFORM), Linux)
  OBJ=.o
  EXE=
  DLL=.so
  LIB=.a
else
# cygwin or mingw
  OBJ=.obj
  EXE=.exe
  DLL=.dll
  LIB=.lib
endif

TARGETS=$(patsubst %, $(BINDIR)/%$(EXE), paper demo charts)

ifeq ($(CC),g++)
  CFLAGS=-c -std=c++17 -Iinclude -Wall -Wextra -Wno-unused-parameter
  LFLAGS=
endif


ifdef OPENMP
  CFLAGS += -fopenmp
  LFLAGS += -fopenmp
endif


ifndef SIMDLEVEL
   CFLAGS += -mfma -mavx2
else ifeq ($(SIMDLEVEL), FMA)
   CFLAGS += -mfma -mavx2
else ifeq ($(SIMDLEVEL), AVX2)
   CFLAGS += -mavx2
else ifeq ($(SIMDLEVEL), AVX)
   CFLAGS += -mavx
else ifeq ($(SIMDLEVEL), SSE42)
   CFLAGS += -msse4.2
else ifeq ($(SIMDLEVEL), SSE41)
   CFLAGS += -msse4.1
else ifeq ($(SIMDLEVEL), SSE2)
   CFLAGS += -msse2
endif


ifdef MKLROOT
	CFLAGS += -DUSE_MKL -m64 -I${MKLROOT}/include
	ifndef CYGWIN
		#LFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_cdft_core.a ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -lpthread -lm -ldl
		#LFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
		#LFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
		LFLAGS += -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
	endif
endif

ifdef DEBUG
   CFLAGS += -g -DDEBUG
else
   CFLAGS += -O3
endif

LIBHEADERS=$(wildcard include/*.h)
PAPERHEADERS=$(wildcard paper/*.h)
PAPEROBJS=$(patsubst paper/%.cpp, $(BINDIR)/%$(OBJ), $(wildcard paper/*.cpp))

all : $(TARGETS)

$(BINDIR)/%$(OBJ) : %.cpp $(LIBHEADERS) Makefile | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@
$(PAPEROBJS) : $(PAPERHEADERS)


$(BINDIR)/%$(EXE) : $(BINDIR)/%$(OBJ)
	$(CC) $(LFLAGS) $< -o $@

# BEGIN NAG SECTION

# END NAG SECTION

# BEGIN AUX SECTION

$(BINDIR)/%.i : %.cpp $(LIBHEADERS) Makefile | $(BINDIR)
	$(CC) -E -c $(CFLAGS) $< -o $@


$(BINDIR)/%.asm : %.cpp $(LIBHEADERS) Makefile | $(BINDIR)
	$(CC) -c -g -Wa,-adhln -masm=intel $(CFLAGS) $< > $@

# END AUX SECTION

# clean
.PHONY: clean
clean:
	rm -rf $(BINDIR)
#	rm -f *.o *.i *.ii *.exe *.asm *.stackdump


# use -p for multithreading
$(BINDIR):
	mkdir -p $(BINDIR)
