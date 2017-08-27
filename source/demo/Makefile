#
# Possible options
# SIMDLEVEL=[ FMA | AVX2 | AVX | SSE44 | SSE41 | SSE2 | NOSIMD ]  # default is FMA
# DEBUG=1                                          # default is DEBUG=0
#

# compiler
CC=g++

# platform
PLATFORM := $(shell uname -s)

$(info platform=$(PLATFORM))
$(info compiler=$(CC))

# directories
BINDIR=bin

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

TARGETS=$(patsubst %, $(BINDIR)/%$(EXE), demo)

ifeq ($(CC),g++)
  CFLAGS=-c -std=c++0x -I../include
  LFLAGS=
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

ifdef DEBUG
   CFLAGS += -g -DDEBUG
else
   CFLAGS += -O3
endif

LIBHEADERS=$(wildcard ../include/*.h)

EXTRADEPS=$(LIBHEADERS) Makefile

all : $(TARGETS)

$(BINDIR)/%$(OBJ) : %.cpp $(EXTRADEPS) | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@


$(BINDIR)/%$(EXE) : $(BINDIR)/%$(OBJ)
	$(CC) $(LFLAGS) $< -o $@

# BEGIN AUX SECTION

$(BINDIR)/%.i : %.cpp $(EXTRADEPS) | $(BINDIR)
	$(CC) -E -c $(CFLAGS) $< -o $@


$(BINDIR)/%.asm : %.cpp $(EXTRADEPS) | $(BINDIR)
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
