#
# Configuration
#
# SIMDLEVEL=[ FMA | AVX2 | AVX | SSE44 | SSE41 | SSE2 | NOSIMD ]  # default is FMA
# OPENMP=1                                         # default is OPENMP=0
# DEBUG=1                                          # default is DEBUG=0
# FLOAT=[single|double]		                   # default is double
# FORTRAN=1                                        # returns indices in Fortran format (default is C format)
#

# compiler
ifndef COMPILER
   CC=g++
#   CC=icpc
endif
$(info compiler=$(CC))

ifndef FCOMPILER
   FC=gfortran
#   FC=ifort
#   FC=nagfor
endif
$(info fcompiler=$(FC))

# platform
PLATFORM := $(shell uname -s)
$(info platform=$(PLATFORM))

# file extensions
ifeq ($(PLATFORM), Linux)
  OBJ=.o
  MOD=.mod
  EXE=
  DLL=.so
  LIB=.a
else
# cygwin or mingw
  OBJ=.obj
# check this is actually module extension on these platforms  
  MOD=.mod  
  EXE=.exe
  DLL=.dll
  LIB=.lib
endif

# select float or double
ifndef FLOAT
  FLOAT := double
endif

# select index type
ifdef FORTRAN
   CFLAGS += -DFORTRAN_LIB
endif


# directories
BINDIR=bin
INCLUDEDIR=../include


ifeq ($(CC),g++)
  CFLAGS += -c  -std=c++0x -I$(INCLUDEDIR) -save-temps=obj -masm=intel  -DFBS_FLOAT=$(FLOAT)
#  LFLAGS=
endif
ifeq ($(CC),icpc)
  CFLAGS += -c  -std=c++0x -I$(INCLUDEDIR) -save-temps=obj -masm=intel  -DFBS_FLOAT=$(FLOAT)
#  LFLAGS=
endif

ifeq ($(FC),gfortran)
  FFLAGS += -ggdb -fPIC -fno-second-underscore -m64 -fcheck=array-temps -fcheck=bounds -fcheck=do -fcheck=pointer -fcheck=recursion -fdefault-integer-8 -Waliasing -Wsurprising -Wcharacter-truncation -Wrealloc-lhs -c
endif
ifeq ($(FC),nagfor)
  FFLAGS += -g -W0,-gline -nan -info -C=all -u -Wc,-m64,-msse,-msse2,-Wno-pointer-sign,-D_GNU_SOURCE,-mfpmath=sse -PIC -Wl,-m64 -Wc=/usr/bin/gcc -thread_safe -ideclient -f2003 -i8 -ieee=stop -c
endif
ifeq ($(FC),ifort)
  FFLAGS += -g -traceback -check all -O0 -auto -fPIC -threads -fexceptions -diag-disable 10212 -diag-disable cpu-dispatch -i8 -c 
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


ifdef DEBUG
   CFLAGS += -g -DDEBUG
else
   CFLAGS += -O3
endif

LIBHEADERS := $(wildcard $(INCLUDEDIR)/*.h)
ALLHEADERS := $(LIBHEADERS) nagfbs.h
ALLDEPS := $(ALLHEADERS) Makefile

ifdef FORTRAN
   TARGET = $(BINDIR)/nagfdemo$(EXE)
else
   TARGET = $(BINDIR)/nagdemo$(EXE)
endif

all: $(TARGET)

# Fortran section

$(BINDIR)/%$(OBJ) : %.f90 $(BINDIR)/%_ib$(MOD) Makefile | $(BINDIR)
	$(FC) $(FFLAGS) $< -o $@

$(BINDIR)/%_ib$(MOD) : %_ib.f90 Makefile | $(BINDIR)
	$(FC) $(FFLAGS) $< -o $@

$(BINDIR)/nagfdemo$(EXE) : $(BINDIR)/nagfdemo$(OBJ) $(BINDIR)/nagfbs$(DLL)
	$(FC) $(LFLAGS) $^ -o $@

# C++ section

$(BINDIR)/%$(OBJ) : %.cpp $(ALLDEPS) | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@

$(BINDIR)/nagfbs$(OBJ) : nagfbs.cpp $(ALLDEPS) | $(BINDIR)
	$(CC) $(CFLAGS) -fPIC $< -o $@

#$(BINDIR)/%$(EXE) : $(BINDIR)/%$(OBJ) $(BINDIR)/nagfbs$(DLL)
$(BINDIR)/nagdemo$(EXE) : $(BINDIR)/nagdemo$(OBJ) $(BINDIR)/nagfbs$(DLL)
	$(CC) $(LFLAGS) $^ -o $@

$(BINDIR)/nagfbs$(DLL) : $(BINDIR)/nagfbs$(OBJ)
	$(CC) $(LFLAGS) -shared $< -o $@

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
	rm *.mod
#	rm -f *.o *.i *.ii *.exe *.asm *.stackdump *.mod


# use -p for multithreading
$(BINDIR):
	mkdir -p $(BINDIR)

