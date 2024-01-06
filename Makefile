CFLAGS = -O0 -fopenmp -Wall

# this allows building on Mac and Linux
ifeq ($(shell uname -s),Darwin)
# Here we use LLVM compiler to load the Mac OpenMP. Tested after installation commands:
# brew install llvm
# brew install libomp
# This may need to be modified with a different installation
ifndef HOMEBREW_PREFIX
HOMEBREW_PREFIX = /usr/local
endif
CC	= ${HOMEBREW_PREFIX}/opt/llvm/bin/clang
LD	= ${HOMEBREW_PREFIX}/opt/llvm/bin/clang
LFLAGS	= -fopenmp -lomp
else
# default (Linux) case
CC	= gcc
LD	= gcc
LFLAGS	= -lgomp
endif

AUNTIE	= demo
AOBJS	= demo.o

.PHONY: main clean

main: $(AUNTIE)

$(AUNTIE):	$(AOBJS) Makefile
	$(LD) $(AOBJS) $(LFLAGS) -o $(AUNTIE)

clean:
	rm -f $(AUNTIE) $(AOBJS)

$(AOBJS): Makefile
