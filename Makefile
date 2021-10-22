## MAKEFILE FOR RascalC. This compiles the grid_covariance.cpp file into the ./cov exececutable.

CC = gcc
CFLAGS = -g -O3 -Wall
CXXFLAGS = -DOPENMP -O3 -Wall -DOPENMP -DJACKKNIFE
#-DOPENMP  # use this to run multi-threaded with OPENMP
#-DPERIODIC # use this to enable periodic behavior
#-DLEGENDRE # use this to compute 2PCF covariances in Legendre bins
#-DJACKKNIFE # use this to compute (r,mu)-space 2PCF covariances and jackknife covariances
#-DTHREE_PCF # use this to compute 3PCF autocovariances

CXX = g++ -fopenmp -lgomp -std=c++0x -ffast-math

AUNTIE	= cov
AOBJS	= grid_covariance.o ./cubature/hcubature.o ./ransampl/ransampl.o

LD	= g++
LFLAGS	= -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas -lgomp -fPIE

.PHONY: main clean

main: $(AUNTIE)

$(AUNTIE):	$(AOBJS) Makefile
	$(LD) $(AOBJS) $(LFLAGS) -o $(AUNTIE)

clean:
	rm $(AUNTIE) $(AOBJS)
