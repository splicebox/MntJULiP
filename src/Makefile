CXX = g++
CXXFLAGS= -Wall -O3 #-g #-std=c++11 #-Wall #-g
#CXXFLAGS= -Wall -g #-std=c++11 #-Wall #-g
LINKPATH= -I./samtools-0.1.19 -L./samtools-0.1.19
LINKFLAGS = -lbam -lz -lm -lpthread 
DEBUG=
OBJECTS = stats.o subexon-graph.o 

junc: junc.o
	if [ ! -f ./samtools-0.1.19/libbam.a ] ; \
	then \
		cd samtools-0.1.19 ; make ;\
	fi ;
	mkdir -p ../bin
	$(CXX) -o ../bin/junc $(LINKPATH) $(CXXFLAGS) junc.o $(LINKFLAGS)

junc.o: FindJunction.cpp
	$(CXX) -c -o $@ $(LINKPATH) $(CXXFLAGS) $< $(LINKFLAGS)

clean:
	rm -f *.o *.gch

