ifeq ($(DEBUG),true)
	FLAGS= -g3 -m64 -c -Wall
	MAINFLAG= -g3 -m64 -Wall
else
	FLAGS=  -O3 -m64 -c -lstdc++ -fopenmp
	MAINFLAG= -O3 -m64 -lstdc++ -fopenmp
endif

COMPILER=g++

OBJS=mdmain.o interactions.o dataManager.o populations.o timestep.o 

main: ${OBJS}
	$(COMPILER) $(MAINFLAG) ${OBJS} -lm -o build/main.exe

interactions.o: interactions.cpp interactions.h mdmain.h populations.h
	$(COMPILER) $(FLAGS) interactions.cpp -o interactions.o

dataManager.o: dataManager.cpp mdmain.h
	$(COMPILER) $(FLAGS) dataManager.cpp -o dataManager.o

mdmain.o: mdmain.cpp mdmain.h interactions.h dataManager.h populations.h
	$(COMPILER) $(FLAGS) mdmain.cpp -o mdmain.o

populations.o: populations.cpp populations.h mdmain.h interactions.h
	$(COMPILER) $(FLAGS)  populations.cpp -o populations.o

timestep.o: timestep.cpp interactions.h mdmain.h populations.h dataManager.h
	$(COMPILER) $(FLAGS) timestep.cpp -o timestep.o

clean: 
	rm -f *.o
	rm -f build/main.exe
