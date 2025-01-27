ifeq ($(DEBUG),true)
	FLAGS= -g3 -m64 -c -Wall -fopenmp
	MAINFLAG= -g3 -m64 -Wall -fopenmp
else
	FLAGS=  -O3 -m64 -c 
	#-fopenmp
	MAINFLAG= -O3 -m64  
	#-fopenmp
endif

OBJS=main.o

main: ${OBJS}
	gcc $(MAINFLAG) ${OBJS} -lm -o nematicField.bin

main.o: main.c main.h
	gcc $(FLAGS) main.c -o main.o

clean: 
	rm -f *.o
	rm -f nematicField.bin

