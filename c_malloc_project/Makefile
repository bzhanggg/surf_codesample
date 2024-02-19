CC = gcc
CFLAGS = -g -std=c99 -Wall -Werror -fsanitize=address,undefined

all: memgrind memtest 

memgrind: memgrind.o mymalloc.o
	$(CC) $(CFLAGS) $^ -o $@

memtest: memtest.o mymalloc.o
	$(CC) $(CFLAGS) $^ -o $@

realmalloc:
	$(CC) -o memgrind $(CFLAGS) -DREALMALLOC memgrind.c mymalloc.c memtest.c

memgrind.o memtest.o mymalloc.o: mymalloc.h

test: memtest
	./memtest

clean:
	rm -rf *.o memtest memgrind
