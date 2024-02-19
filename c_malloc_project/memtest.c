#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

// Compile with -DREALMALLOC to use the real malloc() instead of mymalloc()
#ifndef REALMALLOC
#include "mymalloc.h"
#endif

#define MEMSIZE 4096
#define HEADERSIZE 8
#define OBJECTS 64
#define OBJSIZE (MEMSIZE / OBJECTS - HEADERSIZE)

// default provided test: malloc should be able to handle multiple allocations without overwriting other data
void maintest() {
    printf("Running main test: first init and handle multiple allocations...\t");

    char *obj[OBJECTS];
    int i, j, errors = 0;

    // fill memory with objects
    for (i = 0; i < OBJECTS; i++) {
        obj[i] = malloc(OBJSIZE);
    }

    // fill each object with distinct bytes
    for (i = 0; i < OBJECTS; i++) {
        memset(obj[i], i, OBJSIZE);
    }

    // check that all objects contain the correct bytes
    for (i = 0; i < OBJECTS; i++) {
        for (j = 0; j < OBJSIZE; j++) {
            if (obj[i][j] != i) {
                errors++;
            }
        }
    }
    printf("%d incorrect bytes\n", errors);

    for (i = 0; i < OBJECTS; i++) {
        free(obj[i]);
    }
}

// tests coalesce: should merge on 2 adjacent frees.
void test1(){
    double *testArr[4];
    testArr[0] = malloc(8);
    testArr[1] = malloc(8);

    free(testArr[0]);
    free(testArr[1]);

    testArr[0] = malloc(24);
    free(testArr[0]);
}

// tests double free
void test2() {
    char *doubleFree = malloc(1);
    free(doubleFree);
    free(doubleFree);
}

// tests allocating when no available space for new memory block
void test3() {
    double *largeAlloc = malloc(4088);
    double *smallAlloc = malloc(8);
    free(largeAlloc);
    if (smallAlloc == NULL) {
        return;
    }
    else {
        // this should never run
        fprintf(stderr, "test 3 failed");
        EXIT_FAILURE;
    }
}

// tests memory fragmentation when no available space for new memory block
void test4() {
    char *fragArray[256];
    for (int i = 0; i < 256; i++) {
        fragArray[i] = malloc(8);
    }
    for (int i = 0; i < 128; i++) {
        free(fragArray[2*i]);
    }
    // this will succeed:
    char *successfulAlloc = malloc(8);
    // this will not:
    char *unsuccessfulAlloc = malloc(9);

    if (unsuccessfulAlloc != NULL) {
        fprintf(stderr, "test 4 failed");
        EXIT_FAILURE;
    }

    free(successfulAlloc);
    for (int i = 0; i < 128; i++) {
        free(fragArray[2*i+1]);
    }
}

// tests malloc should refuse incorrect size
void test5() {
    // should throw error
    char *mallocZero = malloc(0);
    
    // should throw error
    char *mallocTooLarge = malloc(4089);

    if (mallocZero != NULL) {
        fprintf(stderr, "test 5 failed");
        EXIT_FAILURE;
    }
    if (mallocTooLarge != NULL) {
        fprintf(stderr, "test 5 failed");
        EXIT_FAILURE;
    }
}

// tests free on out-of-bounds memory address
void test6() {
    int badPtr = 1;
    free(&badPtr);
}

// test edge case of coalescing *end, *end must move up when coalescing
void test7() {
    char *arr[256];
    for (int i = 0; i < 256; i++) {
        arr[i] = malloc(8);
    }
    free(arr[254]);
    free(arr[255]);
    // *end should now be at index 254, not 255
    arr[254] = malloc(24);
    
    for (int i = 0; i < 255; i++) {
        free(arr[i]);
    }
}

int main(int argc, char **argv) {

    if (argc == 1) {
        maintest();
        test1();
        test2();
        test3();
        test4();
        test5();
        test6();
        test7();
    }
    else {
        int c = atoi(argv[1]);
        if (c < 0 || c > 7) {
            fprintf(stderr, "usage: invalid argument\n");
        }
        switch (c) {
            case 0:
                printf("running maintest\n");
                maintest();
                break;
            case 1:
                printf("running test1\n");
                test1();
                break;
            case 2:
                printf("running test2\n");
                test2();
                break;
            case 3:
                printf("running test3\n");
                test3();
                break;
            case 4:
                printf("running test4\n");
                test4();
                break;
            case 5:
                printf("running test5\n");
                test5();
                break;
            case 6:
                printf("running test6\n");
                test6();
                break;
            case 7:
                printf("running test7\n");
                test7();
                break;
        }
    }
   
    return EXIT_SUCCESS;
}
