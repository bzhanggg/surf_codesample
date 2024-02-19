#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

// Compile with -DREALMALLOC to use the real malloc() instead of mymalloc()
#ifndef REALMALLOC
#include "mymalloc.h"
#endif

struct node {
    int val;
    struct node* next;
};


long double timeInMilli() {
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (((long double)tv.tv_sec)*1000)+(((long double)tv.tv_usec)/1000);
}

void task1() {
    for (int i = 0; i < 120; i++) {
        char *ptr = malloc(1);
        free(ptr);
    }
}

void task2() {
    char *ptrs[120];
    for (int i = 0; i < 120; i++) {
        char *newptr = malloc(1);
        ptrs[i] = newptr;
    }
    for (int i = 0; i < 120; i++) {
        free(ptrs[i]);
    }
}

void task3() {
    char *stk[120];
    int stkptr = 0;
    int ctAlloc = 0;
    while (ctAlloc < 120) {
        int coin = rand() % 2;
        if (coin) {
            char *newptr = malloc(1);
            stk[stkptr] = newptr;
            stkptr++;
            ctAlloc++;
        }
        else {
            if (stkptr == 0) {
                continue;
            }
            else {
                --stkptr;
                free(stk[stkptr]);
            }
        }
    }
    while (stkptr > 0) {
        --stkptr;
        free(stk[stkptr]);
    }
}


void task4() {
    double* data[256]; 

    for(int i=0; i<256; i++){
        double *newptr = malloc(8);
        data[i] = newptr;
    }

    for (int i = 0; i < 128; i++) {
        free(data[(i*2)]);
        free(data[(i*2)+1]);
        double *newptr = malloc(24);
        data[i*2] = newptr;
    }

    for(int i = 0; i < 128; i++){
        free(data[i*2]);
    }
}

// This method will malloc() and free() a linked list of 128 16-byte nodes 
void task5() {
    struct node *head = malloc(sizeof(struct node));
    head->val = 0;
    head->next = NULL;

    for (int i = 1; i < 128; i++) {
        struct node *tmp = malloc(sizeof(struct node));
        tmp->val = i;
        tmp->next = head;
        head = tmp;
    }
    struct node *tmp;
    while(head != NULL) {
        tmp = head;
        head = head->next;
        free(tmp);
    }
}

void runTask(void (*task)(), int taskNum){
    printf("Starting task %d...\n", taskNum);
    long double startTime = timeInMilli();
    for (int i = 0; i < 50; i++) {
        long double iterStartTime = timeInMilli();
        task();
        long double iterEndTime = timeInMilli();
        long double iterDur = iterEndTime - iterStartTime;
        fprintf(stdout, "%dth iteration of task %d took %Lf milliseconds\n", i+1, taskNum, iterDur);
    }
    long double endTime = timeInMilli();
    long double avgDur = (endTime - startTime) / 50;
    fprintf(stdout, "Average time for task %d was %Lf milliseconds\n", taskNum, avgDur);
}


int main(int argc, char **argv) {

    void (*task1ptr)() = task1;
    void (*task2ptr)() = task2;
    void (*task3ptr)() = task3;
    void (*task4ptr)() = task4;
    void (*task5ptr)() = task5;
    runTask(task1ptr, 1);
    runTask(task2ptr, 2);
    runTask(task3ptr, 3);
    runTask(task4ptr, 4);
    runTask(task5ptr, 5);

    return EXIT_SUCCESS;
}
