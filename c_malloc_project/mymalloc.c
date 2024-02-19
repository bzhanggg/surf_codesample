#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MEMLENGTH 512
#define heap ((char *) memory)
#define HEAPSIZE (MEMLENGTH * 8)

typedef struct metadata {
    int size;
    bool isAllocated;
} metadata;

static double memory[MEMLENGTH];

/* HELPER FUNCTIONS TO READ/WRITE METADATA */
int readSize(metadata* m) {
    return m->size;
}

void writeSize(metadata* m, int size) {
    m->size = size;
}

void allocate(metadata* m){
    m->isAllocated = true;
}

void unallocate(metadata* m){
    m->isAllocated = false;
}

metadata* nextMetadata(metadata* m) {
    char *arrPtr = (char *) m;
    arrPtr += sizeof(metadata) + readSize(m);
    return (metadata *) arrPtr;
}

metadata* findEnd() {
    metadata *currPtr = (metadata *) heap;
    metadata *nextPtr = (metadata *) heap;
    nextPtr = nextMetadata(nextPtr);

    while ((char *)nextPtr < heap + HEAPSIZE) {
        currPtr = nextMetadata(currPtr);
        nextPtr = nextMetadata(nextPtr);
    }
    return currPtr;
}

/* INITIALIZE MEMORY BLOCK AT START OF PROGRAM */
void init(metadata *start) {
    start = (metadata *) heap;
    writeSize(start, HEAPSIZE - sizeof(metadata));
    unallocate(start);
}

void* mymalloc(unsigned int size, char *file, int line) {

    // round up to the nearest multiple of 8 
    int trueSize = (size + 7) & ~7;
    metadata *start = (metadata *) heap;
    metadata *end = (metadata *) heap;

    // if user tries to initialize a memory block of size 0 or > max size, throw an error
    if (size <= 0 || trueSize + sizeof(metadata) > HEAPSIZE) {
        fprintf(stderr, "%s [%d]: requested memory block of invalid size %u\n", file, line, size);
        return NULL;
    }

    // initialize memory array the first time malloc is called
    if (readSize(start) == 0) {
        init(start);
    }

    end = findEnd();
    metadata *currChunk = start;
    // keep searching for suitable memory block
    while (currChunk->isAllocated || readSize(currChunk) < trueSize) {
        if (currChunk == end) {
            fprintf(stderr, "%s [%d]: requested memory block of size %d is not available\n", file, line, size);
            return NULL;
        }
        currChunk = nextMetadata(currChunk);
    }

    // currChunk is suitable for use, at end of heap
    if (currChunk == end) {
        if (trueSize == readSize(end)) {
            allocate(currChunk);
            writeSize(currChunk, trueSize);
            return (void *)((char *)currChunk + sizeof(metadata));
        }
        if (trueSize + sizeof(metadata) <= readSize(end)) {
            int newEndSize = readSize(end) - (trueSize + sizeof(metadata));

            allocate(currChunk);
            writeSize(currChunk, trueSize);
            
            end = nextMetadata(currChunk);
            unallocate(end);
            writeSize(end, newEndSize);
            return (void *)((char *)currChunk + sizeof(metadata));
        }
    }
    // currChunk is suitable for use, in middle of heap
    allocate(currChunk);
    writeSize(currChunk, trueSize);
    return (void *)((char *)currChunk + sizeof(metadata));
}

void coalesce(metadata *end) {

    metadata *prevChunk = (metadata *) heap;
    metadata *currChunk = nextMetadata(prevChunk);
    
    while (currChunk <= end) {
        // if prevChunk and currChunk both are unallocated, prevChunk should consume currChunk
        if (!prevChunk->isAllocated && !currChunk->isAllocated) {
            if (currChunk == end) {
                prevChunk->size += sizeof(metadata) + readSize(currChunk);
                end = prevChunk;
                return;
            }
            prevChunk->size += sizeof(metadata) + readSize(currChunk);
            currChunk = nextMetadata(currChunk);
        }
        // if either are allocated, move on
        else {
            prevChunk = currChunk;
            currChunk = nextMetadata(currChunk);
        }
    }
    return;
}

void myfree(void *ptr, char *file, int line) {

    metadata *truePtr = (metadata *)((char *)ptr - 8); // move ptr 8 bytes back to match metadata
    metadata *end = findEnd();

    if ((char *)truePtr < heap || truePtr > end) {
        fprintf(stderr, "%s [%d]: ptr is outside bounds\n", file, line);
        return;
    }
    metadata *currChunk = (metadata *)heap;

    // linearly search for correct pointer 
    while (currChunk <= end && currChunk != (metadata *)truePtr) {
        currChunk = nextMetadata(currChunk); 
    }

    // went through all blocks in list and could not find the corresponding block 
    if (currChunk > end) {
        fprintf(stderr, "%s [%d]: requested memory block to free is not found\n", file, line);
        return;
    }

    // if user tries to free already freed memory, throw an error
    if (!currChunk->isAllocated) {
        fprintf(stderr, "%s [%d]: requested memory block to free is already unallocated\n", file, line);
        return;
    }

    unallocate(currChunk);
    coalesce(end);

    return;
}
