Brian Zhang, blz19

My little malloc() project

IMPLEMENTATION
---
Structs, Variables, Macros:

1. MEMLENGTH is the length of our memory array, which is a double array. 

2. heap is a char pointer to the start of our memory array.

3. HEAPSIZE is the size of our array in bytes. It is MEMLENGTH * 8 bytes since memory is a double array.

4. metadata is a struct which has two 4 byte components: size (the size of the chunk representated by this metadata) and isAllocated(weather it is allocated or not). 

5. memory is the double array of MEMLENGTH. It represents our heap.

Primary Methods:

1. init(metadata *start) : 
init() initializes a new memory block the first time we use malloc() by initializing a start pointer to the start of our heap.
init() writes the size of our memory block - the new metadata formed, and sets this space as free. We also set an end pointer to the newly allocated block for future use.

2. mymalloc(unsigned int size, char *file, int line) :
mymalloc() creates start and end pointers both pointing to the start of our heap (which is the start of our linked list of metadatas).
mymalloc() will first check weather the requested value of bytes is 
Once mymalloc() confirms that init() has been run, it sets end to findEnd() which returns a pointer to the last thing in our linked list. 
If init() has not been run yet it runs init(). 
Now we will start iterating through our linked list of metadatas to see if we have a spot that is not allocated and also big enough to hold the requested amount of bytes.
Every iteration we will check if the chunk that is unallocated or is of insufficient size is the end block. If it is, then we throw an error - no block is found.
If we do come across a proper block, we will exit the while loop and use writeSize() and allocate() to update our metadata block.
Depending on weather there is space between the end of the current chunk and the next metadata, we also create a new metadata right after our block is allocated.
Lastly, we return a pointer to the start of the space allocated. 

3. coalesce(metadata *end):
coalesce() creates two pointers, prevChunk and currChunk that iterates through our metadata "linked list" and searches through our metadatas to look for adjacent free metadatas.
If it finds any adjacent free metadatas, it will combine them into one while preserving the first chunk's metadata and ignoring the other.

4.myfree(void *ptr, char *file, int line):
myfree() takes in a pointer of the data our client wants to free, and it creates a pointer to the start of that address's metadata.
After checking weather the chunk requested is in the bounds of our memory block, it declares a pointer "currChunk" and searches through our "linked list" of metadatas for the block.
If it hits a metadata that is the metadata that our client wants to free, it will free the chunk if it's allocated, and if not it will throw an error claiming that it's unallocated. 
If it has not found the chunk that the user requested we look for by the end of our linked list, it will throw an error claiming that the metadata is not found.
Finally, it will run coalesce to ensure that any adjacent free metadatas will be coalesced.

Helper Methods:

1. readSize(metadata* m): Returns the size of the metadata.

2. writeSize(metadata* m, int size): Sets the size of the metadata.

3. allocate(metadata* m): Allocates the metadata.

4. unallocate(metadata* m): Unallocates the metadata.

5. nextMetadata(metadata* m): Returns a pointer to the next metadata in our "linked list" of metadatas.

6. findEnd(): Iterates through our current heap and returns the last metadata in our "linked list" of metadatas. 


ERROR REPORTING
---
There are a few possible errors that can be thrown by malloc() and free():
1. malloc() throws an error and returns NULL if the client tries to allocate a block of invalid size.
2. free() throws an error and returns if the client tries to free an invalid pointer.

For free(), we choose to do nothing and return (instead of forcing the program to exit) for ease of testing in memtest and to mimic the actual behavior of stdlib's free() function. free() in the standard library does not terminate a program, but leads to undefined behavior. Similarly, our implementation does not terminate the program, but it does refuse to free a block if given invalid input.

CORRECTNESS TESTING (memtest.c)
---
There are a few cases for malloc() and free() we must test:
1. for malloc():
  a. If we allocate multiple times, these allocations should not override previously allocated data unless freed.
  b. If we try to allocate space when memory of that size is not available, it should throw an error.
  c. If we have enough space for a chunk in non adjacent chunks, but don't have enough space in adjacent bytes, then we should throw an error if the client tries to allocate that size (i.e. memory fragmentation).
  d. If we try to allocate a pointer of invalid size (i.e. size 0 or larger than our heap), malloc should throw an error.

2. for free() and coalesce():
  a. If block1 and block2 are concurrent blocks in memory, freeing block2 then block1 (or equivalently freeing block1 then block2) should coalesce into one larger block of size block1+block2
  b. If we coalesce the end chunk, our end pointer must move up.
  c. If we free block1 and try to free it again, it should throw an error.
  d. If we try to free something that is out of our memory block's bounds, it should throw an error.

maintest() will test 1a.
test1() will test 2a.
test2() will test 2c.
test3() will test 1b.
test4() will test 1c.
test5() will test 1d.
test6() will test 2d.
test7() will test 2b.

All correctness tests can be run by compiling and running memtest.c. You can either run all at once using `./memtest`, or choose a specific test by adding an optional argument `./memtest <number of test>`, where maintest() has the corresponding argument 0.

**Note that all correctness checks in memtest assume a heap of size 4096 bytes, with a maximum of 4088 bytes of memory available to the client (because we must hold onto 8 bytes to use as a metadata header).**


PERFORMANCE TESTING (memgrind.c)
---
The first three tests are as described in the project writeup.

For our fourth test, we create a linked list of 256 8-byte chunks (16 bytes when including the header), and then frees two sequential bytes at a time, replacing each with a 24-byte chunk (32 bytes when including the header), and then finally freeing everything.

For our fifth test, we create a linked list of 128 16-byte nodes, and free them sequentially.

All performance tests can be run by compiling and running ./memgrind
