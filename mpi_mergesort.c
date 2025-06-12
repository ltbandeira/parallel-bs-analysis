/* MPI recursive merge sort
   Copyright (C) 2011  Atanas Radenski

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation; either version 2 of
 the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public
 License along with this program; if not, write to the Free
 Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 Boston, MA  02110-1301, USA.

*/

/* mpicc mpi_mergesort.c -lm -o mpi_mergesort */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define SMALL 32

void bubble_sort(int *vetor, int tamanho);
void merge(int a[], int size, int temp[]);
void mergesort_parallel_mpi(int a[], int size, int temp[], int level, int my_rank, int max_rank, int tag, MPI_Comm comm);
int my_topmost_level_mpi(int my_rank);
void run_root_mpi(int a[], int size, int temp[], int max_rank, int tag, MPI_Comm comm);
void run_helper_mpi(int my_rank, int max_rank, int tag, MPI_Comm comm);
int main(int argc, char *argv[]);

int main(int argc, char *argv[])
{
  // All processes
  MPI_Init(&argc, &argv);

  // Check processes and their ranks
  // number of processes == communicator size
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int max_rank = comm_size - 1;
  int tag = 123;

  double start_time, end_time, elapsed_time;

  // Set test data
  if (my_rank == 0)
  {
    if (argc != 2)
    {
      printf("Usage: %s array-size\n", argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Get argument
    int size = atoi(argv[1]); // Array size
    printf("Array size = %d\nProcesses = %d\n", size, comm_size);

    // Array allocation
    int *a = malloc(sizeof(int) * size);
    int *temp = malloc(sizeof(int) * size);
    if (a == NULL || temp == NULL)
    {
      printf("Error: Could not allocate array of size %d\n", size);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Random array initialization
    srand(314159);
    int i;
    for (i = 0; i < size; i++)
    {
      a[i] = rand() % size;
    }

    // Sort with root process
    start_time = MPI_Wtime();
    run_root_mpi(a, size, temp, max_rank, tag, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    printf("Start = %.2f\nEnd = %.2f\nElapsed = %.2f\n", start_time, end_time, end_time - start_time);

    // Result check
    for (i = 1; i < size; i++)
    {
      if (!(a[i - 1] <= a[i]))
      {
        printf("Implementation error: a[%d]=%d > a[%d]=%d\n", i - 1, a[i - 1], i, a[i]);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
  }
  else
  {
    run_helper_mpi(my_rank, max_rank, tag, MPI_COMM_WORLD);
  }

  fflush(stdout);
  MPI_Finalize();
  return 0;
}

// Root process code
void run_root_mpi(int a[], int size, int temp[], int max_rank, int tag, MPI_Comm comm)
{
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);
  if (my_rank != 0)
  {
    printf("Error: run_root_mpi called from process %d; must be called from process 0 only\n",
           my_rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  mergesort_parallel_mpi(a, size, temp, 0, my_rank, max_rank, tag, comm);
  return;
}

// Helper process code
void run_helper_mpi(int my_rank, int max_rank, int tag, MPI_Comm comm)
{
  int level = my_topmost_level_mpi(my_rank);
  // probe for a message and determine its size and sender
  MPI_Status status;
  int size;
  MPI_Probe(MPI_ANY_SOURCE, tag, comm, &status);
  MPI_Get_count(&status, MPI_INT, &size);
  int parent_rank = status.MPI_SOURCE;
  // allocate int a[size], temp[size]
  int *a = malloc(sizeof(int) * size);
  int *temp = malloc(sizeof(int) * size);
  MPI_Recv(a, size, MPI_INT, parent_rank, tag, comm, &status);
  mergesort_parallel_mpi(a, size, temp, level, my_rank, max_rank, tag, comm);
  // Send sorted array to parent process
  MPI_Send(a, size, MPI_INT, parent_rank, tag, comm);
  return;
}

// Given a process rank, calculate the top level of the process tree in which the process participates
// Root assumed to always have rank 0 and to participate at level 0 of the process tree
int my_topmost_level_mpi(int my_rank)
{
  int level = 0;
  while (pow(2, level) <= my_rank)
    level++;
  return level;
}

// MPI merge sort
void mergesort_parallel_mpi(int a[], int size, int temp[], int level, int my_rank, int max_rank, int tag, MPI_Comm comm)
{
  int helper_rank = my_rank + pow(2, level);
  if (helper_rank > max_rank)
  {
    bubble_sort(a, size);
  }
  else
  {
    MPI_Request request;
    MPI_Status status;
    // Send second half, asynchronous
    MPI_Isend(a + size / 2, size - size / 2, MPI_INT, helper_rank, tag, comm, &request);
    // Sort first half
    mergesort_parallel_mpi(a, size / 2, temp, level + 1, my_rank, max_rank, tag, comm);
    // Free the async request (matching receive will complete the transfer).
    MPI_Request_free(&request);
    // Receive second half sorted
    MPI_Recv(a + size / 2, size - size / 2, MPI_INT, helper_rank, tag, comm, &status);
    // Merge the two sorted sub-arrays through temp
    merge(a, size, temp);
  }
  return;
}

void merge(int a[], int size, int temp[])
{
  int i1 = 0;
  int i2 = size / 2;
  int tempi = 0;
  while (i1 < size / 2 && i2 < size)
  {
    if (a[i1] < a[i2])
    {
      temp[tempi] = a[i1];
      i1++;
    }
    else
    {
      temp[tempi] = a[i2];
      i2++;
    }
    tempi++;
  }
  while (i1 < size / 2)
  {
    temp[tempi] = a[i1];
    i1++;
    tempi++;
  }
  while (i2 < size)
  {
    temp[tempi] = a[i2];
    i2++;
    tempi++;
  }
  // Copy sorted temp array into main array, a
  memcpy(a, temp, size * sizeof(int));
}

void bubble_sort(int *vetor, int tamanho)
{
  int c=0, d, troca, trocou =1;

  while (c < (tamanho-1) & trocou )
      {
      trocou = 0;
      for (d = 0 ; d < tamanho - c - 1; d++)
          if (vetor[d] > vetor[d+1])
              {
              troca      = vetor[d];
              vetor[d]   = vetor[d+1];
              vetor[d+1] = troca;
              trocou = 1;
              }
      c++;
      }
}
