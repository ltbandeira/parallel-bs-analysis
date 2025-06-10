#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define VETOR_SIZE 1024 
#define DELTA 4

void Inicializa(int *vetor, int n) {
    for (int i = 0; i < n; i++) {
        vetor[i] = VETOR_SIZE - i;
    }
}

void Mostra(int *vetor, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", vetor[i]);
    }
    printf("\n");
}

void bs(int *vetor, int n) {
    int c = 0, d, troca, trocou = 1;

    while ((c < (n - 1)) && trocou) {
        trocou = 0;
        for (d = 0; d < n - c - 1; d++) {
            if (vetor[d] > vetor[d + 1]) {
                troca = vetor[d];
                vetor[d] = vetor[d + 1];
                vetor[d + 1] = troca;
                trocou = 1;
            }
        }
        c++;
    }
}

int *interleaving(int vetor[], int tam) {
    int *vetor_auxiliar;
    int i1, i2, i_aux;

    vetor_auxiliar = (int *)malloc(sizeof(int) * tam);
    i1 = 0;
    i2 = tam / 2;

    for (i_aux = 0; i_aux < tam; i_aux++) {
        if (((vetor[i1] <= vetor[i2]) && (i1 < (tam / 2))) || (i2 == tam))
            vetor_auxiliar[i_aux] = vetor[i1++];
        else
            vetor_auxiliar[i_aux] = vetor[i2++];
    }

    return vetor_auxiliar;
}

int main(int argc, char **argv) {
    int my_rank, comm_sz;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int *vetor = NULL;
    int tam_vetor;
    MPI_Status status;
    
    int recebi_vetor = 0;

    if (my_rank == 0) {
        tam_vetor = VETOR_SIZE;
        vetor = malloc(tam_vetor * sizeof(int));
        Inicializa(vetor, tam_vetor);
       
    } else {
        MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &tam_vetor);
        vetor = malloc(tam_vetor * sizeof(int));
        MPI_Recv(vetor, tam_vetor, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
    }

    if (tam_vetor <= DELTA) {
        bs(vetor, tam_vetor);
    } else {
        int meio = tam_vetor / 2;
        int filho_esq = 2 * my_rank + 1;
        int filho_dir = 2 * my_rank + 2;

        if (filho_esq < comm_sz){
            printf("[rank %d] Enviando para filho_esq %d\n", my_rank, filho_esq); fflush(stdout);
            MPI_Send(vetor, meio, MPI_INT, filho_esq, 0, MPI_COMM_WORLD);
        }else {
            bs(vetor, meio);
        }
        if (filho_dir < comm_sz){
            MPI_Send(vetor + meio, tam_vetor - meio, MPI_INT, filho_dir, 0, MPI_COMM_WORLD);
        }
        else {
            bs(vetor + meio, tam_vetor - meio);
        }
        if (filho_esq < comm_sz){
            printf("[rank %d] Recebendo do filho_esq %d\n", my_rank, filho_esq); fflush(stdout);
            MPI_Recv(vetor, meio, MPI_INT, filho_esq, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (filho_dir < comm_sz)
            MPI_Recv(vetor + meio, tam_vetor - meio, MPI_INT, filho_dir, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        int *vetor_ordenado = interleaving(vetor, tam_vetor);
        free(vetor);
        vetor = vetor_ordenado;
    }

    if (my_rank != 0) {
        int pai = (my_rank - 1) / 2;
        MPI_Send(vetor, tam_vetor, MPI_INT, pai, 0, MPI_COMM_WORLD);
    } else {
        printf("Vetor final ordenado:\n");
        Mostra(vetor, tam_vetor);
    }

    free(vetor);
    MPI_Finalize();
    return 0;
}
