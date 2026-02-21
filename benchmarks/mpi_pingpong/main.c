#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

static uint64_t checksum_bytes(const unsigned char* buf, int n) {
    uint64_t s = 0;
    for (int i = 0; i < n; i++) s += buf[i];
    return s;
}

int main(int argc, char** argv) {
    int msg_size = 1 << 20; // 1MB
    int iters = 200;

    if (argc > 1) msg_size = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        if (rank == 0) fprintf(stderr, "Run with exactly 2 ranks.\n");
        MPI_Finalize();
        return 2;
    }

    unsigned char* buf = (unsigned char*)malloc(msg_size);
    for (int i = 0; i < msg_size; i++) buf[i] = (unsigned char)((i + rank) % 251);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iters; i++) {
        if (rank == 0) {
            MPI_Send(buf, msg_size, MPI_BYTE, 1, 123, MPI_COMM_WORLD);
            MPI_Recv(buf, msg_size, MPI_BYTE, 1, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(buf, msg_size, MPI_BYTE, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf, msg_size, MPI_BYTE, 0, 123, MPI_COMM_WORLD);
        }
    }

    uint64_t cs = checksum_bytes(buf, msg_size);
    if (rank == 0) printf("CHECKSUM=%llu\n", (unsigned long long)cs);

    free(buf);
    MPI_Finalize();
    return 0;
}