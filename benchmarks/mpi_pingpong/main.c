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
    int msg_size = 1 << 20;   // 1MB default
    int iters = 200;
    int chunks = 512;         // much more aggressive fragmentation

    if (argc > 1) msg_size = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);
    if (argc > 3) chunks = atoi(argv[3]);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) fprintf(stderr, "Run with exactly 2 ranks.\n");
        MPI_Finalize();
        return 2;
    }

    if (chunks <= 0) chunks = 1;
    if (chunks > msg_size) chunks = msg_size;

    unsigned char* buf = (unsigned char*)malloc((size_t)msg_size);
    if (!buf) {
        if (rank == 0) fprintf(stderr, "malloc failed\n");
        MPI_Finalize();
        return 3;
    }

    for (int i = 0; i < msg_size; i++) {
        buf[i] = (unsigned char)((i + rank) % 251);
    }

    int base = msg_size / chunks;
    int rem = msg_size % chunks;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int iter = 0; iter < iters; iter++) {
        int offset = 0;

        // Deliberately bad: unnecessary synchronization every iteration
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            for (int c = 0; c < chunks; c++) {
                int this_size = base + (c < rem ? 1 : 0);

                // Deliberately bad: extra sync before each chunk
                MPI_Barrier(MPI_COMM_WORLD);

                // Deliberately bad: many tiny blocking sends
                MPI_Send(buf + offset, this_size, MPI_BYTE, 1, 1000 + c, MPI_COMM_WORLD);

                // Deliberately bad: sync after send
                MPI_Barrier(MPI_COMM_WORLD);

                // Deliberately bad: serialized response path
                MPI_Recv(buf + offset, this_size, MPI_BYTE, 1, 2000 + c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Deliberately bad: sync after receive
                MPI_Barrier(MPI_COMM_WORLD);

                offset += this_size;
            }
        } else {
            for (int c = 0; c < chunks; c++) {
                int this_size = base + (c < rem ? 1 : 0);

                MPI_Barrier(MPI_COMM_WORLD);

                MPI_Recv(buf + offset, this_size, MPI_BYTE, 0, 1000 + c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Barrier(MPI_COMM_WORLD);

                MPI_Send(buf + offset, this_size, MPI_BYTE, 0, 2000 + c, MPI_COMM_WORLD);

                MPI_Barrier(MPI_COMM_WORLD);

                offset += this_size;
            }
        }

        // Deliberately bad: extra global sync at end of iteration
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;

    uint64_t cs = checksum_bytes(buf, msg_size);

    if (rank == 0) {
        printf("TIME_SEC=%f\n", elapsed);
        printf("CHECKSUM=%llu\n", (unsigned long long)cs);
        fflush(stdout);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}