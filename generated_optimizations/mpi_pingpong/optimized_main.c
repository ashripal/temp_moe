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
    int chunks = 64;          // deliberately fragment communication

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

    MPI_Request* reqs = (MPI_Request*)malloc((size_t)(2 * chunks) * sizeof(MPI_Request));
    if (!reqs) {
        if (rank == 0) fprintf(stderr, "malloc failed (reqs)\n");
        free(buf);
        MPI_Finalize();
        return 4;
    }

    for (int iter = 0; iter < iters; iter++) {
        int offset = 0;

        // Post all receives and sends non-blocking, then wait once.
        for (int c = 0; c < chunks; c++) {
            int this_size = base + (c < rem ? 1 : 0);

            if (rank == 0) {
                // Exchange with rank 1: send tag 1000+c, receive reply tag 2000+c
                MPI_Isend(buf + offset, this_size, MPI_BYTE, 1, 1000 + c, MPI_COMM_WORLD, &reqs[2 * c]);
                MPI_Irecv(buf + offset, this_size, MPI_BYTE, 1, 2000 + c, MPI_COMM_WORLD, &reqs[2 * c + 1]);
            } else {
                // Exchange with rank 0: receive tag 1000+c, send reply tag 2000+c
                MPI_Irecv(buf + offset, this_size, MPI_BYTE, 0, 1000 + c, MPI_COMM_WORLD, &reqs[2 * c]);
                MPI_Isend(buf + offset, this_size, MPI_BYTE, 0, 2000 + c, MPI_COMM_WORLD, &reqs[2 * c + 1]);
            }

            offset += this_size;
        }

        MPI_Waitall(2 * chunks, reqs, MPI_STATUSES_IGNORE);
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

    free(reqs);
    free(buf);
    MPI_Finalize();
    return 0;
}