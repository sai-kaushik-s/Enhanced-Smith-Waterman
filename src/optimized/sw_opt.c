#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX(A,B) ((A) > (B) ? (A) : (B))
#define MATCH     2
#define MISMATCH -1
#define GAP      -2

void generate_sequence(char *seq, int n) {
    const char alphabet[] = "ACGT";
    for (int i = 0; i < n; i++)
        seq[i] = alphabet[rand() % 4];
    seq[n] = '\0';
}

int smith_waterman(const char *seq1, const char *seq2, int len1, int len2) {
    int rows = len1 + 1;
    int cols = len2 + 1;
    int total = rows * cols;

    int *H = calloc(total, sizeof(int));

    int max_score = 0;
    for (int i = 1; i <= len1; i++) {
        int row_i   = i * cols;
        int row_im1 = (i - 1) * cols;
        for (int j = 1; j <= len2; j++) {
            int idx      = row_i + j;
            int idx_diag = row_im1 + (j - 1); // (i-1, j-1)
            int idx_up   = row_im1 + j;       // (i-1, j)
            int idx_left = row_i   + (j - 1); // (i,   j-1)

            int match = H[idx_diag] + (seq1[i - 1] == seq2[j - 1] ? MATCH : MISMATCH);
            int del   = H[idx_up]   + GAP;
            int ins   = H[idx_left] + GAP;

            int val = MAX(0, MAX(match, MAX(del, ins)));
            H[idx] = val;

            if (val > max_score)
                max_score = val;
        }
    }

    free(H);

    return max_score;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <sequence_length> <num_of_threads>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    srand(42);

    char *seq1 = malloc((N + 1) * sizeof(char));
    char *seq2 = malloc((N + 1) * sizeof(char));

    generate_sequence(seq1, N);
    generate_sequence(seq2, N);

    clock_t start = clock();
    int score = smith_waterman(seq1, seq2, N, N);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sequence length: %d\n", N);
    printf("Smith-Waterman score: %d\n", score);
    printf("Execution time: %.6f seconds\n", elapsed);

    free(seq1);
    free(seq2);
    return 0;
}
