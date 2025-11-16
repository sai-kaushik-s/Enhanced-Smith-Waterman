#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>
#include <omp.h>

#define MAX(A,B) ((A) > (B) ? (A) : (B))
#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MATCH     2
#define MISMATCH -1
#define GAP      -2

void generate_sequence(char *seq, int n) {
    const char alphabet[] = "ACGT";
    for (int i = 0; i < n; i++)
        seq[i] = alphabet[rand() % 4];
    seq[n] = '\0';
}

static inline int sw_max4(int a, int b, int c, int d) {
    int m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    if (d > m) m = d;
    return m;
}

static inline int get_from_diag(const int *diag,
                                int i, int i_min, int i_max)
{
    if (i < i_min || i > i_max)
        return 0;
    return diag[i - i_min];
}

int smith_waterman_avx2(const char *seq1, const char *seq2,
                                  int len1, int len2)
{
    int max_diag_len, d, d_end, i_min_d, i_max_d, Ld;
    int d1, d2, i_min_d1, i_max_d1, i_min_d2, i_max_d2;\
    int global_max=0;

    int16_t *Dm2, *Dm1, *D, *tmp;
    int16_t diag_buf[16], up_buf[16], left_buf[16], sub_buf[16], h_buf[16];

    __m256i vGap, vZero;

    max_diag_len = (len1 < len2) ? len1 : len2;

    Dm2 = (int16_t *)calloc(max_diag_len, sizeof(int16_t));
    Dm1 = (int16_t *)calloc(max_diag_len, sizeof(int16_t));
    D   = (int16_t *)malloc(max_diag_len * sizeof(int16_t));

    vGap  = _mm256_set1_epi16((int16_t)GAP);
    vZero = _mm256_setzero_si256();

    d_end = len1 + len2;
    for (d = 2; d <= d_end; d++) {

        i_min_d = MAX(1, d - len2); i_max_d = MIN(len1, d - 1);
        
        if (i_min_d > i_max_d) {
            tmp = Dm2; Dm2 = Dm1; Dm1 = D; D = tmp;
            continue;
        }

        Ld = i_max_d - i_min_d + 1;
        d1 = d - 1;
        d2 = d - 2;

        if (d1 >= 2) {
            i_min_d1 = MAX(1, d1 - len2); i_max_d1 = MIN(len1,d1 - 1);
        } else {
            i_min_d1 = 1; i_max_d1 = 0;  /* empty */
        }

        if (d2 >= 2) {
            i_min_d2 = MAX(1,d2 - len2); i_max_d2 = MIN(len1,d2 - 1);
        } else {
            i_min_d2 = 1; i_max_d2 = 0;
        }

        #pragma omp parallel for schedule(static) reduction(max:global_max)
        for (int t = 0; t < Ld; t += 32) {

            /* ---------------------- block 0: offset = t ---------------------- */
            int off0   = t;
            int block0 = MIN(16, Ld - off0);

            if (block0 > 0) {
                int i0 = i_min_d + off0; /* lane0 의 i */
                int j0 = d - i0;         /* lane0 의 j */

                int16_t diag0[16], up0[16], left0[16];
                int16_t sub0[16],  h0[16];

                int base_diag0 = (i0 - 1) - i_min_d2; /* (i-1,j-1)용 */
                int base_up0   = (i0 - 1) - i_min_d1; /* (i-1,j)   용 */
                int base_left0 =  i0      - i_min_d1; /* (i,  j-1) 용 */

                int len_d1 = i_max_d1 - i_min_d1;     /* d-1 길이 (0..len_d1) */

                for (int k = 0; k < 16; k++) {
                    if (k < block0) {
                        int i = i0 + k;
                        int j = j0 - k;

                        diag0[k] = 0;
                        up0[k]   = 0;
                        left0[k] = 0;

                        /* (i-1, j-1) from d-2 */
                        if (d2 >= 2 && i > 1 && j > 1) {
                            int idx = base_diag0 + k;
                            if (idx >= 0 && idx <= (i_max_d2 - i_min_d2))
                                diag0[k] = Dm2[idx];
                        }

                        /* (i-1, j) from d-1 */
                        if (d1 >= 2 && i > 1) {
                            int idx_up = base_up0 + k;
                            if (idx_up >= 0 && idx_up <= len_d1)
                                up0[k] = Dm1[idx_up];
                        }

                        /* (i, j-1) from d-1 */
                        if (d1 >= 2 && j > 1) {
                            int idx_left = base_left0 + k;
                            if (idx_left >= 0 && idx_left <= len_d1)
                                left0[k] = Dm1[idx_left];
                        }

                        /* substitution score */
                        sub0[k] = (seq1[i - 1] == seq2[j - 1])
                                ? (int16_t)MATCH
                                : (int16_t)MISMATCH;
                    } else {
                        diag0[k] = 0;
                        up0[k]   = 0;
                        left0[k] = 0;
                        sub0[k]  = 0;
                    }
                }

                __m256i vDiag0  = _mm256_loadu_si256((__m256i *)diag0);
                __m256i vUp0    = _mm256_loadu_si256((__m256i *)up0);
                __m256i vLeft0  = _mm256_loadu_si256((__m256i *)left0);
                __m256i vSub0   = _mm256_loadu_si256((__m256i *)sub0);

                __m256i vMatch0 = _mm256_add_epi16(vDiag0, vSub0);
                vUp0            = _mm256_add_epi16(vUp0,   vGap);
                vLeft0          = _mm256_add_epi16(vLeft0, vGap);

                __m256i vH0 = _mm256_max_epi16(vMatch0, vUp0);
                vH0         = _mm256_max_epi16(vH0,     vLeft0);
                vH0         = _mm256_max_epi16(vH0,     vZero);

                _mm256_storeu_si256((__m256i *)h0, vH0);

                for (int k = 0; k < block0; k++) {
                    int16_t h = h0[k];
                    D[off0 + k] = h;
                    if (h > global_max)
                        global_max = h;
                }
            }

            /* ---------------------- block 1: offset = t+16 ------------------- */
            int off1   = t + 16;
            int block1 = MIN(16, Ld - off1);

            if (block1 > 0) {
                int i1 = i_min_d + off1;
                int j1 = d - i1;

                int16_t diag1[16], up1[16], left1[16];
                int16_t sub1[16],  h1[16];

                int base_diag1 = (i1 - 1) - i_min_d2;
                int base_up1   = (i1 - 1) - i_min_d1;
                int base_left1 =  i1      - i_min_d1;

                int len_d1 = i_max_d1 - i_min_d1;

                for (int k = 0; k < 16; k++) {
                    if (k < block1) {
                        int i = i1 + k;
                        int j = j1 - k;

                        diag1[k] = 0;
                        up1[k]   = 0;
                        left1[k] = 0;

                        if (d2 >= 2 && i > 1 && j > 1) {
                            int idx = base_diag1 + k;
                            if (idx >= 0 && idx <= (i_max_d2 - i_min_d2))
                                diag1[k] = Dm2[idx];
                        }

                        if (d1 >= 2 && i > 1) {
                            int idx_up = base_up1 + k;
                            if (idx_up >= 0 && idx_up <= len_d1)
                                up1[k] = Dm1[idx_up];
                        }

                        if (d1 >= 2 && j > 1) {
                            int idx_left = base_left1 + k;
                            if (idx_left >= 0 && idx_left <= len_d1)
                                left1[k] = Dm1[idx_left];
                        }

                        sub1[k] = (seq1[i - 1] == seq2[j - 1])
                                ? (int16_t)MATCH
                                : (int16_t)MISMATCH;
                    } else {
                        diag1[k] = 0;
                        up1[k]   = 0;
                        left1[k] = 0;
                        sub1[k]  = 0;
                    }
                }

                __m256i vDiag1  = _mm256_loadu_si256((__m256i *)diag1);
                __m256i vUp1    = _mm256_loadu_si256((__m256i *)up1);
                __m256i vLeft1  = _mm256_loadu_si256((__m256i *)left1);
                __m256i vSub1   = _mm256_loadu_si256((__m256i *)sub1);

                __m256i vMatch1 = _mm256_add_epi16(vDiag1, vSub1);
                vUp1            = _mm256_add_epi16(vUp1,   vGap);
                vLeft1          = _mm256_add_epi16(vLeft1, vGap);

                __m256i vH1 = _mm256_max_epi16(vMatch1, vUp1);
                vH1         = _mm256_max_epi16(vH1,     vLeft1);
                vH1         = _mm256_max_epi16(vH1,     vZero);

                _mm256_storeu_si256((__m256i *)h1, vH1);

                for (int k = 0; k < block1; k++) {
                    int16_t h = h1[k];
                    D[off1 + k] = h;
                    if (h > global_max)
                        global_max = h;
                }
            }
        }


        /* --- 대각선 포인터 롤링 --- */
        tmp = Dm2; Dm2 = Dm1; Dm1 = D; D = tmp;
    }

    free(Dm2);
    free(Dm1);
    free(D);

    return global_max;
}

int smith_waterman(const char *seq1, const char *seq2, int len1, int len2) {

    int *H_prev = (int *)calloc(len2 + 1, sizeof(int));
    int *H_curr = (int *)calloc(len2 + 1, sizeof(int));

    int max_score = 0;

    for (int i = 1; i <= len1; i++) {
        H_curr[0] = 0; 
        int up, left, diag;

        diag = 0;
        up   = H_prev[0]; 
        left = 0;

        for (int j = 1; j <= len2; j++) {
            diag = H_prev[j - 1];
            up   = H_prev[j];
            left = H_curr[j - 1];

            int sub = (seq1[i - 1] == seq2[j - 1]) ? MATCH : MISMATCH;

            int score_diag = diag + sub;
            int score_up   = up   + GAP;
            int score_left = left + GAP;

            int h = score_diag;
            h = MAX(h, score_up);
            h = MAX(h, score_left);
            h = MAX(h, 0);

            H_curr[j] = h;
            if (h > max_score)
                max_score = h;
        }

        int *tmp = H_prev;
        H_prev = H_curr;
        H_curr = tmp;
    }

    free(H_prev);
    free(H_curr);

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

    int score;

    double t0 = omp_get_wtime();

    if (N > 4096) {
        #ifdef __AVX2__
            printf("Using AVX2 optimized Smith-Waterman\n");
            score = smith_waterman_avx2(seq1, seq2, N, N);
        #elif defined(__AVX512F__)
            score = smith_waterman(seq1, seq2, N, N);
        #else
            score = smith_waterman(seq1, seq2, N, N);
        #endif
    } else {
        printf("Using standard Smith-Waterman\n");
        score = smith_waterman(seq1, seq2, N, N);
    }
    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;
    printf("Sequence length: %d\n", N);
    printf("Smith-Waterman score: %d\n", score);
    printf("Execution time: %.6f seconds\n", elapsed);

    free(seq1);
    free(seq2);
    return 0;
}