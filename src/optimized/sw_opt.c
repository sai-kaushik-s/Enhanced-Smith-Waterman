#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>

#include <omp.h>

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

int smith_waterman_avx2(const char *seq1, const char *seq2, int len1, int len2)
{
    /* --- 스칼라 변수 선언부 (한 줄씩 묶어서) --- */
    int max_diag_len, d, d_end, i_min_d, i_max_d, Ld;
    int d1, d2, i_min_d1, i_max_d1, i_min_d2, i_max_d2;
    int has_d1, has_d2, t, block, i0, j0;
    int global_max;

    int16_t *Dm2, *Dm1, *D;
    int16_t *tmp;

    __m256i vGap, vZero;

    /* --- 길이 체크 --- */
    if (len1 <= 0 || len2 <= 0) return 0;

    max_diag_len = (len1 < len2) ? len1 : len2;

    Dm2 = (int16_t *)malloc(max_diag_len * sizeof(int16_t));
    Dm1 = (int16_t *)malloc(max_diag_len * sizeof(int16_t));
    D   = (int16_t *)malloc(max_diag_len * sizeof(int16_t));
    if (!Dm2 || !Dm1 || !D) {
        free(Dm2); free(Dm1); free(D);
        return 0;
    }

    memset(Dm2, 0, max_diag_len * sizeof(int16_t));
    memset(Dm1, 0, max_diag_len * sizeof(int16_t));

    global_max = 0;

    vGap  = _mm256_set1_epi16((int16_t)GAP);
    vZero = _mm256_setzero_si256();

    d_end = len1 + len2;
    
    for (d = 2; d <= d_end; d++) {

        /* --- 현재 대각선 d의 i 범위 계산 --- */
        i_min_d = d - len2;
        if (i_min_d < 1) i_min_d = 1;
        i_max_d = d - 1;
        if (i_max_d > len1) i_max_d = len1;

        if (i_min_d > i_max_d) {
            /* 유효 셀이 없는 대각선 → 포인터만 롤링 */
            tmp = Dm2; Dm2 = Dm1; Dm1 = D; D = tmp;
            continue;
        }

        Ld = i_max_d - i_min_d + 1;

        /* --- d-1, d-2 대각선 범위 미리 계산 --- */
        d1 = d - 1;
        d2 = d - 2;

        has_d1 = (d1 >= 2);
        has_d2 = (d2 >= 2);

        if (has_d1) {
            i_min_d1 = d1 - len2;
            if (i_min_d1 < 1) i_min_d1 = 1;
            i_max_d1 = d1 - 1;
            if (i_max_d1 > len1) i_max_d1 = len1;
        } else {
            i_min_d1 = 1; i_max_d1 = 0;  /* empty */
        }

        if (has_d2) {
            i_min_d2 = d2 - len2;
            if (i_min_d2 < 1) i_min_d2 = 1;
            i_max_d2 = d2 - 1;
            if (i_max_d2 > len1) i_max_d2 = len1;
        } else {
            i_min_d2 = 1; i_max_d2 = 0;
        }

        /* --- 이 대각선을 16셀씩 AVX2로 처리 --- */
        #pragma omp parallel for reduction(max: global_max)
        for (t = 0; t < Ld; t += 16) {

            int16_t diag_buf[16], up_buf[16], left_buf[16], sub_buf[16], h_buf[16];
            int k, i, j, ii, idx;
            int16_t h;

            block = Ld - t;
            if (block > 16) block = 16;

            i0 = i_min_d + t;   /* lane 0의 i */
            j0 = d - i0;        /* lane 0의 j */

            /* lane별로 diag/up/left/sub 값 scalar로 채워넣기 */
            for (k = 0; k < 16; k++) {
                if (k < block) {
                    i = i0 + k;       /* 1..len1 */
                    j = j0 - k;       /* 1..len2 */

                    diag_buf[k] = 0;
                    up_buf[k]   = 0;
                    left_buf[k] = 0;

                    /* (i-1, j-1) → 대각선 d-2 */
                    if (has_d2 && i > 1 && j > 1) {
                        ii = i - 1;
                        if (ii >= i_min_d2 && ii <= i_max_d2) {
                            idx = ii - i_min_d2;
                            diag_buf[k] = Dm2[idx];
                        }
                    }

                    /* (i-1, j) → 대각선 d-1 */
                    if (has_d1 && i > 1) {
                        ii = i - 1;
                        if (ii >= i_min_d1 && ii <= i_max_d1) {
                            idx = ii - i_min_d1;
                            up_buf[k] = Dm1[idx];
                        }
                    }

                    /* (i, j-1) → 대각선 d-1 */
                    if (has_d1 && j > 1) {
                        ii = i;
                        if (ii >= i_min_d1 && ii <= i_max_d1) {
                            idx = ii - i_min_d1;
                            left_buf[k] = Dm1[idx];
                        }
                    }

                    /* substitution score */
                    sub_buf[k] = (seq1[i - 1] == seq2[j - 1]) ? (int16_t)MATCH : (int16_t)MISMATCH;
                } else {
                    /* block 밖은 그냥 0으로 채움 */
                    diag_buf[k] = 0;
                    up_buf[k]   = 0;
                    left_buf[k] = 0;
                    sub_buf[k]  = 0;
                }
            }

            /* --- AVX2로 16셀 DP --- */
            __m256i vDiag  = _mm256_loadu_si256((__m256i *)diag_buf);
            __m256i vUp    = _mm256_loadu_si256((__m256i *)up_buf);
            __m256i vLeft  = _mm256_loadu_si256((__m256i *)left_buf);
            __m256i vSub   = _mm256_loadu_si256((__m256i *)sub_buf);
            __m256i vMatch = _mm256_add_epi16(vDiag, vSub);
            __m256i vDel   = _mm256_add_epi16(vUp,   vGap);
            __m256i vIns   = _mm256_add_epi16(vLeft, vGap);
            __m256i vH     = _mm256_max_epi16(vMatch, vDel);

            vH = _mm256_max_epi16(vH, vIns);
            vH = _mm256_max_epi16(vH, vZero);  /* local alignment */

            _mm256_storeu_si256((__m256i *)h_buf, vH);

            /* 결과 저장 + global_max 갱신 */
            for (k = 0; k < block; k++) {
                h = h_buf[k];
                D[t + k] = h;
                if (h > global_max) global_max = h;
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
    int num_threads = atoi(argv[2]);
    srand(42);

    omp_set_num_threads(num_threads);

    char *seq1 = malloc((N + 1) * sizeof(char));
    char *seq2 = malloc((N + 1) * sizeof(char));

    generate_sequence(seq1, N);
    generate_sequence(seq2, N);

    clock_t start = clock();
    #ifdef __AVX2__
        int score = smith_waterman_avx2(seq1, seq2, N, N);
    #elif defined(__AVX512F__)
        int score = smith_waterman(seq1, seq2, N, N);
    #else
        int score = smith_waterman(seq1, seq2, N, N);
    #endif
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sequence length: %d\n", N);
    printf("Smith-Waterman score: %d\n", score);
    printf("Execution time: %.6f seconds\n", elapsed);

    free(seq1);
    free(seq2);
    return 0;
}
