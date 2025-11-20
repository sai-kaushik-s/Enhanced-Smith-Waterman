#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <cstdint>
#include <unistd.h>

int TILE_SIZE = 64;
int PREFETCH_DIST = 256; 

constexpr int matchScore = 2;
constexpr int mismatchPenalty = -1;
constexpr int gapPenalty = -2;

void computeSystemParams() {
    long l1CacheSize = 32 * 1024;
    long cacheLineSize = 64;

    long detectedL1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    long detectedLine = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    
    if (detectedL1 > 0) l1CacheSize = detectedL1;
    if (detectedLine > 0) cacheLineSize = detectedLine;
  
    int calculatedTile = (l1CacheSize / 2) / (3 * sizeof(int16_t));
    calculatedTile = (calculatedTile / 32) * 32;

    if (calculatedTile > 1024) calculatedTile = 1024;
    if (calculatedTile < 64) calculatedTile = 64;

    TILE_SIZE = calculatedTile;
    PREFETCH_DIST = cacheLineSize * 4;

    std::cout << "[System Detect] L1 Cache: " << l1CacheSize / 1024 << "KB, Line Size: " << cacheLineSize << "B" << std::endl;
    std::cout << "[Auto Tune] TILE_SIZE: " << TILE_SIZE << ", PREFETCH_DIST: " << PREFETCH_DIST << std::endl;
}

std::string generateSequence(int length) {
    const char alphabet[] = "ACGT";
    std::string sequence(length, ' ');
    for (int i = 0; i < length; i++) {
        sequence[i] = alphabet[rand() % 4];
    }
    return sequence;
}

int smithWatermanAvx2(const std::string& sequenceA, const std::string& sequenceB, int lengthA, int lengthB) 
{
    int globalMaxScore = 0;
    int maxDiag = std::min(lengthA, lengthB);

    std::vector<int16_t> pprevDiagScores(maxDiag, 0);
    std::vector<int16_t> prevDiagScores(maxDiag, 0);
    std::vector<int16_t> currDiagScores(maxDiag, 0);

    __m256i vecGapPenalty = _mm256_set1_epi16((int16_t)gapPenalty);
    __m256i vecZero = _mm256_setzero_si256();

    int totaldiags = lengthA + lengthB;

    for (int i = 2; i <= totaldiags; i++) {

        int minIdx = std::max(1, i - lengthB);
        int maxIdx = std::min(lengthA, i - 1);

        if (minIdx > maxIdx) {
            std::swap(pprevDiagScores, prevDiagScores);
            std::swap(prevDiagScores, currDiagScores);
            continue;
        }

        int diagWidth = maxIdx - minIdx + 1;
        int prevDiagIdx = i - 1;
        int pprevDiagIdx = i - 2;

        int prevMinRow = (prevDiagIdx >= 2) ? std::max(1, prevDiagIdx - lengthB) : 1;
        int prevMaxRow = (prevDiagIdx >= 2) ? std::min(lengthA, prevDiagIdx - 1) : 0;

        int pprevMinRow = (pprevDiagIdx >= 2) ? std::max(1, pprevDiagIdx - lengthB) : 1;
        int pprevMaxRow = (pprevDiagIdx >= 2) ? std::min(lengthA, pprevDiagIdx - 1) : 0;

        int prevLen = prevMaxRow - prevMinRow;
        int pprevLen = pprevMaxRow - pprevMinRow;

        #pragma omp parallel for schedule(static) reduction(max:globalMaxScore)
        for (int tileOffset = 0; tileOffset < diagWidth; tileOffset += TILE_SIZE) {
            if (tileOffset + TILE_SIZE < diagWidth) {
                const int16_t* pDiag = pprevDiagScores.data() + (tileOffset + TILE_SIZE);
                const int16_t* pUp   = prevDiagScores.data() + (tileOffset + TILE_SIZE);

                _mm_prefetch(reinterpret_cast<const char*>(pDiag), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(pUp),   _MM_HINT_T0);
                
                _mm_prefetch(reinterpret_cast<const char*>(pDiag + 16), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(pUp + 16),   _MM_HINT_T0);
            }
            
            int prefetchRow = minIdx + tileOffset + (TILE_SIZE / 2); 
            int prefetchCol = i - prefetchRow;
            if (prefetchRow < lengthA && prefetchCol > 0 && prefetchCol < lengthB) {
                _mm_prefetch(&sequenceA[prefetchRow], _MM_HINT_T0);
                _mm_prefetch(&sequenceB[prefetchCol], _MM_HINT_T0);
            }

            int elementsInTile = std::min(TILE_SIZE, diagWidth - tileOffset);

            for (int innerOffset = 0; innerOffset < elementsInTile; innerOffset += 16) {
                
                int currBlockOffset = tileOffset + innerOffset;
                int elementsInBlock = std::min(16, diagWidth - currBlockOffset);

                if (elementsInBlock <= 0) break;

                int startRow = minIdx + currBlockOffset;
                int startCol = i - startRow;

                int16_t arrDiag[16] = {0}, arrUp[16] = {0}, arrLeft[16] = {0};
                int16_t arrSub[16] = {0}, arrResults[16];

                int baseIdxDiag = (startRow - 1) - pprevMinRow;
                int baseIdxUp   = (startRow - 1) - prevMinRow;
                int baseIdxLeft =  startRow      - prevMinRow;

                for (int k = 0; k < elementsInBlock; k++) {
                    int currRow = startRow + k;
                    int currCol = startCol - k;

                    if (pprevDiagIdx >= 2 && currRow > 1 && currCol > 1) {
                        int idx = baseIdxDiag + k;
                        if (idx >= 0 && idx <= pprevLen)
                            arrDiag[k] = pprevDiagScores[idx];
                    }
                    if (prevDiagIdx >= 2 && currRow > 1) {
                        int idxUp = baseIdxUp + k;
                        if (idxUp >= 0 && idxUp <= prevLen)
                            arrUp[k] = prevDiagScores[idxUp];
                    }
                    if (prevDiagIdx >= 2 && currCol > 1) {
                        int idxLeft = baseIdxLeft + k;
                        if (idxLeft >= 0 && idxLeft <= prevLen)
                            arrLeft[k] = prevDiagScores[idxLeft];
                    }

                    arrSub[k] = (sequenceA[currRow - 1] == sequenceB[currCol - 1]) 
                                ? (int16_t)matchScore 
                                : (int16_t)mismatchPenalty;
                }

                __m256i vecDiag = _mm256_loadu_si256((__m256i *)arrDiag);
                __m256i vecUp   = _mm256_loadu_si256((__m256i *)arrUp);
                __m256i vecLeft = _mm256_loadu_si256((__m256i *)arrLeft);
                __m256i vecSub  = _mm256_loadu_si256((__m256i *)arrSub);

                __m256i vecMatchScore = _mm256_add_epi16(vecDiag, vecSub);
                vecUp                 = _mm256_add_epi16(vecUp, vecGapPenalty);
                vecLeft               = _mm256_add_epi16(vecLeft, vecGapPenalty);

                __m256i vecMaxScore = _mm256_max_epi16(vecMatchScore, vecUp);
                vecMaxScore         = _mm256_max_epi16(vecMaxScore, vecLeft);
                vecMaxScore         = _mm256_max_epi16(vecMaxScore, vecZero);

                _mm256_storeu_si256((__m256i *)arrResults, vecMaxScore);

                for (int k = 0; k < elementsInBlock; k++) {
                    int16_t currScore = arrResults[k];
                    currDiagScores[currBlockOffset + k] = currScore;
                    if (currScore > globalMaxScore) globalMaxScore = currScore;
                }
            }
        }

        std::swap(pprevDiagScores, prevDiagScores);
        std::swap(prevDiagScores, currDiagScores);
    }

    return globalMaxScore;
}

int smithWaterman(const std::string& sequenceA, const std::string& sequenceB, int lengthA, int lengthB) {
    std::vector<int> prevRowScores(lengthB + 1, 0);
    std::vector<int> currRowScores(lengthB + 1, 0);

    int maxScore = 0;

    for (int i = 1; i <= lengthA; i++) {
        currRowScores[0] = 0;
        
        int scoreUp   = prevRowScores[0]; 
        int scoreLeft = 0; 
        int scoreDiag;

        for (int j = 1; j <= lengthB; j++) {
            scoreDiag = prevRowScores[j - 1];
            scoreUp   = prevRowScores[j];
            scoreLeft = currRowScores[j - 1];

            int substitutionScore = (sequenceA[i - 1] == sequenceB[j - 1]) 
                                    ? matchScore 
                                    : mismatchPenalty;

            int valDiag = scoreDiag + substitutionScore;
            int valUp   = scoreUp   + gapPenalty;
            int valLeft = scoreLeft + gapPenalty;

            int currScore = valDiag;
            currScore = std::max(currScore, valUp);
            currScore = std::max(currScore, valLeft);
            currScore = std::max(currScore, 0);

            currRowScores[j] = currScore;
            if (currScore > maxScore)
                maxScore = currScore;
        }
        std::swap(prevRowScores, currRowScores);
    }
    return maxScore;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <sequenceLength> <numThreads>" << std::endl;
        return 1;
    }

    int sequenceLength = std::stoi(argv[1]);
    int numThreads = std::stoi(argv[2]);

    computeSystemParams();

    omp_set_num_threads(numThreads);
    srand(42); 

    std::string sequenceA = generateSequence(sequenceLength);
    std::string sequenceB = generateSequence(sequenceLength);

    int finalScore;
    double startTime = omp_get_wtime();

    if (sequenceLength > 4096) {
        #ifdef __AVX2__
            std::cout << "Using AVX2 optimized Smith-Waterman with Prefetching" << std::endl;
            finalScore = smithWatermanAvx2(sequenceA, sequenceB, sequenceLength, sequenceLength);
        #else
            finalScore = smithWaterman(sequenceA, sequenceB, sequenceLength, sequenceLength);
        #endif
    } else {
        std::cout << "Using standard Smith-Waterman" << std::endl;
        finalScore = smithWaterman(sequenceA, sequenceB, sequenceLength, sequenceLength);
    }
    
    double endTime = omp_get_wtime();
    std::cout << "Sequence length: " << sequenceLength << std::endl;
    std::cout << "Smith-Waterman score: " << finalScore << std::endl;
    std::cout << "Execution time: " << (endTime - startTime) << " seconds" << std::endl;

    return 0;
}