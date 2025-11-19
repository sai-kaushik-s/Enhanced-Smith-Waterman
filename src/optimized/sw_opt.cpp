#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <cstdint>

constexpr int matchScore = 2;
constexpr int mismatchPenalty = -1;
constexpr int gapPenalty = -2;

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

    int maxDiagonalLength = std::min(lengthA, lengthB);

    std::vector<int16_t> scoresTwoDiagonalsBack(maxDiagonalLength, 0);
    std::vector<int16_t> scoresOneDiagonalBack(maxDiagonalLength, 0);
    std::vector<int16_t> currentDiagonalScores(maxDiagonalLength, 0);

    __m256i vecGapPenalty = _mm256_set1_epi16((int16_t)gapPenalty);
    __m256i vecZero = _mm256_setzero_si256();

    int totalDiagonals = lengthA + lengthB;

    for (int diagonalIndex = 2; diagonalIndex <= totalDiagonals; diagonalIndex++) {

        int minRowIndex = std::max(1, diagonalIndex - lengthB);
        int maxRowIndex = std::min(lengthA, diagonalIndex - 1);

        if (minRowIndex > maxRowIndex) {
            std::swap(scoresTwoDiagonalsBack, scoresOneDiagonalBack);
            std::swap(scoresOneDiagonalBack, currentDiagonalScores);
            continue;
        }

        int diagonalWidth = maxRowIndex - minRowIndex + 1;
        
        int prevDiagIndex = diagonalIndex - 1;
        int prevPrevDiagIndex = diagonalIndex - 2;

        int minRowPrev, maxRowPrev;
        if (prevDiagIndex >= 2) {
            minRowPrev = std::max(1, prevDiagIndex - lengthB);
            maxRowPrev = std::min(lengthA, prevDiagIndex - 1);
        } else {
            minRowPrev = 1; maxRowPrev = 0;
        }

        int minRowPrevPrev, maxRowPrevPrev;
        if (prevPrevDiagIndex >= 2) {
            minRowPrevPrev = std::max(1, prevPrevDiagIndex - lengthB);
            maxRowPrevPrev = std::min(lengthA, prevPrevDiagIndex - 1);
        } else {
            minRowPrevPrev = 1; maxRowPrevPrev = 0;
        }

        #pragma omp parallel for schedule(static) reduction(max:globalMaxScore)
        for (int tileOffset = 0; tileOffset < diagonalWidth; tileOffset += 32) {
            int currentBlockOffset = tileOffset;
            int elementsInBlock = std::min(16, diagonalWidth - currentBlockOffset);

            if (elementsInBlock > 0) {
                int startRow = minRowIndex + currentBlockOffset;
                int startCol = diagonalIndex - startRow;

                int16_t arrDiag[16] = {0}, arrUp[16] = {0}, arrLeft[16] = {0};
                int16_t arrSub[16] = {0}, arrResults[16];

                int baseIndexDiag = (startRow - 1) - minRowPrevPrev;
                int baseIndexUp   = (startRow - 1) - minRowPrev;
                int baseIndexLeft =  startRow      - minRowPrev;

                int lengthPrev = maxRowPrev - minRowPrev;
                int lengthPrevPrev = maxRowPrevPrev - minRowPrevPrev;

                for (int k = 0; k < elementsInBlock; k++) {
                    int currentRow = startRow + k;
                    int currentCol = startCol - k;

                    if (prevPrevDiagIndex >= 2 && currentRow > 1 && currentCol > 1) {
                        int idx = baseIndexDiag + k;
                        if (idx >= 0 && idx <= lengthPrevPrev)
                            arrDiag[k] = scoresTwoDiagonalsBack[idx];
                    }

                    if (prevDiagIndex >= 2 && currentRow > 1) {
                        int idxUp = baseIndexUp + k;
                        if (idxUp >= 0 && idxUp <= lengthPrev)
                            arrUp[k] = scoresOneDiagonalBack[idxUp];
                    }

                    if (prevDiagIndex >= 2 && currentCol > 1) {
                        int idxLeft = baseIndexLeft + k;
                        if (idxLeft >= 0 && idxLeft <= lengthPrev)
                            arrLeft[k] = scoresOneDiagonalBack[idxLeft];
                    }

                    arrSub[k] = (sequenceA[currentRow - 1] == sequenceB[currentCol - 1]) 
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
                    int16_t calculatedScore = arrResults[k];
                    currentDiagonalScores[currentBlockOffset + k] = calculatedScore;
                    if (calculatedScore > globalMaxScore) globalMaxScore = calculatedScore;
                }
            }

            int nextBlockOffset = tileOffset + 16;
            int elementsInNextBlock = std::min(16, diagonalWidth - nextBlockOffset);

            if (elementsInNextBlock > 0) {
                int startRow = minRowIndex + nextBlockOffset;
                int startCol = diagonalIndex - startRow;

                int16_t arrDiag[16] = {0}, arrUp[16] = {0}, arrLeft[16] = {0};
                int16_t arrSub[16] = {0}, arrResults[16];

                int baseIndexDiag = (startRow - 1) - minRowPrevPrev;
                int baseIndexUp   = (startRow - 1) - minRowPrev;
                int baseIndexLeft =  startRow      - minRowPrev;

                int lengthPrev = maxRowPrev - minRowPrev;
                int lengthPrevPrev = maxRowPrevPrev - minRowPrevPrev;

                for (int k = 0; k < elementsInNextBlock; k++) {
                    int currentRow = startRow + k;
                    int currentCol = startCol - k;

                    if (prevPrevDiagIndex >= 2 && currentRow > 1 && currentCol > 1) {
                        int idx = baseIndexDiag + k;
                        if (idx >= 0 && idx <= lengthPrevPrev)
                            arrDiag[k] = scoresTwoDiagonalsBack[idx];
                    }

                    if (prevDiagIndex >= 2 && currentRow > 1) {
                        int idxUp = baseIndexUp + k;
                        if (idxUp >= 0 && idxUp <= lengthPrev)
                            arrUp[k] = scoresOneDiagonalBack[idxUp];
                    }

                    if (prevDiagIndex >= 2 && currentCol > 1) {
                        int idxLeft = baseIndexLeft + k;
                        if (idxLeft >= 0 && idxLeft <= lengthPrev)
                            arrLeft[k] = scoresOneDiagonalBack[idxLeft];
                    }

                    arrSub[k] = (sequenceA[currentRow - 1] == sequenceB[currentCol - 1]) 
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

                for (int k = 0; k < elementsInNextBlock; k++) {
                    int16_t calculatedScore = arrResults[k];
                    currentDiagonalScores[nextBlockOffset + k] = calculatedScore;
                    if (calculatedScore > globalMaxScore) globalMaxScore = calculatedScore;
                }
            }
        }

        std::swap(scoresTwoDiagonalsBack, scoresOneDiagonalBack);
        std::swap(scoresOneDiagonalBack, currentDiagonalScores);
    }

    return globalMaxScore;
}

int smithWatermanScalar(const std::string& sequenceA, const std::string& sequenceB, int lengthA, int lengthB) {
    std::vector<int> previousRowScores(lengthB + 1, 0);
    std::vector<int> currentRowScores(lengthB + 1, 0);

    int maxScore = 0;

    for (int i = 1; i <= lengthA; i++) {
        currentRowScores[0] = 0;
        
        int scoreUp   = previousRowScores[0]; 
        int scoreLeft = 0; 
        int scoreDiag;

        for (int j = 1; j <= lengthB; j++) {
            scoreDiag = previousRowScores[j - 1];
            scoreUp   = previousRowScores[j];
            scoreLeft = currentRowScores[j - 1];

            int substitutionScore = (sequenceA[i - 1] == sequenceB[j - 1]) 
                                    ? matchScore 
                                    : mismatchPenalty;

            int valDiag = scoreDiag + substitutionScore;
            int valUp   = scoreUp   + gapPenalty;
            int valLeft = scoreLeft + gapPenalty;

            int currentCellScore = valDiag;
            currentCellScore = std::max(currentCellScore, valUp);
            currentCellScore = std::max(currentCellScore, valLeft);
            currentCellScore = std::max(currentCellScore, 0);

            currentRowScores[j] = currentCellScore;
            if (currentCellScore > maxScore)
                maxScore = currentCellScore;
        }

        std::swap(previousRowScores, currentRowScores);
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

    omp_set_num_threads(numThreads);

    srand(42); 

    std::string sequenceA = generateSequence(sequenceLength);
    std::string sequenceB = generateSequence(sequenceLength);

    int finalScore;

    double startTime = omp_get_wtime();

    if (sequenceLength > 4096) {
        #ifdef __AVX2__
            std::cout << "Using AVX2 optimized Smith-Waterman" << std::endl;
            finalScore = smithWatermanAvx2(sequenceA, sequenceB, sequenceLength, sequenceLength);
        #elif defined(__AVX512F__)
            finalScore = smithWatermanScalar(sequenceA, sequenceB, sequenceLength, sequenceLength); 
        #else
            finalScore = smithWatermanScalar(sequenceA, sequenceB, sequenceLength, sequenceLength);
        #endif
    } else {
        std::cout << "Using standard Smith-Waterman" << std::endl;
        finalScore = smithWatermanScalar(sequenceA, sequenceB, sequenceLength, sequenceLength);
    }
    
    double endTime = omp_get_wtime();
    double elapsedTime = endTime - startTime;
    
    std::cout << "Sequence length: " << sequenceLength << std::endl;
    std::cout << "Smith-Waterman score: " << finalScore << std::endl;
    std::cout << "Execution time: " << elapsedTime << " seconds" << std::endl;

    return 0;
}