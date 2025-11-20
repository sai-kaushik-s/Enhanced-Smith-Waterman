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
#include <atomic>
#include <memory>

int BLOCK_SIZE = 64;
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

    BLOCK_SIZE = calculatedTile;
    PREFETCH_DIST = cacheLineSize * 4;

    std::cout << "[System Detect] L1 Cache: " << l1CacheSize / 1024 << "KB, Line Size: " << cacheLineSize << "B" << std::endl;
    std::cout << "[Auto Tune] BLOCK_SIZE: " << BLOCK_SIZE << ", PREFETCH_DIST: " << PREFETCH_DIST << std::endl;
}

std::string generateSequence(int length) {
    const char alphabet[] = "ACGT";
    std::string sequence(length, ' ');
    for (int i = 0; i < length; i++) {
        sequence[i] = alphabet[rand() % 4];
    }
    return sequence;
}

void solveBlock(
    int bRow, int bCol, 
    int numBlockRows, int numBlockCols,
    int L_A, int L_B,
    const std::string& seqA, const std::string& seqB,
    std::vector<int16_t>& globalTop, 
    std::vector<int16_t>& globalLeft, 
    int explicitDiagonal,
    int& threadMax
) {
    int rStart = bRow * BLOCK_SIZE;
    int cStart = bCol * BLOCK_SIZE;
    int rEnd = std::min(rStart + BLOCK_SIZE, L_A);
    int cEnd = std::min(cStart + BLOCK_SIZE, L_B);
    
    int rLen = rEnd - rStart;
    int cLen = cEnd - cStart;

    std::vector<int16_t> prevRow(cLen + 1);
    std::vector<int16_t> currRow(cLen + 1);
    
    for(int j=0; j<cLen; j++) prevRow[j+1] = globalTop[cStart + j];

    prevRow[0] = explicitDiagonal;

    for(int i=0; i<rLen; i++) {
        int gRow = rStart + i;

        if (gRow + PREFETCH_DIST < L_A) {
            _mm_prefetch(&seqA[gRow + PREFETCH_DIST], _MM_HINT_T0);
            _mm_prefetch((const char*)&globalLeft[gRow + (PREFETCH_DIST / 2)], _MM_HINT_T0);
        }
        
        if (bCol == 0) currRow[0] = 0;
        else currRow[0] = globalLeft[gRow];
        
        int left = currRow[0];
        int up   = prevRow[1];
        int diag = prevRow[0]; 

        #pragma omp simd 
        for(int j=0; j<cLen; j++) {
            int gCol = cStart + j;
            
            int nextUp = (j < cLen - 1) ? prevRow[j+2] : 0;
            
            int substitution = (seqA[gRow] == seqB[gCol]) ? matchScore : mismatchPenalty;
            int val = diag + substitution;
            val = std::max(val, up + gapPenalty);
            val = std::max(val, left + gapPenalty);
            val = std::max(val, 0);
            
            currRow[j+1] = (int16_t)val;
            if (val > threadMax) threadMax = val;
            
            diag = up;   
            left = val;  
            up   = nextUp;
        }
        
        globalLeft[gRow] = currRow[cLen];
        std::swap(prevRow, currRow);
    }
    
    for(int j=0; j<cLen; j++) globalTop[cStart + j] = prevRow[j+1];
}

int smithWatermanAsync(const std::string& seqA, const std::string& seqB, int lenA, int lenB) {
    int globalMax = 0;
    
    int numBlockRows = (lenA + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numBlockCols = (lenB + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::vector<int16_t> boundaryTop(lenB, 0);
    std::vector<int16_t> boundaryLeft(lenA, 0);

    auto row_progress = std::make_unique<std::atomic<int>[]>(numBlockRows);
    for(int i = 0; i < numBlockRows; i++) row_progress[i].store(-1);

    #pragma omp parallel 
    {
        int tMax = 0;
        int numThreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        for (int r = tid; r < numBlockRows; r += numThreads) {
            int currentDiag = 0;

            for (int c = 0; c < numBlockCols; c++) {
                if (r > 0) {
                    while (row_progress[r - 1].load(std::memory_order_acquire) < c) {
                        _mm_pause(); 
                    }
                }
                
                int nextDiag = 0;
                int nextBlockEndCol = (c + 1) * BLOCK_SIZE; 
                if (nextBlockEndCol <= lenB) {
                    nextDiag = boundaryTop[nextBlockEndCol - 1];
                } else if ((c * BLOCK_SIZE) < lenB) {
                    nextDiag = boundaryTop[lenB - 1];
                }
                solveBlock(r, c, numBlockRows, numBlockCols, lenA, lenB, seqA, seqB, boundaryTop, boundaryLeft, currentDiag, tMax);
                
                currentDiag = nextDiag;
                row_progress[r].store(c, std::memory_order_release);
            }
        }
        
        #pragma omp critical
        if (tMax > globalMax) globalMax = tMax;
    }

    return globalMax;
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

    finalScore = smithWatermanAsync(sequenceA, sequenceB, sequenceLength, sequenceLength);
    
    double endTime = omp_get_wtime();
    std::cout << "Sequence length: " << sequenceLength << std::endl;
    std::cout << "Smith-Waterman score: " << finalScore << std::endl;
    std::cout << "Execution time: " << (endTime - startTime) << " seconds" << std::endl;

    return 0;
}