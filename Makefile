CC  := gcc
CXX := g++

CFLAGS   := -O3 -march=native -mfma -fopenmp -DNDEBUG -g
CXXFLAGS := -O3 -march=native -mfma -fopenmp -DNDEBUG -g -std=c++17
LDFLAGS  := -fopenmp -lnuma -funroll-loops

BASELINE_DIR   := src/baseline
OPTIMIZED_DIR  := src/optimized 
BUILD_DIR      := bin

BASELINE_TARGET  := $(BUILD_DIR)/sw_baseline
OPTIMIZED_TARGET := $(BUILD_DIR)/sw_opt

BASELINE_SRC  := $(BASELINE_DIR)/sw_baseline.c
OPTIMIZED_SRC := $(OPTIMIZED_DIR)/sw_opt.cpp

all: $(BUILD_DIR) $(OPTIMIZED_TARGET) $(BASELINE_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OPTIMIZED_TARGET): $(OPTIMIZED_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BASELINE_TARGET): $(BASELINE_SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BASELINE_TARGET) $(OPTIMIZED_TARGET)
