# Choose compiler and check TBB location
CXX = g++-16
TBB_DIR := $(shell brew --prefix tbb)

# Build and source directories
BUILD_DIR := build
SRC_DIR := .

# All source files
SRCS := mdmain.cpp interactions.cpp dataManager.cpp SimulationContext.cpp neighborlist.cpp timestep.cpp populations.cpp nutrientField.cpp
# Automatically map .cpp to build/*.o
OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Debug/Release flags
ifeq ($(DEBUG),true)
    OPTFLAGS = -g3
    MSG=Running in debug mode
else
    OPTFLAGS = -O3 -Wuninitialized -Wmaybe-uninitialized
    MSG=Running in release mode
endif

# Add dependency flags to CXXFLAGS
CXXFLAGS = $(OPTFLAGS) -Wall -std=c++23 -m64 -fopenmp -MMD -MP
CPPFLAGS = -I$(TBB_DIR)/include
LDFLAGS  = -L$(TBB_DIR)/lib
LDLIBS   = -ltbb -lpthread -lm

# Default target
.PHONY: all
all: $(BUILD_DIR)/main.exe

# Build directory
$(BUILD_DIR):
	@mkdir -p $@

# Pattern rule for all object files, auto-creates build dir
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Link final executable
$(BUILD_DIR)/main.exe: $(OBJS)
	@echo "Building in $(if $(DEBUG),debug,release) mode"
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Clean
.PHONY: clean
clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d $(BUILD_DIR)/main.exe

# Optional: dependencies auto-generation for more robustness (C++23)
-include $(OBJS:.o=.d)
