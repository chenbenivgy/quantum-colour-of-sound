RACK_DIR ?= $(HOME)/Rack-SDK

FLAGS += -std=c++17
CXXFLAGS += -std=c++17
# Rack's compile.mk adds -funsafe-math-optimizations; the quantum engine must be IEEE-strict so the
# plugin binary reproduces the certified arithmetic (EXTRA_FLAGS is appended after Rack's flags).
EXTRA_FLAGS += -fno-unsafe-math-optimizations -ffp-contract=off

SOURCES += src/plugin.cpp
SOURCES += src/DSM.cpp

DISTRIBUTABLES += res
DISTRIBUTABLES += $(wildcard LICENSE*)

include $(RACK_DIR)/plugin.mk
