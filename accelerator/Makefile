PYTHON ?= python3
BSG ?= ../build/basejump_stl
TARGET ?= targets/u280-ddr-rtl.json
OUT ?= ../build/u280-ddr
MODE ?= hw_emu
PLATFORM_FILE ?=
.PHONY: test lint rtl-test rtl-model-test numerics cpu build
CPU_CASE ?= ../build/cpu-case
RTL_CASE ?= ../build/rtl-case
RTL_OUT ?= ../build/rtl-model-test
VERILATOR ?= verilator
test:
	$(PYTHON) -m pytest tests -q
lint:
	$(PYTHON) scripts/lint.py --bsg $(BSG)
rtl-test:
	$(PYTHON) scripts/rtl_tests.py
cpu:
	mkdir -p ../build
	g++ -std=c++17 -O2 -ffp-contract=off -Wno-unknown-pragmas -DTERNIP_CPU -DTERNIP_LANES=2 hls/ternip_group.cpp hls/cpu_runner.cpp -o ../build/cpu_runner
	$(PYTHON) scripts/prepare_device_test.py $(CPU_CASE)
	../build/cpu_runner $(CPU_CASE)/device
	$(PYTHON) scripts/check_device_test.py $(CPU_CASE)/device
build:
	@test -n "$(PLATFORM_FILE)" || (echo "Set PLATFORM_FILE to the installed U280 development .xpfm"; exit 1)
	$(PYTHON) scripts/build.py --target $(TARGET) --output $(OUT) --mode $(MODE) --bsg $(BSG) --platform-file "$(PLATFORM_FILE)"

rtl-model-test:
	$(PYTHON) scripts/prepare_device_test.py $(RTL_CASE)
	$(PYTHON) scripts/rtl_model_test.py --bsg $(BSG) --verilator $(VERILATOR) --output $(RTL_OUT) --case $(RTL_CASE)/device
numerics:
	mkdir -p ../build
	g++ -std=c++17 -O2 -ffp-contract=off -fPIC -shared -fopenmp hls/numerical_cpu.cpp -o ../build/libternip_numerics.so
