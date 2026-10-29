# RTL tile stream protocol v1

`ternip_tile_group` is a free-running, data-driven AXI4-Stream kernel. Both
streams contain 512-bit beats with TVALID/TREADY and no TLAST. Host code never
submits tiles: the resident model controller issues these transactions on card.
The fixed logical tile dimension is 256. Each lane instantiates `ternip_core`
with `BatchSize=1`, 16-bit signed integer arithmetic (exponent zero), and its own
vector registers. Ternary parallelism is 128 or 256; vector parallelism is 4, 8,
or 16. The model controller retains FP32 vectors and int32 accumulation between
tiles. This numerical path does not invoke the legacy fixed-point RMS/activation
units; those units have not been qualified for the checkpoint.

A command consists of:

1. One header beat. Bits `Lanes-1:0` select active lanes; every other bit is zero.
2. Eight activation beats for each active lane, in increasing lane order. Each
   beat holds 32 little-endian signed-int16 values, lowest vector index first.
   Values must lie in [-127,127]. Inactive lanes receive no vector beats.
3. Exactly 256 shared weight beats. Each beat is one matrix row with 256 packed
   ternary entries, two bits per entry: 0=zero, 1=+1, 3=-1. Code 2 is invalid.

A response contains eight signed-int16 result beats per active lane, in
increasing lane order, followed by one status beat. Status bit 0 reports a
protocol or lane stream error; other bits are reserved zero. A zero active mask
consumes just a header and returns just status (no weights). The maximum legal
tile result magnitude is 256*127=32512, which fits int16 without saturation.

The controller pads partial vectors and matrix rows with zeros, accumulates all
column tiles in int32, and only then applies activation and weight scales.
Normalization and activation absmax always cover the complete logical vector.
The final projection performs deterministic lowest-index argmax over valid rows.

The broadcast buffer tracks consumption independently per lane. A stalled lane
cannot consume a beat twice; incoming weights advance only when every selected
lane has accepted the previous beat. All stream sources must hold data stable
while valid and not ready. Reset abandons the complete transaction and clears
lane/broadcast state. Both ends of a hardware stream must be reset together;
software request reset occurs at a completed command boundary, not mid-packet.
After an internal lane error, reload/reset the kernel before resubmission.

The RTL group and its HLS controller share one SLR and one requested clock.
`target.link_config` generates both CU instances and both stream connections.
External DDR/HBM memory mappings belong to the controller's weight/state/control
AXI masters, independently of the SLR number. Packaging/link recipes are
unbuilt until tested with an installed AMD 2022.2 platform.

`rtl_model_test.py` runs the real RTL through `rtl_transport.cpp`, using random
input gaps and output stalls. Its protocol regression checks all two-lane masks,
maximum positive/negative results, malformed headers, and reset during weight
broadcast. Its recurrent-model test checks unequal prompts and partial tiles.
This is C++/RTL co-simulation; it does not verify synthesized HLS float operators,
physical AXI memory traffic, routed timing, or board performance.
