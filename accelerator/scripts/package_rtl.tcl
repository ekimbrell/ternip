# SPDX-License-Identifier: BSD-3-Clause
# Called by build.py after writing a source manifest and fixed-parameter wrapper.
# Flow follows AMD Vitis 2022.2 bottom-up RTL kernel packaging.
if {$argc != 3} { error "Usage: package_rtl.tcl inputs.tcl FPGA_PART OUTPUT_DIR" }
source [lindex $argv 0]
set part [lindex $argv 1]
set output [file normalize [lindex $argv 2]]
create_project ternip_rtl [file join $output project] -part $part
set_property source_mgmt_mode None [current_project]
add_files -norecurse $rtl_sources
set_property file_type SystemVerilog [get_files $rtl_sources]
add_files -norecurse $rtl_headers
set_property file_type {Verilog Header} [get_files $rtl_headers]
set_property include_dirs $rtl_includes [current_fileset]
set_property verilog_define {TERNIP_CFG=ternip_pkg::RegressionCfg} [current_fileset]
set_property top ternip_rtl [current_fileset]

ipx::package_project -root_dir [file join $output ip] -vendor ternip.org -library accelerator -taxonomy /KernelIP -import_files
set core [ipx::current_core]
set_property name ternip_rtl $core
set_property version 1.0 $core
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
ipx::infer_bus_interface ap_clk xilinx.com:signal:clock_rtl:1.0 $core
ipx::infer_bus_interface ap_rst_n xilinx.com:signal:reset_rtl:1.0 $core
foreach name {command response} {
    if {[llength [ipx::get_bus_interfaces $name -of_objects $core]] != 1} {
        error "AXI stream interface $name was not inferred"
    }
    ipx::associate_bus_interfaces -busif $name -clock ap_clk $core
}
ipx::associate_bus_interfaces -clock ap_clk -reset ap_rst_n $core
# Vitis assigns the descriptor's clock at link time. A fixed FREQ_HZ on the
# packaged interface can conflict with the platform's default kernel clock.
set_property supported_families [list [get_property FAMILY [get_parts $part]] Production] $core
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity -kernel $core
ipx::save_core $core
package_xo -xo_path [file join $output tiles.xo] -kernel_name ternip_rtl -ip_directory [file join $output ip] -ctrl_protocol ap_ctrl_none
close_project
