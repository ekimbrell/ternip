// SPDX-License-Identifier: BSD-3-Clause
#include "ternip_group.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
template<class T> std::vector<T> read(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f) throw std::runtime_error("Cannot read "+path);
    auto bytes=f.tellg();if(bytes%sizeof(T)) throw std::runtime_error("Unaligned input");
    std::vector<T> v(bytes/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),bytes);return v;
}
template<class T> void write(const std::string& path,const std::vector<T>& v) {
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
    if(!f) throw std::runtime_error("Cannot write "+path);
}
int main(int argc,char** argv) {
    if(argc!=2) {std::cerr<<"Usage: cpu_runner CASE_DIRECTORY\n";return 2;}
    std::string dir=argv[1];
    try {
        auto w=read<WeightWord>(dir+"/weights.bin");auto p=read<uint32_t>(dir+"/program.bin");
        auto params=read<float>(dir+"/parameters.bin");auto s=read<float>(dir+"/state.in");
        auto c=read<uint32_t>(dir+"/control.in");auto scratch=read<float>(dir+"/scratch.in");
        auto cfg=read<uint32_t>(dir+"/config.bin");std::vector<uint64_t> counters(8);
        if(cfg.size()!=9) throw std::runtime_error("Bad config");
        ternip_group(w.data(),s.data(),c.data(),p.data(),params.data(),scratch.data(),counters.data(),
                     cfg[0],cfg[1],cfg[2],cfg[3],cfg[4],cfg[5],cfg[6],cfg[7],cfg[8]);
        write(dir+"/state.out",s);write(dir+"/control.out",c);write(dir+"/scratch.out",scratch);write(dir+"/counters.out",counters);
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
