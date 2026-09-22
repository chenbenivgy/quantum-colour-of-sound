// Tick-level dump of the C++ kernel for one bin, to localise divergences from dsm_engine.kernel.
// Build: clang++ -O2 -std=c++17 -o kernel_dump kernel_dump.cpp
// Input (binary): double gx,gz,omega_s,phase,theta; int32 delay; int32 basis_len; char basis[]; int32 enc_len; char enc[];
//                 uint32 seed; int64 T; double amps[T]; double phs[T]
// Output (binary): double SX[T], SY[T], POP[T]; int8 MREC[T]; then the local unitary U (2^L x 2^L complex) as doubles re,im
#include "../src/dsm_engine.hpp"
#include <cstdio>
template <class T> static T rd(FILE* f) { T v; if (fread(&v, sizeof(T), 1, f) != 1) throw std::runtime_error("read"); return v; }
static std::string rds(FILE* f) { int32_t n = rd<int32_t>(f); std::string s(n, ' '); if (n && fread(&s[0], 1, n, f) != (size_t)n) throw std::runtime_error("read"); return s; }
int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb");
    dsm::KernelParams p;
    p.gx = rd<double>(f); p.gz = rd<double>(f); p.omega_s = rd<double>(f); p.phase = rd<double>(f); p.theta_clock = rd<double>(f);
    p.delay = rd<int32_t>(f); p.basis = rds(f); p.encode_target = rds(f);
    uint32_t seed = rd<uint32_t>(f); int64_t T = rd<int64_t>(f);
    std::vector<double> amps(T), phs(T);
    fread(amps.data(), 8, T, f); fread(phs.data(), 8, T, f); fclose(f);
    dsm::KernelOut o = dsm::kernel(amps, phs, p, seed);
    dsm::Collision c = dsm::build_collision(p.gx, p.gz, p.omega_s, 0, 0, 0, p.delay, p.phase, 1.0, p.interaction, p.theta_clock);
    FILE* g = fopen(argv[2], "wb");
    fwrite(o.SX.data(), 8, T, g); fwrite(o.SY.data(), 8, T, g); fwrite(o.POP.data(), 8, T, g); fwrite(o.MREC.data(), 1, T, g);
    for (auto& z : c.U.a) { double re = z.real(), im = z.imag(); fwrite(&re, 8, 1, g); fwrite(&im, 8, 1, g); }
    fclose(g);
    return 0;
}
