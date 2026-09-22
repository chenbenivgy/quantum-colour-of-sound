// Standalone driver for the C++ engine: reads a job file, writes every stage.
// Build: clang++ -O2 -std=c++17 -o engine_test engine_test.cpp
// Job file (binary, little-endian):
//   char magic[8] = "DSMJOB02"; double fs_engine; int32 delay,H,D; double t_l,gx,gz; uint32 seed;
//   double phase, amp_pow, osr; int32 clock, n_traj; int32 basis_len; char basis[]; int32 enc_len; char enc[];
//   int32 render; int32 traj_split; int32 nthreads; int64 n; double x[n]
// Output dir gets: x_model.f64, in_mag.f64, in_phase.f64, out_mag.f64, out_phase.f64, mrec.i8,
//   SX.f64, SY.f64, POP.f64 (layout [ti][k][t]), y_model.f64, y_engine.f64, geometry.txt
#include "../src/dsm_engine.hpp"
#include <cstdio>
#include <chrono>

template <class T> static T rd(FILE* f) { T v; if (fread(&v, sizeof(T), 1, f) != 1) throw std::runtime_error("read"); return v; }
static std::string rds(FILE* f) { int32_t n = rd<int32_t>(f); std::string s(n, ' '); if (n && fread(&s[0], 1, n, f) != (size_t)n) throw std::runtime_error("read"); return s; }
template <class T> static void wr(const std::string& path, const std::vector<T>& v) { FILE* f = fopen(path.c_str(), "wb"); fwrite(v.data(), sizeof(T), v.size(), f); fclose(f); }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: engine_test job.bin outdir\n"); return 1; }
    FILE* f = fopen(argv[1], "rb"); if (!f) { perror("job"); return 1; }
    char magic[8]; if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "DSMJOB02", 8)) { fprintf(stderr, "bad magic\n"); return 1; }
    double fs_e = rd<double>(f);
    dsm::JobParams p;
    p.delay = rd<int32_t>(f); p.H = rd<int32_t>(f); p.D = rd<int32_t>(f);
    p.t_l = rd<double>(f); p.gamma_x = rd<double>(f); p.gamma_z = rd<double>(f); p.seed = rd<uint32_t>(f);
    p.phase = rd<double>(f); p.amp_pow = rd<double>(f); p.omega_s_res = rd<double>(f);
    p.carrier_clock = rd<int32_t>(f); p.n_traj = rd<int32_t>(f);
    p.basis = rds(f); p.encode_target = rds(f);
    p.render = rd<int32_t>(f); int traj_split = rd<int32_t>(f);
    int nthreads = rd<int32_t>(f);
    int64_t n = rd<int64_t>(f);
    std::vector<double> x(n); if (fread(x.data(), sizeof(double), n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    std::string out = argv[2];
    auto t0 = std::chrono::steady_clock::now();
    dsm::Store st;
    dsm::JobResult r = dsm::process_job(x, fs_e, p, nthreads, [](int d, int nb) { if (d % 64 == 0 || d == nb) fprintf(stderr, "  %d/%d\n", d, nb); return true; }, &st, traj_split);
    dsm::PinkFit pf; if (p.render == 4) { dsm::Material m = dsm::prepare_material(x, fs_e, p); dsm::Mat qm, qp, cm, cp; dsm::render_from_store(st, 0, p.n_traj, p.delay, p.amp_pow, m.fed_phase, qm, qp); pf = dsm::pink_render(m.fed_mag, m.fed_phase, qm, qp, p.seed, cm, cp); }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    wr(out + "/x_model.f64", r.x_model); wr(out + "/in_mag.f64", r.in_mag.a); wr(out + "/in_phase.f64", r.in_phase.a);
    wr(out + "/out_mag.f64", r.out_mag.a); wr(out + "/out_phase.f64", r.out_phase.a); wr(out + "/mrec.i8", st.MREC);
    wr(out + "/SX.f64", st.SX); wr(out + "/SY.f64", st.SY); wr(out + "/POP.f64", st.POP);
    wr(out + "/y_model.f64", r.y_model); wr(out + "/y_engine.f64", r.y_engine);
    FILE* g = fopen((out + "/geometry.txt").c_str(), "w");
    fprintf(g, "nf %d nb %d W %d N %d H %d T %d fs_m %.6f scale %.17g secs %.3f nthreads %d ntraj %d", r.nf, r.in_mag.nb, r.W, r.N, r.H, r.T, r.fs_m, r.scale, secs, nthreads, st.ntraj);
    if (p.render == 4) fprintf(g, " pink %.17g %.3f %.17g", pf.depth, pf.P, pf.am);
    fprintf(g, "\n");
    fclose(g);
    fprintf(stderr, "done in %.2f s (%d threads)\n", secs, nthreads);
    return 0;
}
