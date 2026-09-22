// End-to-end test of engine A: DsmCore writes a job into the spool dir, dsm_server.py
// (running separately) returns the trajectory store computed with the unchanged qutip
// kernel, the core synthesises and plays the loop; export for dsm_verify.py.
// Build: clang++ -O2 -std=c++17 -pthread -o core_test_A core_test_A.cpp
// Run:   python dsm_server.py --spool <spool> &   then   ./core_test_A <spool> <exportdir>
#include "../src/DsmCore.hpp"
#include <cstdio>
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: core_test_A spool exportdir\n"); return 1; }
    const double fs = 48000; dsm::Core core; core.setSampleRate(fs); core.setSpoolDir(argv[1]);
    dsm::CoreParams p; p.delay = 2; p.H = 256; p.D = 6; p.t_l = 1.0; p.engine = 1; p.past = true; p.ntraj = 2; p.render = 1; core.setParams(p);
    auto input = [&](size_t n) { double t = n / fs; return (float)(0.5 * std::sin(2 * M_PI * 130.81 * t) * (std::fmod(t, 0.5) < 0.3)); };
    size_t n = 0; std::vector<float> out;
    auto run = [&](double secs) { size_t N = (size_t)(secs * fs); for (size_t i = 0; i < N; i++, n++) out.push_back(core.process(input(n))); };
    run(2.0); core.setOn(true);
    auto t0 = std::chrono::steady_clock::now();
    while (core.state() != dsm::Core::PLAYING) {
        run(0.01); std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (core.hasError()) { printf("ERROR: %s\n", core.lastError().c_str()); return 1; }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 600) { printf("timeout waiting for engine A\n"); return 1; }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    run(2.0);
    double s = 0; for (size_t i = out.size() - 96000; i < out.size(); i++) s += (double)out[i] * out[i];
    printf("engine A render adopted after %.1f s wall; loop %.3f s; %d trajectories stored; output rms %.4f\n", secs, core.loopSeconds(), core.storedTrajectories(), std::sqrt(s / 96000));
    bool ok = core.exportForVerification(argv[2]);
    printf("export %s -> %s\n", ok ? "ok" : "FAILED", argv[2]);
    return ok ? 0 : 1;
}
