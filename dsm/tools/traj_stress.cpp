// Stress test for the TRAJECTORIES knob: wall-clock times to first sound and to each
// appended set at the panel defaults and at heavier settings, knob sweeps, changes mid-job.
// Build: clang++ -O2 -std=c++17 -pthread -o traj_stress traj_stress.cpp
#include "../src/DsmCore.hpp"
#include <cstdio>
static const double fs = 48000; static dsm::Core* core; static size_t n = 0; static float last = 0;
static float input(size_t i) { double t = i / fs; return (float)(0.4 * std::sin(2 * M_PI * 130.81 * t) * (std::fmod(t, 0.5) < 0.3) + 0.1 * std::sin(2 * M_PI * 987 * t)); }
static void feed(double secs) { size_t N = (size_t)(secs * fs); for (size_t i = 0; i < N; i++, n++) last = core->process(input(n)); }
static double now() { static auto t0 = std::chrono::steady_clock::now(); return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
// feed audio in real-time-sized chunks until the core is idle; report wall seconds
static double settle(const char* what) {
    double t0 = now(); int quiet = 0;
    while (quiet < 10 && now() - t0 < 900) { feed(0.02); std::this_thread::sleep_for(std::chrono::milliseconds(20)); quiet = core->idle() ? quiet + 1 : 0; }
    double dt = now() - t0;
    printf("  %-44s %6.1f s wall   state %d  stored %d  loop %.2f s  layers %d%s\n", what, dt, (int)core->state(), core->storedTrajectories(), core->loopSeconds(), core->layerCount(), core->hasError() ? "  ERR" : "");
    if (core->hasError()) printf("     error: %s\n", core->lastError().c_str());
    return dt;
}
int main(int argc, char** argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 4;
    dsm::Core c; core = &c; c.setSampleRate(fs); c.setThreads(threads);
    c.setLogger([](const std::string& s) { printf("      log: %s\n", s.c_str()); });
    dsm::CoreParams p; p.delay = 3; p.H = 256; p.D = 6; p.t_l = 4.0; p.autoCoupling = true; p.engine = 0; p.past = true; p.ntraj = 1; p.render = 0;
    c.setParams(p); c.setTs1(1.0);
    feed(5.0);
    printf("A. panel defaults (d3 H256 /6 t_l 4), %d threads\n", threads);
    c.setOn(true); settle("On, ntraj 1");
    for (int k = 2; k <= 4; k++) { p.ntraj = k; c.setParams(p); settle(("knob -> " + std::to_string(k)).c_str()); }
    p.ntraj = 1; c.setParams(p); settle("knob -> 1 (synth only)");
    p.ntraj = 4; c.setParams(p); feed(0.35); p.delay = 4; c.setParams(p); settle("ntraj 4 then delay 4 mid-job");
    p.ntraj = 4; p.delay = 3; c.setParams(p); settle("delay back to 3 with ntraj 4 (fresh)");
    printf("B. quick sweep 1->4->1->4 within the debounce\n");
    p.ntraj = 1; c.setParams(p); feed(0.05); p.ntraj = 4; c.setParams(p); feed(0.05); p.ntraj = 1; c.setParams(p); feed(0.05); p.ntraj = 4; c.setParams(p); settle("after sweep");
    printf("C. heavier: d5, ntraj 1 then 4\n");
    p.delay = 5; p.ntraj = 1; c.setParams(p); settle("d5 ntraj 1");
    p.ntraj = 4; c.setParams(p); settle("d5 knob -> 4");
    printf("D. heaviest panel setting: d5, hop 1024, rate /1, ntraj 1 then 2\n");
    p.H = 1024; p.D = 1; p.ntraj = 1; c.setParams(p); settle("d5 H1024 /1 ntraj 1");
    p.ntraj = 2; c.setParams(p); settle("d5 H1024 /1 knob -> 2");
    c.setOn(false);
    printf("done\n");
    return 0;
}
