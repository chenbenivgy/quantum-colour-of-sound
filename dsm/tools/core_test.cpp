// Standalone test of the module logic (DsmCore v2) with a fake audio clock.
// Build: clang++ -O2 -std=c++17 -pthread -o core_test core_test.cpp
#include "../src/DsmCore.hpp"
#include <cstdio>
#include <cassert>

static double rms(const std::vector<float>& v, size_t a, size_t b) { double s = 0; for (size_t i = a; i < b; i++) s += (double)v[i] * v[i]; return std::sqrt(s / std::max<size_t>(1, b - a)); }
static const double fs = 48000;
static dsm::Core* core; static size_t n = 0; static std::vector<float> out;
static float input(size_t i) { double t = i / fs; return (float)(0.5 * std::sin(2 * M_PI * 130.81 * t) * (std::fmod(t, 0.5) < 0.3)); }
static void run(double secs) { size_t N = (size_t)(secs * fs); for (size_t i = 0; i < N; i++, n++) out.push_back(core->process(input(n))); }
static void waitIdle(const char* what) {     // feed audio (simulated time) until nothing is queued, running or pending
    // simulated audio time runs far faster than wall time: sleep a little per step so the worker progresses
    int quiet = 0; double w = 0;
    while (quiet < 20 && w < 600) { run(0.02); w += 0.02; std::this_thread::sleep_for(std::chrono::milliseconds(2)); quiet = core->idle() ? quiet + 1 : 0; }
    std::string e = core->lastError(); if (!e.empty()) printf("  [%s] error: %s\n", what, e.c_str());
}

int main() {
    dsm::Core c; core = &c; c.setSampleRate(fs); c.setThreads(8);
    c.setLogger([](const std::string& s) { printf("      log: %s\n", s.c_str()); });
    dsm::CoreParams p; p.delay = 2; p.H = 256; p.D = 6; p.t_l = 1.0; p.autoCoupling = true; p.engine = 0; p.past = true; p.ntraj = 1; p.render = 0;
    c.setParams(p); c.setTs1(1.0);
    // 0. tap fills while Off, output silent
    run(3.0); assert(rms(out, 0, out.size()) == 0.0 && c.state() == dsm::Core::OFF);
    // 1. PAST mode: On computes at once from the last t_l seconds; no recording wait
    size_t onAt = out.size(); c.setOn(true);
    assert(c.state() == dsm::Core::COMPUTING);
    waitIdle("first render");
    assert(c.state() == dsm::Core::PLAYING);
    printf("1. PAST: playing %.3f s loop, material offset %.0f (= press sample %zu - loop length %zu)\n", c.loopSeconds(), c.currentMatOffset(), onAt, (size_t)(c.loopSeconds() * fs));
    assert(std::llround(c.currentMatOffset()) == (long long)onAt - (long long)std::llround(c.loopSeconds() * fs));
    size_t playStart = out.size(); run(2.5);
    size_t L = (size_t)std::lround(c.loopSeconds() * fs);
    double dmax = 0; for (size_t i = playStart + 100; i + L < out.size(); i++) dmax = std::max(dmax, (double)std::fabs(out[i] - out[i + L]));
    printf("   loop periodicity max|y[t]-y[t+L]| = %.2e, rms %.4f\n", dmax, rms(out, playStart, out.size())); assert(dmax < 1e-6 && rms(out, playStart, out.size()) > 1e-3);
    long phys0 = c.physicsJobsRun(), syn0 = c.synthJobsRun();
    // 2. render switch: synth only, no physics job
    p.render = 3; c.setParams(p); run(0.2); waitIdle("twin");
    p.render = 0; c.setParams(p); run(0.2); waitIdle("undep");
    printf("2. render switches: physics jobs %ld -> %ld (unchanged), synth jobs %ld -> %ld\n", phys0, c.physicsJobsRun(), syn0, c.synthJobsRun());
    assert(c.physicsJobsRun() == phys0 && c.synthJobsRun() >= syn0 + 2);
    // 3. trajectories up: incremental physics job; down: synth only
    p.ntraj = 3; c.setParams(p); run(0.4); waitIdle("ntraj 3");
    printf("3. ntraj 1->3: stored %d trajectories, physics jobs %ld\n", c.storedTrajectories(), c.physicsJobsRun()); assert(c.storedTrajectories() == 3);
    long phys1 = c.physicsJobsRun(); p.ntraj = 2; c.setParams(p); run(0.4); waitIdle("ntraj 2");
    printf("   ntraj 3->2: physics jobs %ld (unchanged), stored still %d\n", c.physicsJobsRun(), c.storedTrajectories()); assert(c.physicsJobsRun() == phys1 && c.storedTrajectories() == 3);
    p.render = 1; c.setParams(p); run(0.4); waitIdle("dep0"); assert(c.physicsJobsRun() == phys1);
    // 4. AUTO on: touching gx/gz knobs must not recompute
    p.gx = 0.9; p.gz = 1.0; c.setParams(p); run(0.5); waitIdle("gx knob with AUTO");
    printf("4. gx/gz knobs with AUTO on: physics jobs %ld (unchanged)\n", c.physicsJobsRun()); assert(c.physicsJobsRun() == phys1);
    // 5. t_l change in PAST mode: re-cut from the same capture, aligned on the anchor
    p.t_l = 0.6; c.setParams(p); run(0.4); waitIdle("t_l 0.6");
    printf("5. t_l 1.0->0.6: loop %.3f s, material offset %.0f (press %zu - %zu)\n", c.loopSeconds(), c.currentMatOffset(), onAt, (size_t)(c.loopSeconds() * fs));
    assert(std::llround(c.currentMatOffset()) == (long long)onAt - (long long)std::llround(c.loopSeconds() * fs));
    // 6. physics change during a long computation: the running job is aborted
    p.delay = 5; p.t_l = 3.0; c.setParams(p); run(0.35);      // debounce passes, d=5 job starts (seconds of compute)
    double w = 0; while (!c.computing() && w < 5) { run(0.01); w += 0.01; }
    p.delay = 1; c.setParams(p); run(0.35);                  // new physics -> abort
    waitIdle("abort");
    printf("6. abort on physics change: aborted jobs %ld, final delay %d, loop %.3f s\n", c.physicsJobsAborted(), c.currentParams().delay, c.loopSeconds());
    assert(c.physicsJobsAborted() >= 1 && c.state() == dsm::Core::PLAYING);
    // 7. TS1 = 2: glide, loop takes 2 s
    c.setTs1(2.0); run(4.0); size_t tail = out.size(); run(7.0); L = (size_t)std::lround(c.loopSeconds() * fs);
    double d2 = 0; size_t first = 0; for (size_t i = tail; i + 2 * L < out.size(); i++) { double d = std::fabs(out[i] - out[i + 2 * L]); if (d > 1e-3 && !first) first = i; d2 = std::max(d2, d); }
    printf("7. TS1=2 periodicity at 2L: %.2e (layers %d, rate %.4f, L %zu, first mismatch at +%.3f s, matPos %.1f, physics jobs %ld, synth %ld)\n",
           d2, c.layerCount(), c.currentRate(), L, first ? (first - tail) / fs : -1.0, c.materialPosition(), c.physicsJobsRun(), c.synthJobsRun());
    assert(d2 < 1e-3); c.setTs1(1.0);
    // 8. Off clears; FUTURE mode waits for the material
    c.setOn(false); size_t offAt = out.size(); run(0.2); assert(rms(out, offAt, out.size()) == 0.0 && c.state() == dsm::Core::OFF);
    p.past = false; p.t_l = 1.0; p.delay = 2; c.setParams(p); c.setOn(true); assert(c.state() == dsm::Core::CAPTURING);
    size_t capStart = out.size(); run(0.9); assert(c.state() == dsm::Core::CAPTURING && rms(out, capStart, out.size()) == 0.0);
    run(0.2); waitIdle("future");
    printf("8. FUTURE: loop %.3f s, offset %.0f (must be 0), state PLAYING=%d\n", c.loopSeconds(), c.currentMatOffset(), c.state() == dsm::Core::PLAYING);
    assert(c.state() == dsm::Core::PLAYING && c.currentMatOffset() == 0.0);
    p.t_l = 1.5; c.setParams(p); run(0.4); waitIdle("future t_l 1.5");     // material already captured (2+ s since press)
    printf("   FUTURE t_l 1.0->1.5 after the material exists: loop %.3f s\n", c.loopSeconds()); assert(std::fabs(c.loopSeconds() - 1.5) < 0.05);
    // 9. engine A without a server: falls back to B with an error
    c.setSpoolDir("output/dsm_core_test/spool_none"); p.engine = 1; c.setParams(p); run(0.4); waitIdle("engine A fallback");
    printf("9. engine A with no server: error='%s', playing=%d\n", c.lastError().c_str(), c.state() == dsm::Core::PLAYING);
    assert(c.hasError() && c.state() == dsm::Core::PLAYING);
    p.engine = 0; c.setParams(p); run(0.4); waitIdle("back to B");
    printf("   back to engine B: error cleared=%d\n", !c.hasError()); assert(!c.hasError());
    // lights: state transitions seen by the panel
    c.setOn(false); run(0.1); assert(c.state() == dsm::Core::OFF && !c.hasError() && !c.computing());
    p.past = true; c.setParams(p); c.setOn(true); assert(c.state() == dsm::Core::COMPUTING);
    waitIdle("relight"); assert(c.state() == dsm::Core::PLAYING && !c.computing());
    printf("   lights: OFF -> COMPUTING -> PLAYING, computing() false when idle, no error\n");
    // 9b. reset clears the tap: Off, then On with silence -> silent loop, no old material
    { auto silentRun = [&](double secs) { size_t N = (size_t)(secs * fs); for (size_t i = 0; i < N; i++, n++) out.push_back(c.process(0.f)); };
      run(1.0);                                   // tone playing right up to the reset
      c.setOn(false); p.engine = 0; c.setParams(p);
      silentRun(0.5); c.setOn(true);              // only silence arrived since the reset (PAST mode)
      double w = 0; while (c.state() != dsm::Core::PLAYING && w < 600) { silentRun(0.02); w += 0.02; std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
      size_t a = out.size(); silentRun(1.0);
      printf("9b. Off (reset) then On on silence: output rms %.2e (old material must be gone), physics jobs %ld\n", rms(out, a, out.size()), c.physicsJobsRun());
      assert(rms(out, a, out.size()) == 0.0); }
    // 10. export
    dsm::makeDir("output"); dsm::makeDir("output/dsm_core_test");
    printf("10. before export: state %d, layers %d, stored trajectories %d, error '%s'\n", (int)c.state(), c.layerCount(), c.storedTrajectories(), c.lastError().c_str());
    bool ok = c.exportForVerification("output/dsm_core_test/export"); printf("10. export: %s\n", ok ? "written to output/dsm_core_test/export" : "FAILED"); assert(ok);
    printf("ALL PASS\n");
    return 0;
}
