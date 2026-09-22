// SYNC test: the input is a looping phrase of period P; with SYNC on the render must start
// exactly when the captured material comes round again; without a repeating input it must
// start at the 1.5 t_l fallback. Build: clang++ -O2 -std=c++17 -pthread -o sync_test sync_test.cpp
#include "../src/DsmCore.hpp"
#include <cstdio>
#include <cassert>
static const double fs = 48000; static dsm::Core* core; static size_t n = 0; static std::vector<float> out, inp;
static double P = 2.0;   // input loop period (s)
static float phrase(double t) { double u = std::fmod(t, P); return (float)(0.5 * std::sin(2 * M_PI * 130.81 * u) * (u < 0.7) + 0.3 * std::sin(2 * M_PI * 196 * u) * (u > 0.9 && u < 1.6) + 0.2 * std::sin(2 * M_PI * 440 * u) * (u > 1.7)); }
static float input(size_t i) { return phrase(i / fs); }
static void run(double secs, bool silent = false) { size_t N = (size_t)(secs * fs); for (size_t i = 0; i < N; i++, n++) { float x = silent ? 0.f : input(n); inp.push_back(x); out.push_back(core->process(x)); } }
static void waitState(dsm::Core::State s, double maxWall = 120) { auto t0 = std::chrono::steady_clock::now(); while (core->state() != s && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < maxWall) { run(0.01); std::this_thread::sleep_for(std::chrono::milliseconds(5)); } }
int main() {
    dsm::Core c; core = &c; c.setSampleRate(fs); c.setThreads(8);
    dsm::CoreParams p; p.delay = 2; p.H = 256; p.D = 6; p.t_l = P; p.past = true; p.ntraj = 1; c.setParams(p); c.setSync(true);
    run(2.0 * P + 0.3);                                   // the phrase has looped twice; press at t = 4.3 s (mid-phrase)
    size_t press = n; c.setOn(true);
    // compute finishes quickly; with SYNC the core must HOLD until the material (t_l before the press) recurs
    double w = 0; while (!c.syncWaiting() && w < 60) { run(0.01); std::this_thread::sleep_for(std::chrono::milliseconds(5)); w += 0.01; }
    assert(c.syncWaiting());
    size_t heldAt = n;
    run(P * 0.9);                                         // material start (press - t_l) recurs at press - t_l + P = press (since t_l == P)... next at press + P
    // find first non-zero output sample
    size_t first = 0; for (size_t i = press; i < out.size(); i++) if (out[i] != 0.f) { first = i; break; }
    while (!first && n < press + (size_t)(3 * P * fs)) { run(0.05); for (size_t i = press; i < out.size(); i++) if (out[i] != 0.f) { first = i; break; } }
    double started = (double)(first - press) / fs;        // seconds after the press
    // the loop is t_l rounded to whole hops, so the material start recurs at press + P - (L - P); a synced
    // render starts DET_HOLD after that peak with its material clock advanced by the same lag: in sync iff
    // (input phrase position) == (material phrase position) at the first output sample.
    double L = c.loopSeconds(); double peakT = P - (L - P);
    double err = (started - (peakT + 0.25));
    printf("SYNC on, input loop P=%.1f s, t_l %.3f s: held from %.2f s after the press; render started %.3f s after the press = peak at %.3f s + 250 ms hold (error %.1f ms)\n",
           P, L, (heldAt - press) / fs, started, peakT, err * 1e3);
    // alignment check at the first output sample: material index played = lead + lag where lag = started - peakT;
    // material index m is input phrase position ((press - L*fs + m)/fs) mod P
    double m = c.syncLeadSamples() + (started - peakT) * fs;
    double phraseMat = std::fmod((press - L * fs + m) / fs, P), phraseIn = std::fmod(first / fs, P);
    double align = phraseIn - phraseMat; if (align > P / 2) align -= P; if (align < -P / 2) align += P;
    printf("   alignment: input phrase position %.4f s vs render material phrase position %.4f s -> %.2f ms\n", phraseIn, phraseMat, align * 1e3);
    assert(std::fabs(err) < 0.002 && std::fabs(align) < 0.002);
    // 2. no repeating input: material captured from the phrase, then silence -> fallback 1.5 t_l after ready
    c.setOn(false); run(2.5);                             // Off clears the tap; refill it with the phrase
    c.setOn(true); size_t press2 = n;
    w = 0; while (!c.syncWaiting() && w < 60) { run(0.01, true); std::this_thread::sleep_for(std::chrono::milliseconds(5)); w += 0.01; }
    size_t ready2 = n;
    run(1.5 * P + 0.5, true);
    size_t first2 = 0; for (size_t i = press2; i < out.size(); i++) if (out[i] != 0.f) { first2 = i; break; }
    double fb = first2 ? (double)(first2 - ready2) / fs : -1.0;
    printf("SYNC on, silent input after the press: render started %.2f s after it was ready (expect 1.5 t_l = %.2f s)\n", fb, 1.5 * c.loopSeconds());
    assert(first2 && std::fabs(fb - 1.5 * c.loopSeconds()) < 0.05);
    printf("ALL PASS\n"); return 0;
}
