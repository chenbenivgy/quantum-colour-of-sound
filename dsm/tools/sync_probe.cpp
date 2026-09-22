#include "../src/DsmCore.hpp"
#include <cstdio>
static const double fs = 48000; static double P = 2.0;
static float phrase(double t) { double u = std::fmod(t, P); return (float)(0.5 * std::sin(2 * M_PI * 130.81 * u) * (u < 0.7) + 0.3 * std::sin(2 * M_PI * 196 * u) * (u > 0.9 && u < 1.6) + 0.2 * std::sin(2 * M_PI * 440 * u) * (u > 1.7)); }
int main() {
    dsm::Core c; c.setSampleRate(fs); c.setThreads(8);
    dsm::CoreParams p; p.delay = 2; p.H = 256; p.D = 6; p.t_l = P; p.past = true; c.setParams(p); c.setSync(true);
    size_t n = 0; for (; n < (size_t)((2 * P + 0.3) * fs); n++) c.process(phrase(n / fs));
    size_t press = n; c.setOn(true);
    while (!c.syncWaiting()) { for (int i = 0; i < 480; i++, n++) c.process(phrase(n / fs)); std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
    printf("held at %.3f s after press; lead %.3f s\n", (n - press) / fs, c.syncLeadSamples() / fs);
    double best = -1; size_t bestN = 0; size_t startN = 0;
    for (; n < press + (size_t)(2.6 * fs); n++) {
        float y = c.process(phrase(n / fs));
        if (!startN && y != 0.f) startN = n;
        if ((n % 32) == 0) { double s = c.syncScore(); if (s > best) { best = s; bestN = n; } if ((n % 4800) == 0) printf("  t=%.2f  ncc=%.3f  waiting=%d\n", (n - press) / fs, s, c.syncWaiting()); }
    }
    printf("max ncc %.3f at %.3f s after press; output started at %.3f s\n", best, (bestN - press) / fs, startN ? (startN - press) / fs : -1.0);
}
