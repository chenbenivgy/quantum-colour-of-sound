// Discontinuous Sound Modulator -- the module logic, independent of Rack so it can
// be unit-tested (tools/core_test.cpp).
//
// Material.  A tap always holds the last TAP_SECONDS of input, also when Off.
//   PAST   (default): On snapshots the tap; the loop is the last t_l seconds BEFORE
//          the press (anchor = press moment = loop end); computing starts at once.
//   FUTURE: On starts a capture; the loop is the first t_l seconds AFTER the press
//          (anchor = press moment = loop start); computing starts when t_l seconds
//          are in; the capture keeps filling up to TAP_SECONDS so a later longer
//          t_l can be served without a new press.
//   Off clears capture, trajectories and loops; the tap keeps rolling.
//
// Jobs.  A PHYSICS job (delay, hop, model rate, t_l, couplings, engine, seed, mode)
//   re-cuts the material and computes the trajectory store in the worker thread;
//   a running physics job is aborted at once by a new one with a different physics
//   key. Raising the trajectory count only APPENDS trajectories (own seed each,
//   exact). A SYNTH job (render selector undep/dep0/dep1/twin, trajectory count
//   down) re-synthesises the loop from the stored trajectories in milliseconds.
//   Every physics change also takes a new seed; the store is kept for as long as
//   the physics key is unchanged.
// Playback. Layers of loops crossfade on MATERIAL time (the sample of the capture
//   that is playing), so a re-cut loop of another length stays in place; physics
//   results fade over t_l/2, synth results over 50 ms; TS1 is a varispeed read
//   glided in log-rate over t_l/4.
// Engines. B = native C++ (dsm_engine.hpp). A = job files for dsm_server.py; if
//   the server heartbeat is missing or older than 3 s the job runs on B and the
//   error is reported (ERR light + context menu).
#pragma once
#include "dsm_engine.hpp"
#include <mutex>
#include <condition_variable>
#include <memory>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <chrono>
#include <sys/stat.h>

namespace dsm {

static const double TAP_SECONDS = 10.5;
static const int MAX_TRAJ = 8;

// per-delay auto couplings: table-B optima of DSM_coupling_sweep.ipynb (2026-09-03) with
// gz floored at 0.15 so the resonator (OSR * f/fs * gz) stays active. Index = delay (0 unused).
static const double AUTO_GX[6] = {0.5, 0.555, 0.45, 0.45, 0.45, 0.45};
static const double AUTO_GZ[6] = {0.2, 0.45, 0.15, 0.15, 0.15, 0.15};

struct CoreParams {
    // physics
    int delay = 3, H = 256, D = 6; double t_l = 4.0;
    double gx = 0.5, gz = 0.2; bool autoCoupling = true;
    int engine = 0;                 // 0 = B (C++), 1 = A (Python server)
    uint32_t seedBase = 12345; int seedBump = 0;
    bool past = true;               // PAST / FUTURE material mode
    int ntraj = 1;                  // trajectories (1..MAX_TRAJ)
    uint32_t captureId = 0;         // incremented at every On: a new capture never reuses an old store
    // synthesis
    int render = 0;                 // 0 undep, 1 dep0, 2 dep1, 3 classical FB, 4 pink noise
    double gxEff() const { return autoCoupling ? AUTO_GX[std::min(std::max(delay, 1), 5)] : gx; }
    double gzEff() const { return autoCoupling ? AUTO_GZ[std::min(std::max(delay, 1), 5)] : gz; }
    // the physics key: what the trajectory store depends on (ntraj excluded: it appends)
    bool samePhysics(const CoreParams& o) const {
        return delay == o.delay && H == o.H && D == o.D && t_l == o.t_l && gxEff() == o.gxEff() && gzEff() == o.gzEff()
            && engine == o.engine && seedBase == o.seedBase && seedBump == o.seedBump && past == o.past && captureId == o.captureId;
    }
    uint32_t seed() const { return seedBase + 7919u * (uint32_t)seedBump; }
};

// the result of a physics job: material + trajectory store (immutable once published)
struct Prepared {
    CoreParams p; JobParams jp; Material m; Store st; double fsEngine = 48000; int engineUsed = 0;
    std::vector<float> x;           // the loop material at engine rate (what was fed)
    double matOffset = 0;           // capture-sample index of the material's first sample
};
struct Render {
    std::shared_ptr<const Prepared> prep; std::vector<float> yEngine; int render = 0, n_use = 1;
    double matOffset = 0; JobResult res;
};

// ----------------------------------------------------------- WAV helpers ----
static bool writeWavF32(const std::string& path, const float* x, size_t n, int sr) {
    FILE* f = fopen(path.c_str(), "wb"); if (!f) return false;
    uint32_t dataBytes = (uint32_t)(n * 4), fmtLen = 16; uint16_t fmt = 3, ch = 1, bits = 32, align = 4;
    uint32_t byteRate = sr * 4, riffLen = 36 + dataBytes;
    fwrite("RIFF", 1, 4, f); fwrite(&riffLen, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmtLen, 4, 1, f); fwrite(&fmt, 2, 1, f); fwrite(&ch, 2, 1, f);
    uint32_t sr32 = sr; fwrite(&sr32, 4, 1, f); fwrite(&byteRate, 4, 1, f); fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&dataBytes, 4, 1, f); fwrite(x, 4, n, f); fclose(f); return true;
}
static bool readWavF32(const std::string& path, std::vector<float>& out, int& sr) {
    std::ifstream f(path, std::ios::binary); if (!f) return false;
    char id[4]; f.read(id, 4); if (memcmp(id, "RIFF", 4)) return false; uint32_t sz; f.read((char*)&sz, 4); f.read(id, 4);
    if (memcmp(id, "WAVE", 4)) return false;
    uint16_t fmt = 0, ch = 0, bits = 0; uint32_t rate = 0, dataSize = 0; bool gotFmt = false, gotData = false;
    while (f.good() && !(gotFmt && gotData)) {
        f.read(id, 4); uint32_t len; f.read((char*)&len, 4); if (!f.good()) break;
        if (!memcmp(id, "fmt ", 4)) { f.read((char*)&fmt, 2); f.read((char*)&ch, 2); f.read((char*)&rate, 4); uint32_t br; f.read((char*)&br, 4); uint16_t ba; f.read((char*)&ba, 2); f.read((char*)&bits, 2); if (len > 16) f.seekg(len - 16, std::ios::cur); gotFmt = true; }
        else if (!memcmp(id, "data", 4)) { dataSize = len; gotData = true; }
        else f.seekg(len, std::ios::cur);
    }
    if (!gotFmt || !gotData || ch == 0) return false;
    sr = (int)rate; size_t frames = dataSize / (bits / 8) / ch; out.resize(frames);
    if (fmt == 3 && bits == 32) { std::vector<float> buf(frames * ch); f.read((char*)buf.data(), dataSize); for (size_t i = 0; i < frames; i++) out[i] = buf[i * ch]; }
    else if (fmt == 1 && bits == 16) { std::vector<int16_t> buf(frames * ch); f.read((char*)buf.data(), dataSize); for (size_t i = 0; i < frames; i++) out[i] = buf[i * ch] / 32768.f; }
    else return false;
    return true;
}
template <class T> static void writeRaw(const std::string& path, const std::vector<T>& v) { FILE* f = fopen(path.c_str(), "wb"); if (!f) return; fwrite(v.data(), sizeof(T), v.size(), f); fclose(f); }
template <class T> static bool readRaw(const std::string& path, std::vector<T>& v) { FILE* f = fopen(path.c_str(), "rb"); if (!f) return false; fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); v.resize(n / sizeof(T)); size_t got = fread(v.data(), sizeof(T), v.size(), f); fclose(f); return got == v.size(); }
static bool fileExists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }
static double fileAgeSeconds(const std::string& p) { struct stat st; if (stat(p.c_str(), &st) != 0) return 1e9; return std::difftime(time(nullptr), st.st_mtime); }
static void makeDir(const std::string& p) { mkdir(p.c_str(), 0755); }
static const char* RENDER_NAMES[5] = {"uncond", "cond0", "cond1", "twin", "pink"};

static std::string paramsJson(const JobParams& jp, double fsEngine, int engine, int render, bool past) {
    char buf[1400];
    snprintf(buf, sizeof buf,
        "{\n \"delay\": %d,\n \"H\": %d,\n \"D\": %d,\n \"t_l\": %.10g,\n \"gamma_x\": %.17g,\n \"gamma_z\": %.17g,\n \"seed\": %u,\n"
        " \"phase\": %.17g,\n \"measurement_basis\": \"%s\",\n \"amp_pow\": %.17g,\n \"omega_s_res\": %.17g,\n \"carrier_clock\": %s,\n"
        " \"encode_target\": \"%s\",\n \"interaction\": \"%s\",\n \"n_traj\": %d,\n \"render\": \"%s\",\n \"fs_engine\": %.10g,\n \"engine\": \"%s\",\n \"mode\": \"%s\"\n}\n",
        jp.delay, jp.H, jp.D, jp.t_l, jp.gamma_x, jp.gamma_z, jp.seed, jp.phase, jp.basis.c_str(), jp.amp_pow, jp.omega_s_res,
        jp.carrier_clock ? "true" : "false", jp.encode_target.c_str(), jp.interaction.c_str(), jp.n_traj, RENDER_NAMES[render], fsEngine,
        engine ? "A" : "B", past ? "past" : "future");
    return buf;
}

// ------------------------------------------------------------------ core ----
class Core {
public:
    enum State { OFF = 0, CAPTURING, COMPUTING, PLAYING };

    Core() { setSampleRate(48000); worker = std::thread([this] { workerLoop(); }); }
    ~Core() { { std::lock_guard<std::mutex> lk(mu); quit = true; abortFlag = true; } cv.notify_all(); if (worker.joinable()) worker.join(); }

    void setSampleRate(double fs) {
        if (fs == fsEngine && !tap.empty()) return;
        fsEngine = fs; tap.assign((size_t)(TAP_SECONDS * fs), 0.f); tapPos = 0; tapCount = 0;
    }
    void setSpoolDir(const std::string& d) { std::lock_guard<std::mutex> lk(mu); spoolDir = d; }
    void setThreads(int n) { nthreads = std::max(1, n); }
    int threads() const { return nthreads; }
    void setLogger(std::function<void(const std::string&)> f) { std::lock_guard<std::mutex> lk(mu); logFn = f; }
    // estimated ticks of the next physics job for the current parameters (for the menu)
    double estimatedTicks(const CoreParams& p) const {
        double fs_m = fsEngine / p.D; int nf = loop_frames(p.t_l, fs_m, p.H);
        int W = std::min(warmup_frames(p.delay, p.gxEff(), p.gzEff()), nf);
        return (double)(nf + W + p.delay) * (p.H + 1) * p.ntraj;
    }

    // ---- control (audio thread) ----
    void setOn(bool on) {
        if (on == isOn) return;
        isOn = on;
        if (on) startCapture(); else clearAll();
    }
    // called every sample with the current knob values
    void setParams(const CoreParams& p) {
        CoreParams q = p; q.seedBump = params.seedBump; q.captureId = params.captureId;
        bool physics = !q.samePhysics(params);
        if (physics) q.seedBump = params.seedBump + 1;          // every physics change draws a new trajectory set
        bool trajUp = q.ntraj > params.ntraj, trajDown = q.ntraj < params.ntraj, synth = q.render != params.render;
        params = q;
        if (st == OFF) return;
        if (physics || trajUp) { physDirty = true; physAge = 0; }
        else if (trajDown || synth) { synthDirty = true; synthAge = 0; }
    }
    // SYNC: hold a fresh render until the live input repeats the captured material (an upstream
    // loop), then start the render on that moment so it plays as the echo of the live input.
    // Fallback: start anyway 1.5 t_l after the render is ready. Switching SYNC on while playing
    // re-aligns at the next detection (50 ms crossfade to the aligned position).
    void setSync(bool on) { if (on == syncOn) return; syncOn = on; if (on && !layers.empty()) armDetector(layers.back().r); if (!on) { detArmed = false; if (held) { startHeld(matHeldFallbackPos()); } } }
    bool syncWaiting() const { return held != nullptr; }
    double syncScore() const { return detLast; }
    void setTs1(double ts1) {
        double t = 1.0 / std::min(std::max(ts1, 0.25), 8.0);
        if (t == rateTarget) return;
        rateTarget = t;
        // linear glide in log-rate over t_l/4 (a tape motor slowing down): fixed step per sample
        double glide = layers.empty() ? fsEngine : std::max(1.0, 0.25 * layers.back().r->yEngine.size());
        glideStep = std::fabs(std::log(rateTarget) - std::log(rate)) / glide;
    }
    void newTrajectory() { CoreParams p = params; p.seedBump = params.seedBump + 1; if (st != OFF) { params = p; physDirty = true; physAge = 0; } else params = p; }

    // ---- audio (per sample) ----
    float process(float in) {
        // the tap
        tap[tapPos] = in; tapPos = (tapPos + 1) % tap.size(); if (tapCount < tap.size()) tapCount++;
        if (st != OFF && !params.past && capture.size() < tap.size()) capture.push_back(in);
        // finished jobs
        if (resultReady.load()) { std::unique_lock<std::mutex> lk(mu, std::try_to_lock); if (lk.owns_lock()) { if (ready) { adopt(ready); ready.reset(); } resultReady = false; } }
        // SYNC detector on the live input: 0.5 s template, correlated at fs/8 every 32 input samples
        if (detArmed) {
            detAcc += in; detAccN++;
            if (detAccN == DET_D) {
                detRing[detPos] = detAcc / DET_D; detPos = (detPos + 1) % DET_MD; detAcc = 0.f; detAccN = 0; detCount++;
                if (detCount >= (size_t)DET_MD && (detCount % DET_KD) == 0) {
                    double ncc = detectorScore(); detLast = ncc;
                    // a sustained tone gives a local maximum at every cycle and a phrase gives side lobes;
                    // track the highest score above threshold, fire DET_HOLD after the last new maximum
                    if (ncc > DET_THRESH && ncc > detMax) { detMax = ncc; detMaxCount = detCount; }
                    if (detMax > 0 && (detCount - detMaxCount) * DET_D >= DET_HOLD) {
                        double lag = (double)(detCount - detMaxCount) * DET_D;   // input samples since the maximum
                        double pos = detTemplate->matOffset + detT0 + (double)DET_MD * DET_D + lag;
                        detArmed = false; detMax = 0;
                        if (held) startHeld(pos); else realign(pos);
                    }
                }
            }
        }
        if (held) { heldAge++; if (heldAge >= heldLimit) startHeld(matHeldFallbackPos()); }
        // material becoming available (FUTURE) / debounced jobs
        if (st == CAPTURING && materialAvailable(params)) { st = COMPUTING; requestPhysics(); }
        if (waitingMaterial && materialAvailable(params)) { waitingMaterial = false; requestPhysics(); }
        if (physDirty && st != OFF) { physAge++; if (physAge >= (long)(0.3 * fsEngine)) { physDirty = false;
            if (materialAvailable(params)) requestPhysics(); else { if (layers.empty()) st = CAPTURING; else waitingMaterial = true; } } }
        if (synthDirty && st != OFF) { synthAge++; if (synthAge >= (long)(0.1 * fsEngine)) { synthDirty = false; requestSynth(); } }
        if (layers.empty()) return 0.f;
        return play();
    }

    State state() const { return st; }
    bool computing() const { return busy.load(); }
    // nothing queued, running, pending adoption or debouncing (test hook)
    bool idle() const { return !busy.load() && !pendingFlag.load() && !resultReady.load() && !physDirty && !synthDirty; }
    bool hasError() const { return errFlag.load(); }
    float progress() const { return prog.load(); }
    double loopSeconds() const { return layers.empty() ? 0.0 : layers.back().r->yEngine.size() / fsEngine; }
    std::string lastError() { std::lock_guard<std::mutex> lk(mu); return errorMsg; }
    const CoreParams& currentParams() const { return params; }
    int storedTrajectories() const { return layers.empty() ? 0 : layers.back().r->prep->st.ntraj; }
    double currentMatOffset() const { return layers.empty() ? 0.0 : layers.back().r->matOffset; }
    long physicsJobsRun() const { return physRuns.load(); }
    long physicsJobsAborted() const { return physAborts.load(); }
    long synthJobsRun() const { return synthRuns.load(); }
    int layerCount() const { return (int)layers.size(); }
    double currentRate() const { return rate; }
    double materialPosition() const { return matPos; }

    // export the current render + its material for dsm_verify.py (UI thread)
    bool exportForVerification(const std::string& dir) {
        std::shared_ptr<Render> r; { std::lock_guard<std::mutex> lk(mu); if (!layers.empty()) r = layers.back().r; }
        if (!r) return false;
        const Prepared& pr = *r->prep;
        makeDir(dir);
        writeWavF32(dir + "/input_engine_rate.wav", pr.x.data(), pr.x.size(), (int)pr.fsEngine);
        std::vector<float> xm(pr.m.x_model.begin(), pr.m.x_model.end());
        writeWavF32(dir + "/input_model_rate.wav", xm.data(), xm.size(), (int)std::lround(pr.m.fs_m));
        std::vector<float> ym(r->res.y_model.begin(), r->res.y_model.end());
        writeWavF32(dir + "/loop_model_rate.wav", ym.data(), ym.size(), (int)std::lround(pr.m.fs_m));
        writeWavF32(dir + "/loop_engine_rate.wav", r->yEngine.data(), r->yEngine.size(), (int)pr.fsEngine);
        writeRaw(dir + "/in_mag.f64", pr.m.in_mag.a); writeRaw(dir + "/in_phase.f64", pr.m.in_phase.a);
        writeRaw(dir + "/decoded_mag.f64", r->res.out_mag.a); writeRaw(dir + "/decoded_phase.f64", r->res.out_phase.a);
        if (!pr.st.MREC.empty()) writeRaw(dir + "/mrec.i8", pr.st.MREC);
        JobParams jp = pr.jp; jp.n_traj = r->n_use;
        std::ofstream(dir + "/params.json") << paramsJson(jp, pr.fsEngine, pr.engineUsed, r->render, pr.p.past);
        return true;
    }

private:
    // ---- state ----
    double fsEngine = 48000; int nthreads = 4; std::string spoolDir; std::function<void(const std::string&)> logFn;
    State st = OFF; bool isOn = false;
    CoreParams params; bool physDirty = false, synthDirty = false, waitingMaterial = false; long physAge = 0, synthAge = 0;
    std::vector<float> tap; size_t tapPos = 0, tapCount = 0;
    std::vector<float> capture; double anchor = 0;        // PAST: anchor = capture.size(); FUTURE: 0
    struct Layer { std::shared_ptr<Render> r; double gain, target, rate; };
    std::vector<Layer> layers;
    double matPos = 0, rate = 1.0, rateTarget = 1.0, glideStep = 1.0;
    // ---- sync ----
    // template: up to 2 s of the material from its first energetic window, decimated by 8 (box
    // average), scored every 32 input samples (<= 375 multiply-adds per input sample)
    // DET_HOLD: fire 250 ms after the last new maximum -- longer than the side lobes a few-note
    // phrase produces around the true peak (~60 ms apart); the start is lag-compensated
    static const int DET_D = 8, DET_KD = 4, DET_HOLD = 12000; static constexpr double DET_THRESH = 0.85;
    int DET_MD = 3000;
    bool syncOn = false, detArmed = false; std::vector<float> detRing; size_t detPos = 0, detCount = 0; float detAcc = 0.f; int detAccN = 0;
    std::vector<float> detTpl; double detTplNorm = 1.0; long detT0 = 0; std::shared_ptr<Render> detTemplate;
    double detLast = 0.0, detMax = 0.0; size_t detMaxCount = 0;
    std::shared_ptr<Render> held; long heldAge = 0, heldLimit = 0;
    void armDetector(std::shared_ptr<Render> r) {
        const std::vector<float>& x = r->prep->x; long L = (long)x.size();
        // template length: up to 2 s of material (a sustained note correlates with itself at every
        // cycle; only the phrase structure -- onsets, gaps, note changes -- pins the moment)
        DET_MD = (int)std::min<long>(std::lround(2.0 * fsEngine / DET_D), L / DET_D); int M = DET_MD * DET_D;
        if (DET_MD < 64) { detArmed = false; return; }
        float peak = 0; for (float v : x) peak = std::max(peak, std::fabs(v));
        long t0 = 0; double best = -1;
        for (long s = 0; s + M <= L; s += DET_D * DET_KD) { double e = 0; for (int i = 0; i < M; i++) e += (double)x[s + i] * x[s + i]; if (e > best) { best = e; t0 = s; } if (e > 0.02 * peak * peak * M) { t0 = s; break; } }
        detTpl.assign(DET_MD, 0.f); double en = 0;
        for (int i = 0; i < DET_MD; i++) { float a = 0; for (int j = 0; j < DET_D; j++) a += x[t0 + i * DET_D + j]; a /= DET_D; detTpl[i] = a; en += (double)a * a; }
        detTplNorm = std::sqrt(en) + 1e-12; detT0 = t0; detTemplate = r;
        // pre-fill the ring from the tap (the last M input samples) so scoring starts at once
        detRing.assign(DET_MD, 0.f); detPos = 0; detCount = 0; detAcc = 0.f; detAccN = 0;
        if (tapCount >= (size_t)M) {
            for (int i = 0; i < DET_MD; i++) { float a = 0; for (int j = 0; j < DET_D; j++) a += tap[(tapPos + tap.size() - M + i * DET_D + j) % tap.size()]; detRing[i] = a / DET_D; }
            detPos = 0; detCount = DET_MD;
        }
        detMax = 0.0; detMaxCount = 0; detLast = 0.0; detArmed = true;
    }
    double detectorScore() const {   // normalised cross-correlation of the ring (oldest first) with the template
        double dot = 0, e = 0; size_t p = detPos;
        for (int i = 0; i < DET_MD; i++) { float v = detRing[(p + i) % DET_MD]; dot += (double)v * detTpl[i]; e += (double)v * v; }
        return dot / (std::sqrt(e) * detTplNorm + 1e-12);
    }
public:
    double syncLeadSamples() const { return detT0 + (double)DET_MD * DET_D + DET_KD * DET_D; }   // material index at which a synced start begins
private:
    double matHeldFallbackPos() const { return held ? held->matOffset : matPos; }
    void startHeld(double pos) { auto r = held; held.reset(); detArmed = false; layers.clear(); matPos = pos; layers.push_back(Layer{r, 1.0, 1.0, 1.0}); st = PLAYING; }
    void realign(double pos) {     // jump the material clock with a 50 ms crossfade (same render, new position)
        if (layers.empty()) return;
        auto r = layers.back().r; double fr = 1.0 / std::max(1.0, 0.05 * fsEngine);
        for (auto& l : layers) { l.target = 0; l.rate = fr; l.r = std::make_shared<Render>(*l.r); l.r->matOffset += (matPos - pos); }   // old layers keep their audio position
        layers.push_back(Layer{r, 0.0, 1.0, fr}); matPos = pos;
        if (layers.size() > 4) layers.erase(layers.begin());
    }

    // ---- worker ----
    std::thread worker; std::mutex mu; std::condition_variable cv; bool quit = false;
    struct Job { bool physics; CoreParams p; JobParams jp; std::vector<float> x; double matOffset; double fs; uint32_t id; };
    std::unique_ptr<Job> pendingJob; uint32_t jobCounter = 0;
    std::atomic<bool> abortFlag{false}, busy{false}, resultReady{false}, errFlag{false}, pendingFlag{false}; std::atomic<float> prog{0.f};
    std::atomic<long> physRuns{0}, physAborts{0}, synthRuns{0};
    std::shared_ptr<Render> ready; std::string errorMsg; bool jobWarning = false;   // worker-owned
    std::shared_ptr<const Prepared> latestPrep;           // worker-owned: the store of the current physics key

    // ---- material ----
    size_t needSamples(const CoreParams& p) const {
        double fs_m = fsEngine / p.D; int nf = loop_frames(p.t_l, fs_m, p.H); return (size_t)nf * p.H * p.D;
    }
    bool materialAvailable(const CoreParams& p) const { return p.past ? true : capture.size() >= needSamples(p); }
    void startCapture() {
        layers.clear(); matPos = 0; physDirty = synthDirty = false; held.reset(); detArmed = false; params.captureId++;
        if (params.past) {
            capture.resize(tapCount);
            for (size_t i = 0; i < tapCount; i++) capture[i] = tap[(tapPos + tap.size() - tapCount + i) % tap.size()];
            anchor = (double)capture.size();
            st = COMPUTING; requestPhysics();
        } else {
            capture.clear(); capture.reserve(tap.size()); anchor = 0; st = CAPTURING;
        }
    }
    // Off = reset: capture, loops, pending work AND the tap are cleared, so the next On
    // (PAST mode) sees only what arrived after the reset.
    void clearAll() {
        { std::lock_guard<std::mutex> lk(mu); abortFlag = true; pendingJob.reset(); pendingFlag = false; ready.reset(); resultReady = false; errorMsg.clear(); }
        errFlag = false;
        st = OFF; capture.clear(); layers.clear(); matPos = 0; physDirty = synthDirty = waitingMaterial = false;
        std::fill(tap.begin(), tap.end(), 0.f); tapPos = 0; tapCount = 0;
        held.reset(); detArmed = false; detTemplate.reset();
    }
    JobParams makeJobParams(const CoreParams& p) const {
        JobParams jp; jp.delay = p.delay; jp.H = p.H; jp.D = p.D; jp.t_l = p.t_l;
        jp.gamma_x = p.gxEff(); jp.gamma_z = p.gzEff(); jp.seed = p.seed(); jp.n_traj = p.ntraj; jp.render = p.render;
        return jp;
    }
    // cut the loop material from the capture: PAST = last need samples before the anchor
    // (zero-padded at the front if the tap was not yet full); FUTURE = first need samples
    void cutMaterial(const CoreParams& p, std::vector<float>& x, double& offset) const {
        size_t need = needSamples(p); x.assign(need, 0.f);
        if (p.past) {
            long start = (long)anchor - (long)need; offset = (double)start;
            for (size_t i = 0; i < need; i++) { long j = start + (long)i; if (j >= 0 && j < (long)capture.size()) x[i] = capture[j]; }
        } else {
            offset = 0; for (size_t i = 0; i < need && i < capture.size(); i++) x[i] = capture[i];
        }
    }
    void requestPhysics() {
        std::unique_lock<std::mutex> lk(mu, std::try_to_lock);
        if (!lk.owns_lock()) { physDirty = true; physAge = (long)(0.3 * fsEngine); return; }
        auto j = std::make_unique<Job>(); j->physics = true; j->p = params; j->jp = makeJobParams(params); j->fs = fsEngine; j->id = ++jobCounter;
        cutMaterial(params, j->x, j->matOffset);
        // abort a running job only if its physics differs (an append job may finish and be reused)
        if (busy.load() && runningKey && !runningKey->samePhysics(params)) abortFlag = true;
        pendingJob = std::move(j); pendingFlag = true;
        lk.unlock(); cv.notify_all();
    }
    void requestSynth() {
        std::unique_lock<std::mutex> lk(mu, std::try_to_lock);
        if (!lk.owns_lock()) { synthDirty = true; synthAge = (long)(0.1 * fsEngine); return; }
        if (pendingJob && pendingJob->physics) { pendingJob->p.render = params.render; pendingJob->p.ntraj = params.ntraj; return; }
        auto j = std::make_unique<Job>(); j->physics = false; j->p = params; j->jp = makeJobParams(params); j->fs = fsEngine; j->id = ++jobCounter; j->matOffset = 0;
        pendingJob = std::move(j); pendingFlag = true;
        lk.unlock(); cv.notify_all();
    }
    void adopt(std::shared_ptr<Render> r) {
        if (layers.empty() && syncOn) {           // first sound with SYNC: hold until the input repeats the material
            held = r; heldAge = 0; heldLimit = (long)(1.5 * r->yEngine.size()); armDetector(r); st = COMPUTING; return;
        }
        if (held) { held = r; return; }           // a newer render while holding: hold that one instead
        bool physicsResult = layers.empty() || layers.back().r->prep != r->prep;
        double fadeSec = physicsResult ? 0.5 * r->yEngine.size() / fsEngine : 0.05;
        double fr = 1.0 / std::max(1.0, fadeSec * fsEngine);
        if (layers.empty()) { matPos = r->matOffset; fr = 1.0; }
        for (auto& l : layers) { l.target = 0; l.rate = fr; }
        layers.push_back(Layer{r, layers.empty() ? 1.0 : 0.0, 1.0, fr});
        if (layers.size() > 4) layers.erase(layers.begin());
        st = PLAYING;
    }
    static float readLoop(const std::vector<float>& y, double p) {
        size_t L = y.size(); if (L == 0) return 0.f;
        double q = std::fmod(p, (double)L); if (q < 0) q += L;
        size_t i0 = (size_t)q; double fr = q - i0; size_t i1 = (i0 + 1) % L;
        return (float)((1 - fr) * y[i0] + fr * y[i1]);
    }
    float play() {
        // TS1: glide log(rate) linearly towards the target over t_l/4
        if (rate != rateTarget) {
            double lr = std::log(rate), lt = std::log(rateTarget);
            if (std::fabs(lt - lr) <= glideStep) rate = rateTarget;
            else rate = std::exp(lr + ((lt > lr) ? glideStep : -glideStep));
        }
        float y = 0.f; bool drop = false;
        for (auto& l : layers) {
            if (l.gain < l.target) l.gain = std::min(l.target, l.gain + l.rate); else if (l.gain > l.target) l.gain = std::max(l.target, l.gain - l.rate);
            if (l.gain > 0) y += (float)l.gain * readLoop(l.r->yEngine, matPos - l.r->matOffset);
            if (l.gain <= 0 && l.target <= 0) drop = true;
        }
        if (drop) layers.erase(std::remove_if(layers.begin(), layers.end(), [](const Layer& l) { return l.gain <= 0 && l.target <= 0; }), layers.end());
        matPos += rate;
        const Layer& top = layers.back(); double L = (double)top.r->yEngine.size();
        if (matPos - top.r->matOffset >= L) matPos -= L;          // keep matPos bounded (all layers read modulo their own L)
        return y;
    }

    // ---- worker thread ----
    std::shared_ptr<CoreParams> runningKey;
    void workerLoop() {
        for (;;) {
            std::unique_ptr<Job> job;
            { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return quit || pendingJob; }); if (quit) return; job = std::move(pendingJob); busy = true; pendingFlag = false; abortFlag = false; runningKey = std::make_shared<CoreParams>(job->p); }
            prog = 0.f; jobWarning = false;
            std::shared_ptr<Render> r; std::string err; bool aborted = false;
            try { r = job->physics ? runPhysics(*job) : runSynth(*job); }
            catch (const std::exception& e) { err = e.what(); aborted = (err == "aborted"); }
            busy = false;
            std::lock_guard<std::mutex> lk(mu);
            runningKey.reset();
            if (aborted) physAborts++;
            prog = 0.f;
            // ERR reflects the LAST COMPLETED job: a fallback/warning keeps it on with its message,
            // a clean job clears it, a failed job sets it, an aborted job leaves it untouched.
            if (r) { ready = r; resultReady = true; if (!jobWarning) { errorMsg.clear(); errFlag = false; } else errFlag = true; }
            else if (!err.empty() && !aborted) { errorMsg = err; errFlag = true; }
        }
    }
    void log(const std::string& s) { std::function<void(const std::string&)> f; { std::lock_guard<std::mutex> lk(mu); f = logFn; } if (f) f(s); }
    std::shared_ptr<Render> runPhysics(Job& j) {
        std::shared_ptr<Prepared> pr;
        bool reuse = latestPrep && latestPrep->p.samePhysics(j.p) && latestPrep->engineUsed == j.p.engine;
        auto t0 = std::chrono::steady_clock::now();
        float peak = 0; for (float v : j.x) peak = std::max(peak, std::fabs(v));
        char buf[300]; snprintf(buf, sizeof buf, "physics job: delay %d H %d D %d t_l %.2f gx %.3f gz %.2f engine %s traj %d->%d threads %d (%s, %s, material %zu samples peak %.3g)",
                                j.jp.delay, j.jp.H, j.jp.D, j.jp.t_l, j.jp.gamma_x, j.jp.gamma_z, j.p.engine ? "A" : "B",
                                reuse ? latestPrep->st.ntraj : 0, j.p.ntraj, nthreads, reuse ? "append" : "new", j.p.past ? "past" : "future", j.x.size(), peak);
        log(buf);
        if (reuse) pr = std::make_shared<Prepared>(*latestPrep);          // copy, then append
        else {
            pr = std::make_shared<Prepared>(); pr->p = j.p; pr->jp = j.jp; pr->fsEngine = j.fs; pr->x = j.x; pr->matOffset = j.matOffset;
            std::vector<double> x(j.x.begin(), j.x.end());
            pr->m = prepare_material(x, j.fs, j.jp);
        }
        pr->p = j.p; pr->jp.n_traj = j.p.ntraj; pr->jp.render = j.p.render;
        int have = pr->st.ntraj, want = j.p.ntraj;
        bool silent = true; for (float v : j.x) if (v != 0.f) { silent = false; break; }
        if (want > have) {
            if (silent) {
                // an all-zero material gives an exactly zero channel output (vacuum stays vacuum):
                // build the empty store without running the kernel (bit-identical, no CPU)
                TransformParams tp = transform_params(pr->jp, pr->m);
                Store& s = pr->st; if (s.ntraj == 0) { s.nb = pr->m.fed_mag.nb; s.T = pr->m.fed_mag.nf + tp.delay; s.nfed = pr->m.fed_mag.nf; s.scale = 1.0; }
                size_t add = (size_t)(want - have) * s.nb * s.T;
                s.SX.resize(s.SX.size() + add, 0.0); s.SY.resize(s.SY.size() + add, 0.0); s.POP.resize(s.POP.size() + add, 0.0); s.MREC.resize(s.MREC.size() + add, 0);
                s.ntraj = want; pr->engineUsed = j.p.engine;
                log("physics job: material is silent, empty store built without computation");
            }
            else if (j.p.engine == 1) { if (!runEngineA(*pr, want)) return nullptr; }
            else {
                // one trajectory at a time, each published as soon as it exists: the knob 1->4
                // is heard as 2, 3, 4 instead of silence/old loop until all are done, and an
                // abort keeps the trajectories already finished.
                for (int t = have; t < want; t++) {
                    runEngineB(*pr, t, t + 1);
                    if (t + 1 < want) {
                        auto snap = std::make_shared<Prepared>(*pr);
                        latestPrep = snap;
                        auto r = synthFrom(snap, j.p.render, t + 1);
                        { std::lock_guard<std::mutex> lk(mu); ready = r; resultReady = true; }   // log() locks mu too: keep it outside
                        snprintf(buf, sizeof buf, "  trajectory %d/%d published", t + 1, want); log(buf);
                    }
                }
            }
        }
        physRuns++;
        latestPrep = pr;
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        snprintf(buf, sizeof buf, "physics job done in %.2f s: %d frames (+%d warm-up) x %d bins, %d trajectories stored", secs, pr->m.nf, pr->m.W, pr->m.fed_mag.nb, pr->st.ntraj);
        log(buf);
        return synthFrom(pr, j.p.render, j.p.ntraj);
    }
    std::shared_ptr<Render> runSynth(Job& j) {
        if (!latestPrep) throw std::runtime_error("no trajectories yet");
        return synthFrom(latestPrep, j.p.render, j.p.ntraj);
    }
    std::shared_ptr<Render> synthFrom(std::shared_ptr<const Prepared> pr, int render, int ntraj) {
        auto r = std::make_shared<Render>(); r->prep = pr; r->render = render; r->n_use = std::max(1, std::min(ntraj, pr->st.ntraj)); r->matOffset = pr->matOffset;
        synthesise(pr->m, pr->st, pr->jp, render, r->n_use, r->res);
        r->yEngine.assign(r->res.y_engine.begin(), r->res.y_engine.end());
        synthRuns++;
        return r;
    }
    void runEngineB(Prepared& pr, int from, int to) {
        TransformParams tp = transform_params(pr.jp, pr.m);
        pr.engineUsed = 0;
        compute_store(pr.m.fed_mag, pr.m.fed_phase, pr.m.fbins, tp, from, to, pr.st, nthreads,
                      [&](int d, int tot) { prog = (float)d / tot; return !abortFlag.load(); });
    }
    bool runEngineA(Prepared& pr, int ntraj) {
        std::string spool; { std::lock_guard<std::mutex> lk(mu); spool = spoolDir; }
        if (spool.empty() || fileAgeSeconds(spool + "/server_alive") > 3.0) {
            { std::lock_guard<std::mutex> lk(mu); errorMsg = "engine A: dsm_server.py is not running (no heartbeat) -- rendered with engine B"; }
            jobWarning = true;
            runEngineB(pr, pr.st.ntraj, ntraj); pr.engineUsed = 1; return true;
        }
        makeDir(spool);
        char name[64]; snprintf(name, sizeof name, "job_%06u_%ld", (unsigned)(pr.jp.seed % 1000000u), (long)time(nullptr));
        std::string dir = spool + "/" + name; makeDir(dir);
        writeWavF32(dir + "/input_engine_rate.wav", pr.x.data(), pr.x.size(), (int)pr.fsEngine);
        JobParams jp = pr.jp; jp.n_traj = ntraj;
        std::ofstream(dir + "/params.json") << paramsJson(jp, pr.fsEngine, 1, 0, pr.p.past);
        std::ofstream(dir + "/request") << "1";
        while (!fileExists(dir + "/done")) {
            if (fileExists(dir + "/error.txt")) { std::ifstream f(dir + "/error.txt"); std::stringstream ss; ss << f.rdbuf(); throw std::runtime_error("engine A: " + ss.str().substr(0, 400)); }
            if (abortFlag.load()) { std::ofstream(dir + "/cancel") << "1"; throw std::runtime_error("aborted"); }
            if (fileAgeSeconds(spool + "/server_alive") > 5.0) { std::ofstream(dir + "/cancel") << "1"; throw std::runtime_error("engine A: server stopped responding"); }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        // read the store
        std::ifstream rf(dir + "/result.json"); std::stringstream ss; ss << rf.rdbuf(); std::string js = ss.str();
        auto num = [&](const char* key) { size_t p = js.find(std::string("\"") + key + "\""); if (p == std::string::npos) return 0.0; p = js.find(':', p); return atof(js.c_str() + p + 1); };
        int nb = (int)num("nb"), T = (int)num("T"), nt = (int)num("ntraj"); double scale = num("scale");
        Store st; st.nb = nb; st.T = T; st.ntraj = nt; st.nfed = pr.m.fed_mag.nf; st.scale = scale;
        if (!readRaw(dir + "/SX.f64", st.SX) || !readRaw(dir + "/SY.f64", st.SY) || !readRaw(dir + "/POP.f64", st.POP) || !readRaw(dir + "/MREC.i8", st.MREC))
            throw std::runtime_error("engine A: cannot read the trajectory store");
        if ((int)st.SX.size() != nt * nb * T || nb != pr.m.fed_mag.nb || T != pr.m.fed_mag.nf + pr.jp.delay) throw std::runtime_error("engine A: store geometry mismatch");
        TransformParams tp = transform_params(pr.jp, pr.m);
        if (tp.carrier_clock) st.theta = carrier_clock_rates(pr.m.fed_mag, pr.m.fed_phase, pr.m.fbins, tp.stft_hop, tp.fs, tp.carrier_clock);
        pr.st = st; pr.engineUsed = 1;
        return true;
    }
};

} // namespace dsm
