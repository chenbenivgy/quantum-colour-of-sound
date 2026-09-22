// Discontinuous Sound Modulator -- engine B (native C++), header only, no Rack
// dependency.  A line-by-line port of dsm_engine.py / dsm_chain.py, which are
// themselves certified bit-identical to the original qutip code
// (dependent_trajectory._dependent_colour_kernel).
//
//   * numpy-compatible MT19937 RandomState (init_genrand seeding, 53-bit doubles,
//     RandomState.choice(p=[p0,p1]) semantics) so measurement records match
//   * collision unitary U_A on the local qubits (S, D_0, F) via scaling-and-squaring
//     Taylor expm, spectator phases factored exactly
//   * density-matrix kernel: local U rho U^dag, reduced state of index 1, projective
//     collapse + trace-out in one contraction
//   * smstools STFT conventions (dftAnal / dftSynth), periodic analysis, circular OLA
//   * scipy.signal.decimate(ftype='fir', zero_phase=True) and resample_poly(D, 1)
//
// Qubit order is big-endian as qutip.tensor: index 0 = S (most significant bit),
// 1 = D_0, ..., n-1 = F.
#pragma once
#include <vector>
#include <complex>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <functional>
#include <thread>
#include <atomic>
#include <string>
#include <stdexcept>
#include <random>

namespace dsm {

using cplx = std::complex<double>;
static const double PI = 3.14159265358979323846;
static const double EPS = 2.220446049250313e-16;

// ---------------------------------------------------------------- RNG ----
struct NumpyRandomState {
    // MT19937 exactly as numpy's legacy RandomState(int seed)
    uint32_t mt[624]; int mti = 625;
    explicit NumpyRandomState(uint32_t seed) { init_genrand(seed); }
    void init_genrand(uint32_t s) {
        mt[0] = s;
        for (mti = 1; mti < 624; mti++)
            mt[mti] = (1812433253u * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + (uint32_t)mti);
    }
    uint32_t next32() {
        static const uint32_t mag01[2] = {0x0u, 0x9908b0dfu};
        uint32_t y;
        if (mti >= 624) {
            int kk;
            for (kk = 0; kk < 624 - 397; kk++) {
                y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
                mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1u];
            }
            for (; kk < 623; kk++) {
                y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
                mt[kk] = mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1u];
            }
            y = (mt[623] & 0x80000000u) | (mt[0] & 0x7fffffffu);
            mt[623] = mt[396] ^ (y >> 1) ^ mag01[y & 1u];
            mti = 0;
        }
        y = mt[mti++];
        y ^= (y >> 11); y ^= (y << 7) & 0x9d2c5680u; y ^= (y << 15) & 0xefc60000u; y ^= (y >> 18);
        return y;
    }
    double random_sample() {           // mt19937_next_double
        uint32_t a = next32() >> 5, b = next32() >> 6;
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }
    int choice01(double p0, double p1) { // RandomState.choice([0,1], p=[p0,p1])
        double cdf0 = p0, cdf1 = p0 + p1;
        cdf0 /= cdf1;
        double u = random_sample();
        return (u < cdf0) ? 0 : 1;       // searchsorted(side='right'): idx = #cdf <= u
    }
};

// -------------------------------------------------------- small matrices ----
struct CMat {
    int n = 0; std::vector<cplx> a;
    CMat() {}
    CMat(int n_) : n(n_), a((size_t)n_ * n_, cplx(0, 0)) {}
    cplx& operator()(int i, int j) { return a[(size_t)i * n + j]; }
    const cplx& operator()(int i, int j) const { return a[(size_t)i * n + j]; }
    static CMat eye(int n) { CMat m(n); for (int i = 0; i < n; i++) m(i, i) = 1; return m; }
    CMat operator*(const CMat& b) const {
        CMat c(n);
        for (int i = 0; i < n; i++) for (int k = 0; k < n; k++) { cplx aik = (*this)(i, k); if (aik == cplx(0, 0)) continue;
            for (int j = 0; j < n; j++) c(i, j) += aik * b(k, j); }
        return c;
    }
    CMat operator+(const CMat& b) const { CMat c = *this; for (size_t i = 0; i < a.size(); i++) c.a[i] += b.a[i]; return c; }
    CMat scaled(cplx s) const { CMat c = *this; for (auto& v : c.a) v *= s; return c; }
    double norm1() const { double m = 0; for (int j = 0; j < n; j++) { double s = 0; for (int i = 0; i < n; i++) s += std::abs((*this)(i, j)); m = std::max(m, s); } return m; }
};

static CMat kron(const CMat& A, const CMat& B) {
    CMat C(A.n * B.n);
    for (int i = 0; i < A.n; i++) for (int j = 0; j < A.n; j++) for (int k = 0; k < B.n; k++) for (int l = 0; l < B.n; l++)
        C(i * B.n + k, j * B.n + l) = A(i, j) * B(k, l);
    return C;
}

static CMat expm(const CMat& A) {
    // scaling and squaring with a degree-24 Taylor series; ||A/2^s||_1 <= 0.25
    double nrm = A.norm1(); int s = 0;
    if (nrm > 0.25) s = (int)std::ceil(std::log2(nrm / 0.25));
    CMat X = A.scaled(cplx(std::ldexp(1.0, -s), 0));
    CMat term = CMat::eye(A.n), sum = CMat::eye(A.n);
    for (int k = 1; k <= 24; k++) { term = (term * X).scaled(cplx(1.0 / k, 0)); sum = sum + term; }
    for (int i = 0; i < s; i++) sum = sum * sum;
    return sum;
}

static CMat sx() { CMat m(2); m(0, 1) = 1; m(1, 0) = 1; return m; }
static CMat sy() { CMat m(2); m(0, 1) = cplx(0, -1); m(1, 0) = cplx(0, 1); return m; }
static CMat sz() { CMat m(2); m(0, 0) = 1; m(1, 1) = -1; return m; }
static CMat I2() { return CMat::eye(2); }
static CMat Ry(double th) { CMat m(2); double c = std::cos(th / 2), s = std::sin(th / 2); m(0, 0) = c; m(0, 1) = -s; m(1, 0) = s; m(1, 1) = c; return m; }
static CMat Rz(double ph) { CMat m(2); m(0, 0) = std::exp(cplx(0, -ph / 2)); m(1, 1) = std::exp(cplx(0, ph / 2)); return m; }
static CMat encode_unitary(double a, double phi) {
    a = std::min(std::max(a, 0.0), 1.0);
    return Rz(phi) * Ry(2.0 * std::asin(std::sqrt(a)));
}
static CMat encode_state(double a, double phi) {
    a = std::min(std::max(a, 0.0), 1.0);
    double th = 2.0 * std::asin(std::sqrt(a));
    cplx psi0 = std::cos(th / 2), psi1 = std::exp(cplx(0, phi)) * std::sin(th / 2);
    CMat m(2); m(0, 0) = psi0 * std::conj(psi0); m(0, 1) = psi0 * std::conj(psi1);
    m(1, 0) = psi1 * std::conj(psi0); m(1, 1) = psi1 * std::conj(psi1); return m;
}

// ------------------------------------------------------- collision unitary ----
struct Collision {
    CMat U;                       // 2^L x 2^L on local axes
    std::vector<int> local;       // register indices in U's kron order
    std::vector<int> spec_axes;   // spectator indices
    std::vector<cplx> spec_ph0, spec_ph1;   // per spectator: phase for bit 0 / bit 1
};

static Collision build_collision(double gx, double gz, double omega_s, double omega_cont, double omega_cont_z,
                                 double omega_r, int delay, double phase, double DeltaT,
                                 const std::string& interaction, double theta_clock) {
    int n = delay + 2;
    cplx fb = -std::exp(cplx(0, -phase));
    Collision c;
    if (delay == 0) c.local = {0, 1}; else c.local = {0, 1, n - 1};
    int L = (int)c.local.size();
    auto pos = [&](int q) { for (int i = 0; i < L; i++) if (c.local[i] == q) return i; throw std::runtime_error("not local"); };
    auto op1 = [&](int q, const CMat& m) { CMat r = CMat::eye(1); for (int i = 0; i < L; i++) r = kron(r, i == pos(q) ? m : I2()); return r; };
    auto op2 = [&](int q1, const CMat& m1, int q2, const CMat& m2) {
        CMat r = CMat::eye(1); for (int i = 0; i < L; i++) r = kron(r, i == pos(q1) ? m1 : (i == pos(q2) ? m2 : I2())); return r; };
    int S = 0, F = n - 1;
    CMat H = op1(S, sz()).scaled(omega_s + omega_cont_z) + op1(S, sx()).scaled(omega_cont);
    for (int q : c.local) H = H + op1(q, sz()).scaled(0.5 * theta_clock);
    if (delay > 0) H = H + op1(1, sz()).scaled(omega_r);
    CMat V(1 << L);
    if (interaction == "beamsplitter") {
        V = (op2(S, sx(), F, sx()) + op2(S, sy(), F, sy())).scaled(gx) + op2(S, sz(), F, sz()).scaled(gz);
        if (delay > 0) V = V + ((op2(S, sx(), 1, sx()) + op2(S, sy(), 1, sy())).scaled(gx) + op2(S, sz(), 1, sz()).scaled(gz)).scaled(fb);
    } else if (interaction == "spin") {
        V = op2(S, sx(), F, sx()).scaled(gx) + op2(S, sz(), F, sz()).scaled(gz);
        if (delay > 0) V = V + (op2(S, sx(), 1, sx()).scaled(gx) + op2(S, sz(), 1, sz()).scaled(gz)).scaled(fb);
    } else throw std::runtime_error("interaction");
    c.U = expm((H + V).scaled(cplx(0, -DeltaT)));
    for (int j = 0; j < n; j++) {
        if (std::find(c.local.begin(), c.local.end(), j) != c.local.end()) continue;
        double e = 0.5 * theta_clock + omega_r;
        c.spec_axes.push_back(j);
        c.spec_ph0.push_back(std::exp(cplx(0, -DeltaT * e)));
        c.spec_ph1.push_back(std::exp(cplx(0, DeltaT * e)));
    }
    return c;
}

// ------------------------------------------------------------------ kernel ----
struct KernelOut { std::vector<double> SX, SY, POP; std::vector<int8_t> MREC; };

struct DenseRho {                 // n-qubit density matrix, row-major, big-endian bits
    int n = 0, D = 1; std::vector<cplx> a;
    DenseRho() {}
    DenseRho(int n_) : n(n_), D(1 << n_), a((size_t)(1 << n_) * (1 << n_), cplx(0, 0)) {}
    cplx& at(int i, int j) { return a[(size_t)i * D + j]; }
    const cplx& at(int i, int j) const { return a[(size_t)i * D + j]; }
};

// apply U (2^L x 2^L) on axes `ax` (ascending register indices, U's kron order == ax order)
static void apply_local(DenseRho& rho, const CMat& U, const std::vector<int>& ax) {
    int n = rho.n, D = rho.D, L = (int)ax.size(), M = 1 << L;
    // masks
    std::vector<int> bit(L); for (int i = 0; i < L; i++) bit[i] = 1 << (n - 1 - ax[i]);
    int localMask = 0; for (int b : bit) localMask |= b;
    // enumerate "rest" configurations = indices with local bits cleared
    std::vector<int> rest; rest.reserve(D >> L);
    for (int i = 0; i < D; i++) if ((i & localMask) == 0) rest.push_back(i);
    std::vector<int> off(M);      // local index a -> bit pattern
    for (int a = 0; a < M; a++) { int o = 0; for (int i = 0; i < L; i++) if (a & (1 << (L - 1 - i))) o |= bit[i]; off[a] = o; }
    // Row pass, vectorised along j: for every rest configuration the M local rows
    // are whole contiguous rows of rho, so out[a][:] = sum_b U(a,b) row_b[:].
    std::vector<cplx> buf((size_t)M * D);
    auto row_pass = [&]() {
        for (int r : rest) {
            for (int a = 0; a < M; a++) {
                cplx* o = &buf[(size_t)a * D];
                std::fill(o, o + D, cplx(0, 0));
                for (int b = 0; b < M; b++) {
                    cplx u = U(a, b); if (u == cplx(0, 0)) continue;
                    const cplx* src = &rho.a[(size_t)(r | off[b]) * D];
                    for (int j = 0; j < D; j++) o[j] += u * src[j];
                }
            }
            for (int a = 0; a < M; a++) std::memcpy(&rho.a[(size_t)(r | off[a]) * D], &buf[(size_t)a * D], sizeof(cplx) * D);
        }
    };
    // rho is Hermitian, so rho U^dag = (U rho^dag)^dag = (U rho)^dag:  rho' = U (U rho)^dag
    row_pass();
    for (int i = 0; i < D; i++) for (int j = i; j < D; j++) {
        cplx a_ = rho.at(i, j), b_ = rho.at(j, i);
        rho.at(i, j) = std::conj(b_); rho.at(j, i) = std::conj(a_);
    }
    row_pass();
}

static void apply_spectator_phases(DenseRho& rho, const Collision& c) {
    if (c.spec_axes.empty()) return;
    int n = rho.n, D = rho.D;
    std::vector<cplx> ph(D, cplx(1, 0));
    for (size_t s = 0; s < c.spec_axes.size(); s++) {
        int b = 1 << (n - 1 - c.spec_axes[s]);
        for (int i = 0; i < D; i++) ph[i] *= (i & b) ? c.spec_ph1[s] : c.spec_ph0[s];
    }
    for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) rho.at(i, j) *= ph[i] * std::conj(ph[j]);
}

struct KernelParams {
    double gx = 0.75, gz = 0.3, omega_s = 0, omega_cont = 0, omega_cont_z = 0, omega_r = 0;
    int delay = 5; double phase = PI, DeltaT = 1.0; std::string basis = "x";
    std::string interaction = "beamsplitter"; double theta_clock = 0; std::string encode_target = "system";
};

static KernelOut kernel(const std::vector<double>& amps, const std::vector<double>& phs, const KernelParams& p, uint32_t seed) {
    int n = p.delay + 2, m = n - 1;
    Collision col = build_collision(p.gx, p.gz, p.omega_s, p.omega_cont, p.omega_cont_z, p.omega_r,
                                    p.delay, p.phase, p.DeltaT, p.interaction, p.theta_clock);
    bool no_meas = (p.basis == "none");
    cplx v0[2], v1[2];
    if (p.basis == "z") { v0[0] = 1; v0[1] = 0; v1[0] = 0; v1[1] = 1; }
    else if (p.basis == "x") { double r = 1 / std::sqrt(2.0); v0[0] = r; v0[1] = r; v1[0] = r; v1[1] = -r; }
    else if (p.basis == "y") { double r = 1 / std::sqrt(2.0); v0[0] = r; v0[1] = cplx(0, r); v1[0] = r; v1[1] = cplx(0, -r); }
    else if (!no_meas) throw std::runtime_error("basis");
    bool enc_sys = (p.encode_target == "system");
    NumpyRandomState rng(seed);
    size_t runs = amps.size();
    KernelOut out; out.SX.resize(runs); out.SY.resize(runs); out.POP.resize(runs); out.MREC.resize(runs);
    DenseRho rho(m); rho.at(0, 0) = 1.0;              // ground + vacuum
    DenseRho rp(n);
    int Dm = rho.D, Dn = rp.D, R = 1 << (n - 2);       // rest size for (S, idx1, rest)
    for (size_t t = 0; t < runs; t++) {
        CMat aux(2);
        if (enc_sys) { apply_local(rho, encode_unitary(amps[t], phs[t]), {0}); aux(0, 0) = 1; }
        else aux = encode_state(amps[t], phs[t]);
        // rp = rho (x) aux
        for (int i = 0; i < Dm; i++) for (int j = 0; j < Dm; j++) {
            cplx r = rho.at(i, j);
            for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) rp.at(i * 2 + a, j * 2 + b) = r * aux(a, b);
        }
        apply_local(rp, col.U, col.local);
        apply_spectator_phases(rp, col);
        double trp = 0; for (int i = 0; i < Dn; i++) trp += rp.at(i, i).real();
        if (trp > 1e-12) for (auto& z : rp.a) z /= trp;
        // reduced state of index 1: index i = s*(2R) + a*R + x
        cplx rex[2][2] = {{0, 0}, {0, 0}};
        for (int s = 0; s < 2; s++) for (int x = 0; x < R; x++) for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++)
            rex[a][b] += rp.at(s * 2 * R + a * R + x, s * 2 * R + b * R + x);
        out.SX[t] = (rex[0][1] + rex[1][0]).real();
        out.SY[t] = (cplx(0, 1) * rex[0][1] - cplx(0, 1) * rex[1][0]).real();
        out.POP[t] = rex[1][1].real();
        if (no_meas) {
            out.MREC[t] = -1;
            for (int s = 0; s < 2; s++) for (int x = 0; x < R; x++) for (int tt = 0; tt < 2; tt++) for (int y = 0; y < R; y++) {
                cplx z = 0; for (int a = 0; a < 2; a++) z += rp.at(s * 2 * R + a * R + x, tt * 2 * R + a * R + y);
                rho.at(s * R + x, tt * R + y) = z;
            }
            continue;
        }
        auto expv = [&](const cplx* v) { cplx s = 0; for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) s += std::conj(v[a]) * rex[a][b] * v[b]; return s.real(); };
        double p0 = std::max(expv(v0), 0.0), p1 = std::max(expv(v1), 0.0);
        double ssum = p0 + p1;
        if (ssum < 1e-12) { p0 = 0.5; p1 = 0.5; } else { p0 /= ssum; p1 /= ssum; }
        int o = rng.choice01(p0, p1); out.MREC[t] = (int8_t)o;
        const cplx* v = (o == 0) ? v0 : v1;
        double nr = 0;
        for (int s = 0; s < 2; s++) for (int x = 0; x < R; x++) for (int tt = 0; tt < 2; tt++) for (int y = 0; y < R; y++) {
            cplx z = 0;
            for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) z += std::conj(v[a]) * rp.at(s * 2 * R + a * R + x, tt * 2 * R + b * R + y) * v[b];
            rho.at(s * R + x, tt * R + y) = z;
        }
        for (int i = 0; i < Dm; i++) nr += rho.at(i, i).real();
        if (nr > 1e-12) for (auto& z : rho.a) z /= nr;
    }
    return out;
}

// ---------------------------------------------------------------- transform ----
struct TransformParams {
    int delay = 5; double phase = PI; int n_traj = 1; uint32_t base_seed = 12345;
    double gamma_x = 0.75, gamma_z = 0.3, amp_pow = 1.0, omega_s_res = 23.0;
    std::string basis = "x", interaction = "beamsplitter", encode_target = "system";
    int carrier_clock = 1;        // 0 off, 1 input IF, 2 grid ('bin')
    int stft_hop = 256; double fs = 8000;
};

struct Mat { int nf = 0, nb = 0; std::vector<double> a; Mat() {} Mat(int f, int b) : nf(f), nb(b), a((size_t)f * b, 0.0) {}
    double& at(int i, int k) { return a[(size_t)i * nb + k]; } double at(int i, int k) const { return a[(size_t)i * nb + k]; } };

struct TransformOut { Mat out_mag, out_phase; double scale = 1; std::vector<int8_t> mrec; int T = 0; };
static std::vector<double> carrier_clock_rates(const Mat& im, const Mat& ip, const std::vector<double>& fbins, int hop, double fs, int mode);

// ------------------------------------------------------ trajectory store ----
// Raw kernel outputs per trajectory and bin (dsm_engine.trajectories). Layout
// [ti][k][t] so that appending trajectories is a block append; every trajectory
// has its own seed (base + 10007 k + ti), so incremental computation is exact.
struct Store {
    int nb = 0, ntraj = 0, T = 0, nfed = 0; double scale = 1.0;
    std::vector<double> theta;              // per-bin clock rate (empty = clock off)
    std::vector<double> SX, SY, POP; std::vector<int8_t> MREC;
    size_t idx(int ti, int k, int t) const { return ((size_t)ti * nb + k) * T + t; }
};

struct BinSetup {
    double scale = 1.0; Mat mag_norm; std::vector<double> theta; int T = 0, nf = 0, nb = 0;
    // per-bin kernel inputs
    void per_bin(int k, const Mat& in_phase, const TransformParams& p, const std::vector<double>& fbins,
                 std::vector<double>& amps, std::vector<double>& phs, KernelParams& kp) const {
        int delay = p.delay; amps.assign(T, 0.0); phs.assign(T, 0.0);
        for (int t = 0; t < nf; t++) { amps[t] = std::pow(std::min(std::max(mag_norm.at(t, k), 0.0), 1.0), p.amp_pow); phs[t] = in_phase.at(t, k); }
        kp.gx = p.gamma_x; kp.gz = p.gamma_z; kp.delay = delay; kp.phase = p.phase; kp.basis = p.basis;
        kp.interaction = p.interaction; kp.encode_target = p.encode_target;
        kp.omega_s = (p.omega_s_res != 0) ? p.omega_s_res * (fbins[k] / p.fs) * p.gamma_z : 0.0;
        kp.theta_clock = theta.empty() ? 0.0 : theta[k];
    }
};

static BinSetup bin_setup(const Mat& in_mag, const Mat& in_phase, const std::vector<double>& fbins, const TransformParams& p) {
    BinSetup s; s.nf = in_mag.nf; s.nb = in_mag.nb; s.T = s.nf + p.delay;
    double sc = 0; for (double v : in_mag.a) sc = std::max(sc, std::fabs(v));
    s.mag_norm = Mat(s.nf, s.nb);
    if (sc < 1e-12) sc = 1.0; else for (size_t i = 0; i < s.mag_norm.a.size(); i++) s.mag_norm.a[i] = in_mag.a[i] / sc;
    s.scale = sc;
    if (p.carrier_clock) s.theta = carrier_clock_rates(in_mag, in_phase, fbins, p.stft_hop, p.fs, p.carrier_clock);
    return s;
}

// compute trajectories [traj_from, traj_to) of every bin and append them to the store
static void compute_store(const Mat& in_mag, const Mat& in_phase, const std::vector<double>& fbins, const TransformParams& p,
                          int traj_from, int traj_to, Store& st, int nthreads = 1, std::function<bool(int, int)> progress = nullptr) {
    BinSetup su = bin_setup(in_mag, in_phase, fbins, p);
    if (p.basis == "none") traj_to = std::min(traj_to, 1);
    if (st.ntraj == 0) { st.nb = su.nb; st.T = su.T; st.nfed = su.nf; st.scale = su.scale; st.theta = su.theta; }
    if (st.ntraj != traj_from || st.nb != su.nb || st.T != su.T) throw std::runtime_error("store/trajectory range mismatch");
    int n_new = traj_to - traj_from; if (n_new <= 0) return;
    size_t oldSize = st.SX.size(), add = (size_t)n_new * st.nb * st.T;
    st.SX.resize(oldSize + add, 0.0); st.SY.resize(oldSize + add, 0.0); st.POP.resize(oldSize + add, 0.0); st.MREC.resize(oldSize + add, -1);
    std::atomic<int> next(0), done(0); std::atomic<bool> abort(false); int total = n_new * st.nb;
    auto work = [&]() {
        std::vector<double> amps, phs;
        for (;;) {
            int j = next.fetch_add(1); if (j >= total || abort.load()) break;
            int ti = traj_from + j / st.nb, k = j % st.nb;
            KernelParams kp; su.per_bin(k, in_phase, p, fbins, amps, phs, kp);
            KernelOut ko = kernel(amps, phs, kp, p.base_seed + 10007u * (uint32_t)k + (uint32_t)ti);
            size_t base = st.idx(ti, k, 0);
            for (int t = 0; t < st.T; t++) { st.SX[base + t] = ko.SX[t]; st.SY[base + t] = ko.SY[t]; st.POP[base + t] = ko.POP[t]; st.MREC[base + t] = ko.MREC[t]; }
            int d = done.fetch_add(1) + 1;
            if (progress && !progress(d, total)) abort.store(true);
        }
    };
    std::vector<std::thread> th; for (int i = 1; i < nthreads; i++) th.emplace_back(work);
    work(); for (auto& t : th) t.join();
    if (abort.load()) { st.SX.resize(oldSize); st.SY.resize(oldSize); st.POP.resize(oldSize); st.MREC.resize(oldSize); throw std::runtime_error("aborted"); }
    st.ntraj = traj_to;
}

// decoded (scaled magnitude, phase) over the fed frames from the first n_use trajectories
// version: 0 uncond, 1 cond0, 2 cond1 (dsm_engine.render_from_store)
static void render_from_store(const Store& st, int version, int n_use, int delay, double amp_pow, const Mat& in_phase,
                              Mat& out_mag, Mat& out_phase) {
    int T = st.T, nf = st.nfed, nb = st.nb; n_use = std::max(1, std::min(n_use, st.ntraj));
    out_mag = Mat(nf, nb); out_phase = in_phase;
    std::vector<char> mask(n_use);
    for (int k = 0; k < nb; k++) {
        for (int t = 0; t < T; t++) {
            int cnt = 0;
            bool cond = (version != 0) && (t + delay < T);
            if (cond) { int a = version - 1; for (int ti = 0; ti < n_use; ti++) { mask[ti] = (st.MREC[st.idx(ti, k, t + delay)] == a); cnt += mask[ti]; } }
            bool useAll = !cond || cnt == 0;
            double sx = 0, sy = 0, pop = 0; int n = 0;
            for (int ti = 0; ti < n_use; ti++) { if (!useAll && !mask[ti]) continue; sx += st.SX[st.idx(ti, k, t)]; sy += st.SY[st.idx(ti, k, t)]; pop += st.POP[st.idx(ti, k, t)]; n++; }
            sx /= n; sy /= n; pop /= n;
            if (t >= delay && t - delay < nf) {
                out_mag.at(t - delay, k) = std::pow(std::min(std::max(pop, 0.0), 1.0), 1.0 / amp_pow) * st.scale;
                out_phase.at(t - delay, k) = std::atan2(sy, sx);
            }
        }
    }
}

// the classical twin (dsm_engine.twin / twin_bin): linear one-excitation model with
// site energies relative to the vacuum branch (clock, resonator, sigma_z sigma_z)
static void twin_render(const Mat& in_mag, const Mat& in_phase, const std::vector<double>& fbins, const TransformParams& p,
                        Mat& out_mag, Mat& out_phase) {
    BinSetup su = bin_setup(in_mag, in_phase, fbins, p);
    int delay = p.delay, n = delay + 2, S = 0, D0 = 1, F = delay + 1, T = su.T, nf = su.nf, nb = su.nb;
    out_mag = Mat(nf, nb); out_phase = in_phase;
    cplx fb = -std::exp(cplx(0, -p.phase)); double J = 2 * p.gamma_x;
    std::vector<double> amps, phs; std::vector<cplx> c(n), c2(n);
    for (int k = 0; k < nb; k++) {
        KernelParams kp; su.per_bin(k, in_phase, p, fbins, amps, phs, kp);
        CMat Hm(n); Hm(S, F) = J; Hm(F, S) = J; Hm(S, D0) = fb * J; Hm(D0, S) = std::conj(fb) * J;
        double th = kp.theta_clock, gz = p.gamma_z;
        Hm(S, S) += -4 * gz - 2 * kp.omega_s - th; Hm(F, F) += -2 * gz - th; Hm(D0, D0) += -2 * gz - th;
        for (int j = 2; j <= delay; j++) Hm(j, j) += -th;
        CMat U1 = expm(Hm.scaled(cplx(0, -1.0)));
        std::fill(c.begin(), c.end(), cplx(0, 0));
        for (int t = 0; t < T; t++) {
            if (p.encode_target == "system") c[S] = std::exp(cplx(0, phs[t])) * (c[S] + std::sqrt(amps[t]));
            else c[F] = std::sqrt(amps[t]) * std::exp(cplx(0, phs[t]));
            for (int i = 0; i < n; i++) { cplx s = 0; for (int j = 0; j < n; j++) s += U1(i, j) * c[j]; c2[i] = s; }
            if (t >= delay && t - delay < nf) {
                out_mag.at(t - delay, k) = std::pow(std::min(std::max(std::norm(c2[D0]), 0.0), 1.0), 1.0 / p.amp_pow) * su.scale;
                out_phase.at(t - delay, k) = std::arg(c2[D0]);
            }
            c[0] = c2[0]; for (int i = 1; i < n - 1; i++) c[i] = c2[i + 1]; c[n - 1] = 0.0;
        }
    }
}

// carrier clock rates per bin (dsm_engine.carrier_clock_rates)
static std::vector<double> carrier_clock_rates(const Mat& im, const Mat& ip, const std::vector<double>& fbins, int hop, double fs, int mode) {
    int nb = im.nb, nf = im.nf; std::vector<double> th(nb);
    for (int k = 0; k < nb; k++) th[k] = 2.0 * PI * fbins[k] * hop / fs;
    if (mode == 2) return th;
    for (int k = 0; k < nb; k++) {
        cplx zc = 0;
        for (int t = 1; t < nf; t++) { double w = im.at(t, k) * im.at(t, k); zc += w * std::exp(cplx(0, (ip.at(t, k) - ip.at(t - 1, k)) - th[k])); }
        th[k] += std::arg(zc + cplx(1e-30, 0));
    }
    return th;
}

// uncond version only; bins processed in `nthreads` worker threads (bin-independent)
static TransformOut transform(const Mat& in_mag, const Mat& in_phase, const std::vector<double>& fbins,
                              const TransformParams& p, int nthreads = 1,
                              std::function<bool(int)> progress = nullptr) {
    int nf = in_mag.nf, nb = in_mag.nb, delay = p.delay, T = nf + delay;
    bool no_meas = (p.basis == "none"); int n_traj = no_meas ? 1 : p.n_traj;
    double scale = 0; for (double v : in_mag.a) scale = std::max(scale, std::fabs(v));
    Mat mag_norm(nf, nb);
    if (scale < 1e-12) scale = 1.0; else for (size_t i = 0; i < mag_norm.a.size(); i++) mag_norm.a[i] = in_mag.a[i] / scale;
    TransformOut out; out.scale = scale; out.out_mag = Mat(nf, nb); out.out_phase = in_phase; out.T = T;
    for (size_t i = 0; i < out.out_mag.a.size(); i++) out.out_mag.a[i] = std::min(std::max(mag_norm.a[i], 0.0), 1.0);
    out.mrec.assign((size_t)nb * T * n_traj, -1);
    std::vector<double> th;
    if (p.carrier_clock) th = carrier_clock_rates(in_mag, in_phase, fbins, p.stft_hop, p.fs, p.carrier_clock);
    std::atomic<int> next(0), done(0); std::atomic<bool> abort(false);
    auto work = [&]() {
        std::vector<double> amps(T), phs(T);
        std::vector<double> SXm(T), SYm(T), POPm(T);
        for (;;) {
            int k = next.fetch_add(1); if (k >= nb || abort.load()) break;
            KernelParams kp; kp.gx = p.gamma_x; kp.gz = p.gamma_z; kp.delay = delay; kp.phase = p.phase;
            kp.basis = p.basis; kp.interaction = p.interaction; kp.encode_target = p.encode_target;
            if (p.omega_s_res != 0) kp.omega_s = p.omega_s_res * (fbins[k] / p.fs) * p.gamma_z;
            kp.theta_clock = p.carrier_clock ? th[k] : 0.0;
            for (int t = 0; t < nf; t++) { amps[t] = std::pow(std::min(std::max(mag_norm.at(t, k), 0.0), 1.0), p.amp_pow); phs[t] = in_phase.at(t, k); }
            for (int t = nf; t < T; t++) { amps[t] = 0; phs[t] = 0; }
            std::fill(SXm.begin(), SXm.end(), 0); std::fill(SYm.begin(), SYm.end(), 0); std::fill(POPm.begin(), POPm.end(), 0);
            for (int ti = 0; ti < n_traj; ti++) {
                KernelOut ko = kernel(amps, phs, kp, p.base_seed + 10007u * (uint32_t)k + (uint32_t)ti);
                for (int t = 0; t < T; t++) { SXm[t] += ko.SX[t]; SYm[t] += ko.SY[t]; POPm[t] += ko.POP[t]; out.mrec[((size_t)k * n_traj + ti) * T + t] = ko.MREC[t]; }
            }
            for (int t = 0; t < nf; t++) {
                double amp = POPm[t + delay] / n_traj, sx = SXm[t + delay] / n_traj, sy = SYm[t + delay] / n_traj;
                amp = std::min(std::max(amp, 0.0), 1.0);
                out.out_mag.at(t, k) = std::pow(amp, 1.0 / p.amp_pow);
                out.out_phase.at(t, k) = std::atan2(sy, sx);
            }
            int d = done.fetch_add(1) + 1;
            if (progress && !progress(d)) abort.store(true);
        }
    };
    std::vector<std::thread> th_;
    for (int i = 1; i < nthreads; i++) th_.emplace_back(work);
    work();
    for (auto& t : th_) t.join();
    if (abort.load()) throw std::runtime_error("aborted");
    return out;
}

// -------------------------------------------------------------------- FFT ----
static void fft_inplace(std::vector<cplx>& a, bool inverse) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) { int bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) std::swap(a[i], a[j]); }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (inverse ? 1 : -1);
        for (int i = 0; i < n; i += len) for (int j = 0; j < len / 2; j++) {
            cplx w = std::exp(cplx(0, ang * j));          // direct twiddle: accuracy over speed
            cplx u = a[i + j], v = a[i + j + len / 2] * w;
            a[i + j] = u + v; a[i + j + len / 2] = u - v;
        }
    }
    if (inverse) for (auto& z : a) z /= (double)n;
}

static std::vector<double> hann_sym(int N) {           // scipy.signal.windows.hann(N) (sym)
    std::vector<double> w(N); double step = 2 * PI / (N - 1);
    for (int i = 0; i < N; i++) w[i] = 0.5 + 0.5 * std::cos(-PI + i * step);
    return w;
}

// periodic STFT of loop x (length L = nf*H), frames centred at mH (smstools dftAnal conventions)
static void periodic_stft(const std::vector<double>& x, int N, int H, Mat& mag, Mat& phase) {
    int L = (int)x.size(), nf = L / H, hM1 = (N + 1) / 2, hM2 = N / 2, nb = N / 2 + 1;
    std::vector<double> w = hann_sym(N); double ws = 0; for (double v : w) ws += v; for (double& v : w) v /= ws;
    mag = Mat(nf, nb); phase = Mat(nf, nb);
    std::vector<cplx> buf(N);
    for (int m = 0; m < nf; m++) {
        std::vector<double> xw(N);
        for (int i = 0; i < N; i++) { int idx = ((m * H - hM2 + i) % L + L) % L; xw[i] = x[idx] * w[i]; }
        for (int i = 0; i < hM1; i++) buf[i] = xw[hM2 + i];
        for (int i = 0; i < hM2; i++) buf[N - hM2 + i] = xw[i];
        fft_inplace(buf, false);
        for (int k = 0; k < nb; k++) { mag.at(m, k) = std::max(std::abs(buf[k]), EPS); phase.at(m, k) = std::arg(buf[k]); }
    }
}

static std::vector<double> circular_ola(const Mat& mag, const Mat& phase, int N, int H, int L) {
    int nf = mag.nf, hM1 = (N + 1) / 2, hM2 = N / 2, nb = N / 2 + 1;
    std::vector<double> y(L, 0.0); std::vector<cplx> Y(N);
    for (int m = 0; m < nf; m++) {
        for (int k = 0; k < nb; k++) Y[k] = mag.at(m, k) * std::exp(cplx(0, phase.at(m, k)));
        Y[0] = cplx(Y[0].real(), 0); Y[N / 2] = cplx(Y[N / 2].real(), 0);       // np.fft.irfft semantics
        for (int k = 1; k < N / 2; k++) Y[N - k] = std::conj(Y[k]);
        fft_inplace(Y, true);
        std::vector<double> yw(N);
        for (int i = 0; i < hM2; i++) yw[i] = Y[N - hM2 + i].real();
        for (int i = 0; i < hM1; i++) yw[hM2 + i] = Y[i].real();
        for (int i = 0; i < N; i++) { int idx = ((m * H - hM1 + i) % L + L) % L; y[idx] += H * yw[i]; }
    }
    return y;
}

// -------------------------------------------------------------- resampling ----
static double sinc(double x) { return x == 0 ? 1.0 : std::sin(PI * x) / (PI * x); }
static double bessel_i0(double x) { double s = 1, t = 1, h = x / 2; for (int k = 1; k < 60; k++) { t *= (h / k) * (h / k); s += t; if (t < 1e-18 * s) break; } return s; }

// scipy.signal.firwin(numtaps, cutoff, window=...) lowpass, pass_zero, scale=True
static std::vector<double> firwin(int numtaps, double cutoff, const std::string& window, double beta = 5.0) {
    std::vector<double> h(numtaps), w(numtaps); double alpha = 0.5 * (numtaps - 1);
    for (int i = 0; i < numtaps; i++) h[i] = cutoff * sinc(cutoff * (i - alpha));
    double step = 2 * PI / (numtaps - 1);
    for (int i = 0; i < numtaps; i++) {
        if (window == "hamming") w[i] = 0.54 + 0.46 * std::cos(-PI + i * step);
        else if (window == "kaiser") { double r = (i - alpha) / alpha; w[i] = bessel_i0(beta * std::sqrt(std::max(0.0, 1 - r * r))) / bessel_i0(beta); }
        else throw std::runtime_error("window");
        h[i] *= w[i];
    }
    double s = 0; for (double v : h) s += v; for (double& v : h) v /= s;
    return h;
}

// scipy.signal.decimate(x, q, ftype='fir', zero_phase=True): y[m] = sum_j h[j] x[m q + half - j]
static std::vector<double> decimate_fir(const std::vector<double>& x, int q) {
    if (q == 1) return x;
    int half = 10 * q; std::vector<double> h = firwin(2 * half + 1, 1.0 / q, "hamming");
    int n_in = (int)x.size(), n_out = n_in / q + (n_in % q ? 1 : 0);
    std::vector<double> y(n_out, 0.0);
    for (int m = 0; m < n_out; m++) { double s = 0; for (int j = 0; j <= 2 * half; j++) { int i = m * q + half - j; if (i >= 0 && i < n_in) s += h[j] * x[i]; } y[m] = s; }
    return y;
}

// scipy.signal.resample_poly(x, up, 1): y[m] = up * sum_j h[j] x_up[m + half - j]
static std::vector<double> upsample_poly(const std::vector<double>& x, int up) {
    if (up == 1) return x;
    int half = 10 * up; std::vector<double> h = firwin(2 * half + 1, 1.0 / up, "kaiser", 5.0);
    int n_in = (int)x.size(), n_out = n_in * up;
    std::vector<double> y(n_out, 0.0);
    for (int m = 0; m < n_out; m++) {
        double s = 0;
        for (int j = 0; j <= 2 * half; j++) { int i = m + half - j; if (i < 0 || i >= n_out || i % up) continue; s += h[j] * x[i / up]; }
        y[m] = up * s;
    }
    return y;
}

// ------------------------------------------------------------ pink noise ----
// The PINK NOISE render (dsm_engine.pink_rival / summary notebook): the input's own carrier
// dressed with 1/f^P noise whose phase depth, memory exponent P and AM depth are fitted to
// the quantum undep render. Same definitions as the Python reference; differences: the noise
// generator (mt19937_64 + Box-Muller instead of numpy's PCG64), spectral shaping on a
// power-of-two FFT of the white field (first nf samples kept). depth and AM depth are
// deterministic and match Python; P is a coarse fit that depends on the noise realisation
// (see test_dsm_cpp.py for its seed-to-seed spread); the realisation is engine-specific.
static void pink_field(int nf, int nb, double p, uint64_t seed, Mat& out) {
    std::mt19937_64 g(seed); std::normal_distribution<double> nd(0.0, 1.0);
    int N2 = 1; while (N2 < nf) N2 <<= 1;
    out = Mat(nf, nb);
    std::vector<cplx> buf(N2);
    for (int k = 0; k < nb; k++) {
        for (int i = 0; i < N2; i++) buf[i] = nd(g);
        fft_inplace(buf, false);
        for (int i = 0; i < N2; i++) { int fi = std::min(i, N2 - i); double f = (fi == 0 ? 1.0 : (double)fi) / N2; buf[i] /= std::pow(f, p / 2); }
        fft_inplace(buf, true);
        double m = 0, v = 0; for (int i = 0; i < nf; i++) m += buf[i].real(); m /= nf;
        for (int i = 0; i < nf; i++) { double d = buf[i].real() - m; v += d * d; } v = std::sqrt(v / nf) + 1e-9;
        for (int i = 0; i < nf; i++) out.at(i, k) = (buf[i].real() - m) / v;
    }
}
static void cacf(const Mat& f, const Mat& W, const std::vector<int>& bins, int L, std::vector<double>& out) {
    out.assign(L, 1.0); int nf = f.nf;
    for (int l = 1; l < L; l++) { double num = 0, den = 0;
        for (int k : bins) for (int t = l; t < nf; t++) { double wl = W.at(t, k) * W.at(t - l, k); num += wl * std::cos(f.at(t, k) - f.at(t - l, k)); den += wl; }
        out[l] = num / (den + 1e-30); }
}
struct PinkFit { double depth = 0, P = 0, am = 0, residual = 0; };
static PinkFit pink_render(const Mat& in_mag, const Mat& in_phase, const Mat& out_mag, const Mat& out_phase, uint64_t seed,
                           Mat& cl_mag, Mat& cl_phase) {
    int nf = in_mag.nf, nb = in_mag.nb; PinkFit fit;
    Mat dev(nf, nb), W(nf, nb); double num = 0, den = 0; std::vector<double> ebin(nb, 0.0);
    for (int t = 0; t < nf; t++) for (int k = 0; k < nb; k++) {
        double d = std::arg(std::exp(cplx(0, out_phase.at(t, k) - in_phase.at(t, k)))), w = in_mag.at(t, k) * in_mag.at(t, k);
        dev.at(t, k) = d; W.at(t, k) = w; num += w * d * d; den += w; ebin[k] += w; }
    fit.depth = std::sqrt(num / (den + 1e-30));
    int L = std::min(25, nf - 1);
    std::vector<int> all(nb); for (int k = 0; k < nb; k++) all[k] = k;
        std::vector<double> target; cacf(dev, W, all, L, target);
    double bestCost = 1e300; Mat field;
    for (int gi = 0; gi < 20; gi++) {
        double p = 0.3 + gi * (6.0 - 0.3) / 19.0;
        pink_field(nf, nb, p, seed + 3, field);
        for (auto& v : field.a) v *= fit.depth;
        std::vector<double> c; cacf(field, W, all, L, c);
        double cost = 0; for (int l = 0; l < L; l++) cost += (c[l] - target[l]) * (c[l] - target[l]);
        if (cost < bestCost) { bestCost = cost; fit.P = p; }
    }
    fit.residual = std::sqrt(bestCost / L);
    pink_field(nf, nb, fit.P, seed + 7, field);
    cl_phase = in_phase; for (int t = 0; t < nf; t++) for (int k = 0; k < nb; k++) cl_phase.at(t, k) += fit.depth * field.at(t, k);
    double mx = 0; for (double v : in_mag.a) mx = std::max(mx, v);
    std::vector<double> r; for (int t = 0; t < nf; t++) for (int k = 0; k < nb; k++) if (in_mag.at(t, k) > 0.1 * mx) r.push_back(out_mag.at(t, k) / std::max(in_mag.at(t, k), 1e-12) - 1.0);
    if (r.empty()) fit.am = 0.3; else { double m = 0; for (double v : r) m += v; m /= r.size(); double s = 0; for (double v : r) s += (v - m) * (v - m); fit.am = std::min(std::max(std::sqrt(s / r.size()), 0.05), 1.0); }
    pink_field(nf, nb, fit.P, seed + 8, field);
    cl_mag = in_mag; for (int t = 0; t < nf; t++) for (int k = 0; k < nb; k++) cl_mag.at(t, k) = std::max(in_mag.at(t, k) * (1.0 + fit.am * field.at(t, k)), 0.0);
    return fit;
}

// ---------------------------------------------------------------- the job ----
struct JobParams {
    int delay = 5, H = 256, D = 6; double t_l = 4.0; double gamma_x = 0.75, gamma_z = 0.3; uint32_t seed = 12345;
    double phase = PI; std::string basis = "x"; double amp_pow = 1.0, omega_s_res = 23.0; int carrier_clock = 1;
    std::string encode_target = "system", interaction = "beamsplitter"; int n_traj = 1;
    int render = 0;             // 0 undep (uncond), 1 dep0, 2 dep1, 3 classical FB (linear twin), 4 pink noise
};

struct JobResult {
    std::vector<double> x_model, y_model, y_engine; Mat in_mag, in_phase, out_mag, out_phase; std::vector<int8_t> mrec;
    int nf = 0, W = 0, N = 0, H = 0, T = 0; double fs_m = 0, scale = 1;
};

// the analysed material: decimated loop, its periodic STFT and the fed (warm-up + loop) frames
struct Material {
    std::vector<double> x_model; int nf = 0, W = 0, N = 0, H = 0, D = 1; double fs_m = 0;
    Mat in_mag, in_phase, fed_mag, fed_phase; std::vector<double> fbins;
};

static int loop_frames(double t_l, double fs_m, int H) { return std::max(2, (int)std::floor(t_l * fs_m / H + 0.5)); }

// exact one-excitation impulse response of the channel (dsm_chain.impulse_model)
static std::vector<double> impulse_model(double gx, double gz, int delay, const std::string& encode_target, double phase, int steps) {
    int n = delay + 2, S = 0, D0 = 1, F = delay + 1;
    cplx fb = -std::exp(cplx(0, -phase)); double J = 2 * gx;
    CMat Hm(n);
    Hm(S, F) = J; Hm(F, S) = J; Hm(S, D0) = fb * J; Hm(D0, S) = std::conj(fb) * J;
    for (int i = 0; i < n; i++) Hm(i, i) += 2 * gz;
    Hm(S, S) += -4 * gz; Hm(F, F) += -2 * gz; Hm(D0, D0) += -2 * gz;
    CMat U1 = expm(Hm.scaled(cplx(0, -1.0)));
    std::vector<cplx> c(n, 0.0), c2(n); std::vector<double> pop(steps);
    for (int t = 0; t < steps; t++) {
        if (t == 0) c[encode_target == "aux" ? F : S] = 1.0;
        for (int i = 0; i < n; i++) { cplx s = 0; for (int j = 0; j < n; j++) s += U1(i, j) * c[j]; c2[i] = s; }
        pop[t] = std::norm(c2[D0]);
        c[0] = c2[0]; for (int i = 1; i < n - 1; i++) c[i] = c2[i + 1]; c[n - 1] = 0.0;
    }
    return pop;
}

// warm-up frames: smallest W after which < eps of an impulse's decoded energy is still to come
// (dsm_chain.warmup_frames); at least 4d and 12; the caller caps at the loop length.
static int warmup_frames(int delay, double gx, double gz, const std::string& encode_target = "system", double eps = 1e-3, int cap = 400) {
    std::vector<double> pop = impulse_model(gx, gz, delay, encode_target, PI, cap + 1);
    double tot = 0; for (double v : pop) tot += v; tot = std::max(tot, 1e-300);
    double tail = 0; int W = cap;
    std::vector<double> cum(pop.size());
    for (int i = (int)pop.size() - 1; i >= 0; i--) { tail += pop[i]; cum[i] = tail / tot; }
    for (int i = 0; i < (int)cum.size(); i++) if (cum[i] < eps) { W = i; break; }
    return std::max(W, std::max(4 * delay, 12));
}

static Material prepare_material(const std::vector<double>& x_engine, double fs_engine, const JobParams& p) {
    Material m; int H = p.H, N = 2 * H, D = p.D;
    m.fs_m = fs_engine / D; m.H = H; m.N = N; m.D = D;
    std::vector<double> xm = decimate_fir(x_engine, D);
    int nf = loop_frames(p.t_l, m.fs_m, H), L = nf * H;
    xm.resize(L, 0.0);
    m.x_model = xm; m.nf = nf;
    int W = std::min(warmup_frames(p.delay, p.gamma_x, p.gamma_z, p.encode_target), nf); m.W = W;
    periodic_stft(xm, N, H, m.in_mag, m.in_phase);
    int nb = m.in_mag.nb;
    m.fed_mag = Mat(W + nf, nb); m.fed_phase = Mat(W + nf, nb);
    for (int i = 0; i < W + nf; i++) { int src = (i < W) ? ((nf - W + i) % nf + nf) % nf : i - W;
        for (int k = 0; k < nb; k++) { m.fed_mag.at(i, k) = m.in_mag.at(src, k); m.fed_phase.at(i, k) = m.in_phase.at(src, k); } }
    m.fbins.resize(nb); for (int k = 0; k < nb; k++) m.fbins[k] = k * m.fs_m / N;
    return m;
}

static TransformParams transform_params(const JobParams& p, const Material& m) {
    TransformParams tp; tp.delay = p.delay; tp.phase = p.phase; tp.n_traj = p.n_traj; tp.base_seed = p.seed;
    tp.gamma_x = p.gamma_x; tp.gamma_z = p.gamma_z; tp.amp_pow = p.amp_pow; tp.omega_s_res = p.omega_s_res;
    tp.basis = p.basis; tp.interaction = p.interaction; tp.encode_target = p.encode_target; tp.carrier_clock = p.carrier_clock;
    tp.stft_hop = m.H; tp.fs = m.fs_m;
    return tp;
}

// loop render from a store (or the twin): decoded loop frames, circular OLA, upsampling
static void synthesise(const Material& m, const Store& st, const JobParams& p, int render, int n_use, JobResult& r) {
    TransformParams tp = transform_params(p, m);
    Mat om, oph;
    if (render == 3) twin_render(m.fed_mag, m.fed_phase, m.fbins, tp, om, oph);
    else if (render == 4) { Mat qm, qp; render_from_store(st, 0, n_use, p.delay, p.amp_pow, m.fed_phase, qm, qp); pink_render(m.fed_mag, m.fed_phase, qm, qp, p.seed, om, oph); }
    else render_from_store(st, render, n_use, p.delay, p.amp_pow, m.fed_phase, om, oph);
    int nf = m.nf, W = m.W, nb = m.in_mag.nb;
    r.out_mag = Mat(nf, nb); r.out_phase = Mat(nf, nb);
    for (int i = 0; i < nf; i++) for (int k = 0; k < nb; k++) { r.out_mag.at(i, k) = om.at(W + i, k); r.out_phase.at(i, k) = oph.at(W + i, k); }
    r.y_model = circular_ola(r.out_mag, r.out_phase, m.N, m.H, nf * m.H);
    r.y_engine = upsample_poly(r.y_model, m.D);
    r.x_model = m.x_model; r.in_mag = m.in_mag; r.in_phase = m.in_phase; r.nf = nf; r.W = W; r.N = m.N; r.H = m.H;
    r.fs_m = m.fs_m; r.scale = st.scale; r.T = st.T; r.mrec = st.MREC;
}

static JobResult process_job(const std::vector<double>& x_engine, double fs_engine, const JobParams& p, int nthreads = 1,
                             std::function<bool(int, int)> progress = nullptr, Store* store_out = nullptr, int traj_split = 0) {
    Material m = prepare_material(x_engine, fs_engine, p);
    TransformParams tp = transform_params(p, m);
    Store st;
    if (traj_split > 0 && traj_split < p.n_traj) {      // test hook: incremental computation
        compute_store(m.fed_mag, m.fed_phase, m.fbins, tp, 0, traj_split, st, nthreads, progress);
        compute_store(m.fed_mag, m.fed_phase, m.fbins, tp, traj_split, p.n_traj, st, nthreads, progress);
    } else compute_store(m.fed_mag, m.fed_phase, m.fbins, tp, 0, p.n_traj, st, nthreads, progress);
    JobResult r; synthesise(m, st, p, p.render, p.n_traj, r);
    if (store_out) *store_out = st;
    return r;
}

} // namespace dsm
