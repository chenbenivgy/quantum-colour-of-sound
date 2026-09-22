"""Regression certificate: dsm_engine (NumPy blueprint) vs the qutip kernel.

Run:  conda activate basic_pitch_env && python test_dsm_engine.py
Pass criterion: max |diff| of SX, SY, POP <= 1e-12 and identical MREC for
every configuration; the transform-level check compares decoded matrices.
"""
import time
import numpy as np
import dependent_trajectory as dt
import dsm_engine as eng

rng0 = np.random.RandomState(7)


def _cmp(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def kernel_cases():
    steps = 40
    amps = rng0.rand(steps) * 0.9; phs = rng0.rand(steps) * 2 * np.pi - np.pi
    amps[:3] = 0.0                                     # leading silence
    for delay in (0, 1, 2, 3, 5):
        for tgt in ('aux', 'system'):
            for basis in ('x', 'z', 'none'):
                for clock, osr in ((0.0, 0.0), (0.37, 0.0), (0.37, 0.61)):
                    for interaction in ('beamsplitter', 'spin'):
                        if interaction == 'spin' and (clock or basis != 'x'):
                            continue
                        yield dict(amps=amps, phs=phs, delay=delay, tgt=tgt, basis=basis,
                                   clock=clock, os_=osr, interaction=interaction)
    # non-Hermitian feedback phase (renormalised path)
    yield dict(amps=amps, phs=phs, delay=2, tgt='aux', basis='x', clock=0.0, os_=0.0,
               interaction='beamsplitter', phase=np.pi / 3)


def run_kernel_tests():
    worst = 0.0; nfail = 0; ncase = 0
    for c in kernel_cases():
        ncase += 1
        phase = c.get('phase', np.pi)
        args = (c['amps'], c['phs'], 0.75, 0.3, c['os_'], 0.0, 0.0, 0.0, c['delay'], phase, 1.0,
                c['basis'])
        ref = dt._dependent_colour_kernel(*args, 'Pur.Deph', 12345, interaction=c['interaction'],
                                          fast=False, theta_clock=c['clock'], encode_target=c['tgt'])
        got = eng.kernel(*args, 12345, interaction=c['interaction'], theta_clock=c['clock'],
                         encode_target=c['tgt'])
        d = max(_cmp(ref[0], got[0]), _cmp(ref[1], got[1]), _cmp(ref[2], got[2]))
        same_rec = np.array_equal(ref[3], got[3])
        ok = d <= 1e-12 and same_rec
        worst = max(worst, d); nfail += (not ok)
        tag = 'ok ' if ok else 'FAIL'
        print(f"  [{tag}] d={c['delay']} {c['tgt']:6s} basis={c['basis']:4s} clock={c['clock']} "
              f"osr={c['os_']} {c['interaction']:12s} phase={phase:.2f}: max|diff|={d:.1e} "
              f"MREC {'identical' if same_rec else 'DIFFER'}")
    print(f"kernel: {ncase} cases, {nfail} failures, worst {worst:.1e}")
    return nfail == 0


def run_transform_test():
    fs, N, H = 8000, 1024, 256
    t = np.arange(int(1.2 * fs)) / fs
    x = 0.5 * np.sin(2 * np.pi * 130.81 * t) * (t < 0.6) + 0.3 * np.sin(2 * np.pi * 196 * t) * (t > 0.7)
    x[:int(0.1 * fs)] = 0
    import quantum_audio_master as qam
    from scipy.signal.windows import hann
    mdB, ph = qam.STFT.stftAnal(x, hann(N), N, H)
    mag = 10 ** (mdB / 20.0)
    nb = mag.shape[1]; fb = np.arange(nb) * fs / N
    bins = np.arange(10, 30)
    kw = dict(n_traj=2, base_seed=12345, gamma_x=0.75, gamma_z=0.3, interaction='beamsplitter',
              measurement_basis='x', amp_pow=1.0, omega_s_res=23.0, fs=fs, verbose=False,
              carrier_clock=True, stft_hop=H, encode_target='system')
    t0 = time.perf_counter()
    om_r, oph_r, sc_r, dg_r = dt.dependent_colour_transform_per_bin(
        mag, ph, fb, bins, 5, np.pi, amp_mode='power', fast=False, **kw)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    om_g, oph_g, sc_g, dg_g = eng.transform(mag, ph, fb, bins, 5, np.pi, **kw)
    t_got = time.perf_counter() - t0
    d = max(max(_cmp(om_r[v], om_g[v]) for v in eng._VERSIONS),
            max(_cmp(oph_r[v], oph_g[v]) for v in eng._VERSIONS), _cmp(sc_r, sc_g))
    ok = d <= 1e-10
    print(f"transform (d=5, 20 bins x {mag.shape[0]} frames, 2 traj, system/x/clock/osr): "
          f"max|diff|={d:.1e} -> {'ok' if ok else 'FAIL'};  qutip {t_ref:.1f}s  numpy {t_got:.1f}s")
    return ok


def benchmark():
    steps = 200
    amps = rng0.rand(steps); phs = rng0.rand(steps) * 6.28
    print('per-tick cost (system encoding, basis x, clock on):')
    for delay in (1, 2, 3, 4, 5):
        t0 = time.perf_counter()
        eng.kernel(amps, phs, 0.75, 0.3, 0.1, 0, 0, 0, delay, np.pi, 1.0, 'x', 1,
                   theta_clock=0.3, encode_target='system')
        dt_np = (time.perf_counter() - t0) / steps
        t0 = time.perf_counter()
        dt._dependent_colour_kernel(amps, phs, 0.75, 0.3, 0.1, 0, 0, 0, delay, np.pi, 1.0, 'x',
                                    'Pur.Deph', 1, theta_clock=0.3, encode_target='system')
        dt_qt = (time.perf_counter() - t0) / steps
        job = 513 * 313          # 10 s @ 8 kHz, H=256, all bins
        print(f"  d={delay}: numpy {dt_np*1e3:.3f} ms/tick  qutip {dt_qt*1e3:.3f} ms/tick  "
              f"-> 10 s input, 513 bins: numpy {dt_np*job:.0f} s, qutip {dt_qt*job:.0f} s")


if __name__ == '__main__':
    ok1 = run_kernel_tests()
    ok2 = run_transform_test()
    benchmark()
    print('\nALL PASS' if (ok1 and ok2) else '\nFAILURES PRESENT')
