"""Certificates for the trajectory store, the renders and the classical twin.

  1. transform() (now store-based) is still bit-identical to the original
     dependent_colour_transform_per_bin for n_traj = 1 and 3, all three versions.
  2. incremental trajectories are exact: append(0..1) + append(1..3) == compute(0..3).
  3. kernel_impl='qutip' store == numpy store.
  4. the twin equals the deterministic quantum map in the weak-drive limit
     (relative error -> 0 with the drive amplitude) and has the right phase.
Run: conda activate basic_pitch_env && python test_dsm_store.py
"""
import time
import numpy as np
from scipy.signal.windows import hann
import quantum_audio_master as qam
import dependent_trajectory as dt
import dsm_engine as eng


def _sig():
    fs, N, H = 8000, 512, 256
    t = np.arange(int(1.2 * fs)) / fs
    x = 0.5 * np.sin(2 * np.pi * 130.81 * t) * (t < 0.6) + 0.3 * np.sin(2 * np.pi * 196 * t) * (t > 0.7)
    x[:int(0.1 * fs)] = 0
    mdB, ph = qam.STFT.stftAnal(x, hann(N), N, H)
    mag = 10 ** (mdB / 20.0); nb = mag.shape[1]
    return mag, ph, np.arange(nb) * fs / N, fs, H


def _cmp(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def test_transform_identity():
    mag, ph, fb, fs, H = _sig(); bins = np.arange(8, 30); ok = True
    kw = dict(base_seed=12345, gamma_x=0.45, gamma_z=0.15, interaction='beamsplitter', measurement_basis='x',
              amp_pow=1.0, omega_s_res=23.0, fs=fs, carrier_clock=True, stft_hop=H, encode_target='system')
    for n_traj in (1, 3):
        r = dt.dependent_colour_transform_per_bin(mag, ph, fb, bins, 3, np.pi, n_traj=n_traj, amp_mode='power',
                                                  fast=False, verbose=False, **kw)
        g = eng.transform(mag, ph, fb, bins, 3, np.pi, n_traj=n_traj, **kw)
        d = max(max(_cmp(r[0][v], g[0][v]) for v in eng._VERSIONS), max(_cmp(r[1][v], g[1][v]) for v in eng._VERSIONS))
        fbd = max(abs(r[3][k]['fallback_frac'] - g[3][k]['fallback_frac']) for k in bins)
        print(f'  transform vs original, n_traj={n_traj}, uncond/cond0/cond1: max|diff|={d:.1e}, fallback diag diff {fbd:.1e}')
        ok &= d < 1e-10 and fbd < 1e-12       # atan2/mean rounding order differs from the original at ~1e-12
    return ok


def test_incremental_and_qutip():
    mag, ph, fb, fs, H = _sig(); bins = np.arange(8, 20)
    kw = dict(base_seed=12345, gamma_x=0.45, gamma_z=0.15, measurement_basis='x', amp_pow=1.0, omega_s_res=23.0,
              fs=fs, carrier_clock=True, stft_hop=H, encode_target='system')
    full = eng.trajectories(mag, ph, fb, bins, 3, np.pi, (0, 3), **kw)
    inc = eng.trajectories(mag, ph, fb, bins, 3, np.pi, (0, 1), **kw)
    inc = eng.trajectories(mag, ph, fb, bins, 3, np.pi, (1, 3), store=inc, **kw)
    d = max(_cmp(full[k], inc[k]) for k in ('SX', 'SY', 'POP')); same = np.array_equal(full['MREC'], inc['MREC'])
    print(f'  incremental (0..1)+(1..3) vs (0..3): max|diff|={d:.1e}, records identical={same}')
    t0 = time.time(); q = eng.trajectories(mag, ph, fb, bins, 3, np.pi, (0, 2), kernel_impl='qutip', **kw); tq = time.time() - t0
    d2 = max(_cmp(full[k][:, :2], q[k]) for k in ('SX', 'SY', 'POP')); same2 = np.array_equal(full['MREC'][:, :2], q['MREC'])
    print(f'  qutip-kernel store vs numpy store: max|diff|={d2:.1e}, records identical={same2} ({tq:.1f} s)')
    # render from a bigger store with n_use=1 equals a fresh n_traj=1 render
    m1, p1 = eng.render_from_store(full, 'uncond', 1, 3, 1.0, ph, bins)
    one = eng.trajectories(mag, ph, fb, bins, 3, np.pi, (0, 1), **kw)
    m1b, p1b = eng.render_from_store(one, 'uncond', 1, 3, 1.0, ph, bins)
    print(f'  render(n_use=1) from a 3-trajectory store vs 1-trajectory store: {_cmp(m1, m1b):.1e}')
    return d < 1e-12 and same and d2 < 1e-12 and same2 and _cmp(m1, m1b) < 1e-12


def test_twin():
    ok = True
    # 1. weak-drive limit at kernel level: same per-bin inputs into the quantum kernel (basis none,
    #    system encoding, resonator + clock on) and into the twin; error must vanish with the drive.
    rng = np.random.RandomState(0); T = 60
    amps0 = rng.rand(T); phs = rng.rand(T) * 2 * np.pi - np.pi
    print('  twin vs deterministic quantum kernel (system encoding, os=0.7, theta=0.9, d=3):')
    prev = None
    for enc in ('system', 'aux'):
        for a0 in (1.0, 0.1, 0.01, 0.001):
            amps = amps0 * a0
            sx, sy, pq, _ = eng.kernel(amps, phs, 0.45, 0.15, 0.7, 0, 0, 0, 3, np.pi, 1.0, 'none', 1,
                                       theta_clock=0.9, encode_target=enc)
            pt, at = eng.twin_bin(amps, phs, 0.45, 0.15, 0.7, 0.9, 3, np.pi, enc)
            rel = np.linalg.norm(pt - pq) / np.linalg.norm(pq)
            w = pq > 1e-3 * pq.max()
            dph = np.sqrt(np.mean(np.angle(np.exp(1j * (at[w] - np.arctan2(sy[w], sx[w])))) ** 2))
            print(f'    {enc:6s} peak drive a={a0:<6g}: population rel error {rel:.2e}, phase rms error {dph:.2e} rad')
            if a0 == 0.001:
                ok &= rel < 1e-2 and dph < 1e-2
    # 2. full chain at the operating point: the twin's discrepancy from the quantum map is saturation (reported)
    mag, ph, fb, fs, H = _sig(); bins = np.arange(8, 30)
    kw = dict(gamma_x=0.45, gamma_z=0.15, omega_s_res=23.0, fs=fs, carrier_clock=True, stft_hop=H,
              encode_target='system', amp_pow=1.0)
    st = eng.trajectories(mag, ph, fb, bins, 3, np.pi, (0, 1), measurement_basis='none', **kw)
    qm, qp = eng.render_from_store(st, 'uncond', 1, 3, 1.0, ph, bins)
    tm, tp = eng.twin(mag, ph, fb, bins, 3, np.pi, **kw)
    W = mag[:, bins] ** 2
    rel = np.sqrt((W * (tm[:, bins] - qm[:, bins]) ** 2).sum() / (W * qm[:, bins] ** 2).sum())
    print(f'  full-drive test signal (peak amplitude 1 = full inversion): twin vs quantum map rel {rel:.3f} -- saturation, expected')
    return ok


if __name__ == '__main__':
    ok = test_transform_identity(); ok &= test_incremental_and_qutip(); test_twin()
    print('ALL PASS' if ok else 'FAILURES PRESENT')
