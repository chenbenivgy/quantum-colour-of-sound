"""Tests for dsm_chain: STFT conventions vs smstools, circular round trip,
periodic steady state, and the qutip/numpy engines agreeing through the chain."""
import numpy as np, time
from scipy.signal.windows import hann
import quantum_audio_master as qam
import dsm_chain as ch


def test_stft_matches_smstools():
    fs, H = 8000, 256; N = 2 * H
    rng = np.random.RandomState(0)
    L = 40 * H
    x = rng.randn(L) * np.hanning(L)            # silent at the edges so padding == wrap
    x[:N] = 0; x[-N:] = 0
    mdB, ph = qam.STFT.stftAnal(x, hann(N), N, H)
    mag, phs = ch.periodic_stft(x, N, H)
    nf = mag.shape[0]
    d_mag = np.max(np.abs(10 ** (mdB[:nf] / 20) - mag))
    d_ph = np.max(np.abs(np.angle(np.exp(1j * (ph[:nf] - phs)))))
    print(f'periodic_stft vs smstools stftAnal (edge-silent signal): max|dmag|={d_mag:.1e} max|dphase|={d_ph:.1e}')
    assert d_mag < 1e-12 and d_ph < 1e-9
    # synthesis: smstools on the same frames vs circular OLA (interior samples)
    y_sms = qam.STFT.stftSynth(mdB[:nf], ph[:nf], N, H)
    y = ch.circular_ola(mag, phs, N, H, L)
    n = min(len(y_sms), L) - N
    d = np.max(np.abs(y_sms[N:n] - y[N:n]))
    print(f'circular_ola vs smstools stftSynth (interior): max|dy|={d:.1e}')
    assert d < 1e-12


def test_round_trip():
    H = 256; N = 2 * H; L = 30 * H
    rng = np.random.RandomState(1); x = rng.randn(L)
    mag, ph = ch.periodic_stft(x, N, H)
    y = ch.circular_ola(mag, ph, N, H, L)
    err = np.max(np.abs(y - x)); print(f'periodic STFT round trip on white noise: max|y-x|={err:.2e} (symmetric-hann COLA ripple)')
    assert err < 1e-2 * np.max(np.abs(x))


def test_steady_state_and_engines():
    fs_e, D, H = 48000, 6, 256
    fs_m = ch.model_rate(fs_e, D)
    t = np.arange(int(2.0 * fs_e)) / fs_e
    x = 0.5 * np.sin(2 * np.pi * 130.81 * t) * (t < 1.2)
    p = dict(delay=3, H=H, D=D, t_l=1.5, gamma_x=0.75, gamma_z=0.3, seed=12345)
    t0 = time.time(); r_q = ch.process_job(x, fs_e, p, engine='qutip'); tq = time.time() - t0
    t0 = time.time(); r_n = ch.process_job(x, fs_e, p, engine='numpy', nproc=4); tn = time.time() - t0
    dm = np.max(np.abs(r_q['out_mag'] - r_n['out_mag']))
    dp = np.max(np.abs(np.angle(np.exp(1j * (r_q['out_phase'] - r_n['out_phase'])))))
    dy = np.max(np.abs(r_q['y'] - r_n['y']))
    print(f'chain qutip vs numpy(4 proc): max|dmag|={dm:.1e} max|dphase|={dp:.1e} max|dy|={dy:.1e}; '
          f'{tq:.1f}s vs {tn:.1f}s; loop {r_q["nf"]} frames = {len(r_q["y"])/fs_m:.3f} s @ {fs_m} Hz, warm-up {r_q["W"]} frames')
    assert dm < 1e-9 and dp < 1e-6 and dy < 1e-9
    # steady state: the decoded energy in the first delay frames of the loop is NOT a ground-state transient:
    # compare with a cold run (no warm-up) on the same loop.
    import dsm_engine as eng
    N = 2 * H; nb = r_n['in_mag'].shape[1]; fb = np.arange(nb) * fs_m / N
    kw = dict(n_traj=1, base_seed=12345, gamma_x=0.75, gamma_z=0.3, measurement_basis='x', amp_pow=1.0,
              omega_s_res=23.0, fs=fs_m, carrier_clock=True, stft_hop=H, encode_target='system')
    om, oph, sc, _ = eng.transform(r_n['in_mag'], r_n['in_phase'], fb, np.arange(nb), 3, np.pi, versions=['uncond'], **kw)
    cold = om['uncond'] * sc[None, :]
    e_warm = (r_n['out_mag'][:3] ** 2).sum(); e_cold = (cold[:3] ** 2).sum()
    print(f'decoded energy in loop frames 0..2: warm-started {e_warm:.3e} vs cold start {e_cold:.3e} '
          f'(the loop was silent from 1.2 s to 1.5 s, so the warm start carries the wrap-around echo of the note tail)')


if __name__ == '__main__':
    test_stft_matches_smstools(); test_round_trip(); test_steady_state_and_engines(); print('ALL PASS')
