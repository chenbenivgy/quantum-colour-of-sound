"""
Discontinuous Sound Modulator -- the signal chain around the quantum channel,
shared by engine A (Python server), the verification harness and (as the
specification) engine B (C++).

    recorded buffer @ engine rate
      -> zero-phase FIR decimation by D          (scipy.signal.decimate, ftype='fir')
      -> loop buffer: L = nf * H samples @ model rate (whole STFT frames)
      -> PERIODIC STFT of the loop, fed to the channel after a warm-up prefix
         of the loop's last W frames (the delay line and S then hold the end of
         the loop when frame 0 arrives: the periodic steady state of the channel,
         instead of a ground-state transient at every wrap)
      -> dependent transform (original qutip code, or the certified NumPy engine)
      -> frames [W, W+nf) of the decoded matrices (B16 alignment)
      -> CIRCULAR overlap-add on L samples: the loop, seamless at the wrap
      -> polyphase upsampling by D for playback (scipy.signal.resample_poly)

STFT conventions are those of smstools (dftAnal/dftSynth): window normalised
by its sum, zero-phase buffer, |X| floored at machine eps, synthesis frames
scaled by H. The module's TS1 (varispeed) is a fractional linear-interpolation
read of the engine-rate loop, i.e. quantum_audio_master.time_stretch.
"""
import json
import numpy as np
from scipy.signal import decimate, resample_poly
from scipy.signal.windows import hann

EPS = np.finfo(float).eps
RENDERS = ['uncond', 'cond0', 'cond1', 'twin', 'pink']   # module render selector 0..4 (undep, dep0, dep1, classical FB, pink noise)
DEFAULT_PARAMS = dict(
    delay=5, H=256, D=6, t_l=4.0, gamma_x=0.75, gamma_z=0.3, seed=12345,
    phase=np.pi, measurement_basis='x', amp_pow=1.0, omega_s_res=23.0,
    carrier_clock=True, encode_target='system', interaction='beamsplitter', n_traj=1,
    render='uncond')


# ------------------------------------------------------------- geometry ----
def model_rate(fs_engine, D):
    fs = fs_engine / D
    return int(round(fs)) if abs(fs - round(fs)) < 1e-9 else fs


def loop_frames(t_l, fs_m, H):
    """Number of STFT frames in the loop: t_l quantised to whole hops (>= 2)."""
    return max(2, int(np.floor(t_l * fs_m / H + 0.5)))     # half-up, as the C++ engine


def impulse_model(gx, gz, delay, encode_target='system', phase=np.pi, DeltaT=1.0, steps=60):
    """Exact one-excitation propagation of an isolated impulse (summary notebook §16):
    decoded population per tick, the unconditional impulse response of the channel."""
    from scipy.linalg import expm
    n = delay + 2; S, D0, F = 0, 1, delay + 1
    fb = -np.exp(-1j * phase); J = 2 * gx
    Hm = np.zeros((n, n), complex)
    Hm[S, F] = Hm[F, S] = J
    Hm[S, D0] = fb * J; Hm[D0, S] = np.conj(fb) * J
    E = np.full(n, 2 * gz); E[S] = -2 * gz; E[F] = 0.0; E[D0] = 0.0
    U1 = expm(-1j * DeltaT * (Hm + np.diag(E)))
    c = np.zeros(n, complex); pop = np.zeros(steps)
    for t in range(steps):
        if t == 0:
            c[F if encode_target == 'aux' else S] = 1.0
        c = U1 @ c
        pop[t] = abs(c[D0]) ** 2
        c = np.concatenate([c[:1], c[2:], [0.0]])
    return pop


def warmup_frames(delay, gx=0.75, gz=0.3, eps=1e-3, cap=400, encode_target='system'):
    """Frames of the loop's tail fed before frame 0: the smallest W after which less
    than `eps` of an impulse's decoded energy is still to come (exact impulse model,
    no detuning = the longest memory), at least 4d and at least 12. The caller caps
    W at the loop length nf (one full extra period)."""
    pop = impulse_model(gx, gz, delay, encode_target=encode_target, steps=cap + 1)
    tail = np.cumsum(pop[::-1])[::-1] / max(pop.sum(), 1e-300)
    below = np.where(tail < eps)[0]
    W = int(below[0]) if len(below) else cap
    return max(W, 4 * int(delay), 12)


def decimate_buffer(x, D):
    return np.asarray(x, float) if D == 1 else decimate(np.asarray(x, float), D, ftype='fir', zero_phase=True)


def upsample_loop(y, D):
    return np.asarray(y, float) if D == 1 else resample_poly(np.asarray(y, float), D, 1)


# ------------------------------------------------------- periodic STFT ----
def _window(N):
    w = hann(N)                      # symmetric, as the notebooks
    return w / w.sum()


def periodic_stft(x, N, H):
    """Frames m = 0..L/H-1 of the periodic extension of x, each centred at mH
    (smstools dftAnal: zero-phase buffer, |X| >= eps). Returns mag, phase."""
    x = np.asarray(x, float); L = len(x)
    assert L % H == 0 and N == 2 * H
    nf = L // H; hM1 = (N + 1) // 2; hM2 = N // 2
    w = _window(N)
    idx = (np.arange(-hM2, hM1)[None, :] + (np.arange(nf) * H)[:, None]) % L
    xw = x[idx] * w[None, :]
    fftbuf = np.concatenate([xw[:, hM2:], xw[:, :hM2]], axis=1)   # zero-phase
    X = np.fft.rfft(fftbuf, axis=1)
    mag = np.maximum(np.abs(X), EPS)
    return mag, np.angle(X)


def circular_ola(mag, phase, N, H, L):
    """Inverse of periodic_stft for the loop (smstools dftSynth/stftSynth with
    circular overlap-add): y of length L."""
    nf = mag.shape[0]; hM1 = (N + 1) // 2; hM2 = N // 2
    assert nf * H == L
    Y = mag * np.exp(1j * phase)
    yb = np.fft.irfft(Y, n=N, axis=1)
    yw = np.concatenate([yb[:, -hM2:], yb[:, :hM1]], axis=1)        # undo zero-phase
    y = np.zeros(L)
    idx = (np.arange(-hM1, hM2)[None, :] + (np.arange(nf) * H)[:, None]) % L
    np.add.at(y, idx, H * yw)
    return y


# --------------------------------------------------------- the channel ----
def fed_frame_indices(nf, W):
    """Loop frame index fed at each tick: the last W frames, then 0..nf-1."""
    pre = [(nf - W + i) % nf for i in range(W)]
    return np.array(pre + list(range(nf)))


def run_channel(x_loop, fs_m, params, engine='qutip', nproc=1, verbose=False):
    """x_loop: loop buffer at model rate, length nf*H. Returns dict with the loop
    render y (length L), decoded matrices (loop frames only), measurement
    records, and the fed input matrices."""
    p = dict(DEFAULT_PARAMS); p.update(params)
    H = int(p['H']); N = 2 * H; delay = int(p['delay'])
    L = len(x_loop); nf = L // H
    W = min(warmup_frames(delay, float(p['gamma_x']), float(p['gamma_z']), encode_target=p['encode_target']), nf)
    fed = fed_frame_indices(nf, W)
    mag0, ph0 = periodic_stft(x_loop, N, H)
    in_mag, in_phase = mag0[fed], ph0[fed]
    nb = in_mag.shape[1]
    freq_bins = np.arange(nb) * fs_m / N
    kw = dict(n_traj=int(p['n_traj']), base_seed=int(p['seed']), gamma_x=float(p['gamma_x']),
              gamma_z=float(p['gamma_z']), interaction=p['interaction'],
              measurement_basis=p['measurement_basis'], amp_pow=float(p['amp_pow']),
              omega_s_res=float(p['omega_s_res']), fs=fs_m, carrier_clock=p['carrier_clock'],
              stft_hop=H, encode_target=p['encode_target'], verbose=verbose)
    bins = np.arange(nb)
    render = p.get('render', 'uncond'); store = None
    import dsm_engine as eng
    if render == 'twin':
        tkw = {k: kw[k] for k in ('gamma_x', 'gamma_z', 'omega_s_res', 'fs', 'carrier_clock', 'stft_hop',
                                  'encode_target', 'amp_pow')}
        om_s, oph_s = eng.twin(in_mag, in_phase, freq_bins, bins, delay, float(p['phase']), **tkw)
        sc = np.full(nb, float(np.max(np.abs(in_mag))) or 1.0)
        out_mag, out_phase = om_s[W:W + nf], oph_s[W:W + nf]
    elif engine == 'qutip':          # the UNCHANGED transform (oracle)
        import dependent_trajectory as dt
        om, oph, sc, diag = dt.dependent_colour_transform_per_bin(
            in_mag, in_phase, freq_bins, bins, delay, float(p['phase']), amp_mode='power',
            fast=False, **kw)
        v = 'uncond' if render == 'pink' else render
        om_s, oph_s = om[v] * sc[None, :], oph[v]
        if render == 'pink':
            om_s, oph_s = eng.pink_rival(in_mag, in_phase, om_s, oph_s, seed=int(p['seed']))
        out_mag, out_phase = om_s[W:W + nf], oph_s[W:W + nf]
    elif engine in ('numpy', 'qutip-kernel'):      # store path (module semantics), numpy or qutip kernel
        skw = {k: kw[k] for k in kw if k not in ('n_traj', 'base_seed', 'verbose')}
        store = _store(in_mag, in_phase, freq_bins, bins, delay, float(p['phase']), int(p['n_traj']),
                       int(p['seed']), skw, nproc, 'qutip' if engine == 'qutip-kernel' else 'numpy')
        om_s, oph_s = eng.render_from_store(store, 'uncond' if render == 'pink' else render, int(p['n_traj']),
                                            delay, float(p['amp_pow']), in_phase, bins)
        if render == 'pink':
            om_s, oph_s = eng.pink_rival(in_mag, in_phase, om_s, oph_s, seed=int(p['seed']))
        sc = np.full(nb, store['scale'])
        out_mag, out_phase = om_s[W:W + nf], oph_s[W:W + nf]
    else:
        raise ValueError(engine)
    y = circular_ola(out_mag, out_phase, N, H, L)
    return dict(y=y, out_mag=out_mag, out_phase=out_phase, in_mag=mag0, in_phase=ph0,
                fed_mag=in_mag, fed_phase=in_phase, store=store,
                scale=sc, W=W, nf=nf, N=N, H=H, fs_m=fs_m, freq_bins=freq_bins, params=p)


def _store_worker(args):
    in_mag, in_phase, freq_bins, bins, delay, phase, n_traj, seed, kw, impl = args
    import dsm_engine as eng
    st = eng.trajectories(in_mag, in_phase, freq_bins, bins, delay, phase, (0, n_traj), seed,
                          kernel_impl=impl, **kw)
    return bins, st


def _store(in_mag, in_phase, freq_bins, bins, delay, phase, n_traj, seed, kw, nproc, impl):
    """Trajectory store over all bins, optionally split across processes (bins are independent)."""
    import dsm_engine as eng
    if nproc <= 1:
        return eng.trajectories(in_mag, in_phase, freq_bins, bins, delay, phase, (0, n_traj), seed,
                                kernel_impl=impl, **kw)
    from multiprocessing import get_context
    chunks = [c for c in np.array_split(bins, nproc * 4) if len(c)]
    full = None
    with get_context('fork').Pool(nproc) as pool:
        for b, st in pool.imap_unordered(_store_worker, [(in_mag, in_phase, freq_bins, c, delay, phase, n_traj, seed, kw, impl) for c in chunks]):
            if full is None:
                full = st; full['bins'] = np.asarray(bins)
            else:
                for key in ('SX', 'SY', 'POP', 'MREC'):
                    full[key][b] = st[key][b]
    return full


def _numpy_worker(args):
    in_mag, in_phase, freq_bins, bins, delay, phase, kw = args
    import dsm_engine as eng
    kw = dict(kw); kw.pop('verbose', None)
    om, oph, sc, diag = eng.transform(in_mag, in_phase, freq_bins, bins, delay, phase,
                                      versions=['uncond'], **kw)
    return bins, om['uncond'][:, bins], oph['uncond'][:, bins], sc


def _numpy_transform(in_mag, in_phase, freq_bins, bins, delay, phase, kw, nproc):
    if nproc <= 1:
        import dsm_engine as eng
        kw = dict(kw); kw.pop('verbose', None)
        return eng.transform(in_mag, in_phase, freq_bins, bins, delay, phase, versions=['uncond'], **kw)
    from multiprocessing import Pool
    chunks = np.array_split(bins, nproc * 4)
    om = {'uncond': np.zeros_like(in_mag)}; oph = {'uncond': np.zeros_like(in_phase)}; sc = None
    with Pool(nproc) as pool:
        for b, m, ph, s in pool.imap_unordered(
                _numpy_worker, [(in_mag, in_phase, freq_bins, c, delay, phase, kw) for c in chunks if len(c)]):
            om['uncond'][:, b] = m; oph['uncond'][:, b] = ph; sc = s
    return om, oph, sc, {}


# --------------------------------------------------------- full job ----
def process_job(x_engine, fs_engine, params, engine='qutip', nproc=1, verbose=False):
    """Module job: engine-rate recording -> loop render at model rate and at
    engine rate. Returns the run_channel dict plus x_model, y_engine, D, fs_e."""
    p = dict(DEFAULT_PARAMS); p.update(params)
    D = int(p['D']); H = int(p['H'])
    fs_m = model_rate(fs_engine, D)
    xm = decimate_buffer(x_engine, D)
    nf = loop_frames(float(p['t_l']), fs_m, H)
    L = nf * H
    if len(xm) < L:
        xm = np.concatenate([xm, np.zeros(L - len(xm))])
    x_loop = xm[:L]
    r = run_channel(x_loop, fs_m, p, engine=engine, nproc=nproc, verbose=verbose)
    r.update(x_model=x_loop, y_engine=upsample_loop(r['y'], D), D=D, fs_e=fs_engine)
    return r


def varispeed_read(loop_engine, ts1, n_out, start=0.0):
    """TS1 playback: fractional linear-interpolation read of the engine-rate loop
    at 1/ts1 samples per output sample, wrapping. == time_stretch(loop, ts1) up
    to the periodic wrap."""
    L = len(loop_engine)
    pos = (start + np.arange(n_out) / ts1) % L
    i0 = np.floor(pos).astype(int); fr = pos - i0; i1 = (i0 + 1) % L
    return (1 - fr) * loop_engine[i0] + fr * loop_engine[i1]


def params_to_json(p):
    q = dict(p); q['phase'] = float(q['phase']); return json.dumps(q, indent=1)
