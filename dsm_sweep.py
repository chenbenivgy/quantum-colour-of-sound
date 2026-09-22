"""
Coupling sweep for the Discontinuous Sound Modulator, encode_target='system'.

For every delay 1..5 and a (gx, gz) grid, on TWO inputs (synthetic bass notes
and a 4 s vibrato bass-line excerpt of audio/nightfall.wav), computes
  (A) the RECONSTRUCTED T17 metrics                         [C, tau, M, P]
      (the cell that produced T17_parameter_map.json is lost; these follow the
      pink_rival definitions of Dependent_trajectory_summary.ipynb, the only
      surviving code computing quantities with those names)
  (B) the NEW metrics proposed here                          [E, G, tau_int, R, D, T, Gm]
and renders (all 513 bins, no declick) the per-delay optimum of each rule plus
the summary-notebook reference point (gx 0.75, gz 0.3).

Operating point otherwise identical to Dependent_trajectory_summary.ipynb §2:
fs 8000, N 1024, H 256, basis x, OSR 23, carrier clock on, amp_pow 1,
phase pi, n_traj 1, seed 12345, beamsplitter.

Run:  conda activate basic_pitch_env && python dsm_sweep.py [nproc]
Outputs -> output/dsm_sweep/ (sweep_results.npz, selection.json, *.wav)
"""
import os, sys, time, json
import numpy as np
from multiprocessing import Pool
from scipy.linalg import expm
from scipy.signal import decimate
from scipy.signal.windows import hann
import soundfile
import quantum_audio_master as qam
import dsm_engine as eng

OUT_DIR = 'output/dsm_sweep'
FS, N, H = 8000, 1024, 256
BASIS, OSR, CLOCK, AMP_POW, PHASE, SEED = 'x', 23.0, True, 1.0, np.pi, 12345
ENC = 'system'
DELAYS = [1, 2, 3, 4, 5]
GXS = [0.15, 0.25, 0.35, 0.45, 0.555, 0.65, 0.75, 0.9, 1.05]
GZS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.9, 1.2, 1.5]
REF = (0.75, 0.3)
NOTE_FREQS = [130.81, 164.81, 196.0]
NOTE_DUR = [0.3, 0.5, 0.7]
SIL_DUR = 0.5
VIB_PATH, VIB_START, VIB_TRIM = 'audio/nightfall.wav', 505.0, 4.0
ENERGETIC_DB = -80.0     # bins whose peak input magnitude is above this (re global max) are simulated
INPUTS = ['bass', 'vib']


# ---------------------------------------------------------------- inputs ----
def make_bass_notes(fs=FS):
    seg = [np.zeros(int(0.1 * fs))]
    for i, f in enumerate(NOTE_FREQS):
        t = np.arange(int(NOTE_DUR[i] * fs)) / fs
        env = 0.5 * (1 - np.cos(2 * np.pi * t / NOTE_DUR[i]))
        seg.append(0.5 * np.sin(2 * np.pi * f * t) * env)
        if i < len(NOTE_FREQS) - 1:
            seg.append(np.zeros(int(SIL_DUR * fs)))
    seg.append(np.zeros(int(0.4 * fs)))
    return np.concatenate(seg)


def load_decimated(path, start_s, trim_s, target_fs=FS):
    """Mono, peak-normalise, trim, 0.2 s edge fades, FIR zero-phase decimate (summary §3)."""
    x, sr = soundfile.read(path, dtype='float64')
    if x.ndim > 1: x = x.mean(axis=1)
    x = x[int(start_s * sr):int((start_s + trim_s) * sr)]
    x = x / max(np.max(np.abs(x)), 1e-9)
    nf = int(0.2 * sr)
    x[:nf] *= np.linspace(0, 1, nf); x[-nf:] *= np.linspace(1, 0, nf)
    D = max(1, int(round(sr / target_fs)))
    if D > 1: x = decimate(x, D, ftype='fir', zero_phase=True)
    return x, sr // D


def stft_pair(x):
    mdB, ph = qam.STFT.stftAnal(x, hann(N), N, H)
    return 10 ** (mdB / 20.0), ph


def istft(mag, phase):
    return qam.STFT.stftSynth(20 * np.log10(np.maximum(mag, 1e-10)), phase, N, H)


def note_windows(nf):
    """(on frames, off frames, bin index) per bass note."""
    t0 = 0.1; out = []
    for i, f in enumerate(NOTE_FREQS):
        a, b = t0, t0 + NOTE_DUR[i]
        s_end = b + (SIL_DUR if i < len(NOTE_FREQS) - 1 else 0.4)
        out.append(dict(f=f, k=int(round(f / (FS / N))), on=(int(a * FS / H), int(b * FS / H)),
                        off=(int(b * FS / H) + 1, min(nf, int(s_end * FS / H)))))
        t0 = s_end
    return out


def prepare_inputs():
    x_b = make_bass_notes()
    x_v, fs_v = load_decimated(VIB_PATH, VIB_START, VIB_TRIM)
    assert fs_v == FS, fs_v
    inputs = {}
    for name, x in (('bass', x_b), ('vib', x_v)):
        im, ip = stft_pair(x)
        peak = im.max(0)
        bins = np.where(20 * np.log10(peak / peak.max() + 1e-30) > ENERGETIC_DB)[0]
        inputs[name] = dict(x=x, in_mag=im, in_phase=ip, bins=bins,
                            freq_bins=np.arange(im.shape[1]) * FS / N)
    return inputs


# ------------------------------------------------- exact impulse model ----
def impulse_model(gx, gz, delay, encode_target=ENC, phase=PHASE, DeltaT=1.0, steps=60):
    """One-excitation propagation of an isolated impulse (summary notebook §16):
    decoded population per tick. Exact for the unconditional map."""
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


def impulse_metrics(gx, gz, delay):
    pop = impulse_model(gx, gz, delay)
    tot = pop.sum()
    return dict(E=float(pop[delay:].sum() / max(tot, 1e-12)), G=float(tot), direct=float(pop[0]))


# -------------------------------------------------------- run one point ----
_G = {}


def _init(inputs):
    _G.update(inputs)


def run_point(args):
    name, delay, gx, gz, all_bins = args
    inp = _G[name]
    bins = np.arange(inp['in_mag'].shape[1]) if all_bins else inp['bins']
    t0 = time.perf_counter()
    om, oph, sc, _ = eng.transform(
        inp['in_mag'], inp['in_phase'], inp['freq_bins'], bins, delay, PHASE, n_traj=1,
        base_seed=SEED, gamma_x=gx, gamma_z=gz, measurement_basis=BASIS, amp_pow=AMP_POW,
        omega_s_res=OSR, fs=FS, carrier_clock=CLOCK, stft_hop=H, encode_target=ENC,
        versions=['uncond'])
    return dict(name=name, delay=delay, gx=gx, gz=gz, out_mag=om['uncond'] * sc[None, :],
                out_phase=oph['uncond'], secs=time.perf_counter() - t0)


# --------------------------------------------------------------- metrics ----
def pink_field(nf, nb, p, seed=0):
    rng = np.random.default_rng(seed); w = rng.standard_normal((nf, nb))
    f = np.fft.rfftfreq(nf).copy(); f[0] = f[1] if len(f) > 1 else 1.0
    x = np.fft.irfft(np.fft.rfft(w, axis=0) / (f[:, None] ** (p / 2)), n=nf, axis=0)
    return (x - x.mean(0)) / (x.std(0) + 1e-9)


def _cacf(f, W, L):
    out = np.empty(L); out[0] = 1.0
    for l in range(1, L):
        wl = W[l:] * W[:-l]
        out[l] = float((wl * np.cos(f[l:] - f[:-l])).sum() / (wl.sum() + 1e-30))
    return out


def old_metrics(im, ip, om, op, seed=0):
    """Reconstructed T17 metrics (pink_rival definitions of the summary notebook):
      C   : lag-1 energy-weighted circular ACF of the phase deviation dev = angle(e^{i(phi_out - phi_in)})
      tau : first lag (frames) where that ACF drops below 1/e
      M   : AM depth = std(out/in - 1) over energetic cells, clipped to [0.05, 1]
      P   : 1/f^P exponent whose pink field best matches the ACF curve"""
    nf, nb = im.shape
    dev_q = np.angle(np.exp(1j * (op - ip))); W = im ** 2
    depth = float(np.sqrt((W * dev_q ** 2).sum() / (W.sum() + 1e-30)))
    L = min(25, nf - 1)
    target = _cacf(dev_q, W, L)
    below = np.where(target < 1 / np.e)[0]
    tau = int(below[0]) if len(below) else L
    grid = np.linspace(0.3, 6.0, 20)
    cost = [float(np.sum((_cacf(depth * pink_field(nf, nb, p, seed + 3), W, L) - target) ** 2)) for p in grid]
    P = float(grid[int(np.argmin(cost))])
    en = im > 0.1 * im.max()
    M = float(np.clip(np.std((om / np.maximum(im, 1e-12))[en] - 1.0), 0.05, 1.0)) if en.any() else 0.3
    return dict(C=float(target[1]), tau=tau, M=M, P=P, depth=depth)


def new_metrics(im, ip, om, op, bins, notes=None):
    """Proposed metrics:
      tau_int : integrated memory of the phase deviation, 1 + 2 sum_{l>=1} ACF(l) (frames; continuous,
                unlike the 1/e lag which is 1 almost everywhere for a single x-basis trajectory)
      R       : envelope roughness = std(frame diff of decoded magnitude)/mean over energetic cells
      Gm      : level = decoded energy / input energy
      D       : sideband distortion (bass notes only) = decoded energy outside +-2 bins of each
                note's fundamental during the note / in-band energy  (Rocchesso's bass catch)
      T       : tail (bass notes only) = decoded energy in the silence after each note / during it"""
    nf = im.shape[0]
    dev_q = np.angle(np.exp(1j * (op - ip))); W = im ** 2
    L = min(25, nf - 1); acf = _cacf(dev_q, W, L)
    tau_int = float(1.0 + 2.0 * np.sum(acf[1:]))
    en = im > 0.1 * im.max()
    rough = []
    for k in range(im.shape[1]):
        fr = np.where(en[:, k])[0]
        if len(fr) < 4: continue
        e = om[fr, k]
        if e.mean() > 1e-9:
            rough.append(np.std(np.diff(e)) / e.mean())
    R = float(np.mean(rough)) if rough else 0.0
    Gm = float((om ** 2).sum() / max((im ** 2).sum(), 1e-30))
    D = T = float('nan')
    if notes is not None:
        inb = outb = tail = dur = 0.0
        for nt in notes:
            a, b = nt['on']; c, d = nt['off']
            near = np.abs(bins - nt['k']) <= 2
            blk = om[a:b] ** 2
            inb += blk[:, near].sum(); outb += blk[:, ~near].sum()
            dur += blk.sum(); tail += (om[c:d] ** 2).sum()
        D = float(outb / max(inb, 1e-30)); T = float(tail / max(dur, 1e-30))
    return dict(tau_int=tau_int, R=R, Gm=Gm, D=D, T=T)


KEYS = ['input', 'delay', 'gx', 'gz', 'secs', 'E', 'G', 'direct',
        'C', 'tau', 'M', 'P', 'depth', 'tau_int', 'R', 'Gm', 'D', 'T']


def select(arr, keys):
    """Per-delay optima.
    old : reconstructed-T17 rule on the bass notes,  score_old = M * tau  (modulation depth x memory)
    new : geometric mean over both inputs of  E * tau_int * sqrt(Gm) / (1 + D/0.05)
          (echo fraction x memory x level, penalised by sideband distortion beyond -13 dB;
          D is bass-only, so the vibrato factor has no penalty)."""
    ix = {k: i for i, k in enumerate(keys)}
    out = {}
    delays = np.unique(arr[:, ix['delay']])
    for d in delays:
        sub_b = arr[(arr[:, ix['delay']] == d) & (arr[:, ix['input']] == 0)]
        sub_v = arr[(arr[:, ix['delay']] == d) & (arr[:, ix['input']] == 1)]
        # align on (gx, gz)
        key_b = [(round(r[ix['gx']], 4), round(r[ix['gz']], 4)) for r in sub_b]
        key_v = {(round(r[ix['gx']], 4), round(r[ix['gz']], 4)): r for r in sub_v}
        s_old = sub_b[:, ix['M']] * sub_b[:, ix['tau']]
        s_b = sub_b[:, ix['E']] * sub_b[:, ix['tau_int']] * np.sqrt(np.maximum(sub_b[:, ix['Gm']], 0)) \
            / (1.0 + sub_b[:, ix['D']] / 0.05)
        s_v = np.array([key_v[k][ix['E']] * key_v[k][ix['tau_int']] * np.sqrt(max(key_v[k][ix['Gm']], 0))
                        for k in key_b])
        s_new = np.sqrt(np.maximum(s_b, 0) * np.maximum(s_v, 0))
        out[str(int(d))] = {}
        for tag, s in (('old', s_old), ('new', s_new), ('new_bass_only', s_b), ('new_vib_only', s_v)):
            j = int(np.argmax(s))
            row = sub_b[j]
            out[str(int(d))][tag] = dict(gx=float(row[ix['gx']]), gz=float(row[ix['gz']]), score=float(s[j]),
                                         **{k: float(row[ix[k]]) for k in
                                            ('E', 'G', 'C', 'tau', 'M', 'P', 'tau_int', 'R', 'Gm', 'D', 'T')})
    return out


# ----------------------------------------------------------------- main ----
def main(nproc=8):
    os.makedirs(OUT_DIR, exist_ok=True)
    inputs = prepare_inputs()
    for name, inp in inputs.items():
        nf, nb = inp['in_mag'].shape
        print(f'{name}: {len(inp["x"])/FS:.2f} s, {nf} frames x {nb} bins, '
              f'{len(inp["bins"])} energetic bins (> {ENERGETIC_DB} dB)', flush=True)
        soundfile.write(os.path.join(OUT_DIR, f'{name}_input.wav'), inp['x'].astype(np.float32), FS)
    notes = note_windows(inputs['bass']['in_mag'].shape[0])

    jobs = [(nm, d, gx, gz, False) for nm in INPUTS for d in DELAYS for gx in GXS for gz in GZS]
    t0 = time.time(); rows = []
    with Pool(nproc, initializer=_init, initargs=(inputs,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_point, jobs)):
            inp = inputs[r['name']]; b = inp['bins']
            im, ip = inp['in_mag'][:, b], inp['in_phase'][:, b]
            om, op = r['out_mag'][:, b], r['out_phase'][:, b]
            imp = impulse_metrics(r['gx'], r['gz'], r['delay'])
            old = old_metrics(im, ip, om, op, seed=1)
            new = new_metrics(im, ip, om, op, b, notes if r['name'] == 'bass' else None)
            rows.append(dict(input=INPUTS.index(r['name']), delay=r['delay'], gx=r['gx'], gz=r['gz'],
                             secs=r['secs'], **imp, **old, **new))
            if (i + 1) % 40 == 0 or i + 1 == len(jobs):
                print(f'  {i+1}/{len(jobs)} grid points done, {time.time()-t0:.0f} s elapsed', flush=True)
    arr = np.array([[row[k] for k in KEYS] for row in rows], float)
    np.savez(os.path.join(OUT_DIR, 'sweep_results.npz'), data=arr, keys=np.array(KEYS),
             GXS=np.array(GXS), GZS=np.array(GZS), DELAYS=np.array(DELAYS), INPUTS=np.array(INPUTS),
             bins_bass=inputs['bass']['bins'], bins_vib=inputs['vib']['bins'])
    print(f'grid done in {time.time()-t0:.0f} s -> {OUT_DIR}/sweep_results.npz', flush=True)

    sel = select(arr, KEYS)
    json.dump(sel, open(os.path.join(OUT_DIR, 'selection.json'), 'w'), indent=1)
    for d in DELAYS:
        s = sel[str(d)]
        print(f"d={d}: old -> gx {s['old']['gx']:.3f} gz {s['old']['gz']:.2f} | "
              f"new -> gx {s['new']['gx']:.3f} gz {s['new']['gz']:.2f}  "
              f"(bass-only {s['new_bass_only']['gx']:.3f}/{s['new_bass_only']['gz']:.2f}, "
              f"vib-only {s['new_vib_only']['gx']:.3f}/{s['new_vib_only']['gz']:.2f})", flush=True)

    # full-bin renders: bass at old/new/ref for every delay; vibrato at new/ref
    want = {}
    for d in DELAYS:
        for tag in ('old', 'new'):
            want.setdefault(('bass', d, sel[str(d)][tag]['gx'], sel[str(d)][tag]['gz']), []).append(tag)
        want.setdefault(('bass', d, REF[0], REF[1]), []).append('ref')
        want.setdefault(('vib', d, sel[str(d)]['new']['gx'], sel[str(d)]['new']['gz']), []).append('new')
        want.setdefault(('vib', d, REF[0], REF[1]), []).append('ref')
    t0 = time.time()
    with Pool(nproc, initializer=_init, initargs=(inputs,)) as pool:
        for r in pool.imap_unordered(run_point, [(nm, d, gx, gz, True) for (nm, d, gx, gz) in want]):
            y = istft(r['out_mag'], r['out_phase'])
            tags = want[(r['name'], r['delay'], r['gx'], r['gz'])]
            name = f"{r['name']}_d{r['delay']}_gx{r['gx']:.3f}_gz{r['gz']:.2f}_{'+'.join(tags)}.wav"
            soundfile.write(os.path.join(OUT_DIR, name), y.astype(np.float32), FS)
            print(f'  rendered {name}  ({r["secs"]:.0f} s, peak {np.max(np.abs(y)):.3f})', flush=True)
    print(f'renders done in {time.time()-t0:.0f} s', flush=True)


if __name__ == '__main__':
    main(nproc=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
