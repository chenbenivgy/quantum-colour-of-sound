"""
Verification harness: re-run the ORIGINAL Python code (qutip kernel, smstools
STFT conventions) on a buffer the module recorded, and compare with what the
module (engine B, C++) produced for the same parameters.

Usage:  conda activate basic_pitch_env && python dsm_verify.py <export_dir> [--engine qutip|numpy] [--nproc 8]

<export_dir> is written by the module's "Export for verification" menu item and holds
    params.json, input_engine_rate.wav, input_model_rate.wav, loop_model_rate.wav,
    loop_engine_rate.wav, decoded_mag.f64, decoded_phase.f64  [, mrec.i8]

Report: max differences per stage with pass thresholds
    decimated buffer      1e-6   (float32 wav quantisation)
    decoded magnitude     1e-6   (row-major nf x nb)  -- tighter numbers are printed
    decoded phase         1e-6   (modulo 2 pi)
    loop @ model rate     1e-5
    loop @ engine rate    1e-5
    measurement records   identical (if exported)
A projective-outcome flip (probability within ~1e-13 of the random draw) would show up
as a localised divergence from that frame on in one bin; the report lists such bins.
"""
import os, sys, json, argparse, time
import numpy as np
import soundfile
import dsm_chain as ch


def load(d):
    p = json.load(open(os.path.join(d, 'params.json')))
    x_e, fs_e = soundfile.read(os.path.join(d, 'input_engine_rate.wav'), dtype='float64')
    if x_e.ndim > 1: x_e = x_e.mean(axis=1)
    out = dict(params=p, x_e=x_e, fs_e=fs_e)
    for nm, key in (('input_model_rate.wav', 'x_m'), ('loop_model_rate.wav', 'y_m'), ('loop_engine_rate.wav', 'y_e')):
        f = os.path.join(d, nm)
        if os.path.exists(f):
            out[key] = soundfile.read(f, dtype='float64')[0]
    nb = None
    for nm, key in (('decoded_mag.f64', 'mag'), ('decoded_phase.f64', 'phase')):
        f = os.path.join(d, nm)
        if os.path.exists(f):
            out[key] = np.fromfile(f, dtype=np.float64)
    f = os.path.join(d, 'mrec.i8')
    if os.path.exists(f):
        out['mrec'] = np.fromfile(f, dtype=np.int8)
    for nm, key in (('in_mag.f64', 'in_mag'), ('in_phase.f64', 'in_phase')):
        f = os.path.join(d, nm)
        if os.path.exists(f):
            out[key] = np.fromfile(f, dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('export_dir')
    ap.add_argument('--engine', default='qutip', choices=['qutip', 'numpy'])
    ap.add_argument('--nproc', type=int, default=8)
    a = ap.parse_args()
    E = load(a.export_dir)
    p = E['params']
    print(f"params: {json.dumps({k: p[k] for k in sorted(p)})}")
    if p.get('render', 'uncond') == 'twin':
        print('render = classical twin: the reference is dsm_engine.twin (Python), not the quantum code')
    t0 = time.time()
    r = ch.process_job(E['x_e'], E['fs_e'], p, engine=a.engine, nproc=a.nproc)
    print(f'reference ({a.engine}) computed in {time.time()-t0:.1f} s: loop {r["nf"]} frames x {r["out_mag"].shape[1]} bins '
          f'@ {r["fs_m"]} Hz, warm-up {r["W"]} frames')
    ok = True

    def rep(name, d, thr):
        nonlocal ok
        flag = d <= thr; ok &= flag
        print(f'  {name:28s} max|diff| = {d:.2e}   ({"ok" if flag else "FAIL"}, threshold {thr:g})')

    if 'x_m' in E:
        n = min(len(E['x_m']), len(r['x_model']))
        rep('decimated buffer', float(np.max(np.abs(E['x_m'][:n] - r['x_model'][:n]))), 1e-6)
        if len(E['x_m']) != len(r['x_model']):
            print(f'    length differs: module {len(E["x_m"])} vs reference {len(r["x_model"])}'); ok = False
    nf, nb = r['out_mag'].shape
    # 1) the module's STFT of the loop vs the reference STFT, as complex numbers
    ref_mag, ref_phase = r['in_mag'], r['in_phase']
    same_stft = False
    if 'in_mag' in E and 'in_phase' in E and E['in_mag'].size == nf * nb:
        Mi = E['in_mag'].reshape(nf, nb); Pi = E['in_phase'].reshape(nf, nb)
        Xc = Mi * np.exp(1j * Pi); Xr = ref_mag * np.exp(1j * ref_phase)
        rep('input STFT (complex, rel.)', float(np.max(np.abs(Xc - Xr)) / np.max(np.abs(Xr))), 1e-9)
        # 2) re-run the reference transform on the MODULE's STFT: the system encoding applies Rz(phi)
        #    to the memory qubit even at zero amplitude, so the round-off phase of numerically silent
        #    bins leaks into the output at ~1e-8 -- feeding both engines the same frames isolates the kernel.
        W = r['W']; fed = ch.fed_frame_indices(nf, W); H = r['H']; N = 2 * H
        fb = np.arange(nb) * r['fs_m'] / N
        kw = dict(n_traj=int(p.get('n_traj', 1)), base_seed=int(p['seed']), gamma_x=float(p['gamma_x']), gamma_z=float(p['gamma_z']),
                  measurement_basis=p.get('measurement_basis', 'x'), amp_pow=float(p.get('amp_pow', 1.0)),
                  omega_s_res=float(p.get('omega_s_res', 23.0)), fs=r['fs_m'], carrier_clock=p.get('carrier_clock', True),
                  stft_hop=H, encode_target=p.get('encode_target', 'system'))
        render = p.get('render', 'uncond'); delay = int(p['delay']); phase = float(p.get('phase', np.pi))
        import dsm_engine as eng
        if render == 'pink':
            print('render = pink noise: a classical comparison whose noise realisation is engine-specific; the fitted depth/P/AM are'
                  ' certified by test_dsm_cpp.py, the loop itself is not expected to match sample-wise')
        if render == 'twin':
            tkw = {k: kw[k] for k in ('gamma_x', 'gamma_z', 'omega_s_res', 'fs', 'carrier_clock', 'stft_hop', 'encode_target', 'amp_pow')}
            om_s, oph_s = eng.twin(Mi[fed], Pi[fed], fb, np.arange(nb), delay, phase, **tkw)
        elif a.engine == 'qutip':      # the UNCHANGED transform on the module's frames
            import dependent_trajectory as dt
            om, oph, sc, _ = dt.dependent_colour_transform_per_bin(Mi[fed], Pi[fed], fb, np.arange(nb), delay, phase,
                                                                   amp_mode='power', fast=False, verbose=False, interaction=p.get('interaction', 'beamsplitter'), **kw)
            om_s, oph_s = om[render] * sc[None, :], oph[render]
        else:
            skw = {k: v for k, v in kw.items() if k not in ('n_traj', 'base_seed')}
            st = ch._store(Mi[fed], Pi[fed], fb, np.arange(nb), delay, phase, kw['n_traj'], kw['base_seed'],
                           dict(skw, interaction=p.get('interaction', 'beamsplitter')), a.nproc, 'numpy')
            om_s, oph_s = eng.render_from_store(st, render, kw['n_traj'], delay, kw['amp_pow'], Pi[fed], np.arange(nb))
        r['out_mag_same'] = om_s[W:W + nf]; r['out_phase_same'] = oph_s[W:W + nf]
        r['y_same'] = ch.circular_ola(r['out_mag_same'], r['out_phase_same'], N, H, nf * H)
        same_stft = True
    if 'mag' in E:
        if E['mag'].size == nf * nb:
            M = E['mag'].reshape(nf, nb)
            Yc = M * np.exp(1j * E['phase'].reshape(nf, nb)) if 'phase' in E else None
            Yr = r['out_mag'] * np.exp(1j * r['out_phase'])
            if Yc is not None:
                rep('decoded STFT e2e (rel.)', float(np.max(np.abs(Yc - Yr)) / np.max(np.abs(Yr))), 1e-6)
            if same_stft:
                rep('decoded magnitude (same STFT)', float(np.max(np.abs(M - r['out_mag_same']))), 1e-9)
                if Yc is not None:
                    Ys = r['out_mag_same'] * np.exp(1j * r['out_phase_same'])
                    rep('decoded STFT (same STFT, rel.)', float(np.max(np.abs(Yc - Ys)) / np.max(np.abs(Ys))), 1e-9)
                dm = np.abs(M - r['out_mag_same']); bad = np.where(dm.max(0) > 1e-9)[0]
                for k in bad[:10]:
                    print(f'    bin {k}: first frame with |diff|>1e-9 is {int(np.argmax(dm[:, k] > 1e-9))}')
        else:
            print(f'  decoded magnitude: size mismatch {E["mag"].size} vs {nf*nb}'); ok = False
    if 'y_m' in E:
        n = min(len(E['y_m']), len(r['y']))
        rep('loop @ model rate (e2e)', float(np.max(np.abs(E['y_m'][:n] - r['y'][:n]))), 1e-5)
        if same_stft:
            rep('loop @ model rate (same STFT)', float(np.max(np.abs(E['y_m'][:n] - r['y_same'][:n]))), 1e-6)
    if 'y_e' in E:
        n = min(len(E['y_e']), len(r['y_engine']))
        rep('loop @ engine rate', float(np.max(np.abs(E['y_e'][:n] - r['y_engine'][:n]))), 1e-5)
    if 'mrec' in E:
        print('  measurement records exported: compare with dsm_engine per bin (not done by the qutip path)')
    print('\nVERIFIED: module output == original Python code on this recording' if ok else '\nDIVERGENCE FOUND -- see above')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
