"""Certificate for engine B (C++): build dsm/tools/engine_test, run it on synthetic
jobs, and compare every stage with the Python chain: decimation, STFT, trajectory
store (SX/SY/POP/MREC), the undep/dep0/dep1 renders, the classical twin, the
loops; incremental trajectory computation; the qutip oracle on the operating
point. Also times the C++ engine for the t_c projection.

Run: conda activate basic_pitch_env && python test_dsm_cpp.py
"""
import os, sys, struct, subprocess, time
import numpy as np
import dsm_chain as ch
import dsm_engine as eng

ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(ROOT, 'dsm', 'tools'); EXE = os.path.join(TOOLS, 'engine_test')
SCR = os.path.join(ROOT, 'output', 'dsm_cpp_test'); os.makedirs(SCR, exist_ok=True)
RENDER_ID = {'uncond': 0, 'cond0': 1, 'cond1': 2, 'twin': 3, 'pink': 4}


def build():
    flags = os.environ.get('DSM_CXXFLAGS', '-std=c++17 -stdlib=libc++ -O3 -fno-omit-frame-pointer -march=armv8-a+fp+simd '
                           '-fno-unsafe-math-optimizations -ffp-contract=off').split()
    cmd = ['clang++'] + flags + ['-o', EXE, os.path.join(TOOLS, 'engine_test.cpp')]
    print(' '.join(cmd)); subprocess.check_call(cmd)


def write_job(path, x, fs_e, p, nthreads, traj_split=0):
    with open(path, 'wb') as f:
        f.write(b'DSMJOB02'); f.write(struct.pack('<d', fs_e))
        f.write(struct.pack('<iii', p['delay'], p['H'], p['D']))
        f.write(struct.pack('<dddI', p['t_l'], p['gamma_x'], p['gamma_z'], p['seed']))
        f.write(struct.pack('<ddd', float(p['phase']), p['amp_pow'], p['omega_s_res']))
        clock = {False: 0, True: 1, 'bin': 2}[p['carrier_clock']]
        f.write(struct.pack('<ii', clock, p['n_traj']))
        for s in (p['measurement_basis'], p['encode_target']):
            b = s.encode(); f.write(struct.pack('<i', len(b))); f.write(b)
        f.write(struct.pack('<ii', RENDER_ID[p.get('render', 'uncond')], traj_split))
        f.write(struct.pack('<i', nthreads)); f.write(struct.pack('<q', len(x)))
        f.write(np.asarray(x, np.float64).tobytes())


def run_cpp(x, fs_e, p, nthreads, tag, traj_split=0):
    job = os.path.join(SCR, f'{tag}.bin'); out = os.path.join(SCR, tag); os.makedirs(out, exist_ok=True)
    write_job(job, x, fs_e, p, nthreads, traj_split)
    t0 = time.time(); subprocess.check_call([EXE, job, out], stderr=subprocess.DEVNULL); secs = time.time() - t0
    g = open(os.path.join(out, 'geometry.txt')).read().split(); g = dict(zip(g[::2], g[1::2]))
    nf, nb, T, ntraj = int(g['nf']), int(g['nb']), int(g['T']), int(g['ntraj'])
    ld = lambda nm, dt: np.fromfile(os.path.join(out, nm), dtype=dt)
    pinkfit = [float(v) for v in open(os.path.join(out, 'geometry.txt')).read().split('pink')[1].split()[:3]] if 'pink' in open(os.path.join(out, 'geometry.txt')).read() else None
    return dict(pinkfit=pinkfit, x_model=ld('x_model.f64', np.float64), in_mag=ld('in_mag.f64', np.float64).reshape(nf, nb),
                in_phase=ld('in_phase.f64', np.float64).reshape(nf, nb),
                out_mag=ld('out_mag.f64', np.float64).reshape(nf, nb), out_phase=ld('out_phase.f64', np.float64).reshape(nf, nb),
                SX=ld('SX.f64', np.float64).reshape(ntraj, nb, T), SY=ld('SY.f64', np.float64).reshape(ntraj, nb, T),
                POP=ld('POP.f64', np.float64).reshape(ntraj, nb, T), mrec=ld('mrec.i8', np.int8).reshape(ntraj, nb, T),
                y=ld('y_model.f64', np.float64), y_engine=ld('y_engine.f64', np.float64),
                secs=float(g['secs']), wall=secs, nf=nf, nb=nb, W=int(g['W']), T=T, ntraj=ntraj, scale=float(g['scale']))


def compare(c, r, p, fs_m):
    """c: C++ outputs; r: Python chain result (numpy store path) for the same job."""
    ok = True
    def rep(name, d, thr):
        nonlocal ok; f = d <= thr; ok &= f
        print(f'    {name:34s} {d:.2e}  {"ok" if f else "FAIL"}')
    rep('decimated buffer', float(np.max(np.abs(c['x_model'] - r['x_model']))), 1e-12)
    Xc = c['in_mag'] * np.exp(1j * c['in_phase']); Xr = r['in_mag'] * np.exp(1j * r['in_phase'])
    rep('input STFT (complex, rel.)', float(np.max(np.abs(Xc - Xr)) / np.max(np.abs(Xr))), 1e-12)
    # end-to-end (own STFT each; silent-bin phases leak at ~1e-8 through Rz on S)
    Yc = c['out_mag'] * np.exp(1j * c['out_phase']); Yr = r['out_mag'] * np.exp(1j * r['out_phase'])
    if p.get('render', 'uncond') != 'pink':      # pink: noise realisation is engine-specific, no sample-wise match
        rep('decoded STFT e2e (rel.)', float(np.max(np.abs(Yc - Yr)) / np.max(np.abs(Yr))), 1e-6)
        rep('loop @ model rate e2e', float(np.max(np.abs(c['y'] - r['y']))), 1e-7)
        rep('loop @ engine rate e2e', float(np.max(np.abs(c['y_engine'] - r['y_engine']))), 1e-7)
    # same-STFT: Python store + renders on the C++ fed frames
    nf, nb, W, H = c['nf'], c['nb'], c['W'], p['H']; N = 2 * H
    fed = ch.fed_frame_indices(nf, W); fb = np.arange(nb) * fs_m / N
    kw = dict(gamma_x=p['gamma_x'], gamma_z=p['gamma_z'], measurement_basis=p['measurement_basis'], amp_pow=p['amp_pow'],
              omega_s_res=p['omega_s_res'], fs=fs_m, carrier_clock=p['carrier_clock'], stft_hop=H, encode_target=p['encode_target'])
    fm, fp = c['in_mag'][fed], c['in_phase'][fed]
    st = ch._store(fm, fp, fb, np.arange(nb), p['delay'], float(p['phase']), p['n_traj'], p['seed'], kw, 8, 'numpy')
    ntr = st['ntraj']
    for key, ck in (('SX', 'SX'), ('SY', 'SY'), ('POP', 'POP')):
        rep(f'store {key} (same STFT)', float(np.max(np.abs(np.transpose(st[key], (1, 0, 2)) - c[ck][:ntr]))), 1e-11)
    same = np.array_equal(np.transpose(st['MREC'], (1, 0, 2)), c['mrec'][:ntr])
    print(f'    measurement records               {"identical in all bins and trajectories" if same else "DIFFER"}'); ok &= same
    render = p.get('render', 'uncond')
    if render == 'twin':
        tkw = {k: kw[k] for k in ('gamma_x', 'gamma_z', 'omega_s_res', 'fs', 'carrier_clock', 'stft_hop', 'encode_target', 'amp_pow')}
        om, oph = eng.twin(fm, fp, fb, np.arange(nb), p['delay'], float(p['phase']), **tkw)
    elif render == 'pink':
        # the noise realisation is engine-specific; certify the FIT (depth, AM exact; P within a grid step)
        qm, qp = eng.render_from_store(st, 'uncond', p['n_traj'], p['delay'], p['amp_pow'], fp, np.arange(nb))
        om, oph, fit = eng.pink_rival(fm, fp, qm, qp, seed=p['seed'], return_fit=True)
        cf = c['pinkfit']
        Ps = [eng.pink_rival(fm, fp, qm, qp, seed=s_, return_fit=True)[2]['P'] for s_ in range(5)]   # seed-to-seed spread of the P fit
        print(f"    pink fit  python: depth {fit['depth']:.6f} P {fit['P']:.2f} am {fit['am']:.6f} | C++: depth {cf[0]:.6f} P {cf[1]:.2f} am {cf[2]:.6f} | python P over 5 seeds: {min(Ps):.2f}..{max(Ps):.2f}")
        okp = abs(fit['depth'] - cf[0]) < 1e-9 and abs(fit['am'] - cf[2]) < 1e-9 and min(Ps) - 0.31 <= cf[1] <= max(Ps) + 0.31
        print(f"    pink fit                          {'ok' if okp else 'FAIL'}"); ok &= okp
        # spectral sanity of the C++ loop: it must keep the input's band (carrier preserved)
        return ok
    else:
        om, oph = eng.render_from_store(st, render, p['n_traj'], p['delay'], p['amp_pow'], fp, np.arange(nb))
    Ys = (om[W:W + nf]) * np.exp(1j * oph[W:W + nf])
    rep(f'render {render} (same STFT, rel.)', float(np.max(np.abs(Yc - Ys)) / np.max(np.abs(Ys))), 1e-10)
    ys = ch.circular_ola(np.abs(Ys), np.angle(Ys), N, H, nf * H)
    rep('loop @ model rate (same STFT)', float(np.max(np.abs(c['y'] - ys))), 1e-10)
    return ok


def main():
    build()
    fs_e = 48000
    t = np.arange(int(3.0 * fs_e)) / fs_e
    rng = np.random.RandomState(3)
    x = (0.5 * np.sin(2 * np.pi * 130.81 * t) * (t < 1.0) + 0.3 * np.sin(2 * np.pi * 196 * t) * ((t > 1.3) & (t < 2.2))
         + 0.05 * rng.randn(len(t)) * (t > 2.3))
    x[:int(0.05 * fs_e)] = 0
    cases = [
        dict(tag='d1_aux_z', delay=1, D=6, H=256, t_l=1.0, gamma_x=0.4, gamma_z=0.6, measurement_basis='z', encode_target='aux', carrier_clock='bin', omega_s_res=0.0, n_traj=1, seed=7),
        dict(tag='d2_none', delay=2, D=3, H=128, t_l=1.2, gamma_x=0.75, gamma_z=0.3, measurement_basis='none', encode_target='system', carrier_clock=True, omega_s_res=23.0, n_traj=1, seed=12345),
        dict(tag='d3_cond0_3traj', delay=3, D=6, H=256, t_l=1.5, gamma_x=0.555, gamma_z=0.45, measurement_basis='x', encode_target='system', carrier_clock=True, omega_s_res=23.0, n_traj=3, seed=99, render='cond0'),
        dict(tag='d3_cond1_3traj_incremental', delay=3, D=6, H=256, t_l=1.5, gamma_x=0.45, gamma_z=0.15, measurement_basis='x', encode_target='system', carrier_clock=True, omega_s_res=23.0, n_traj=3, seed=99, render='cond1', traj_split=1),
        dict(tag='d3_twin', delay=3, D=6, H=256, t_l=1.5, gamma_x=0.45, gamma_z=0.15, measurement_basis='x', encode_target='system', carrier_clock=True, omega_s_res=23.0, n_traj=1, seed=99, render='twin'),
        dict(tag='d3_pink', delay=3, D=6, H=256, t_l=1.5, gamma_x=0.45, gamma_z=0.15, measurement_basis='x', encode_target='system', carrier_clock=True, omega_s_res=23.0, n_traj=1, seed=99, render='pink'),
        dict(tag='d5_operating', delay=5, D=6, H=256, t_l=2.0, gamma_x=0.45, gamma_z=0.15, measurement_basis='x', encode_target='system', carrier_clock=True, omega_s_res=23.0, n_traj=1, seed=12345),
    ]
    allok = True
    for c in cases:
        p = dict(ch.DEFAULT_PARAMS); p.update({k: v for k, v in c.items() if k not in ('tag', 'traj_split')})
        fs_m = ch.model_rate(fs_e, p['D'])
        print(f"\n[{c['tag']}] delay {p['delay']} D {p['D']} H {p['H']} t_l {p['t_l']} basis {p['measurement_basis']} enc {p['encode_target']} "
              f"clock {p['carrier_clock']} n_traj {p['n_traj']} render {p.get('render', 'uncond')}")
        t0 = time.time(); r = ch.process_job(x, fs_e, p, engine='numpy', nproc=8); tpy = time.time() - t0
        cc = run_cpp(x, fs_e, p, 8, c['tag'], c.get('traj_split', 0))
        print(f"  python(numpy, 8 proc) {tpy:.1f} s   C++ (8 threads) {cc['secs']:.2f} s   loop {cc['nf']} frames x {cc['nb']} bins, warm-up {cc['W']}, {cc['ntraj']} traj")
        allok &= compare(cc, r, p, fs_m)
    # qutip oracle (the UNCHANGED transform) on the operating point and on the cond0 case
    for tag in ('d5_operating', 'd3_cond0_3traj'):
        c = next(cc for cc in cases if cc['tag'] == tag)
        p = dict(ch.DEFAULT_PARAMS); p.update({k: v for k, v in c.items() if k not in ('tag', 'traj_split')})
        print(f'\n[qutip oracle on {tag}]')
        t0 = time.time(); rq = ch.process_job(x, fs_e, p, engine='qutip'); tq = time.time() - t0
        cc = run_cpp(x, fs_e, p, 8, tag)
        Yc = cc['out_mag'] * np.exp(1j * cc['out_phase']); Yq = rq['out_mag'] * np.exp(1j * rq['out_phase'])
        # e2e (own STFTs) -- and same-STFT via the unchanged transform on the C++ frames
        fed = ch.fed_frame_indices(cc['nf'], cc['W']); fb = np.arange(cc['nb']) * ch.model_rate(fs_e, p['D']) / (2 * p['H'])
        import dependent_trajectory as dt
        om, oph, sc, _ = dt.dependent_colour_transform_per_bin(
            cc['in_mag'][fed], cc['in_phase'][fed], fb, np.arange(cc['nb']), p['delay'], float(p['phase']), n_traj=p['n_traj'],
            base_seed=p['seed'], gamma_x=p['gamma_x'], gamma_z=p['gamma_z'], measurement_basis=p['measurement_basis'], amp_mode='power',
            amp_pow=p['amp_pow'], omega_s_res=p['omega_s_res'], fs=ch.model_rate(fs_e, p['D']), fast=False, carrier_clock=p['carrier_clock'],
            stft_hop=p['H'], encode_target=p['encode_target'], verbose=False)
        v = p.get('render', 'uncond'); W = cc['W']; nf = cc['nf']
        Ys = (om[v][W:W + nf] * sc[None, :]) * np.exp(1j * oph[v][W:W + nf])
        d_e2e = float(np.max(np.abs(Yc - Yq)) / np.max(np.abs(Yq))); d_same = float(np.max(np.abs(Yc - Ys)) / np.max(np.abs(Ys)))
        print(f'  qutip {tq:.1f} s vs C++ {cc["secs"]:.2f} s: decoded STFT e2e {d_e2e:.2e}, same-STFT vs UNCHANGED transform {d_same:.2e} -> {"ok" if d_same < 1e-10 else "FAIL"}')
        allok &= d_same < 1e-10
    print('\nC++ single-thread per-tick cost (operating point, all bins):')
    p = dict(ch.DEFAULT_PARAMS); p.update({k: v for k, v in cases[-1].items() if k != 'tag'})
    for d in (1, 3, 5):
        q = dict(p); q['delay'] = d
        cc = run_cpp(x, fs_e, q, 1, f'time_d{d}')
        ticks = (cc['nf'] + cc['W'] + d) * cc['nb']
        Wd = min(ch.warmup_frames(d, q['gamma_x'], q['gamma_z']), 313)
        print(f'  d={d}: {cc["secs"]:.2f} s for {ticks} ticks = {cc["secs"]/ticks*1e3:.4f} ms/tick  -> 10 s loop @ 8 kHz (513 bins x (313+W={Wd}) frames): {cc["secs"]/ticks*513*(313+Wd):.1f} s single-thread')
    print('\nALL PASS' if allok else '\nFAILURES PRESENT')


if __name__ == '__main__':
    main()
