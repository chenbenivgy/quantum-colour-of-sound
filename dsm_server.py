"""
Engine A of the Discontinuous Sound Modulator: a Python job server that runs the
UNCHANGED quantum kernel (dependent_trajectory._dependent_colour_kernel, qutip) on
jobs the VCV Rack module drops into a spool directory, and returns the raw
trajectory store; the module does the averaging, synthesis and playback itself
(the same code path as engine B, so switching renders needs no new job).

Protocol (file based):
    <spool>/server_alive                   heartbeat, touched every loop (module falls back to
                                           engine B when it is older than 3 s)
    <spool>/<jobid>/params.json            written by the module (delay, H, D, t_l, gamma_x,
                                           gamma_z, seed, ..., n_traj)
    <spool>/<jobid>/input_engine_rate.wav  the loop material, mono float32 @ engine rate
    <spool>/<jobid>/request                marker: job may be processed
    <spool>/<jobid>/cancel                 marker: abandon this job (running jobs are killed)
  server writes
    <spool>/<jobid>/SX.f64 SY.f64 POP.f64  trajectory store, layout [ti][k][t] (nb, T from result.json)
    <spool>/<jobid>/MREC.i8
    <spool>/<jobid>/in_mag.f64 in_phase.f64   fed frames (warm-up + loop), row-major (W+nf) x nb
    <spool>/<jobid>/input_model_rate.wav   the decimated loop
    <spool>/<jobid>/result.json            geometry + scale + timings
    <spool>/<jobid>/done                   marker  (or error.txt on failure)

Run:  conda activate basic_pitch_env && python dsm_server.py [--spool DIR] [--engine qutip|numpy] [--nproc 8]
Default spool: ~/Library/Application Support/Rack2/dsm_spool
"""
import os, sys, time, json, argparse, traceback
from multiprocessing import Process
import numpy as np
import soundfile
import dsm_chain as ch

DEFAULT_SPOOL = os.path.expanduser('~/Library/Application Support/Rack2/dsm_spool')


def process(jobdir, engine, nproc):
    p = json.load(open(os.path.join(jobdir, 'params.json')))
    x, fs_e = soundfile.read(os.path.join(jobdir, 'input_engine_rate.wav'), dtype='float64')
    if x.ndim > 1:
        x = x.mean(axis=1)
    t0 = time.time()
    p['render'] = 'uncond'
    r = ch.process_job(x, fs_e, p, engine='qutip-kernel' if engine == 'qutip' else 'numpy', nproc=nproc)
    st = r['store']; secs = time.time() - t0
    # store layout [ti][k][t] as the C++ engine
    for key, dt in (('SX', np.float64), ('SY', np.float64), ('POP', np.float64), ('MREC', np.int8)):
        np.ascontiguousarray(np.transpose(st[key], (1, 0, 2))).astype(dt).tofile(os.path.join(jobdir, key + ('.f64' if dt is np.float64 else '.i8')))
    r['fed_mag'].astype(np.float64).tofile(os.path.join(jobdir, 'in_mag.f64'))
    r['fed_phase'].astype(np.float64).tofile(os.path.join(jobdir, 'in_phase.f64'))
    soundfile.write(os.path.join(jobdir, 'input_model_rate.wav'), r['x_model'].astype(np.float32), int(r['fs_m']), subtype='FLOAT')
    res = dict(engine=engine, secs=secs, fs_m=r['fs_m'], nf=r['nf'], L=len(r['y']), W=r['W'], N=r['N'], H=r['H'],
               nb=st['SX'].shape[0], T=int(st['T']), ntraj=int(st['ntraj']), scale=float(st['scale']),
               theta=(st['theta'].tolist() if st['theta'] is not None else None), peak=float(np.max(np.abs(r['y']))))
    json.dump(res, open(os.path.join(jobdir, 'result.json'), 'w'))
    return res


def _run(jobdir, engine, nproc):
    try:
        res = process(jobdir, engine, nproc)
        open(os.path.join(jobdir, 'done'), 'w').close()
        print(f"[dsm_server]   done in {res['secs']:.1f} s: {res['ntraj']} trajectories, loop {res['nf']} frames "
              f"({res['L']/res['fs_m']:.2f} s @ {res['fs_m']} Hz)", flush=True)
    except Exception:
        open(os.path.join(jobdir, 'error.txt'), 'w').write(traceback.format_exc())
        print('[dsm_server]   ERROR\n' + traceback.format_exc(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spool', default=DEFAULT_SPOOL)
    ap.add_argument('--engine', default='qutip', choices=['qutip', 'numpy'])
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--once', action='store_true', help='process pending jobs and exit')
    a = ap.parse_args()
    os.makedirs(a.spool, exist_ok=True)
    print(f'[dsm_server] engine={a.engine} nproc={a.nproc} spool={a.spool}', flush=True)
    heartbeat = os.path.join(a.spool, 'server_alive')
    while True:
        open(heartbeat, 'w').write(str(time.time()))
        pending = []
        for name in sorted(os.listdir(a.spool)):
            d = os.path.join(a.spool, name)
            if (os.path.isdir(d) and os.path.exists(os.path.join(d, 'request'))
                    and not os.path.exists(os.path.join(d, 'done'))
                    and not os.path.exists(os.path.join(d, 'error.txt'))
                    and not os.path.exists(os.path.join(d, 'cancel'))):
                pending.append(d)
        for d in pending:
            print(f'[dsm_server] job {os.path.basename(d)} ...', flush=True)
            proc = Process(target=_run, args=(d, a.engine, a.nproc)); proc.start()
            while proc.is_alive():
                open(heartbeat, 'w').write(str(time.time()))
                if os.path.exists(os.path.join(d, 'cancel')):
                    proc.terminate(); proc.join()
                    print(f'[dsm_server]   cancelled {os.path.basename(d)}', flush=True)
                    break
                time.sleep(0.2)
            proc.join()
        if a.once:
            break
        time.sleep(0.25)


if __name__ == '__main__':
    main()
