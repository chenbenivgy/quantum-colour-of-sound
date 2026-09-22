"""
Dependent-trajectory mode: one quantum delay line per STFT bin, output decoded
from the exiting probes. Runs N trajectories per bin and produces three renders:
uncond (plain average) and cond0/cond1, where frame n is conditioned on the
delayed probe's measurement outcome -- the retrodicted, delayed-choice picture:

    b_a(n) = E[ (sx,sy,sz)(n) | outcome(n+delay) = a ]

Amplitude comes from population, phase from atan2(<sy>,<sx>) of the averaged
Bloch vector (average the vector first, never the angles).
"""
import os
import numpy as np
import scipy.sparse as _sp
import soundfile
from qutip import (Qobj, qeye, sigmax, sigmay, sigmaz, fock_dm, tensor, destroy,
                   basis, ket2dm, expect)
import quantum_audio_master as qam


def _sparsify_local_unitary(U, tol=1e-12):
    """Store the collision unitary sparse. H+V only touches the system, D_0 and
    the fresh probe, so U factorises as a local unitary tensored with identity
    on the spectator qubits -- structurally sparse. Verified bit-identical to
    the dense path; `tol` just drops expm round-off dust."""
    A = U.full()
    A[np.abs(A) < tol] = 0.0
    return Qobj(_sp.csr_matrix(A), dims=U.dims)

_VERSIONS = ['uncond', 'cond0', 'cond1']
_Z11 = Qobj([[0, 0], [0, 1]])


def adaptive_bandwidth_params(data, fs, M, N, H, nH=60, minf0=100.0, enable=False,
                              floor_db=-80.0, margin=1.3, min_decim=2, verbose=True):
    """Bandwidth adaptation, OFF by default. If the content sits well below
    Nyquist, integer-decimate and shrink (M, N, H) by the same factor: identical
    time-frequency resolution, far fewer bins to process. Returns
    (data, fs, M, N, H, nH, info); info['changed'] says whether it acted."""
    info = {'enabled': bool(enable), 'changed': False}
    if not enable:
        info['reason'] = 'flag off (default)'
        return data, fs, M, N, H, nH, info

    x = data if np.ndim(data) == 1 else np.mean(data, axis=1)
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w))
    fr = np.fft.rfftfreq(len(x), 1.0 / fs)
    db = 20.0 * np.log10(X / (X.max() + 1e-12) + 1e-12)
    f_cut = float(fr[db > floor_db].max())
    target_nyq = f_cut * margin
    info.update(f_cut_Hz=round(f_cut, 1), target_nyquist_Hz=round(target_nyq, 1),
                orig=dict(fs=fs, M=M, N=N, H=H, bins=N // 2 + 1))

    decim_max = int(np.floor((fs / 2.0) / target_nyq)) if target_nyq > 0 else 1
    if decim_max < min_decim:
        info['reason'] = (f'content fills the band (f_cut={f_cut:.0f} Hz, '
                          f'max decim={decim_max}) -- left unchanged')
        if verbose:
            print(f'[adaptive bandwidth] {info["reason"]}')
        return data, fs, M, N, H, nH, info
    # prefer the largest decim in [min_decim, decim_max] that DIVIDES fs -> clean
    # integer output rate + exact integer decimation; else fall back to decim_max.
    fs_int = int(round(fs))
    divisors = [d for d in range(decim_max, min_decim - 1, -1) if fs_int % d == 0]
    decim = divisors[0] if divisors else decim_max
    info['decim_max'] = decim_max

    from scipy.signal import resample_poly
    x2 = resample_poly(x, 1, decim).astype(np.float64)
    fs2_f = fs / decim
    fs2 = int(round(fs2_f)) if abs(fs2_f - round(fs2_f)) < 1e-6 else fs2_f

    def _odd(v):
        v = max(3, int(round(v)))
        return v if v % 2 == 1 else v + 1

    def _pow2(v):
        return int(2 ** int(np.ceil(np.log2(max(int(round(v)), 8)))))

    M2 = _odd(M / decim)
    H2 = max(1, int(round(H / decim)))
    N2 = _pow2(N / decim)
    if N2 < M2:
        N2 = _pow2(M2)
    # cap nH to the harmonics that still fit under the new Nyquist (oversized
    # tracks were harmless but pointless once the band shrinks).
    nH2 = max(1, min(int(nH), int(0.95 * (fs2 / 2.0) / max(float(minf0), 1.0))))
    info.update(changed=True, decim=decim,
                new=dict(fs=fs2, M=M2, N=N2, H=H2, nH=nH2, bins=N2 // 2 + 1),
                bin_reduction=round((N // 2 + 1) / (N2 // 2 + 1), 1),
                nH=dict(old=int(nH), new=nH2))
    if verbose:
        print(f'[adaptive bandwidth] f_cut={f_cut:.0f} Hz (>{floor_db:.0f} dB), '
              f'margin {margin} -> decim {decim}x')
        print(f'   fs {fs}->{fs2}   N {N}->{N2}   bins {N//2+1}->{N2//2+1} '
              f'({info["bin_reduction"]}x fewer)   M {M}->{M2}   H {H}->{H2}   nH {nH}->{nH2}')
    return x2, fs2, M2, N2, H2, nH2, info


def _basis_projectors(measurement_basis):
    if measurement_basis == 'none':      # no projective measurement (deterministic channel)
        return None, None
    if measurement_basis == 'z':
        return fock_dm(2, 0), fock_dm(2, 1)
    if measurement_basis == 'x':
        p = (basis(2, 0) + basis(2, 1)).unit(); m = (basis(2, 0) - basis(2, 1)).unit()
        return ket2dm(p), ket2dm(m)
    if measurement_basis == 'y':
        p = (basis(2, 0) + 1j * basis(2, 1)).unit(); m = (basis(2, 0) - 1j * basis(2, 1)).unit()
        return ket2dm(p), ket2dm(m)
    raise ValueError(measurement_basis)


def _interaction_beamsplitter(g, gz, delay, phase, dim=2):
    """Excitation-preserving exchange g(sx sx + sy sy) plus a QND dephasing term
    gz(sz sz), to the fresh probe and, with the feedback phase, to D_0.
    Annihilates |00>: silence stays silent."""
    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    ox = [g * sx] + [qeye(dim)] * delay + [sx]
    oy = [g * sy] + [qeye(dim)] * delay + [sy]
    oz = [gz * sz] + [qeye(dim)] * delay + [sz]
    V = tensor(ox) + tensor(oy) + tensor(oz)
    if delay > 0:
        fb = -np.exp(-1j * phase)
        ox = [fb * g * sx, sx] + [qeye(dim)] * (delay - 1) + [qeye(dim)]
        oy = [fb * g * sy, sy] + [qeye(dim)] * (delay - 1) + [qeye(dim)]
        oz = [fb * gz * sz, sz] + [qeye(dim)] * (delay - 1) + [qeye(dim)]
        V = V + tensor(ox) + tensor(oy) + tensor(oz)
    return V


def _dependent_colour_kernel(aux_amps, aux_phases, gx, gz, omega_s, omega_cont,
                             omega_cont_z, omega_r, delay, phase, DeltaT,
                             measurement_basis, evuType, seed, dim=2,
                             interaction='beamsplitter', fast=False,
                             theta_clock=0.0, encode_target='aux'):
    """One trajectory of the colour-encoded kernel: encode the probe, collide,
    passively decode the exiting probe, then measure it projectively (the
    conditioning outcome). interaction='beamsplitter' or 'spin'.

    encode_target: where the bin's (magnitude, phase) is written each tick.
      'aux' (default, original behaviour): the fresh probe is PREPARED in the
        encoded state |ψ(a,φ)> (qam.encode_aux_state); the system S is only
        ever driven through the collision.
      'system': the fresh probe is vacuum and the encoding is applied as a
        ROTATION of the persistent S qubit, R(a,φ) = Rz(φ)·Ry(2·asin√a)
        (qam.encode_unitary; R|0> = |ψ(a,φ)>, so from ground it is the same
        Bloch point) -- the sensing route's picture, where the sample rotates S
        on top of whatever S already holds. Collision, readout of D_0,
        measurement and trace-out are identical in both cases.

    theta_clock (0 = off): the bin's per-tick carrier advance. Adds the same
    splitting (theta/2)*sigma_z to every qubit of the register, so phases ride
    the carrier while equal splittings keep the exchange resonant; sigma_z on
    vacuum does nothing, so silence is untouched."""
    qam._check_encode_target(encode_target)
    enc_on_sys = (encode_target == 'system')
    N = [dim]; b = destroy(dim)
    H = qam._get_SPbHam(omega_s, omega_cont, omega_cont_z, omega_r, b, delay, N, evuType)
    if theta_clock:
        nq = delay + 2
        Hck = 0
        for j in range(nq):
            ops = [qeye(dim)] * nq
            ops[j] = sigmaz()
            Hck = Hck + tensor(ops)
        H = H + 0.5 * theta_clock * Hck
    if interaction == 'beamsplitter':
        V = _interaction_beamsplitter(gx, gz, delay, phase, dim)   # gx=exchange, gz=σz dephasing
    else:
        V = qam._get_interaction_ham(gx, gz, delay, phase, N)
    U = (-1j * DeltaT * (H + V)).expm()
    if fast:
        U = _sparsify_local_unitary(U)   # exact: U = U_A ⊗ I_spectators is sparse
    Ud = U.dag()
    keep = qam._keep_indices(delay)
    m0, m1 = _basis_projectors(measurement_basis)
    no_meas = (measurement_basis == 'none')   # deterministic: trace exiting aux, no collapse
    if no_meas:
        P0 = P1 = None
    elif delay == 0:
        P0, P1 = tensor(qeye(dim), m0), tensor(qeye(dim), m1)
    else:
        P0 = tensor([qeye(dim), m0] + [qeye(dim)] * (delay - 1) + [qeye(dim)])
        P1 = tensor([qeye(dim), m1] + [qeye(dim)] * (delay - 1) + [qeye(dim)])
    rng = np.random.RandomState(seed)
    rho = qam._initial_state_with_delay(evuType, delay, dim)
    runs = len(aux_amps)
    SX = np.empty(runs); SY = np.empty(runs); POP = np.empty(runs); MREC = np.empty(runs, int)
    vac = fock_dm(dim, 0)
    for n in range(runs):
        if enc_on_sys:
            # rotate the persistent S (register index 0) by the encoding gate;
            # the fresh probe enters as vacuum
            R = qam._embed_1q_gate(qam.encode_unitary(aux_amps[n], aux_phases[n]),
                                   0, delay + 1)
            if fast:
                R = qam._sparsify_op(R)
            rho = R * rho * R.dag()
            aux = vac
        else:
            aux = qam.encode_aux_state(aux_amps[n], aux_phases[n], N)   # amplitude INTO the state
        rp = U * tensor(rho, aux) * Ud
        # feedback phases outside {0, pi} make the map non-trace-preserving;
        # renormalise per step (exact no-op at 0/pi)
        trp = float(np.real(rp.tr()))
        if trp > 1e-12:
            rp = rp / trp
        rex = rp.ptrace(1)                                          # exiting aux (D_0)
        SX[n] = float(np.real(expect(sigmax(), rex)))              # passive decode
        SY[n] = float(np.real(expect(sigmay(), rex)))
        POP[n] = float(np.real(expect(_Z11, rex)))
        if no_meas:                       # deterministic: trace out, no collapse, no record
            MREC[n] = -1
            rho = rp.ptrace(keep)
            continue
        p0 = max(float(np.real((P0 * rp).tr())), 0.0)
        p1 = max(float(np.real((P1 * rp).tr())), 0.0)   # explicit tr(P1 rho), not 1-p0 (B15)
        s = p0 + p1
        p0, p1 = (0.5, 0.5) if s < 1e-12 else (p0 / s, p1 / s)
        o = rng.choice([0, 1], p=[p0, p1]); MREC[n] = o            # projective conditioning
        Pp = P0 if o == 0 else P1
        rc = Pp * rp * Pp; nr = float(np.real(rc.tr()))
        if nr > 1e-12: rc = rc / nr
        rho = rc.ptrace(keep)
    return SX, SY, POP, MREC


def dependent_colour_transform_per_bin(
        in_mag, in_phase, freq_bins, bin_indices, delay, phase,
        n_traj=8, base_seed=0, DeltaT=1.0,
        gamma_x=0.1, gamma_z=0.1, omega_cont=0.0, omega_cont_z=0.0,
        omega_s=0.0, omega_r=0.0, freq_scale=False, interaction='beamsplitter',
        measurement_basis='x', evuType='Pur.Deph', amp_gain=1.0, verbose=True,
        fast=False, omega_s_res=0.0, fs=None, soft_gain=False,
        amp_mode=None, amp_pow=0.5, carrier_clock=False, stft_hop=None,
        encode_target='aux'):
    """Colour-encoded dependent transform over the given bins.

    Amplitude is companded (amp_mode 'power': enc a**amp_pow, dec p**(1/amp_pow);
    'soft' and 'hard' are legacy), encoded into the probes, and read back from
    the exiting probe's population -- silent in stays silent out. The input is
    padded by `delay` frames and the decoded arrays sliced back, so the outputs
    are time-aligned with the input (B16); output frame n is conditioned on the
    probe measured at n+delay.

    encode_target: 'aux' (default) prepares each fresh probe in the encoded
    state; 'system' keeps the probes as vacuum and applies the encoding as a
    rotation of the persistent system qubit (see _dependent_colour_kernel).
    Companding and decoding are identical for both.

    omega_s_res > 0 adds system-only precession os = omega_s_res*(f_k/fs)*gz,
    a detuning that dries the channel towards high frequency.

    carrier_clock (needs stft_hop and fs): same splitting on every register
    qubit at the bin's carrier rate, so decoded phases ride the carrier. True
    estimates the rate from the input's per-frame phase advance (energy-weighted
    circular mean); 'bin' uses the grid rate 2*pi*f_k*H/fs -- audibly worse for
    off-grid content, kept for A/B.

    Returns out_mag (dict of 0-1 magnitude tables), out_phase, scale_per_bin,
    diag (per bin: fallback_frac = conditional frames that fell back to the
    unconditional average, and both_outcomes_frac)."""
    if amp_mode is None:
        amp_mode = 'soft' if soft_gain else 'hard'
    if amp_mode not in ('hard', 'soft', 'power'):
        raise ValueError(f"amp_mode '{amp_mode}' not in hard|soft|power")
    qam._check_encode_target(encode_target)
    no_meas = (measurement_basis == 'none')
    if no_meas:
        n_traj = 1        # deterministic: every trajectory identical; cond0/1 := uncond
    num_frames, num_bins = in_mag.shape
    T = num_frames + delay          # padded kernel length (B16 latency compensation)
    # one global scale; per-bin normalisation would re-inject each bin's level
    mag_norm, scale_f = qam.normalize_amplitudes(in_mag)
    scale_per_bin = np.full(num_bins, scale_f)
    out_mag = {v: np.clip(mag_norm.copy(), 0.0, 1.0) for v in _VERSIONS}
    out_phase = {v: in_phase.copy() for v in _VERSIONS}

    def _encode(a):
        if amp_mode == 'power':
            return np.clip(a, 0.0, 1.0) ** amp_pow
        if amp_mode == 'soft':
            return 1.0 - np.exp(-amp_gain * a)
        return np.clip(a * amp_gain, 0.0, 1.0)

    def _decode(p):
        p = np.clip(p, 0.0, 1.0)
        if amp_mode == 'power':
            return p ** (1.0 / amp_pow)
        if amp_mode == 'soft':
            return np.clip(-np.log(1.0 - np.clip(p, 0.0, 1.0 - 1e-3)) / amp_gain, 0.0, 1.0)
        return np.clip(p / amp_gain, 0.0, 1.0)

    th_clock_arr = None
    if carrier_clock:
        if not stft_hop:
            raise ValueError('carrier_clock requires stft_hop (hop size in samples)')
        _fsc = fs if fs else 2.0 * float(np.max(freq_bins))
        th_bin = 2.0 * np.pi * np.asarray(freq_bins, float) * float(stft_hop) / float(_fsc)
        if carrier_clock == 'bin':
            th_clock_arr = th_bin
        else:   # True / 'input': phase-vocoder IF, energy-weighted circular mean
            w = in_mag[1:, :] ** 2
            zc = np.sum(w * np.exp(1j * (np.diff(in_phase, axis=0) - th_bin[None, :])), axis=0)
            th_clock_arr = th_bin + np.angle(zc + 1e-30)   # silent bins fall back to grid rate

    diag = {}
    total = len(bin_indices)
    for count, k in enumerate(bin_indices):
        if verbose and (count % max(1, total // 10) == 0 or count == total - 1):
            print(f"  bin {count+1}/{total}  (k={k}, f={freq_bins[k]:.1f} Hz)")
        sc = (freq_bins[k] / 100.0) if freq_scale else 1.0
        gx, gz = gamma_x * sc, gamma_z * sc
        oc, ocz, os_, orr = omega_cont * sc, omega_cont_z * sc, omega_s * sc, omega_r * sc
        if omega_s_res:   # resonator: detunes the exchange towards high frequency
            _fs = fs if fs else 2.0 * float(np.max(freq_bins))
            os_ = os_ + omega_s_res * (freq_bins[k] / _fs) * gz
        th_ck = float(th_clock_arr[k]) if th_clock_arr is not None else 0.0
        amps = np.concatenate([_encode(mag_norm[:, k]), np.zeros(delay)])   # B16 pad
        phs = np.concatenate([in_phase[:, k], np.zeros(delay)])
        SX = np.empty((n_traj, T)); SY = np.empty((n_traj, T))
        POP = np.empty((n_traj, T)); MR = np.empty((n_traj, T), int)
        for ti in range(n_traj):
            sx, sy, pop, mr = _dependent_colour_kernel(
                amps, phs, gx, gz, os_, oc, ocz, orr, delay, phase, DeltaT,
                measurement_basis, evuType, base_seed + 10007 * int(k) + ti,
                interaction=interaction, fast=fast, theta_clock=th_ck,
                encode_target=encode_target)
            SX[ti], SY[ti], POP[ti], MR[ti] = sx, sy, pop, mr
        n_fallback = 0; n_cond = 0
        for v in _VERSIONS:
            amp = np.empty(T); ph = np.empty(T)
            for n in range(T):
                if v == 'uncond' or no_meas:
                    sel = slice(None)
                else:
                    a = 0 if v == 'cond0' else 1
                    cf = n + delay
                    if cf < T:
                        mask = (MR[:, cf] == a)
                        n_cond += 1
                        if mask.any():
                            sel = mask
                        else:
                            sel = slice(None); n_fallback += 1
                    else:
                        sel = slice(None)
                amp[n] = POP[sel, n].mean()
                ph[n] = np.arctan2(SY[sel, n].mean(), SX[sel, n].mean())
            # B16: slice the delay-line latency away -> output aligned with input
            out_mag[v][:, k] = _decode(amp[delay:delay + num_frames])
            out_phase[v][:, k] = ph[delay:delay + num_frames]
        both = np.mean([(MR[:, n] == 0).any() and (MR[:, n] == 1).any() for n in range(T)])
        diag[k] = dict(fallback_frac=n_fallback / max(n_cond, 1),
                       both_outcomes_frac=float(both))
    return out_mag, out_phase, scale_per_bin, diag


def dependent_trajectory_transform_per_bin(
        in_mag, in_phase, freq_bins, bin_indices, delay, phase,
        n_traj=24, base_seed=0, DeltaT=1.0, lam=1.0 / 100.0,
        gamma_x=0.2, gamma_z=0.5, omega_drive_xy=1.0, omega_drive_z=0.0,
        measurement_basis='x', normperbin=False, N=None,
        evuType='Pur.Deph', verbose=True):
    """Returns {version: out_mag(0-1)}, {version: out_phase}, scale_per_bin, diag."""
    if N is None:
        N = [2]
    num_frames, num_bins = in_mag.shape

    if normperbin:
        mag_norm, scale_per_bin = qam.normalize_amplitudes_per_bin(in_mag)
    else:
        mag_norm, sf = qam.normalize_amplitudes(in_mag)
        scale_per_bin = np.full(num_bins, sf)

    out_mag = {v: in_mag.copy() for v in _VERSIONS}     # placeholder; bins overwritten
    out_phase = {v: in_phase.copy() for v in _VERSIONS}
    # normalized magnitude tables (0-1) for the processed bins
    out_mag_n = {v: np.clip(mag_norm.copy(), 0.0, 1.0) for v in _VERSIONS}
    diag = {}
    total = len(bin_indices)

    for count, k in enumerate(bin_indices):
        if verbose and (count % max(1, total // 10) == 0 or count == total - 1):
            print(f"  bin {count+1}/{total}  (k={k}, f={freq_bins[k]:.1f} Hz)")

        qp = qam._compute_sms_quantum_params(freq_bins[k], lam=lam)
        gx = gamma_x * qp['omega_s']
        gz = gamma_z * qp['omega_s']
        drive_amps = mag_norm[:, k]
        drive_phases = in_phase[:, k]               # full-STFT: bin carries its own phase

        SX = np.empty((n_traj, num_frames))
        SY = np.empty((n_traj, num_frames))
        POP = np.empty((n_traj, num_frames))
        MREC = np.empty((n_traj, num_frames), dtype=int)
        for ti in range(n_traj):
            res = qam.trotterize_trajectory(
                DeltaT=DeltaT, omega_s=qp['omega_s'], omega_r=qp['omega_r'],
                gamma_x=gx, gamma_z=gz, delay=delay, N=N, phase=phase,
                drive_amplitudes=drive_amps, drive_phases=drive_phases,
                omega_drive_xy=omega_drive_xy * qp['omega_s'],
                omega_drive_z=omega_drive_z * qp['omega_s'],
                omega_s_static=0, measurement_basis=measurement_basis,
                evuType=evuType, seed=base_seed + 10007 * int(k) + ti)
            SX[ti] = res['sys_sigmax']
            SY[ti] = res['sys_sigmay']
            POP[ti] = res['sys_pop']
            MREC[ti] = res['measurement_record']

        # ---- route-1 conditional averages ----
        for v in _VERSIONS:
            amp = np.empty(num_frames)
            ph = np.empty(num_frames)
            for n in range(num_frames):
                if v == 'uncond':
                    sel = slice(None)
                else:
                    a = 0 if v == 'cond0' else 1
                    cf = n + delay
                    if cf < num_frames:
                        mask = (MREC[:, cf] == a)
                        sel = mask if mask.any() else slice(None)   # fallback if empty class
                    else:
                        sel = slice(None)                            # no future -> unconditional
                bx = SX[sel, n].mean()
                by = SY[sel, n].mean()
                amp[n] = POP[sel, n].mean()
                ph[n] = np.arctan2(by, bx)
            out_mag_n[v][:, k] = np.clip(amp, 0.0, 1.0)
            out_phase[v][:, k] = ph

        # outcome balance diagnostic at this bin (fraction of frames with both classes)
        both = np.mean([(MREC[:, n] == 0).any() and (MREC[:, n] == 1).any()
                        for n in range(num_frames)])
        diag[k] = dict(both_outcomes_frac=float(both))

    return out_mag_n, out_phase, scale_per_bin, diag


def dependent_trajectory_pipeline(
        audio_path, bins='stochastic_only', delay=3, phase=np.pi,
        gamma_x=0.2, gamma_z=0.5, measurement_basis='x',
        n_traj=24, base_seed=0,
        omega_drive_xy=1.0, omega_drive_z=0.0, lam=1.0 / 100.0,
        normperbin=False, DeltaT=1.0,
        use_quantum_amp=True, use_quantum_phase=True, softclip=True,
        evuType='Pur.Deph',
        M=2501, N_fft=4096, H=1024, t=-80, nH=23,
        minf0=100, maxf0=400, f0et=5, harmDevSlope=0.01, minSineDur=0.02,
        Ns=2048, stocf=0.2, verbose=True):
    if verbose:
        print("=== MODE: dependent_trajectory (full-STFT, route-1 conditional) ===")

    data, fs = qam.load_wav_full(audio_path)
    ana = qam.sms_analyse(data, fs, M=M, N=N_fft, H=H, t=t, nH=nH,
                          minf0=minf0, maxf0=maxf0, f0et=f0et,
                          harmDevSlope=harmDevSlope, minSineDur=minSineDur,
                          Ns=Ns, stocf=stocf)
    freq_bins = ana['freq_bins']
    num_frames, num_bins = ana['Xr_mag'].shape

    if bins == 'full_stft':
        # every bin; sinusoidal+residual merged -> the trajectory's stochastic
        # collapse touches tonal bins too (noisy, but full-band).
        bin_indices = np.arange(num_bins)
        in_mag, in_phase, sin_bin_map = qam.merge_sms_to_stft(
            ana['Xr_mag'], ana['Xr_phase'], ana['sin_freqs'],
            ana['sin_amps'], ana['sin_phases'], freq_bins, num_bins)
    else:  # 'stochastic_only' — colour only the residual, leave sinusoids clean
        bin_indices = np.where(
            qam.get_stochastic_only_bins(ana['sin_freqs'], freq_bins, None))[0]
        in_mag, in_phase, sin_bin_map = ana['Xr_mag'], ana['Xr_phase'], None

    if verbose:
        print(f"  bins={bins} ({len(bin_indices)}/{num_bins})  frames={num_frames}  "
              f"n_traj={n_traj}  delay={delay}  gx={gamma_x} gz={gamma_z} basis={measurement_basis}")

    out_mag_n, out_phase, scale_per_bin, diag = dependent_trajectory_transform_per_bin(
        in_mag, in_phase, freq_bins, bin_indices, delay, phase,
        n_traj=n_traj, base_seed=base_seed, DeltaT=DeltaT, lam=lam,
        gamma_x=gamma_x, gamma_z=gamma_z,
        omega_drive_xy=omega_drive_xy, omega_drive_z=omega_drive_z,
        measurement_basis=measurement_basis, normperbin=normperbin,
        N=[2], evuType=evuType, verbose=verbose)

    stem = os.path.splitext(os.path.basename(audio_path))[0]
    out_dir = os.path.dirname(os.path.abspath(audio_path))
    paths = {}
    synths = {}
    tag = 'DEPfull' if bins == 'full_stft' else 'DEPstoch'
    for v in _VERSIONS:
        om_scaled = out_mag_n[v] * scale_per_bin[np.newaxis, :]
        if bins == 'full_stft':
            new_sin_amps, _, res_mag, res_phase = qam.separate_sinusoidal_from_stft(
                om_scaled, out_phase[v], ana['sin_freqs'], ana['sin_amps'],
                freq_bins, sin_bin_map)
            synth = qam.synthesise_sms(
                ana, res_mag=res_mag, res_phase=res_phase, sin_amps_mod=new_sin_amps,
                use_quantum_amp=use_quantum_amp, use_quantum_phase=use_quantum_phase,
                softclip=softclip, win_norm_dB_override=ana['win_norm_dB'])
        else:  # stochastic_only: residual only, sinusoidal tracks left untouched
            synth = qam.synthesise_sms(
                ana, res_mag=om_scaled, res_phase=out_phase[v],
                use_quantum_amp=use_quantum_amp, use_quantum_phase=use_quantum_phase,
                softclip=softclip, win_norm_dB_override=ana['win_norm_dB'])
        synths[v] = synth
        p = os.path.join(out_dir, f"{stem}_{tag}_{v}.wav")
        soundfile.write(p, synth['y_quantum'].astype(np.float32), fs)
        paths[v] = p
    p_orig = os.path.join(out_dir, f"{stem}_{tag}_original.wav")
    soundfile.write(p_orig, synths['uncond']['y_original'].astype(np.float32), fs)
    paths['original'] = p_orig
    if verbose:
        print(f"  Saved: {', '.join(os.path.basename(p) for p in paths.values())}")

    return dict(
        y_uncond=synths['uncond']['y_quantum'],
        y_cond0=synths['cond0']['y_quantum'],
        y_cond1=synths['cond1']['y_quantum'],
        y_original=synths['uncond']['y_original'],
        fs=fs, mode='dependent_trajectory', audio_path=audio_path,
        analysis=ana, diagnostics=diag, output_paths=paths,
        params=dict(bins=bins, delay=delay, phase=phase, gamma_x=gamma_x,
                    gamma_z=gamma_z, measurement_basis=measurement_basis,
                    n_traj=n_traj, base_seed=base_seed, normperbin=normperbin))
