"""
'dependent_trajectory' mode  --  full-STFT, per-bin quantum delay line, where the
per-bin readout is the ROUTE-1 conditional (retrodicted) Bloch trajectory.

Same encode/decode as mode_sms='full_stft' (one system per STFT bin, amplitude &
phase), but instead of one forward trajectory it runs N trajectories per bin and
extracts b_a(n) by sorting on the DELAYED probe outcome and averaging:

    b_a(n) = E[ (sx,sy,sz)(n) | measurement_record[n+delay] = a ]    (over seeds)

The probe injected at frame n exits/is measured at frame n+delay; conditioning
b(n) on that future outcome is the delayed-choice re-painting of S@n.

Three renderings are produced so you can A/B the difference:
    uncond  : average over all trajectories          (= expectation / no future)
    cond0   : conditioned on delayed outcome a = 0
    cond1   : conditioned on delayed outcome a = 1

Amplitude <- population (1 - <sz>)/2 ; phase <- atan2(<sy>, <sx>) of the AVERAGED
Bloch vector (average the vector, THEN take the phase -- never average angles).
"""
import os
import numpy as np
import scipy.sparse as _sp
import soundfile
from qutip import (Qobj, qeye, sigmax, sigmay, sigmaz, fock_dm, tensor, destroy,
                   basis, ket2dm, expect)
import quantum_audio_master as qam


def _sparsify_local_unitary(U, tol=1e-12):
    """Store the collision unitary U as a SPARSE Qobj (exact speed optimisation).

    Physics: H + V couples the system only to its fresh ancilla and the
    recirculating D_0 qubit; the remaining delay-line qubits D_1..D_{d-1} are
    spectators this step, so U factorises EXACTLY as U_A (on system/D_0/fresh)
    tensored with the identity on the spectators.  U is therefore structurally
    sparse (~8*D nonzeros out of D^2).  Storing it sparse skips the
    multiply-by-zero / multiply-by-identity inside U @ X @ U.dag().

    This does NOT change the dynamics: the partial trace (dissipation) and the
    projective measurement (collapse) that follow are left completely untouched.
    Verified bit-identical to the dense path (max abs diff ~1e-15, identical
    measurement records).  `tol` (1e-12) only removes expm round-off dust; real
    couplings here are O(1e-2..1).  If U is genuinely dense this is just a
    slower no-op, still exact."""
    A = U.full()
    A[np.abs(A) < tol] = 0.0
    return Qobj(_sp.csr_matrix(A), dims=U.dims)

_VERSIONS = ['uncond', 'cond0', 'cond1']
_Z11 = Qobj([[0, 0], [0, 1]])


def adaptive_bandwidth_params(data, fs, M, N, H, nH=60, minf0=100.0, enable=False,
                              floor_db=-80.0, margin=1.3, min_decim=2, verbose=True):
    """OPTION-A exact-reconstruction bandwidth adaptation (OFF by default).

    Studies the input spectrum; if the content sits well below Nyquist it
    integer-decimates the signal and shrinks (M, N, H) by the SAME factor, so
    window DURATIONS and bin spacing are preserved -> identical time-frequency
    resolution, all bins land in-band, far fewer bins (~decim x fewer => ~decim x
    faster per-bin quantum transform). This does NOT beat the Gabor limit; it
    only stops spending bins on the empty high-frequency band.

    enable=False (default) -> returns (data, fs, M, N, H) UNCHANGED.
    Also returns unchanged if the signal already fills its band (decim<min_decim).

    Bandwidth = highest frequency whose Hann-windowed spectral envelope exceeds
    `floor_db` (relative to peak); new Nyquist target = bandwidth * margin.
    nH is intentionally left alone (oversized harmonic tracks are skipped by
    sineModelSynth, so they cost nothing -- see synthesise_sms).

    Returns
    -------
    (data, fs, M, N, H, info)  -- info['changed'] tells you whether it acted.
    """
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
    """Excitation-preserving exchange  g·(σx⊗σx + σy⊗σy)  PLUS a QND dephasing
    term  gz·(σz⊗σz),  both system<->aux. The beamsplitter annihilates |00>
    (vacuum fixed -> no flood); σz⊗σz is QND on |0> (also vacuum-safe) and
    injects phase / conditional variation. Mirrors _get_interaction_ham's
    fresh + D_0 feedback."""
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
                             interaction='beamsplitter', fast=False):
    """One trajectory of the COLOUR-encoded dependent kernel:
    encode_aux_state(amp,phi) -> delay-line collision -> PASSIVE decode of the
    exiting aux (Bloch) + projective measurement (conditioning outcome).
    interaction='beamsplitter' (excitation-preserving, silent->silent) or 'spin'
    (original σxσx+σzσz)."""
    N = [dim]; b = destroy(dim)
    H = qam._get_SPbHam(omega_s, omega_cont, omega_cont_z, omega_r, b, delay, N, evuType)
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
    if delay == 0:
        P0, P1 = tensor(qeye(dim), m0), tensor(qeye(dim), m1)
    else:
        P0 = tensor([qeye(dim), m0] + [qeye(dim)] * (delay - 1) + [qeye(dim)])
        P1 = tensor([qeye(dim), m1] + [qeye(dim)] * (delay - 1) + [qeye(dim)])
    rng = np.random.RandomState(seed)
    rho = qam._initial_state_with_delay(evuType, delay, dim)
    runs = len(aux_amps)
    SX = np.empty(runs); SY = np.empty(runs); POP = np.empty(runs); MREC = np.empty(runs, int)
    for n in range(runs):
        aux = qam.encode_aux_state(aux_amps[n], aux_phases[n], N)   # amplitude INTO the state
        rp = U * tensor(rho, aux) * Ud
        rex = rp.ptrace(1)                                          # exiting aux (D_0)
        SX[n] = float(np.real(expect(sigmax(), rex)))              # passive decode
        SY[n] = float(np.real(expect(sigmay(), rex)))
        POP[n] = float(np.real(expect(_Z11, rex)))
        p0 = max(float(np.real((P0 * rp).tr())), 0.0); p1 = max(1.0 - p0, 0.0)
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
        fast=False, omega_s_res=0.0, fs=None, soft_gain=False):
    """Colour-encoded dependent transform. Amplitude is encoded in the circuit
    (encode_aux_state) and read back from the decoded exiting aux (pop) -> silent
    in stays silent out, no re-injection. freq_scale=False -> flat couplings
    (omega_s hook OFF); True -> couplings x freq/100 (old behaviour).
    amp_gain boosts the encoded population off the |00> vacuum fixed point
    (effective rate ~ g^2*DeltaT*a, gated by input population a). Encode
    clip(a*amp_gain,1), decode, divide back by amp_gain -> energy-fair, still
    silent->silent (0*gain=0); loud bins above 1/amp_gain saturate (compression).
    soft_gain=True -> replace the hard clip with an invertible knee enc=1-exp(-g*a),
    dec=-ln(1-p)/g: same weak-bin lift, NO loud-bin saturation (ordering preserved) ->
    excitation transfer stays visible in the amplitude. omega_s_res>0 -> per-bin system
    precession os = omega_s_res*(freq/fs)*gz (a resonator tuned to the bin frequency);
    detunes the exchange at high frequency (treble goes static/dry) and writes a phase
    carrier where the exchange is still alive."""
    num_frames, num_bins = in_mag.shape
    # GLOBAL normalization (one scale) — per-bin would re-inject each bin's level,
    # which is exactly the amplitude inheritance we are avoiding.
    mag_norm, scale_f = qam.normalize_amplitudes(in_mag)
    scale_per_bin = np.full(num_bins, scale_f)
    out_mag = {v: np.clip(mag_norm.copy(), 0.0, 1.0) for v in _VERSIONS}
    out_phase = {v: in_phase.copy() for v in _VERSIONS}
    diag = {}
    total = len(bin_indices)
    for count, k in enumerate(bin_indices):
        if verbose and (count % max(1, total // 10) == 0 or count == total - 1):
            print(f"  bin {count+1}/{total}  (k={k}, f={freq_bins[k]:.1f} Hz)")
        sc = (freq_bins[k] / 100.0) if freq_scale else 1.0
        gx, gz = gamma_x * sc, gamma_z * sc
        oc, ocz, os_, orr = omega_cont * sc, omega_cont_z * sc, omega_s * sc, omega_r * sc
        if omega_s_res:   # RESONATOR: system precesses at bin freq -> detunes exchange at high f, writes phase carrier
            _fs = fs if fs else 2.0 * float(np.max(freq_bins))
            os_ = os_ + omega_s_res * (freq_bins[k] / _fs) * gz
        # soft_gain: invertible knee 1-exp(-g*a) instead of clip(g*a,1) -> weak-bin lift kept, loud-bin saturation removed
        amps = (1.0 - np.exp(-amp_gain * mag_norm[:, k])) if soft_gain else np.clip(mag_norm[:, k] * amp_gain, 0.0, 1.0)
        phs = in_phase[:, k]
        SX = np.empty((n_traj, num_frames)); SY = np.empty((n_traj, num_frames))
        POP = np.empty((n_traj, num_frames)); MR = np.empty((n_traj, num_frames), int)
        for ti in range(n_traj):
            sx, sy, pop, mr = _dependent_colour_kernel(
                amps, phs, gx, gz, os_, oc, ocz, orr, delay, phase, DeltaT,
                measurement_basis, evuType, base_seed + 10007 * int(k) + ti,
                interaction=interaction, fast=fast)
            SX[ti], SY[ti], POP[ti], MR[ti] = sx, sy, pop, mr
        for v in _VERSIONS:
            amp = np.empty(num_frames); ph = np.empty(num_frames)
            for n in range(num_frames):
                if v == 'uncond':
                    sel = slice(None)
                else:
                    a = 0 if v == 'cond0' else 1
                    cf = n + delay
                    if cf < num_frames:
                        mask = (MR[:, cf] == a); sel = mask if mask.any() else slice(None)
                    else:
                        sel = slice(None)
                amp[n] = POP[sel, n].mean()
                ph[n] = np.arctan2(SY[sel, n].mean(), SX[sel, n].mean())
            if soft_gain:   # invert the soft knee: a = -ln(1-p)/g  (clip p<1 so a fully-excited exit doesn't blow up)
                out_mag[v][:, k] = np.clip(-np.log(1.0 - np.clip(amp, 0.0, 1.0 - 1e-3)) / amp_gain, 0.0, 1.0)
            else:
                out_mag[v][:, k] = np.clip(amp / amp_gain, 0.0, 1.0)   # undo boost -> energy-fair
            out_phase[v][:, k] = ph
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
