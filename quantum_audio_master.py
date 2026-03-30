#!/usr/bin/env python
# coding: utf-8
"""
quantum_audio_master.py
=======================
Unified quantum-inspired audio processor — all four modes in one file.

MODES
-----
  'sensing'        (I)   NV-center quantum sensing
                         audio → phase → qubit Ramsey/Hahn → ancilla readout
  'sms_colour'     (II)  SMS spectral colouring (deterministic)
                         SMS residual bins → encode into fresh aux → decode output
  'sms_trajectory' (III) SMS quantum trajectory (stochastic)
                         SMS spectral data drives system; vacuum aux bath; projective collapse
  'harmony'        (IV)  Quantum tonal brightness
                         Woolhouse TA matrices → qutrit collision → spectral brightness

USAGE
-----
    result = quantum_audio_pipeline(
        'audio/satie.wav',
        mode='sensing',       # or 'sms_colour' | 'sms_trajectory' | 'harmony'
        delay=3,
        phase=np.pi,
    )
    plot_quantum_audio(result, show_details=True)

Output WAV files are named  <stem>_I_sensing.wav,  <stem>_II_colour_combined.wav, etc.

BUGS FIXED vs. original notebooks
----------------------------------
B1  trotterize_nv_sensing_linear removed (called undefined _embed_two_qubit_gate)
B2  trotterize_trajectory delay=0 projector dimension was wrong
B3  save_wav_with_resampling defined before first use
B4  soundfile imported as 'soundfile' (not 'sf') to avoid name clashes
B5  trotterize_trajectory omega_s_static=0 explicit param instead of silent override
B6  debug print removed from harmony pipeline
B7  sin_phases added to all inline analysis dicts
B9  all kernels named trotterize_* (was trotzierize_* in SMS notebook)
B10 step label fixed in trajectory pipeline
B11 local variable 'sf' in compute_stretch_factor renamed to 'stretch'
B12 unnecessary tensor() wrapping of single Qobj removed
B13 harmony pipeline calls sms_analyse_for_quantum instead of inlining it
"""

# ============================================================
# SECTION 1 — IMPORTS
# ============================================================
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import soundfile                               # B4: explicit name, never aliased 'sf'
from scipy.signal import butter, filtfilt, resample, resample_poly, spectrogram as _spectrogram
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from scipy.io import wavfile
from math import gcd
from fractions import Fraction

warnings.filterwarnings('ignore')

from qutip import (
    Qobj, basis, ket2dm, tensor, qeye,
    sigmax, sigmay, sigmaz, destroy,
    expect, fock_dm
)

# sms-tools — path may need adjusting
sys.path.insert(0, '/Users/xxxxx/Documents/Projects/SMS/sms-tools')
from smstools.models import sineModel as SM
from smstools.models import harmonicModel as HM
from smstools.models import stochasticModel as STC
from smstools.models import hpsModel as HPS
from smstools.models import utilFunctions as UF
from smstools.models import stft as STFT


# ============================================================
# SECTION 2 — SHARED QUANTUM PRIMITIVES
# ============================================================

# ---- Gates ----

def gate_Rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return Qobj([[c, -1j*s], [-1j*s, c]])

def gate_Ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return Qobj([[c, -s], [s, c]])

def gate_Rz(phi):
    return Qobj([[np.exp(-1j*phi/2), 0], [0, np.exp(1j*phi/2)]])

def gate_SWAP():
    return Qobj([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dims=[[2,2],[2,2]])

def gate_partial_SWAP(theta):
    I4 = tensor(qeye(2), qeye(2))
    return np.cos(theta)*I4 + 1j*np.sin(theta)*gate_SWAP()

def _embed_gate_2q(gate_2q, idx_a, idx_b, n_qubits):
    """Embed a 2-qubit gate into an n-qubit Hilbert space."""
    dim = 2**n_qubits
    U = np.zeros((dim, dim), dtype=complex)
    gm = gate_2q.full()
    for i in range(dim):
        bits = [(i >> (n_qubits-1-k)) & 1 for k in range(n_qubits)]
        ba, bb = bits[idx_a], bits[idx_b]
        for j in range(4):
            coeff = gm[j, ba*2 + bb]
            if abs(coeff) > 1e-15:
                nb = bits.copy()
                nb[idx_a], nb[idx_b] = (j >> 1) & 1, j & 1
                oi = sum(b << (n_qubits-1-k) for k, b in enumerate(nb))
                U[oi, i] += coeff
    return Qobj(U, dims=[[2]*n_qubits, [2]*n_qubits])

# ---- NV sensing unitaries ----

def _ramsey_unitary_sinphi(phi):
    """Rx(π/2) → Rz(φ) → Ry(-π/2)  →  ⟨σz⟩ = sin(φ)  (linear for small φ)."""
    return gate_Ry(-np.pi/2) * gate_Rz(phi) * gate_Rx(np.pi/2)

def _hahn_echo_unitary(phi_plus, phi_minus):
    """Rx(π/2)→Rz(φ+)→Rx(π)→Rz(φ-)→Ry(-π/2)."""
    return (gate_Ry(-np.pi/2) * gate_Rz(phi_minus) *
            gate_Rx(np.pi) * gate_Rz(phi_plus) * gate_Rx(np.pi/2))

def _qpsd_subcircuit(phi_plus, phi_minus, theta_k, sequence='ramsey'):
    """One QPSD sub-circuit with MW2 phase offset θ_k."""
    MW2 = gate_Rz(-theta_k) * gate_Ry(-np.pi/2) * gate_Rz(theta_k)
    if sequence == 'ramsey':
        return MW2 * gate_Rz(phi_plus) * gate_Rx(np.pi/2)
    else:
        return (MW2 * gate_Rz(phi_minus) * gate_Rx(np.pi) *
                gate_Rz(phi_plus) * gate_Rx(np.pi/2))

# ---- Bloch-sphere encoding / decoding (used by SMS colour mode) ----

def encode_aux_state(amplitude, phase_val, N):
    """Encode (amplitude, phase) → qubit density matrix on Bloch sphere.
    θ = 2·arcsin(√a), |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}·sin(θ/2)|1⟩.
    """
    a = np.clip(float(amplitude), 0.0, 1.0)
    theta = 2.0 * np.arcsin(np.sqrt(a))
    c0 = np.cos(theta / 2.0)
    c1 = np.exp(1j * float(phase_val)) * np.sin(theta / 2.0)
    psi = Qobj([[c0], [c1]])
    return ket2dm(psi)

def decode_aux_state(rho_aux):
    """Extract (amplitude, phase) from output auxiliary density matrix."""
    pop = expect(Qobj([[0, 0], [0, 1]]), rho_aux)
    sx  = expect(sigmax(), rho_aux)
    sy  = expect(sigmay(), rho_aux)
    return float(np.real(pop)), float(np.arctan2(np.real(sy), np.real(sx)))

# ---- Amplitude normalisation helpers ----

def normalize_amplitudes(Xr_mag):
    scale = np.max(np.abs(Xr_mag))
    if scale < 1e-12:
        return np.zeros_like(Xr_mag), 1.0
    return Xr_mag / scale, scale

def normalize_amplitudes_per_bin(Xr_mag):
    scale_per_bin = np.max(Xr_mag, axis=0)
    scale_per_bin[scale_per_bin < 1e-12] = 1.0
    return Xr_mag / scale_per_bin[np.newaxis, :], scale_per_bin

# ---- Initial state helpers ----

def _initial_system_state(evuType):
    """Return single-qubit initial density matrix.  B12: no spurious tensor() wrap."""
    if evuType == 'Pur.Deph':
        return 0.5 * Qobj([[1, 1], [1, 1]])     # |+⟩⟨+|
    return Qobj([[0, 0], [0, 1]])                # |1⟩⟨1|

def _vacuum_delay(delay, dim):
    """Tensor product of `delay` ground-state qubits/qutrits."""
    states = [fock_dm(dim, 0) for _ in range(delay)]
    return tensor(states)

def _initial_state_with_delay(evuType, delay, dim):
    """System ⊗ delay-line initial state."""
    rho_sys = _initial_system_state(evuType)
    if delay > 0:
        return tensor(rho_sys, _vacuum_delay(delay, dim))
    return rho_sys

# ---- Partial-trace index helpers ----

def _keep_indices(delay):
    """Indices to keep after tracing out D_0 (index 1): system + D_1…D_{d-1} + fresh."""
    return [0] + list(range(2, 2 + delay))

def _delay_indices_in_kept(delay):
    """Within the kept state (system, D_1…D_{d-1}, fresh), indices of the new delay."""
    if delay > 1:
        return list(range(1, delay + 1))   # D_1…D_{d-1} + fresh  →  new D_0…D_{d-1}
    return [1]                             # just fresh  →  new D_0

# ---- Gell-Mann operators (qutrit mode) ----

def gell_mann_x_operators():
    """Off-diagonal Gell-Mann matrices: λ₁, λ₄, λ₆."""
    lam1 = Qobj([[0,1,0],[1,0,0],[0,0,0]])
    lam4 = Qobj([[0,0,1],[0,0,0],[1,0,0]])
    lam6 = Qobj([[0,0,0],[0,0,1],[0,1,0]])
    return lam1, lam4, lam6

# ---- TA density matrix (harmony mode) ----

def ta_to_density_matrix(TA, method='symmetrize'):
    """Convert 3×3 Woolhouse TA matrix → valid qutrit density matrix."""
    if TA.shape != (3, 3):
        raise ValueError(f"TA must be 3×3, got {TA.shape}")
    if method == 'symmetrize':
        M = (TA + TA.T) / 2.0
        tr = np.trace(M)
        M = M / tr if tr > 1e-15 else np.eye(3) / 3.0
        vals, vecs = np.linalg.eigh(M)
        vals = np.maximum(vals, 0)
        vals /= vals.sum()
        M = vecs @ np.diag(vals) @ vecs.T
    elif method == 'gram':
        M = TA.T @ TA
        tr = np.trace(M)
        M = M / tr if tr > 1e-15 else np.eye(3) / 3.0
    else:
        raise ValueError(f"Unknown method '{method}'")
    return Qobj(M)


# ============================================================
# SECTION 3 — HAMILTONIANS
# ============================================================

def _get_Hs(omega_s, delay, N, evuType='Pur.Deph'):
    ops = [omega_s * (sigmax() if 'Rot' in evuType else sigmaz())]
    ops += [qeye(N[0])] * delay
    return tensor(ops)

def _get_Hcont(omega_cont, delay, N, evuType='Pur.Deph'):
    ops = [omega_cont * (sigmaz() if 'Rot' in evuType else sigmax())]
    ops += [qeye(N[0])] * delay
    return tensor(ops)

def _get_Hcont_z(omega_cont_z, delay, N, evuType='Pur.Deph'):
    ops = [omega_cont_z * (sigmax() if 'Rot' in evuType else sigmaz())]
    ops += [qeye(N[0])] * delay
    return tensor(ops)

def _get_Hr(omega_r, b, delay, N):
    if delay == 0:
        return 0
    H_r = 0
    for k in range(delay):
        row = [qeye(2)]
        for l in range(delay):
            row.append(omega_r * sigmaz() if l == k else qeye(N[0]))
        H_r = H_r + tensor(row)
    return H_r

def _get_SPbHam(omega_s, omega_cont, omega_cont_z, omega_r, b, delay, N,
                evuType='Pur.Deph'):
    """System + pump + delay free Hamiltonians, tensored with fresh-aux identity."""
    H = (_get_Hs(omega_s, delay, N, evuType) +
         _get_Hcont(omega_cont, delay, N, evuType) +
         _get_Hcont_z(omega_cont_z, delay, N, evuType) +
         _get_Hr(omega_r, b, delay, N))
    return tensor(H, qeye(N[0]))

def _get_interaction_ham(gamma_x, gamma_z, delay, phase, N):
    """σx⊗σx + σz⊗σz coupling to fresh aux + feedback to D_0."""
    dim = N[0]
    # Fresh (last index)
    ops_xf = [gamma_x * sigmax()] + [qeye(dim)] * delay + [sigmax()]
    ops_zf = [gamma_z * sigmaz()] + [qeye(dim)] * delay + [sigmaz()]
    V = tensor(ops_xf) + tensor(ops_zf)
    # Delayed (D_0, index 1) with feedback phase
    if delay > 0:
        fb = -np.exp(-1j * phase)
        ops_xd = [fb * gamma_x * sigmax(), sigmax()] + [qeye(dim)] * (delay-1) + [qeye(dim)]
        ops_zd = [fb * gamma_z * sigmaz(), sigmaz()] + [qeye(dim)] * (delay-1) + [qeye(dim)]
        V = V + tensor(ops_xd) + tensor(ops_zd)
    return V

def _build_qutrit_interaction(g, delay, phase):
    """Gell-Mann coupling for qutrit collision model."""
    dim = 3
    lambdas = gell_mann_x_operators()
    V = 0
    for lam in lambdas:
        ops = [g * lam] + [qeye(dim)] * delay + [lam]
        V = V + tensor(ops)
    if delay > 0:
        fb = -np.exp(-1j * phase)
        for lam in lambdas:
            ops = [fb * g * lam, lam] + [qeye(dim)] * (delay-1) + [qeye(dim)]
            V = V + tensor(ops)
    return V

def _build_qutrit_free_ham(omega, delay):
    """H = ω·diag(0,1,2) on system qutrit."""
    dim = 3
    H_sys = omega * Qobj([[0,0,0],[0,1,0],[0,0,2]])
    ops = [H_sys] + [qeye(dim)] * delay + [qeye(dim)]
    return tensor(ops)


# ============================================================
# SECTION 4 — AUDIO I/O
# ============================================================

def load_wav(filepath, max_freq=2000):
    """Load wav, convert to mono, low-pass, peak-normalise."""
    data, sr = soundfile.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]
    if max_freq < sr / 2.0:
        b, a = butter(5, max_freq / (sr / 2.0), btype='low')
        data = filtfilt(b, a, data)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak
    return data, sr

def load_wav_full(filepath):
    """Load wav, mono, float normalised — no low-pass (for SMS analysis)."""
    data, sr = soundfile.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.9
    return data, sr

def save_wav(filepath, signal, sr, normalise=True):
    """Save wav, optionally peak-normalise to 0.8."""
    s = signal.copy().astype(np.float64)
    if normalise:
        p = np.max(np.abs(s))
        if p > 0:
            s = s / p * 0.8
    soundfile.write(filepath, s.astype(np.float32), sr)
    return filepath

def save_wav_with_resampling(filepath, signal, sr, target_sr=44100, normalise=True):
    """Save wav, upsampling to target_sr if needed."""          # B3: defined early
    if sr < target_sr:
        duration = len(signal) / sr
        t_old = np.linspace(0, duration, len(signal))
        t_new = np.linspace(0, duration, int(duration * target_sr))
        signal = interp1d(t_old, signal, kind='linear', fill_value='extrapolate')(t_new)
        sr = target_sr
    s = signal.copy().astype(np.float64)
    if normalise:
        p = np.max(np.abs(s))
        if p > 0:
            s = s / p * 0.8
    soundfile.write(filepath, s.astype(np.float32), sr)
    return filepath

def compute_stretch_factor(audio_max_freq, sensing_rate):
    """Return time-stretch factor so signal fits in sensing bandwidth."""
    nyq = sensing_rate / 2.0
    stretch = max(1.0, np.ceil(audio_max_freq / nyq))   # B11: renamed from 'sf'
    return stretch, {'nyquist': nyq, 'stretch': stretch,
                     'compressed_bw': audio_max_freq / stretch}

def time_stretch(signal, factor):
    n_out = int(len(signal) * factor)
    return np.interp(np.linspace(0, len(signal)-1, n_out),
                     np.arange(len(signal)), signal)

def time_compress(signal, factor):
    return resample(signal, max(int(len(signal) / factor), 2))

def _output_path(audio_path, mode, suffix=''):
    """Build output filename: <stem>_<roman>_<mode>[_suffix].wav."""
    from pathlib import Path
    stem = Path(audio_path).stem
    labels = {
        'sensing':        'I_sensing',
        'sms_colour':     'II_colour',
        'sms_trajectory': 'III_trajectory',
        'harmony':        'IV_harmony',
    }
    label = labels.get(mode, mode)
    name = f"{stem}_{label}"
    if suffix:
        name += f"_{suffix}"
    return str(Path(audio_path).parent / (name + '.wav'))


# ============================================================
# SECTION 5 — SMS ANALYSIS & SYNTHESIS
# ============================================================

def sms_analyse(x, fs, M=2501, N=4096, H=1024, t=-80, nH=60,
                minf0=100, maxf0=400, f0et=5,
                harmDevSlope=0.01, minSineDur=0.02,
                Ns=2048, stocf=0.2):
    """HPS analysis → residual STFT → unified analysis dict.
    Returns all fields needed by both colour and trajectory modes.
    """
    w = np.blackman(M)
    hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(
        x, fs, w, N, H, t, nH, minf0, maxf0, f0et,
        harmDevSlope, minSineDur, Ns, stocf)

    xr = UF.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)

    N_stoc = H * 2
    hN = N_stoc // 2 + 1
    stft_w = hann(N_stoc)
    Xr_mag_dB_norm, Xr_phase = STFT.stftAnal(xr, stft_w, N_stoc, H)

    win_norm_dB = 20.0 * np.log10(np.sum(stft_w))
    Xr_mag_dB_unnorm = Xr_mag_dB_norm + win_norm_dB

    nf = min(Xr_mag_dB_norm.shape[0], Xr_phase.shape[0],
             stocEnv.shape[0], hfreq.shape[0])

    Xr_mag_dB_norm  = Xr_mag_dB_norm[:nf]
    Xr_phase        = Xr_phase[:nf]
    Xr_mag          = 10.0 ** (Xr_mag_dB_norm / 20.0)           # normalised linear
    Xr_mag_stoc     = 10.0 ** (Xr_mag_dB_unnorm[:nf] / 20.0)    # unnorm linear (synth)
    freq_bins       = np.arange(hN) * fs / float(N_stoc)

    sin_freqs  = hfreq[:nf].copy()
    sin_amps   = 10.0 ** (hmag[:nf] / 20.0)
    sin_phases = hphase[:nf].copy()
    sin_freqs[sin_freqs <= 0] = np.nan
    sin_amps[np.isnan(sin_freqs)] = 0.0

    print(f"  HPS: {nf} frames × {hN} bins | "
          f"freq res {fs/N_stoc:.1f} Hz | frame rate {fs/H:.1f} Hz")

    return dict(
        hfreq=hfreq, hmag=hmag, hphase=hphase,
        stocEnv=stocEnv, xr=xr,
        Xr_mag=Xr_mag, Xr_mag_stoc=Xr_mag_stoc,
        Xr_mag_dB=Xr_mag_dB_norm,
        Xr_phase=Xr_phase,
        freq_bins=freq_bins,
        num_frames=nf, hN=hN, N_stoc=N_stoc,
        sin_freqs=sin_freqs, sin_amps=sin_amps, sin_phases=sin_phases,   # B7
        win_norm_dB=win_norm_dB,
        H=H, Ns=Ns, N=N, fs=fs,
    )

def make_quantum_stocEnv(quantum_mag, num_frames, win_norm_dB=0.0):
    safe = np.maximum(quantum_mag[:num_frames], 1e-10)
    return 20.0 * np.log10(safe) + win_norm_dB

def make_quantum_phase_func(quantum_phase):
    def phase_func(freq_bins, frame_idx, fs_inner):
        if frame_idx < quantum_phase.shape[0]:
            return quantum_phase[frame_idx]
        return 2 * np.pi * np.random.rand(len(freq_bins))
    return phase_func

def synthesise_sms(ana, res_mag=None, res_phase=None,
                   sin_amps_mod=None, use_quantum_amp=True,
                   use_quantum_phase=True, softclip=True,
                   win_norm_dB_override=None):
    """Synthesise audio from HPS parameters with optional quantum modifications."""
    H       = ana['H']
    N       = ana['N']
    N_stoc  = ana['N_stoc']
    nf      = ana['num_frames']
    win_dB  = win_norm_dB_override if win_norm_dB_override is not None else ana['win_norm_dB']

    # Harmonics
    hfreq, hmag, hphase = ana['hfreq'], ana['hmag'], ana['hphase']
    if sin_amps_mod is not None:
        hmag_use = hmag.copy()
        nf2 = min(nf, hmag.shape[0], sin_amps_mod.shape[0])
        for n in range(nf2):
            for tr in range(hfreq.shape[1]):
                if hfreq[n, tr] > 0 and sin_amps_mod[n, tr] > 0:
                    hmag_use[n, tr] = 20.0 * np.log10(
                        np.clip(sin_amps_mod[n, tr], 1e-10, None))
        y_harm = SM.sineModelSynth(hfreq, hmag_use, hphase, N, H, ana['fs'])
    else:
        y_harm = SM.sineModelSynth(hfreq, hmag, hphase, N, H, ana['fs'])

    y_harm_orig = SM.sineModelSynth(hfreq, hmag, hphase, N, H, ana['fs'])

    # Stochastic — quantum path
    if use_quantum_amp and res_mag is not None:
        stocEnv_q = make_quantum_stocEnv(res_mag, nf, win_dB)
    else:
        stocEnv_q = ana['Xr_mag_dB'][:nf] + win_dB

    phase_fn_q = (make_quantum_phase_func(res_phase)
                  if use_quantum_phase and res_phase is not None
                  else make_quantum_phase_func(ana['Xr_phase']))

    y_stoc_q = STC.stochasticModelSynth(
        stocEnv_q, H, N_stoc, ana['fs'], melScale=0, phase_func=phase_fn_q)

    # Stochastic — original path
    y_stoc_o = STC.stochasticModelSynth(
        ana['stocEnv'], H, N_stoc, ana['fs'], melScale=1, phase_func=None)

    min_len = min(len(y_harm), len(y_harm_orig), len(y_stoc_q), len(y_stoc_o))

    y_q = y_harm[:min_len] + y_stoc_q[:min_len]
    y_o = y_harm_orig[:min_len] + y_stoc_o[:min_len]

    # Clipping / normalisation
    y_q = _fix_clipping(y_q, ana['fs'])
    if softclip:
        y_q = np.where(np.abs(y_q) > 0.9, np.sign(y_q) * 0.9, y_q)

    return dict(
        y_quantum=y_q,
        y_original=y_o,
        y_harm_quantum=y_harm[:min_len],
        y_harm_original=y_harm_orig[:min_len],
        y_stoc_quantum=y_stoc_q[:min_len],
        y_stoc_original=y_stoc_o[:min_len],
    )

def _fix_clipping(signal, fs):
    peak = np.max(np.abs(signal))
    if peak <= 1.0:
        return signal
    over = np.abs(signal) > 1.0
    margin = int(0.5 * fs)
    over_exp = over.copy()
    for idx in np.where(over)[0]:
        lo, hi = max(0, idx - margin), min(len(signal), idx + margin)
        over_exp[lo:hi] = True
    diff = np.diff(over_exp.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1
    if over_exp[0]:  starts = np.insert(starts, 0, 0)
    if over_exp[-1]: ends   = np.append(ends, len(signal))
    for s, e in zip(starts, ends):
        rp = np.max(np.abs(signal[s:e]))
        if rp > 0.95:
            signal[s:e] = signal[s:e] / rp * 0.9
    return signal

# ---- SMS bin-selection helpers ----

def get_stochastic_only_bins(sin_freqs, freq_bins, tolerance_hz=None):
    num_bins = len(freq_bins)
    if tolerance_hz is None:
        tolerance_hz = float(freq_bins[1] - freq_bins[0]) if num_bins > 1 else 1.0
    occupied = np.zeros(num_bins, dtype=bool)
    for frame in range(sin_freqs.shape[0]):
        for track in range(sin_freqs.shape[1]):
            f = sin_freqs[frame, track]
            if np.isnan(f) or f <= 0:
                continue
            occupied |= (np.abs(freq_bins - f) <= tolerance_hz)
    return ~occupied

def get_stochastic_bins_with_peaks(sin_freqs, sin_phases, freq_bins, tolerance_hz=None):
    """Return mask + nearest-peak-phase and delta_f arrays for trajectory mode."""
    num_frames, num_bins = sin_freqs.shape[0], len(freq_bins)
    stoch_mask = get_stochastic_only_bins(sin_freqs, freq_bins, tolerance_hz)
    nearest_phase = np.zeros((num_frames, num_bins))
    delta_f       = np.zeros((num_frames, num_bins))

    for n in range(num_frames):
        active = (~np.isnan(sin_freqs[n])) & (sin_freqs[n] > 0)
        af = sin_freqs[n, active]
        ap = sin_phases[n, active]
        if len(af) == 0:
            continue
        for k in range(num_bins):
            dists = np.abs(af - freq_bins[k])
            idx = np.argmin(dists)
            nearest_phase[n, k] = ap[idx]
            delta_f[n, k]       = freq_bins[k] - af[idx]

    delta_f_max = np.max(np.abs(delta_f), axis=0)
    delta_f_max[delta_f_max < 1e-6] = 1.0
    return stoch_mask, nearest_phase, delta_f, delta_f_max

def _compute_sms_quantum_params(freq_hz, lam=1.0/100.0):
    omega = lam * freq_hz
    return dict(omega_s=omega, omega_r=omega, gamma_x=0.1, gamma_z=0.1)


# ============================================================
# SECTION 6 — WOOLHOUSE HARMONY FUNCTIONS (mode IV)
# ============================================================

def pitch_distance_matrix(X, Y):
    X, Y = np.array(X), np.array(Y)
    return np.abs(Y[np.newaxis, :] - X[:, np.newaxis])

def interval_cycle_matrix(PD):
    IC = np.zeros_like(PD, dtype=float)
    for i in range(PD.shape[0]):
        for j in range(PD.shape[1]):
            d = int(PD[i, j]) % 12
            IC[i, j] = 1.0 if d == 0 else 12.0 / gcd(d, 12)
    return IC

def voice_leading_matrix(PD, alpha=50.0):
    if alpha == np.inf:
        return np.ones_like(PD, dtype=float)
    return alpha / (PD + alpha)

def root_salience_matrix(X, Y, root_X=None, root_Y=None, beta=2.0, gamma_rs=4.0):
    RS = np.ones((len(X), len(Y)))
    if root_X is not None: RS[root_X, :] *= beta
    if root_Y is not None: RS[:, root_Y] *= gamma_rs
    total = RS.sum()
    return RS / total if total > 0 else RS

def _is_dissonant(chord):
    pcs = [p % 12 for p in chord]
    for i in range(len(pcs)):
        for j in range(i+1, len(pcs)):
            d = abs(pcs[i] - pcs[j])
            if min(d, 12-d) in [1, 2, 6]:
                return True
    return False

def consonance_dissonance_scale(X, Y, RS3, delta=0.1):
    dis_X, dis_Y = _is_dissonant(X), _is_dissonant(Y)
    if dis_X and not dis_Y:
        return RS3 * (1 + delta)
    elif not dis_X and dis_Y:
        return RS3 * (1 - delta)
    return RS3.copy()

def tonal_attraction_matrix(X, Y, root_X=None, root_Y=None,
                             alpha=50.0, beta=2.0, gamma_rs=4.0, delta=0.1,
                             use_voice_leading=True, use_root_salience=True,
                             use_consonance_dissonance=True):
    PD = pitch_distance_matrix(X, Y)
    IC = interval_cycle_matrix(PD)
    VL = voice_leading_matrix(PD, alpha) if use_voice_leading else np.ones_like(PD)
    ICVL = IC * VL
    if use_root_salience:
        RS3 = root_salience_matrix(X, Y, root_X, root_Y, beta, gamma_rs)
    else:
        RS3 = np.ones((len(X), len(Y))) / (len(X) * len(Y))
    CD = consonance_dissonance_scale(X, Y, RS3, delta) if use_consonance_dissonance else RS3.copy()
    TA = ICVL * CD
    A  = TA.sum() / 12.0
    return TA, A, dict(PD=PD, IC=IC, VL=VL, ICVL=ICVL, RS3=RS3, CD=CD, TA=TA, A=A)

def score_to_density_chain(events, **woolhouse_kwargs):
    rho_chain, ta_chain, a_chain, pairs = [], [], [], []
    for i in range(len(events) - 1):
        ev0, ev1 = events[i], events[i+1]
        TA, A, _ = tonal_attraction_matrix(
            ev0['midi'], ev1['midi'],
            root_X=ev0.get('root_index', 0),
            root_Y=ev1.get('root_index', 0),
            **woolhouse_kwargs)
        rho_chain.append(ta_to_density_matrix(TA))
        ta_chain.append(TA)
        a_chain.append(A)
        pairs.append((ev0, ev1))
    return rho_chain, ta_chain, a_chain, pairs

TRIAD_TEMPLATES = {
    (0,4,7): 'major', (0,3,7): 'minor',
    (0,3,6): 'diminished', (0,4,8): 'augmented',
}
SEVENTH_TEMPLATES = {
    (0,4,7,11): 'major7',   (0,4,7,10): 'dominant7',
    (0,3,7,10): 'minor7',   (0,3,6,10): 'half-dim7',
    (0,3,6,9):  'diminished7',
}

def midi_to_note_name(midi):
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[midi % 12]}{midi // 12 - 1}"

def detect_chord_root(midi_notes):
    pcs_unique = sorted(set([m % 12 for m in midi_notes]))
    templates  = SEVENTH_TEMPLATES if len(pcs_unique) >= 4 else TRIAD_TEMPLATES
    for rotation in range(len(pcs_unique)):
        root_pc   = pcs_unique[rotation]
        intervals = tuple(sorted([(pc - root_pc) % 12 for pc in pcs_unique]))
        if intervals in templates:
            chord_type   = templates[intervals]
            sorted_notes = sorted(midi_notes)
            root_index   = next((i for i, m in enumerate(sorted_notes)
                                 if m % 12 == root_pc), 0)
            return root_index, chord_type, sorted_notes[root_index]
    sorted_notes = sorted(midi_notes)
    return 0, 'unknown', sorted_notes[0]

def extract_chord_events(audio_path, chord_size=3, onset_tolerance=0.05,
                          min_velocity=0, verbose=True):
    from basic_pitch.inference import predict
    from itertools import combinations
    if verbose:
        print(f"  basic-pitch on: {audio_path}")
    _, _, note_events = predict(str(audio_path))
    note_events = [ne for ne in note_events if ne[3] >= min_velocity]
    if verbose:
        print(f"  {len(note_events)} notes after velocity filter")

    groups = {}
    for start, end, pitch, velocity, *_ in note_events:
        found = next((k for k in groups if abs(start - k) < onset_tolerance), None)
        key = found if found is not None else start
        groups.setdefault(key, []).append(
            dict(start=start, end=end, pitch=int(pitch), velocity=velocity))

    events = []
    for onset in sorted(groups):
        notes = groups[onset]
        if len(notes) < chord_size:
            continue
        notes.sort(key=lambda n: n['pitch'])
        seen, unique = set(), []
        for n in notes:
            pc = n['pitch'] % 12
            if pc not in seen:
                seen.add(pc)
                unique.append(n)
        if len(unique) < chord_size:
            continue
        midi_all  = [n['pitch'] for n in unique]
        best_combo, best_type = None, 'unknown'
        for combo in combinations(range(len(midi_all)), chord_size):
            mc = [midi_all[i] for i in combo]
            _, ctype, _ = detect_chord_root(mc)
            if ctype != 'unknown':
                best_combo, best_type = mc, ctype
                break
        if best_combo is None:
            best_combo = [n['pitch'] for n in unique[:chord_size]]
        midi_sorted = sorted(best_combo)
        root_idx, chord_type, root_midi = detect_chord_root(midi_sorted)
        min_end  = min(n['end'] for n in notes)
        duration = max(0.1, min_end - onset)
        events.append(dict(
            onset=onset, duration=duration, midi=midi_sorted,
            root_index=root_idx, chord_type=chord_type,
            root_name=midi_to_note_name(root_midi)))
    if verbose:
        print(f"  {len(events)} chord events extracted")
    return events, note_events

def identify_voice_harmonics(event, n_harmonics=8):
    voice_freqs = []
    for midi_note in event['midi']:
        f0 = 440.0 * 2**((midi_note - 69) / 12.0)
        voice_freqs.append([f0 * h for h in range(1, n_harmonics+1)])
    return voice_freqs

def identify_voice_bins(event, freq_bins, n_harmonics=8):
    voice_bins = []
    for midi_note in event['midi']:
        f0 = 440.0 * 2**((midi_note - 69) / 12.0)
        bins = []
        for h in range(1, n_harmonics+1):
            fh = f0 * h
            if fh > freq_bins[-1]:
                break
            bins.append(int(np.argmin(np.abs(freq_bins - fh))))
        voice_bins.append(bins)
    return voice_bins

def apply_brightness_modulation(stoc_mag, freq_bins, events, winners, diagonals,
                                 brightness=1.0, n_harmonics=8,
                                 sin_freqs=None, sin_amps=None):
    out_stoc  = stoc_mag.copy()
    out_sin   = sin_amps.copy() if sin_amps is not None else None
    num_frames = stoc_mag.shape[0]
    mod_log    = []

    for t in range(len(winners)):
        ev     = events[t + 1]
        onset  = ev['onset']
        next_o = events[t+2]['onset'] if t+2 < len(events) else onset + ev.get('duration', 1.0)
        sr_ev  = ev.get('_sr', 44100)
        hop_ev = ev.get('_hop', 512)
        fs     = max(0, min(int(onset * sr_ev / hop_ev), num_frames))
        fe     = max(fs,  min(int(next_o * sr_ev / hop_ev), num_frames))

        w     = winners[t]
        prob  = diagonals[t, w]
        boost = 1.0 + brightness * max(0, prob - 1.0/3.0)

        voice_harmonics = identify_voice_harmonics(ev, n_harmonics)
        stoc_boosted = sin_boosted = 0

        if w < len(voice_harmonics):
            vbins = identify_voice_bins(ev, freq_bins, n_harmonics)
            if w < len(vbins):
                for k in vbins[w]:
                    if 0 <= k < stoc_mag.shape[1]:
                        out_stoc[fs:fe, k] *= boost
                        stoc_boosted += 1
            if sin_freqs is not None and out_sin is not None:
                tol = freq_bins[1] - freq_bins[0] if len(freq_bins) > 1 else 20.0
                target_freqs = voice_harmonics[w]
                for frame in range(fs, fe):
                    if frame >= sin_freqs.shape[0]:
                        break
                    for tr in range(sin_freqs.shape[1]):
                        f = sin_freqs[frame, tr]
                        if np.isnan(f) or f <= 0:
                            continue
                        if any(abs(f - fh) < tol for fh in target_freqs):
                            out_sin[frame, tr] *= boost
                            sin_boosted += 1

        mod_log.append(dict(
            transition=t, winner=w,
            winner_name=['root','3rd','5th'][w] if w < 3 else f'voice{w}',
            probability=prob, boost_factor=boost,
            frames=(fs, fe), event_midi=ev['midi'],
            stoc_bins_boosted=stoc_boosted, sin_tracks_boosted=sin_boosted))

    return out_stoc, out_sin, mod_log

def synthesise_brightened_audio(analysis, brightened_stoc_mag, fs,
                                 brightened_sin_amps=None):
    from smstools.models import sineModel as SM
    from smstools.models import stochasticModel as STC

    H      = analysis['H']
    N      = analysis['N']
    N_stoc = analysis['N_stoc']
    nf     = analysis['num_frames']
    win_dB = analysis['win_norm_dB']

    hfreq, hphase = analysis['hfreq'], analysis['hphase']
    if brightened_sin_amps is not None:
        hmag_mod = analysis['hmag'].copy()
        nf2 = min(nf, hmag_mod.shape[0], brightened_sin_amps.shape[0])
        for n in range(nf2):
            for tr in range(hfreq.shape[1]):
                if hfreq[n,tr] > 0 and brightened_sin_amps[n,tr] > 0:
                    hmag_mod[n,tr] = 20.0 * np.log10(
                        np.clip(brightened_sin_amps[n,tr], 1e-10, None))
        y_harm = SM.sineModelSynth(hfreq, hmag_mod, hphase, N, H, fs)
    else:
        y_harm = SM.sineModelSynth(hfreq, analysis['hmag'], hphase, N, H, fs)

    safe_mag = np.maximum(brightened_stoc_mag[:nf], 1e-10)
    stocEnv  = 20.0 * np.log10(safe_mag) + win_dB
    orig_ph  = analysis['Xr_phase'][:nf]

    def phase_func(fb, fidx, fs_in):
        return orig_ph[fidx] if fidx < orig_ph.shape[0] else 2*np.pi*np.random.rand(len(fb))

    y_stoc = STC.stochasticModelSynth(stocEnv, H, N_stoc, fs, melScale=0)
    ml     = min(len(y_harm), len(y_stoc))
    y      = y_harm[:ml] + y_stoc[:ml]
    peak   = np.max(np.abs(y))
    if peak > 0.95:
        y = y / peak * 0.9
    return y, y_harm[:ml], y_stoc[:ml]


# ============================================================
# SECTION 7 — NV SENSING HELPERS (mode I)
# ============================================================

def _switching_function(n_sub, sequence):
    g = np.ones(n_sub)
    if sequence == 'hahn_echo':
        g[n_sub//2:] = -1.0
    return g

def _compute_phases_from_signal(B_ac, hires_sr, T_phi, T_seq,
                                 n_cycles, sequence='ramsey'):
    dt    = 1.0 / hires_sr
    n_sub = max(int(T_phi * hires_sr), 2)
    n_seq = max(int(T_seq * hires_sr), 1)
    g     = _switching_function(n_sub, sequence)
    half  = n_sub // 2

    phases = np.zeros(n_cycles)
    phi_p  = np.zeros(n_cycles)
    phi_m  = np.zeros(n_cycles)

    for N in range(n_cycles):
        i0 = N * n_seq
        i1 = i0 + n_sub
        if i1 > len(B_ac):
            seg = np.zeros(n_sub)
            av  = max(len(B_ac) - i0, 0)
            if av > 0:
                seg[:av] = B_ac[i0:i0+av]
        else:
            seg = B_ac[i0:i1]
        phases[N] = np.sum(g * seg) * dt
        phi_p[N]  = np.sum(seg[:half]) * dt
        phi_m[N]  = np.sum(seg[half:]) * dt

    return phases, phi_p, phi_m

def _demodulate_qpsd(sub_samples, delta_f, dt_sub):
    N   = len(sub_samples)
    k   = np.arange(N)
    ref = 2 * np.pi * delta_f * k * dt_sub
    I   = (2.0/N) * np.sum(sub_samples * np.sin(ref))
    Q   = (2.0/N) * np.sum(sub_samples * np.cos(ref))
    return np.arctan2(Q, I), np.sqrt(I**2 + Q**2)


# ============================================================
# SECTION 8 — TROTTERIZE KERNELS
# ============================================================

# ---- 8a. MODE I — NV sensing ----

def trotterize_sensing(B_ac_hires, hires_sr, T_phi, T_seq,
                        phase_scale=0.5,
                        delay=0, swap_angle=np.pi/2,
                        sequence='ramsey',
                        method='sin_phi',
                        N_mod=10, delta_f=None):
    """
    NV-center quantum sensing.

    method='sin_phi'  : one Ramsey/Hahn circuit per sample, readout sin(φ)
    method='qpsd'     : N_mod QPSD sub-circuits per sample, IQ demodulation

    Returns dict with 'phases', 'aux_sigma_z' or 'qpsd_phases', 'tlist'.
    """
    T_seq_ = 1.0 / (1.0 / T_seq)   # keep as float
    gamma_eff = phase_scale / T_phi

    if method == 'sin_phi':
        n_cycles = max(int(len(B_ac_hires) / (hires_sr * T_seq)), 1)
        phases, pp, pm = _compute_phases_from_signal(
            B_ac_hires * gamma_eff, hires_sr, T_phi, T_seq, n_cycles, sequence)

        U_swap     = gate_partial_SWAP(swap_angle)
        U_swap_dag = U_swap.dag()

        if delay > 0:
            rho_delay = _vacuum_delay(delay, 2)

        aux_sz = np.zeros(n_cycles)

        for n in range(n_cycles):
            if sequence == 'ramsey':
                U = _ramsey_unitary_sinphi(phases[n])
            else:
                U = _hahn_echo_unitary(pp[n], pm[n])

            rho = ket2dm(basis(2, 0))
            rho = U * rho * U.dag()
            anc = ket2dm(basis(2, 0))

            if delay == 0:
                rho_sa   = tensor(rho, anc)
                rho_post = U_swap * rho_sa * U_swap_dag
                aux_sz[n] = expect(sigmaz(), rho_post.ptrace(1)).real
            else:
                tq        = 1 + delay + 1
                rho_full  = tensor(rho, rho_delay, anc)
                Usw       = _embed_gate_2q(U_swap, 0, tq-1, tq)
                rho_post  = Usw * rho_full * Usw.dag()
                aux_sz[n] = expect(sigmaz(), rho_post.ptrace(1)).real
                rn = rho_post.ptrace(_keep_indices(delay))
                rho_delay = rn.ptrace(_delay_indices_in_kept(delay))

        return dict(phases=phases, aux_sigma_z=aux_sz,
                    tlist=np.arange(n_cycles) * T_seq,
                    sensing_rate=1.0/T_seq, method='sin_phi')

    elif method == 'qpsd':
        n_total = int(len(B_ac_hires) / (hires_sr * T_seq))
        n_cycles = max(n_total // N_mod, 1)
        dt_sub   = 2 * T_seq
        if delta_f is None:
            delta_f = 1.0 / (N_mod * dt_sub)

        all_phases, all_pp, all_pm = _compute_phases_from_signal(
            B_ac_hires * gamma_eff, hires_sr, T_phi, T_seq,
            n_cycles * N_mod, sequence)

        U_swap     = gate_partial_SWAP(swap_angle)
        U_swap_dag = U_swap.dag()

        if delay > 0:
            rho_delay = _vacuum_delay(delay, 2)

        qpsd_phases = np.zeros(n_cycles)
        qpsd_amps   = np.zeros(n_cycles)
        true_phases = np.zeros(n_cycles)

        for N in range(n_cycles):
            subs = np.zeros(N_mod)
            for k in range(N_mod):
                si     = N * N_mod + k
                theta_k = 2 * np.pi * delta_f * k * dt_sub
                pp_v    = all_phases[si] if si < len(all_phases) else 0.0
                pp_val  = all_pp[si]     if si < len(all_pp)     else 0.0
                pm_val  = all_pm[si]     if si < len(all_pm)     else 0.0

                U   = _qpsd_subcircuit(pp_v if sequence=='ramsey' else pp_val,
                                       pm_val, theta_k, sequence)
                rho = ket2dm(basis(2, 0))
                rho = U * rho * U.dag()
                anc = ket2dm(basis(2, 0))

                if delay == 0:
                    rho_sa    = tensor(rho, anc)
                    rho_post  = U_swap * rho_sa * U_swap_dag
                    subs[k]   = expect(sigmaz(), rho_post.ptrace(1)).real
                else:
                    tq       = 1 + delay + 1
                    rho_full = tensor(rho, rho_delay, anc)
                    Usw      = _embed_gate_2q(U_swap, 0, tq-1, tq)
                    rho_post = Usw * rho_full * Usw.dag()
                    subs[k]  = expect(sigmaz(), rho_post.ptrace(1)).real
                    rn = rho_post.ptrace(_keep_indices(delay))
                    rho_delay = rn.ptrace(_delay_indices_in_kept(delay))

            phi_d, amp_d   = _demodulate_qpsd(subs, delta_f, dt_sub)
            qpsd_phases[N] = phi_d
            qpsd_amps[N]   = amp_d
            i0, i1 = N*N_mod, min(N*N_mod+N_mod, len(all_phases))
            true_phases[N] = np.mean(all_phases[i0:i1]) if i1 > i0 else 0.0

        return dict(phases=true_phases, qpsd_phases=qpsd_phases,
                    qpsd_amplitudes=qpsd_amps,
                    tlist=np.arange(n_cycles) * N_mod * dt_sub,
                    sensing_rate=1.0/T_seq,
                    phase_readout_rate=1.0/(N_mod*dt_sub),
                    method='qpsd')
    else:
        raise ValueError(f"Unknown sensing method '{method}'")


# ---- 8b. MODE II — SMS colour (deterministic) ----

def trotterize_colour(DeltaT, omega_s, omega_cont, omega_cont_z, omega_r,
                      gamma_x, gamma_z, delay, N, phase,
                      aux_amplitudes, aux_phases,
                      evuType='Pur.Deph'):
    """
    Collision model — SMS spectral data encoded in fresh auxiliaries.
    System memory accumulates through the delay line.
    Output: decoded (amplitude, phase) of exiting auxiliary per step.
    """
    runs      = len(aux_amplitudes)
    indexlist = _keep_indices(delay)
    b         = destroy(N[0])
    i_unit    = 1j

    H_free = _get_SPbHam(omega_s, omega_cont, omega_cont_z, omega_r,
                          b, delay, N, evuType)
    V_int  = _get_interaction_ham(gamma_x, gamma_z, delay, phase, N)
    U      = (-i_unit * DeltaT * (H_free + V_int)).expm()
    Udag   = U.dag()

    rho = _initial_state_with_delay(evuType, delay, N[0])

    sys_sx, sys_sz, sys_pop = [], [], []
    amp_out, ph_out = [], []

    for n in range(runs):
        rho_s = rho.ptrace(0)
        sys_sx.append(float(np.real(expect(sigmax(),  rho_s))))
        sys_sz.append(float(np.real(expect(sigmaz(),  rho_s))))
        sys_pop.append(float(np.real(expect(Qobj([[0,0],[0,1]]), rho_s))))

        aux = encode_aux_state(aux_amplitudes[n], aux_phases[n], N)
        rho_e = tensor(rho, aux)
        rho_e_post = U * rho_e * Udag

        aux_out = rho_e_post.ptrace(1)
        amp, ph = decode_aux_state(aux_out)
        amp_out.append(amp)
        ph_out.append(ph)

        rho = rho_e_post.ptrace(indexlist)

    return dict(
        sys_sigmax=np.array(sys_sx),
        sys_sigmaz=np.array(sys_sz),
        sys_pop=np.array(sys_pop),
        aux_amp_out=np.array(amp_out),
        aux_phase_out=np.array(ph_out),
    )


# ---- 8c. MODE III — SMS trajectory (stochastic) ----

def trotterize_trajectory(DeltaT, omega_s, omega_r,
                           gamma_x, gamma_z, delay, N, phase,
                           drive_amplitudes, drive_phases,
                           omega_drive_xy=1.0,
                           omega_drive_z=1.0,
                           omega_s_static=0,       # B5: explicit param, default=0
                           measurement_basis='z',
                           evuType='Pur.Deph',
                           seed=None):
    """
    Quantum trajectory — system is driven by time-dependent field,
    delay line with white-noise auxiliaries, projective measurement
    on exiting auxiliary.

    omega_s_static : frequency for the system's static free Hamiltonian.
                     Default 0 — all precession comes from drive terms.
                     omega_s (parameter) is used for scaling by the caller
                     but does not appear in H_static unless you set omega_s_static != 0.
    """
    rng  = np.random.RandomState(seed)
    runs = len(drive_amplitudes)
    dim  = N[0]
    b    = destroy(dim)
    i_u  = 1j

    # Static unitary: interaction + free (omega_s_static defaults to 0 → B5 fixed)
    H_free   = _get_SPbHam(omega_s_static, 0.0, 0.0, omega_r,
                            b, delay, N, evuType)
    V_int    = _get_interaction_ham(gamma_x, gamma_z, delay, phase, N)
    U_static = (-i_u * DeltaT * (H_free + V_int)).expm()

    # Measurement projectors on exiting auxiliary (index 1 = D_0)
    # B2 fix: build projectors correctly for any delay value
    total_q = 1 + delay + 1   # system + delay + fresh

    if measurement_basis == 'z':
        m0_local, m1_local = fock_dm(2, 0), fock_dm(2, 1)
    elif measurement_basis == 'x':
        plus  = (basis(2,0) + basis(2,1)).unit()
        minus = (basis(2,0) - basis(2,1)).unit()
        m0_local, m1_local = ket2dm(plus), ket2dm(minus)
    elif measurement_basis == 'y':
        plus_y  = (basis(2,0) + 1j*basis(2,1)).unit()
        minus_y = (basis(2,0) - 1j*basis(2,1)).unit()
        m0_local, m1_local = ket2dm(plus_y), ket2dm(minus_y)
    else:
        raise ValueError(f"Unknown measurement_basis '{measurement_basis}'")

    def _build_projector(m_local):
        if delay == 0:
            # Full space: system(0) ⊗ fresh(1) — no D_0 slot
            # measure on fresh (index 1), which is the ONLY auxiliary
            return tensor(qeye(dim), m_local)
        else:
            # Full space: system(0) ⊗ D_0(1) ⊗ D_1..D_{d-1}(2..d) ⊗ fresh(d+1)
            ops = [qeye(dim), m_local]
            ops += [qeye(dim)] * (delay - 1)
            ops += [qeye(dim)]    # fresh
            return tensor(ops)

    P0 = _build_projector(m0_local)
    P1 = _build_projector(m1_local)

    indexlist = _keep_indices(delay)
    rho = _initial_state_with_delay(evuType, delay, dim)

    sys_sx, sys_sz, sys_pop = [], [], []
    sys_amp, sys_phase_arr  = [], []
    meas_record = []

    for n in range(runs):
        rho_s = rho.ptrace(0)
        sx  = float(np.real(expect(sigmax(), rho_s)))
        sz  = float(np.real(expect(sigmaz(), rho_s)))
        sy  = float(np.real(expect(sigmay(), rho_s)))
        pop = float(np.real(expect(Qobj([[0,0],[0,1]]), rho_s)))

        sys_sx.append(sx);  sys_sz.append(sz)
        sys_pop.append(pop);  sys_amp.append(pop)
        sys_phase_arr.append(float(np.arctan2(sy, sx)))

        a   = float(drive_amplitudes[n])
        phi = float(drive_phases[n])
        H_drive_2x2 = DeltaT * (
            omega_drive_xy * (np.cos(phi) * sigmax() + np.sin(phi) * sigmay()) +
            a * omega_drive_z * sigmaz()
        )
        R_drive = (-i_u * H_drive_2x2).expm()
        drive_ops = [R_drive] + [qeye(dim)] * delay
        rho_driven = tensor(drive_ops) * rho * tensor(drive_ops).dag()

        aux_fresh = fock_dm(dim, 0)
        rho_full  = tensor(rho_driven, aux_fresh)
        rho_post  = U_static * rho_full * U_static.dag()

        p0 = max(float(np.real((P0 * rho_post).tr())), 0.0)
        p1 = max(1.0 - p0, 0.0)
        p_tot = p0 + p1
        if p_tot < 1e-12:
            p0, p1 = 0.5, 0.5
        else:
            p0, p1 = p0/p_tot, p1/p_tot

        outcome = rng.choice([0, 1], p=[p0, p1])
        meas_record.append(outcome)

        P_out = P0 if outcome == 0 else P1
        rho_c = P_out * rho_post * P_out
        norm  = float(np.real(rho_c.tr()))
        if norm > 1e-12:
            rho_c = rho_c / norm

        rho = rho_c.ptrace(indexlist)

    return dict(
        sys_amp=np.array(sys_amp),
        sys_phase=np.array(sys_phase_arr),
        sys_sigmax=np.array(sys_sx),
        sys_sigmaz=np.array(sys_sz),
        sys_pop=np.array(sys_pop),
        measurement_record=np.array(meas_record),
    )


# ---- 8d. MODE IV — Qutrit harmony ----

def trotterize_harmony(rho_chain, coupling_g, delay, phase,
                        omega_sys=10.0, DeltaT=1.0):
    """
    Qutrit collision model — fresh auxiliaries carry Woolhouse TA density matrices.
    System accumulates tonal context through the delay line.
    Output: diagonal of each exiting auxiliary → voice selection.
    """
    dim     = 3
    N_steps = len(rho_chain)
    if N_steps == 0:
        return [], np.zeros((0,3)), np.zeros(0, dtype=int), []

    g = np.sqrt(coupling_g / DeltaT) if coupling_g > 0 else 0.0

    V      = _build_qutrit_interaction(g, delay, phase)
    H_free = _build_qutrit_free_ham(omega_sys, delay)
    U      = (-1j * DeltaT * (H_free + V)).expm()
    Udag   = U.dag()

    # Early unitary (steps < delay): no feedback
    if delay > 0:
        lambdas = gell_mann_x_operators()
        V_fresh = sum(tensor([g*lam] + [qeye(dim)]*delay + [lam])
                      for lam in lambdas)
        U_early     = (-1j * DeltaT * (H_free + V_fresh)).expm()
        U_early_dag = U_early.dag()

    rho_sd = (tensor([rho_chain[0]] + [fock_dm(dim,0)]*delay)
              if delay > 0 else rho_chain[0])

    output_rhos = []
    diagonals   = np.zeros((N_steps, 3))
    winners     = np.zeros(N_steps, dtype=int)
    sys_states  = []

    for n in range(N_steps):
        sys_states.append(rho_sd.ptrace(0) if delay > 0 else rho_sd)

        rho_fresh = rho_chain[n]
        rho_full  = tensor(rho_sd, rho_fresh)

        if delay > 0 and n < delay:
            rho_post = U_early * rho_full * U_early_dag
        else:
            rho_post = U * rho_full * Udag

        # Read fresh auxiliary (last index = delay+1)
        fresh_idx = delay + 1 if delay > 0 else 1
        rho_out   = rho_post.ptrace(fresh_idx)

        # Shift delay line
        if delay > 0:
            keep   = [0] + list(range(2, 2+delay))
            rho_sd = rho_post.ptrace(keep)
        else:
            rho_sd = rho_post.ptrace([0])

        diag = np.real(np.diag(rho_out.full()))
        diag = np.maximum(diag, 0)
        if diag.sum() > 0:
            diag /= diag.sum()

        output_rhos.append(rho_out)
        diagonals[n] = diag
        winners[n]   = np.argmax(diag)

    return output_rhos, diagonals, winners, sys_states


# ============================================================
# SECTION 9 — PER-BIN QUANTUM TRANSFORMS (modes II & III)
# ============================================================

def _quantum_colour_transform_per_bin(Xr_mag, Xr_phase, freq_bins,
                                       bin_indices, delay, phase,
                                       DeltaT=1.0, lam=1.0/100.0,
                                       omega_cont=50.0, omega_cont_z=0.0,
                                       gamma_x=None, gamma_z=None,
                                       headroom=None,
                                       N=None, evuType='Pur.Deph',
                                       verbose=True):
    if N is None:
        N = [2]
    num_frames = Xr_mag.shape[0]
    out_mag    = Xr_mag.copy()
    out_phase  = Xr_phase.copy()
    diagnostics = {}
    total_bins  = len(bin_indices)

    for count, k in enumerate(bin_indices):
        if verbose and (count % max(1, total_bins//10) == 0 or count == total_bins-1):
            print(f"  bin {count+1}/{total_bins}  (k={k}, f={freq_bins[k]:.1f} Hz)")

        qp  = _compute_sms_quantum_params(freq_bins[k], lam=lam)
        gx  = (gamma_x if gamma_x is not None else qp['gamma_x']) * qp['omega_s']
        gz  = (gamma_z if gamma_z is not None else qp['gamma_z']) * qp['omega_s']

        pad_amps   = np.concatenate([Xr_mag[:,k],   np.zeros(delay)])
        pad_phases = np.concatenate([Xr_phase[:,k], np.zeros(delay)])

        res = trotterize_colour(
            DeltaT=DeltaT,
            omega_s=qp['omega_s'],
            omega_cont=omega_cont * qp['omega_s'],
            omega_cont_z=omega_cont_z * qp['omega_s'],
            omega_r=qp['omega_r'],
            gamma_x=gx, gamma_z=gz,
            delay=delay, N=N, phase=phase,
            aux_amplitudes=pad_amps, aux_phases=pad_phases,
            evuType=evuType,
        )

        aligned_amp   = res['aux_amp_out'][delay: delay + num_frames]
        aligned_phase = res['aux_phase_out'][delay: delay + num_frames]

        out_mag[:,k]   = np.clip(aligned_amp, 0.0, 1.0)
        out_phase[:,k] = aligned_phase
        diagnostics[k] = dict(
            sys_sigmax=res['sys_sigmax'], sys_sigmaz=res['sys_sigmaz'],
            sys_pop=res['sys_pop'],
            aux_amp_out=res['aux_amp_out'], aux_phase_out=res['aux_phase_out'])

    if headroom is not None:
        for k in bin_indices:
            for n in range(num_frames):
                if Xr_mag[n,k] > 1e-12 and out_mag[n,k] > headroom * Xr_mag[n,k]:
                    out_mag[n,k] = headroom * Xr_mag[n,k]

    return out_mag, out_phase, diagnostics


def _quantum_trajectory_transform_per_bin(Xr_mag, Xr_phase, freq_bins,
                                           bin_indices, delay, phase,
                                           sin_freqs, sin_phases,
                                           DeltaT=1.0, lam=1.0/100.0,
                                           gamma_x=None, gamma_z=None,
                                           omega_drive_xy=1.0,
                                           omega_drive_z=0.0,
                                           measurement_basis='z',
                                           normperbin=False,
                                           N=None, evuType='Pur.Deph',
                                           seed=None, verbose=True):
    if N is None:
        N = [2]
    num_frames, num_bins = Xr_mag.shape

    if normperbin:
        mag_norm, scale_per_bin = normalize_amplitudes_per_bin(Xr_mag)
    else:
        mag_norm, scale_f = normalize_amplitudes(Xr_mag)
        scale_per_bin     = np.full(num_bins, scale_f)

    _, nearest_peak_phase, _, _ = get_stochastic_bins_with_peaks(
        sin_freqs, sin_phases, freq_bins)

    out_mag   = Xr_mag.copy()
    out_phase = Xr_phase.copy()
    diagnostics = {}
    total_bins  = len(bin_indices)

    for count, k in enumerate(bin_indices):
        if verbose and (count % max(1, total_bins//10) == 0 or count == total_bins-1):
            print(f"  bin {count+1}/{total_bins}  (k={k}, f={freq_bins[k]:.1f} Hz)")

        qp  = _compute_sms_quantum_params(freq_bins[k], lam=lam)
        gx  = (gamma_x if gamma_x is not None else qp['gamma_x']) * qp['omega_s']
        gz  = (gamma_z if gamma_z is not None else qp['gamma_z']) * qp['omega_s']

        drive_amps   = mag_norm[:, k]
        drive_phases = nearest_peak_phase[:num_frames, k]
        bin_seed     = (seed + k) if seed is not None else None

        res = trotterize_trajectory(
            DeltaT=DeltaT,
            omega_s=qp['omega_s'],      # used for scaling only (omega_s_static=0)
            omega_r=qp['omega_r'],
            gamma_x=gx, gamma_z=gz,
            delay=delay, N=N, phase=phase,
            drive_amplitudes=drive_amps,
            drive_phases=drive_phases,
            omega_drive_xy=omega_drive_xy * qp['omega_s'],
            omega_drive_z=omega_drive_z  * qp['omega_s'],
            omega_s_static=0,            # B5: explicit
            measurement_basis=measurement_basis,
            evuType=evuType, seed=bin_seed,
        )

        out_mag[:,k]   = np.clip(res['sys_amp'], 0.0, 1.0) * scale_per_bin[k]
        out_phase[:,k] = res['sys_phase']

        # Energy matching
        orig_e = np.sum(Xr_mag[:,k]**2)
        traj_e = np.sum(out_mag[:,k]**2)
        if traj_e > 1e-20:
            out_mag[:,k] *= np.sqrt(orig_e / traj_e)

        diagnostics[k] = dict(
            sys_sigmax=res['sys_sigmax'], sys_sigmaz=res['sys_sigmaz'],
            sys_pop=res['sys_pop'], sys_amp=res['sys_amp'],
            sys_phase=res['sys_phase'],
            measurement_record=res['measurement_record'])

    return out_mag, out_phase, diagnostics


# ============================================================
# SECTION 10 — MODE-SPECIFIC PIPELINE WRAPPERS
# ============================================================

def _pipeline_sensing(audio_path, delay=3, phase=np.pi,
                       sensing_rate=2000, T_phi=None, max_freq=2000,
                       phase_scale=0.5, method='sin_phi',
                       N_mod=10, swap_angle=np.pi/2, sequence='ramsey',
                       audio_sr=44100, verbose=True):
    if verbose:
        print("=== MODE I: NV Quantum Sensing ===")

    data, sr = load_wav(audio_path, max_freq=max_freq)
    orig_dur = len(data) / sr

    stretch, _ = compute_stretch_factor(max_freq, sensing_rate)
    B_hires    = time_stretch(data, stretch) if stretch > 1 else data

    T_seq = 1.0 / sensing_rate
    if T_phi is None:
        T_phi = 0.8 * T_seq

    if verbose:
        print(f"  sensing_rate={sensing_rate} Hz  T_phi={T_phi:.5f}  delay={delay}  "
              f"stretch={stretch:.1f}  method={method}")

    res = trotterize_sensing(
        B_hires, sr, T_phi, T_seq,
        phase_scale=phase_scale, delay=delay, swap_angle=swap_angle,
        sequence=sequence, method=method, N_mod=N_mod)

    out = res['qpsd_phases'] if method == 'qpsd' else res['aux_sigma_z']
    if stretch > 1:
        out = time_compress(out, stretch)
    out = out - np.mean(out)

    n_audio = int(orig_dur * audio_sr)
    audio   = resample(out, max(n_audio, 2))
    peak    = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.8

    # Save
    out_path = _output_path(audio_path, 'sensing')
    soundfile.write(out_path, audio.astype(np.float32), audio_sr)
    if verbose:
        print(f"  Saved: {out_path}")

    return dict(
        y_quantum=audio, y_original=data,
        fs=audio_sr, fs_original=sr,
        mode='sensing', audio_path=audio_path,
        trotterize_result=res,
        output_paths=dict(sensing=out_path),
        params=dict(sensing_rate=sensing_rate, T_phi=T_phi, delay=delay,
                    phase_scale=phase_scale, method=method, sequence=sequence),
    )


def _pipeline_sms_colour(audio_path, delay=3, phase=np.pi,
                          mode_sms='stochastic_only',
                          lam=1.0/100.0, omega_cont=50.0, omega_cont_z=0.0,
                          gamma_x=None, gamma_z=None,
                          headroom=None, softclip=True,
                          normperbin=False, DeltaT=1.0,
                          use_quantum_amp=True, use_quantum_phase=True,
                          evuType='Pur.Deph', tolerance_hz=None,
                          M=2501, N_fft=4096, H=1024, t=-80, nH=60,
                          minf0=100, maxf0=400, f0et=5,
                          harmDevSlope=0.01, minSineDur=0.02,
                          Ns=2048, stocf=0.2,
                          verbose=True):
    if verbose:
        print("=== MODE II: SMS Quantum Colour (deterministic) ===")

    data, fs = load_wav_full(audio_path)
    ana = sms_analyse(data, fs, M=M, N=N_fft, H=H, t=t, nH=nH,
                      minf0=minf0, maxf0=maxf0, f0et=f0et,
                      harmDevSlope=harmDevSlope, minSineDur=minSineDur,
                      Ns=Ns, stocf=stocf)

    # Select bins.
    # mode_sms controls which frequency bins are passed through the quantum kernel:
    #   'stochastic_only'   — only bins that carry stochastic (noise) energy, not harmonic partials (default)
    #   'all'               — every STFT bin (slow but full-spectrum)
    # Other modes (e.g. 'reverse_sinusoidal', 'full_stft') can be added here later.
    if mode_sms == 'all':
        bin_indices = np.arange(len(ana['freq_bins']))
    else:  # 'stochastic_only'
        mask        = get_stochastic_only_bins(ana['sin_freqs'], ana['freq_bins'], tolerance_hz)
        bin_indices = np.where(mask)[0]
    if verbose:
        print(f"  Bins ({mode_sms}): {len(bin_indices)} / {len(ana['freq_bins'])} | "
              f"delay={delay}  phase={phase:.3f}")

    # Normalise
    if normperbin:
        mag_norm, scale_per_bin = normalize_amplitudes_per_bin(ana['Xr_mag'])
    else:
        mag_norm, sf = normalize_amplitudes(ana['Xr_mag'])
        scale_per_bin = np.full(len(ana['freq_bins']), sf)

    out_mag, out_phase, diag = _quantum_colour_transform_per_bin(
        mag_norm, ana['Xr_phase'], ana['freq_bins'],
        bin_indices, delay, phase,
        DeltaT=DeltaT, lam=lam,
        omega_cont=omega_cont, omega_cont_z=omega_cont_z,
        gamma_x=gamma_x, gamma_z=gamma_z,
        headroom=headroom, N=[2], evuType=evuType, verbose=verbose)

    res_mag   = out_mag   * scale_per_bin[np.newaxis, :]
    res_phase = out_phase

    synth = synthesise_sms(ana, res_mag=res_mag, res_phase=res_phase,
                            use_quantum_amp=use_quantum_amp,
                            use_quantum_phase=use_quantum_phase,
                            softclip=softclip,
                            win_norm_dB_override=ana['win_norm_dB'])

    # Save
    paths = {}
    for suffix, sig in [('combined', synth['y_quantum']),
                         ('original', synth['y_original']),
                         ('harmonic_q', synth['y_harm_quantum']),
                         ('stoc_q',     synth['y_stoc_quantum']),
                         ('stoc_orig',  synth['y_stoc_original'])]:
        p = _output_path(audio_path, 'sms_colour', suffix)
        soundfile.write(p, sig.astype(np.float32), fs)
        paths[suffix] = p
    if verbose:
        print(f"  Saved {len(paths)} files with prefix II_colour")

    return dict(
        y_quantum=synth['y_quantum'], y_original=synth['y_original'],
        fs=fs, mode='sms_colour', audio_path=audio_path,
        analysis=ana, quantum_result=dict(
            res_mag=res_mag, res_phase=res_phase,
            diagnostics=diag, bin_indices=bin_indices,
            scale_per_bin=scale_per_bin),
        synth=synth, output_paths=paths,
        params=dict(delay=delay, phase=phase, gamma_x=gamma_x, gamma_z=gamma_z,
                    omega_cont=omega_cont, lam=lam, normperbin=normperbin),
    )


def _pipeline_sms_trajectory(audio_path, delay=3, phase=np.pi,
                               lam=1.0/100.0,
                               gamma_x=0.1, gamma_z=1.0,
                               omega_drive_xy=1.0, omega_drive_z=0.0,
                               measurement_basis='z',
                               normperbin=False, DeltaT=1.0,
                               use_quantum_amp=True, use_quantum_phase=True,
                               softclip=True, seed=None,
                               evuType='Pur.Deph', tolerance_hz=None,
                               M=2501, N_fft=4096, H=1024, t=-80, nH=23,
                               minf0=100, maxf0=400, f0et=5,
                               harmDevSlope=0.01, minSineDur=0.02,
                               Ns=2048, stocf=0.2,
                               verbose=True):
    if verbose:
        print("=== MODE III: SMS Quantum Trajectory (stochastic) ===")

    data, fs = load_wav_full(audio_path)
    ana = sms_analyse(data, fs, M=M, N=N_fft, H=H, t=t, nH=nH,
                      minf0=minf0, maxf0=maxf0, f0et=f0et,
                      harmDevSlope=harmDevSlope, minSineDur=minSineDur,
                      Ns=Ns, stocf=stocf)

    mask        = get_stochastic_only_bins(ana['sin_freqs'], ana['freq_bins'], tolerance_hz)
    bin_indices = np.where(mask)[0]
    if verbose:
        print(f"  Stochastic bins: {len(bin_indices)} / {len(ana['freq_bins'])} | "
              f"delay={delay}  basis={measurement_basis}")

    out_mag, out_phase, diag = _quantum_trajectory_transform_per_bin(
        ana['Xr_mag'], ana['Xr_phase'], ana['freq_bins'],
        bin_indices, delay, phase,
        sin_freqs=ana['sin_freqs'], sin_phases=ana['sin_phases'],
        DeltaT=DeltaT, lam=lam,
        gamma_x=gamma_x, gamma_z=gamma_z,
        omega_drive_xy=omega_drive_xy, omega_drive_z=omega_drive_z,
        measurement_basis=measurement_basis,
        normperbin=normperbin, N=[2], evuType=evuType,
        seed=seed, verbose=verbose)

    synth = synthesise_sms(ana, res_mag=out_mag, res_phase=out_phase,
                            use_quantum_amp=use_quantum_amp,
                            use_quantum_phase=use_quantum_phase,
                            softclip=softclip,
                            win_norm_dB_override=ana['win_norm_dB'])

    paths = {}
    for suffix, sig in [('combined', synth['y_quantum']),
                         ('original', synth['y_original']),
                         ('harmonic', synth['y_harm_quantum']),
                         ('stoc_q',   synth['y_stoc_quantum']),
                         ('stoc_orig',synth['y_stoc_original'])]:
        p = _output_path(audio_path, 'sms_trajectory', suffix)
        soundfile.write(p, sig.astype(np.float32), fs)
        paths[suffix] = p
    if verbose:
        print(f"  Saved {len(paths)} files with prefix III_trajectory")

    return dict(
        y_quantum=synth['y_quantum'], y_original=synth['y_original'],
        fs=fs, mode='sms_trajectory', audio_path=audio_path,
        analysis=ana,
        diagnostics=diag, bin_indices=bin_indices,
        synth=synth, output_paths=paths,
        params=dict(delay=delay, phase=phase, gamma_x=gamma_x, gamma_z=gamma_z,
                    omega_drive_xy=omega_drive_xy, lam=lam,
                    measurement_basis=measurement_basis, seed=seed,
                    normperbin=normperbin),
    )


def _pipeline_harmony(audio_path, delay=2, phase=np.pi,
                       coupling_g=0.5, omega_sys=10.0,
                       brightness=1.5, n_harmonics=8,
                       alpha=50.0, beta=2.0, gamma_rs=4.0, delta=0.1,
                       use_voice_leading=True, use_root_salience=True,
                       use_consonance_dissonance=True,
                       chord_size=3, onset_tolerance=0.05, min_velocity=0,
                       M=2501, N_fft=4096, H=1024, t=-80, nH=60,
                       minf0=50, maxf0=2000, f0et=5,
                       harmDevSlope=0.01, minSineDur=0.02,
                       Ns=2048, stocf=0.2,
                       verbose=True):
    if verbose:
        print("=== MODE IV: Quantum Tonal Brightness (Harmony) ===")

    # Load
    data, fs = load_wav_full(audio_path)

    # Chord events via basic-pitch
    events, _ = extract_chord_events(audio_path, chord_size=chord_size,
                                      onset_tolerance=onset_tolerance,
                                      min_velocity=min_velocity, verbose=verbose)
    if len(events) < 2:
        raise ValueError(f"Need ≥2 chord events, got {len(events)}. "
                         f"Lower min_velocity or onset_tolerance.")

    # Woolhouse TA chain
    wk = dict(alpha=alpha, beta=beta, gamma_rs=gamma_rs, delta=delta,
               use_voice_leading=use_voice_leading,
               use_root_salience=use_root_salience,
               use_consonance_dissonance=use_consonance_dissonance)
    rho_chain, ta_chain, a_chain, pairs = score_to_density_chain(events, **wk)
    if verbose:
        print(f"  {len(rho_chain)} TA transitions  "
              f"A: [{min(a_chain):.3f}, {max(a_chain):.3f}]")

    # Qutrit collision
    _, diagonals, winners, sys_states = trotterize_harmony(
        rho_chain, coupling_g=coupling_g, delay=delay,
        phase=phase, omega_sys=omega_sys)

    # SMS analysis (B13: call sms_analyse instead of inlining)
    if verbose:
        print("  Running SMS analysis...")
    ana = sms_analyse(data, fs, M=M, N=N_fft, H=H, t=t, nH=nH,
                      minf0=minf0, maxf0=maxf0, f0et=f0et,
                      harmDevSlope=harmDevSlope, minSineDur=minSineDur,
                      Ns=Ns, stocf=stocf)

    # Inject frame alignment into events
    events_aligned = [dict(ev, _sr=fs, _hop=H) for ev in events]

    # Brightness modulation
    brightened_stoc, brightened_sin, mod_log = apply_brightness_modulation(
        ana['Xr_mag'], ana['freq_bins'],
        events_aligned, winners, diagonals,
        brightness=brightness, n_harmonics=n_harmonics,
        sin_freqs=ana['sin_freqs'], sin_amps=ana['sin_amps'])

    # Synthesise
    y_q, y_harm, y_stoc = synthesise_brightened_audio(
        ana, brightened_stoc, fs, brightened_sin_amps=brightened_sin)

    # Baseline
    y_baseline = SM.sineModelSynth(
        ana['hfreq'], ana['hmag'], ana['hphase'], N_fft, H, fs)
    y_stoc_bl  = STC.stochasticModelSynth(
        ana['stocEnv'], H, ana['N_stoc'], fs, melScale=1, phase_func=None)
    ml_bl = min(len(y_baseline), len(y_stoc_bl))
    y_baseline = y_baseline[:ml_bl] + y_stoc_bl[:ml_bl]

    if verbose:
        vc = [0,0,0]
        for e in mod_log: vc[e['winner']] += 1
        print(f"  Voice wins: root={vc[0]}  3rd={vc[1]}  5th={vc[2]}")

    # Save
    paths = {}
    for suffix, sig, sr_out in [
            ('quantum',  y_q,       fs),
            ('baseline', y_baseline, fs)]:
        p = _output_path(audio_path, 'harmony', suffix)
        soundfile.write(p, sig.astype(np.float32), sr_out)
        paths[suffix] = p
    if verbose:
        print(f"  Saved {len(paths)} files with prefix IV_harmony")

    return dict(
        y_quantum=y_q, y_original=y_baseline,
        fs=fs, mode='harmony', audio_path=audio_path,
        analysis=ana,
        events=events, rho_chain=rho_chain,
        ta_chain=ta_chain, a_chain=a_chain,
        diagonals=diagonals, winners=winners,
        modulation_log=mod_log,
        brightened_stoc=brightened_stoc,
        brightened_sin=brightened_sin,
        output_paths=paths,
        params=dict(delay=delay, phase=phase, coupling_g=coupling_g,
                    brightness=brightness, omega_sys=omega_sys),
    )


# ============================================================
# SECTION 11 — MASTER PIPELINE DISPATCHER
# ============================================================

def quantum_audio_pipeline(audio_path, mode='sensing', **kwargs):
    """
    Run one of the four quantum audio modes.

    Parameters
    ----------
    audio_path : str
        Input WAV file.
    mode : str
        'sensing'        — (I)   NV-center quantum sensing
        'sms_colour'     — (II)  SMS spectral colouring (deterministic)
        'sms_trajectory' — (III) SMS quantum trajectory (stochastic)
        'harmony'        — (IV)  Quantum tonal brightness

    Common kwargs (accepted by all modes)
    --------------------------------------
    delay : int    (default 3)
    phase : float  (default np.pi)

    Mode-specific kwargs are forwarded to the relevant pipeline.
    See _pipeline_sensing / _pipeline_sms_colour / etc. for full lists.

    Returns
    -------
    result : dict   always contains 'y_quantum', 'y_original', 'fs', 'mode',
                    'output_paths', 'params', plus mode-specific keys.
    """
    dispatch = {
        'sensing':        _pipeline_sensing,
        'sms_colour':     _pipeline_sms_colour,
        'sms_trajectory': _pipeline_sms_trajectory,
        'harmony':        _pipeline_harmony,
    }
    if mode not in dispatch:
        raise ValueError(f"Unknown mode '{mode}'. "
                         f"Choose from: {list(dispatch.keys())}")
    return dispatch[mode](audio_path, **kwargs)


# ============================================================
# SECTION 12 — UNIFIED PLOT FUNCTION
# ============================================================

def plot_quantum_audio(result, show_details=True, max_freq=5000,
                       figsize_main=(14, 8), figsize_detail=(14, 8)):
    """
    Universal plot covering all four modes.

    Always shows
    ------------
    Fig 1 (2×2):
      [0,0] Original waveform       [0,1] Quantum waveform
      [1,0] Original spectrogram    [1,1] Quantum spectrogram

    With show_details=True, adds a mode-specific Fig 2:
      sensing        — phase trajectory + sin(φ) scatter + spectrum
      sms_colour     — per-bin system evolution (sys_sigmax, sys_pop, amp in/out)
      sms_trajectory — per-bin: drive amp vs sys_amp, phase, measurement record
      harmony        — voice selection bar chart + attraction values + boost log
    """
    mode = result['mode']
    fs   = result['fs']
    y_q  = result['y_quantum']
    y_o  = result['y_original']

    # ---- Fig 1: waveform + spectrogram ----
    fig1, axes = plt.subplots(2, 2, figsize=figsize_main)
    fig1.suptitle(f"Quantum Audio — Mode {_mode_label(mode)}", fontsize=13)

    t_q = np.arange(len(y_q)) / fs
    t_o = np.arange(len(y_o)) / fs

    axes[0,0].plot(t_o, y_o, lw=0.4, color='steelblue')
    axes[0,0].set_title('Original'); axes[0,0].set_ylabel('Amplitude')
    axes[0,0].set_xlim([0, t_o[-1]])

    axes[0,1].plot(t_q, y_q, lw=0.4, color='darkorchid')
    axes[0,1].set_title(f'Quantum ({_mode_short(mode)})')
    axes[0,1].set_xlim([0, t_q[-1]])

    for ax, sig, title in [(axes[1,0], y_o, 'Original'),
                            (axes[1,1], y_q, f'Quantum {_mode_short(mode)}')]:
        N_spect = 2048
        f_s, t_s, Sxx = _spectrogram(sig, fs, nperseg=N_spect, noverlap=N_spect//2)
        fm = f_s <= max_freq
        ax.pcolormesh(t_s, f_s[fm], 10*np.log10(Sxx[fm] + 1e-12),
                      shading='gouraud', cmap='inferno')
        ax.set_ylabel('Frequency (Hz)'); ax.set_xlabel('Time (s)')
        ax.set_title(f'Spectrogram — {title}')

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    if not show_details:
        return

    # ---- Fig 2: mode-specific ----
    if mode == 'sensing':
        _plot_details_sensing(result, figsize_detail)
    elif mode == 'sms_colour':
        _plot_details_sms_colour(result, figsize_detail)
    elif mode == 'sms_trajectory':
        _plot_details_sms_trajectory(result, figsize_detail)
    elif mode == 'harmony':
        _plot_details_harmony(result, figsize_detail)


def _mode_label(mode):
    return {'sensing':'I — Sensing', 'sms_colour':'II — SMS Colour',
            'sms_trajectory':'III — SMS Trajectory', 'harmony':'IV — Harmony'}[mode]

def _mode_short(mode):
    return {'sensing':'I', 'sms_colour':'II', 'sms_trajectory':'III', 'harmony':'IV'}[mode]


def _plot_details_sensing(result, figsize):
    res  = result['trotterize_result']
    fs   = result['fs']
    meth = res['method']

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Mode I — Sensing: Phase & Readout', fontsize=12)

    tl    = res['tlist']
    phi   = res['phases']
    readout = res['qpsd_phases'] if meth == 'qpsd' else res['aux_sigma_z']
    lbl     = 'QPSD phase' if meth == 'qpsd' else 'sin(φ)'

    axes[0,0].plot(tl, phi, 'r-', lw=0.6)
    axes[0,0].axhline(np.pi/2,  color='grey', ls='--', alpha=0.4)
    axes[0,0].axhline(-np.pi/2, color='grey', ls='--', alpha=0.4)
    axes[0,0].set_title('Accumulated phase φ'); axes[0,0].set_ylabel('rad')

    axes[0,1].plot(tl, readout, 'm-', lw=0.6, label=lbl)
    axes[0,1].set_title('Readout'); axes[0,1].legend(fontsize=9)

    axes[1,0].scatter(phi, readout, s=1, alpha=0.3, color='darkorchid')
    phi_ref = np.linspace(-np.pi/2, np.pi/2, 200)
    axes[1,0].plot(phi_ref, np.sin(phi_ref), 'r-', lw=1.5, label='sin(φ)')
    axes[1,0].plot(phi_ref, phi_ref, 'k--', lw=1, label='linear')
    axes[1,0].set_xlabel('True φ'); axes[1,0].set_ylabel('Readout')
    axes[1,0].set_title('Linearity'); axes[1,0].legend(fontsize=8)

    c    = readout - np.mean(readout)
    rate = res.get('sensing_rate', res.get('phase_readout_rate', 1))
    sp   = np.abs(fft(c)); fr = fftfreq(len(c), 1/rate)
    pos  = fr >= 0
    mx   = np.max(sp[pos]) + 1e-10
    axes[1,1].plot(fr[pos], sp[pos]/mx, 'k-', lw=0.8)
    axes[1,1].set_title('Output spectrum'); axes[1,1].set_xlabel('Hz')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def _plot_details_sms_colour(result, figsize):
    diag      = result['quantum_result']['diagnostics']
    freq_bins = result['analysis']['freq_bins']
    Xr_mag    = result['analysis']['Xr_mag']

    bin_keys = sorted(diag.keys())
    if not bin_keys:
        print("No diagnostics to plot."); return

    show_keys = _pick_representative_bins(bin_keys, 4)
    n_cols    = len(show_keys)

    fig, axes = plt.subplots(3, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle('Mode II — SMS Colour: per-bin system & I/O', fontsize=12)

    for col, k in enumerate(show_keys):
        d      = diag[k]
        steps  = np.arange(len(d['sys_pop']))
        f_hz   = freq_bins[k] if k < len(freq_bins) else k
        in_amp = Xr_mag[:, k] if k < Xr_mag.shape[1] else np.zeros(len(d['sys_pop']))

        ax = axes[0, col]
        ax.plot(steps, d['sys_pop'],    lw=0.8, label='ρ₁₁', color='steelblue')
        ax.plot(steps, d['sys_sigmax'], lw=0.8, label='⟨σx⟩', color='tomato')
        ax.set_title(f'bin {k} ({f_hz:.0f} Hz)')
        ax.legend(fontsize=7); ax.set_ylim([-1.1, 1.1])
        ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('System')

        ax = axes[1, col]
        ax.plot(np.arange(len(in_amp)), in_amp,         lw=0.6, color='steelblue', label='in amp')
        ax.plot(np.arange(len(d['aux_amp_out'])), d['aux_amp_out'], lw=0.6, color='darkorchid', label='out amp')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Amplitude')

        ax = axes[2, col]
        ax.plot(steps, np.unwrap(d['aux_phase_out']), lw=0.5, color='seagreen')
        ax.set_xlabel('Frame'); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Output phase (rad)')

    plt.tight_layout(); plt.show()


def _plot_details_sms_trajectory(result, figsize):
    diag      = result.get('diagnostics', {})
    freq_bins = result['analysis']['freq_bins']
    Xr_mag    = result['analysis']['Xr_mag']
    sin_freqs = result['analysis']['sin_freqs']
    sin_phases= result['analysis']['sin_phases']
    num_frames = Xr_mag.shape[0]

    if not diag:
        print("No diagnostics to plot."); return

    normperbin = result.get('params', {}).get('normperbin', False)
    if normperbin:
        mag_norm, _ = normalize_amplitudes_per_bin(Xr_mag)
    else:
        mag_norm, _ = normalize_amplitudes(Xr_mag)

    _, nearest_peak_phase, _, _ = get_stochastic_bins_with_peaks(
        sin_freqs, sin_phases, freq_bins)

    bin_keys  = sorted(diag.keys())
    show_keys = _pick_representative_bins(bin_keys, 4)
    n_cols    = len(show_keys)

    fig, axes = plt.subplots(3, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle('Mode III — Trajectory: drive vs system & measurement', fontsize=12)

    for col, k in enumerate(show_keys):
        d       = diag[k]
        sys_amp = d['sys_amp']
        drv_amp = mag_norm[:num_frames, k] if k < mag_norm.shape[1] else np.zeros(num_frames)
        steps   = np.arange(len(sys_amp))
        f_hz    = freq_bins[k] if k < len(freq_bins) else k

        # Row 0: drive amp vs sys amp (mean-centred, peak-normalised)
        ax = axes[0, col]
        def _normalise_for_plot(arr):
            c = arr - np.mean(arr)
            p = np.max(np.abs(c)) or 1.0
            return c / p
        ax.plot(np.arange(num_frames), _normalise_for_plot(drv_amp),
                lw=0.5, color='steelblue', alpha=0.8, label='drive')
        ax.plot(steps, _normalise_for_plot(sys_amp),
                lw=0.5, color='darkorchid', alpha=0.8, label='sys')
        ax.set_title(f'bin {k} ({f_hz:.0f} Hz)')
        ax.legend(fontsize=6); ax.set_ylim([-1.2, 1.2]); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Normalised amp')

        # Row 1: Bloch components
        ax = axes[1, col]
        ax.plot(steps, d['sys_sigmax'], lw=0.6, label='⟨σx⟩')
        ax.plot(steps, d['sys_sigmaz'], lw=0.6, label='⟨σz⟩')
        ax.plot(steps, d['sys_pop'],    lw=0.6, ls='--', label='ρ₁₁')
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Bloch')

        # Row 2: measurement record
        ax = axes[2, col]
        meas = d['measurement_record']
        ax.fill_between(np.arange(len(meas)), 0, meas, step='mid',
                        alpha=0.5, color='orange')
        ax.set_ylim([-0.1, 1.1]); ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Outcome (0/1)')

    plt.tight_layout(); plt.show()


def _plot_details_harmony(result, figsize):
    diagonals = result['diagonals']
    winners   = result['winners']
    a_chain   = result['a_chain']
    mod_log   = result['modulation_log']
    events    = result['events']

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Mode IV — Harmony: voice selection & attraction', fontsize=12)

    # [0,0] Stacked probability of each voice
    t = np.arange(len(diagonals))
    axes[0,0].stackplot(t,
        diagonals[:,0], diagonals[:,1], diagonals[:,2],
        labels=['root', '3rd', '5th'],
        colors=['steelblue', 'darkorange', 'seagreen'], alpha=0.7)
    axes[0,0].set_title('Voice probabilities per transition')
    axes[0,0].set_ylabel('Probability'); axes[0,0].legend(fontsize=8)
    axes[0,0].set_xlabel('Transition index')

    # [0,1] Winner at each transition
    vc = ['steelblue', 'darkorange', 'seagreen']
    axes[0,1].bar(t, np.ones(len(winners)), color=[vc[w] for w in winners],
                  alpha=0.7, edgecolor='none')
    axes[0,1].set_title('Winning voice (colour = root/3rd/5th)')
    axes[0,1].set_yticks([])

    # [1,0] Tonal attraction A over transitions
    axes[1,0].plot(np.arange(len(a_chain)), a_chain, 'ko-', ms=4, lw=1)
    axes[1,0].set_title('Woolhouse attraction A'); axes[1,0].set_ylabel('A')
    axes[1,0].set_xlabel('Transition')

    # [1,1] Boost factor histogram
    boosts = [e['boost_factor'] for e in mod_log]
    axes[1,1].hist(boosts, bins=20, color='darkorchid', alpha=0.7, edgecolor='white')
    axes[1,1].set_title('Boost factor distribution')
    axes[1,1].set_xlabel('Boost ×'); axes[1,1].set_ylabel('Count')

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.show()


def _pick_representative_bins(bin_keys, n=4):
    if len(bin_keys) <= n:
        return bin_keys
    indices = np.linspace(0, len(bin_keys)-1, n, dtype=int)
    return [bin_keys[i] for i in indices]


# ============================================================
# SECTION 13 — EXAMPLE EXECUTION
# ============================================================
# Run any of the four modes by changing `mode` below.
# Keep delay <= 9 for tractable run times (delay=3 is fast, delay=9 is slow).

if __name__ == '__main__':
    AUDIO = 'audio/satie.wav'

    # ---- Choose mode ----
    MODE = 'sensing'         # 'sensing' | 'sms_colour' | 'sms_trajectory' | 'harmony'

    # ---- Common parameters ----
    DELAY = 3
    PHASE = np.pi

    # ---- Mode-specific overrides ----
    kwargs = {}

    if MODE == 'sensing':
        kwargs = dict(sensing_rate=2000, phase_scale=0.5,
                      method='sin_phi', sequence='ramsey',
                      max_freq=1000, audio_sr=44100)

    elif MODE == 'sms_colour':
        kwargs = dict(gamma_x=0, gamma_z=1,
                      omega_cont=0, omega_cont_z=0.1,
                      lam=1.0/100.0, headroom=1.0,
                      softclip=True, normperbin=True,
                      use_quantum_amp=False, use_quantum_phase=True)

    elif MODE == 'sms_trajectory':
        kwargs = dict(gamma_x=1, gamma_z=1,
                      omega_drive_xy=1, omega_drive_z=1,
                      measurement_basis='z',
                      use_quantum_amp=True, use_quantum_phase=True,
                      seed=42)

    elif MODE == 'harmony':
        kwargs = dict(coupling_g=0.5, omega_sys=10.0,
                      brightness=1.5, chord_size=3,
                      onset_tolerance=0.05)

    result = quantum_audio_pipeline(AUDIO, mode=MODE,
                                     delay=DELAY, phase=PHASE,
                                     **kwargs)

    print(f"\nOutput files:")
    for k, p in result['output_paths'].items():
        print(f"  [{k}] {p}")

    plot_quantum_audio(result, show_details=True, max_freq=5000)
