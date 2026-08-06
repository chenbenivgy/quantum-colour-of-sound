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
B14 stocEnv is mel-analysed at the hardcoded default fs=44100 (hpsModelAnal calls
    stochasticModelAnal without fs); every melScale=1 resynth of ana['stocEnv'] must
    use ana['stoc_fs']=44100, NOT ana['fs'] — else at low sample rates (e.g. after
    decimation) the mel grids mismatch and the original residual inflates/warps.
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
import scipy.sparse as _sp
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


def _embed_1q_gate(gate_2x2, qubit_idx, n_qubits):
    """Embed a single-qubit gate at qubit_idx in n_qubits Hilbert space (Kronecker)."""
    g      = gate_2x2.full()
    before = 2**qubit_idx
    after  = 2**(n_qubits - qubit_idx - 1)
    U = np.kron(np.kron(np.eye(before, dtype=complex), g),
                np.eye(after, dtype=complex))
    return Qobj(U, dims=[[2]*n_qubits, [2]*n_qubits])


def _embed_2q_gate_at_01(gate_4x4, n_qubits):
    """Embed 4×4 two-qubit gate acting on qubits (0,1) into n_qubits-qubit space."""
    rest = 2**(n_qubits - 2)
    U = np.kron(gate_4x4.full(), np.eye(rest, dtype=complex))
    return Qobj(U, dims=[[2]*n_qubits, [2]*n_qubits])


def _sparsify_op(U, tol=1e-12):
    """Store an embedded gate sparse: 1- and 2-qubit gates are identity on every
    untouched qubit, so their embedded matrices are structurally sparse.
    Identical to the dense path; `tol` only drops round-off dust."""
    A = U.full()
    A[np.abs(A) < tol] = 0.0
    return Qobj(_sp.csr_matrix(A), dims=U.dims)


def _build_joint_sensing_unitary(joint_mode, xx_coupling=0.3):
    """
    Build the 4×4 joint unitary U_{S1,S2} for dual-channel sensing.

    'xx'           : U = exp(-iθ σx⊗σx), θ = xx_coupling  [transverse Ising]
    'fdn_hadamard' : circulant Hadamard A (QFDN Eq. 2.3, Rocchesso 2025)
                     A = ½[[-1,1,1,1],[1,-1,1,1],[1,1,-1,1],[1,1,1,-1]]
                     Maximally diffusive, circulant, Householder reflector.
    'hadamard_id'  : H⊗I — Hadamard combgate on S1 only (identity on S2)
    """
    if joint_mode == 'xx':
        XX = tensor(sigmax(), sigmax())
        U  = (-1j * xx_coupling * XX).expm()
        U.dims = [[2, 2], [2, 2]]
        return U
    elif joint_mode == 'fdn_hadamard':
        A = 0.5 * np.array([[-1, 1, 1, 1],
                             [ 1,-1, 1, 1],
                             [ 1, 1,-1, 1],
                             [ 1, 1, 1,-1]], dtype=complex)
        return Qobj(A, dims=[[2, 2], [2, 2]])
    elif joint_mode == 'hadamard_id':
        H2 = Qobj(np.array([[1, 1],[1,-1]], dtype=complex) / np.sqrt(2))
        return tensor(H2, qeye(2))
    else:
        raise ValueError(f"Unknown joint_mode '{joint_mode}'. "
                         f"Choose 'xx', 'fdn_hadamard', or 'hadamard_id'.")


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
    """Single-qubit initial state: GROUND |0><0| for 'Pur.Deph' (B21). |+> would
    carry population 0.5, and in an excitation-preserving channel population is
    energy -- it lifted the noise floor by ~28 dB on tonal material. Trace must
    be 1 (B15) so outcome sampling stays unbiased."""
    if evuType == 'Pur.Deph':
        return Qobj([[1, 0], [0, 0]])            # |0⟩⟨0| ground (B21)
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

def _resolve_audio_path(filepath):
    """Resolve an audio path: try as-is, then relative to this script's directory,
    then relative to this script's audio/ subdirectory.  Returns the first hit."""
    import os
    p = filepath
    if os.path.isfile(p):
        return p
    # Try relative to the directory containing this module
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, filepath)
    if os.path.isfile(candidate):
        return candidate
    # Try just the basename under audio/ next to this module
    basename = os.path.basename(filepath)
    candidate2 = os.path.join(here, 'audio', basename)
    if os.path.isfile(candidate2):
        return candidate2
    # Return original path and let soundfile raise the natural error
    return filepath

def load_wav(filepath, max_freq=2000):
    """Load wav, convert to mono, low-pass, peak-normalise."""
    data, sr = soundfile.read(_resolve_audio_path(filepath))
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
    data, sr = soundfile.read(_resolve_audio_path(filepath))
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
        'sensing_dual':   'Ia_sensing_dual',
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
        # B14: stocEnv is mel-warped by hpsModelAnal's stochasticModelAnal(xr, H, H*2, stocf)
        # which is called WITHOUT fs -> always uses the smstools default 44100. Any melScale=1
        # resynth of ana['stocEnv'] MUST use this same fs, else (esp. after decimation to a low
        # rate) the mel grids mismatch and the residual inflates/warps. Use ana['stoc_fs'].
        stoc_fs=44100,
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
                   sin_amps_mod=None, sin_phases_mod=None, use_quantum_amp=True,
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
    hmag_use = hmag
    if sin_amps_mod is not None:
        hmag_use = hmag.copy()
        nf2 = min(nf, hmag.shape[0], sin_amps_mod.shape[0])
        for n in range(nf2):
            for tr in range(hfreq.shape[1]):
                if hfreq[n, tr] > 0 and sin_amps_mod[n, tr] > 0:
                    hmag_use[n, tr] = 20.0 * np.log10(
                        np.clip(sin_amps_mod[n, tr], 1e-10, None))
    # Optionally drive the harmonic PHASE with the quantum-decoded phase at each
    # track's bin (else keep the analysed hphase). Lets the delayed-choice phase
    # reach the tonal partials, not just the residual.
    hphase_use = hphase
    if sin_phases_mod is not None and np.size(hphase) > 0:
        hphase_use = hphase.copy()
        nf3 = min(nf, hphase.shape[0], sin_phases_mod.shape[0])
        for n in range(nf3):
            for tr in range(hfreq.shape[1]):
                if hfreq[n, tr] > 0:
                    hphase_use[n, tr] = sin_phases_mod[n, tr]
    y_harm = SM.sineModelSynth(hfreq, hmag_use, hphase_use, N, H, ana['fs'])

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

    # Stochastic — original path (B14: resynth at the fs the stocEnv was ANALYSED at, not ana['fs'])
    y_stoc_o = STC.stochasticModelSynth(
        ana['stocEnv'], H, N_stoc, ana.get('stoc_fs', 44100), melScale=1, phase_func=None)

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

def get_reverse_sinusoidal_mapping(sin_freqs, freq_bins, tolerance_hz=None):
    """Build per-frame remapping for the 'reverse_sinusoidal' mode.

    When a sinusoidal track moves from bin k to bin k' between frames n-1 and n,
    the quantum auxiliary for bin k' at frame n is initialised with the value
    that was at bin k — preserving quantum coherence as the sinusoid slides in frequency.

    Returns
    -------
    remap : list of dicts, length num_frames
        remap[n][k'] = k  means "at frame n, bin k' reads input from bin k".
        Bins absent from the dict keep their own index (identity mapping).
    occupied_per_frame : ndarray bool (num_frames, num_bins)
        True where a sinusoidal occupies that bin in that frame.
    """
    num_frames = sin_freqs.shape[0]
    num_tracks = sin_freqs.shape[1]
    num_bins   = len(freq_bins)
    if tolerance_hz is None:
        tolerance_hz = float(freq_bins[1] - freq_bins[0]) if num_bins > 1 else 1.0

    def _freq_to_bin(f):
        if np.isnan(f) or f <= 0:
            return None
        return int(np.argmin(np.abs(freq_bins - f)))

    occupied_per_frame = np.zeros((num_frames, num_bins), dtype=bool)
    track_bins         = np.full((num_frames, num_tracks), -1, dtype=int)

    for n in range(num_frames):
        for t in range(num_tracks):
            b = _freq_to_bin(sin_freqs[n, t])
            if b is not None:
                track_bins[n, t] = b
                occupied_per_frame[n] |= (np.abs(freq_bins - sin_freqs[n, t]) <= tolerance_hz)

    remap = [dict() for _ in range(num_frames)]
    for n in range(1, num_frames):
        for t in range(num_tracks):
            b_prev, b_curr = track_bins[n-1, t], track_bins[n, t]
            if b_prev >= 0 and b_curr >= 0 and b_prev != b_curr:
                remap[n][b_curr] = b_prev   # sinusoidal moved k → k'

    return remap, occupied_per_frame


def merge_sms_to_stft(Xr_mag, Xr_phase, sin_freqs, sin_amps, sin_phases,
                      freq_bins, num_bins):
    """Merge sinusoidal + stochastic residual into a single STFT magnitude/phase table.

    Adds each sinusoidal track's complex contribution into the nearest STFT bin
    (complex-vector addition).  Used by 'full_stft' mode.

    Returns
    -------
    merged_mag, merged_phase : ndarray (num_frames, num_bins)
    sin_bin_map : bool ndarray (num_frames, num_bins)
        True at bins that received a sinusoidal contribution (for separation later).
    """
    num_frames  = Xr_mag.shape[0]
    merged_mag  = Xr_mag.copy()
    merged_phase = Xr_phase.copy()
    sin_bin_map = np.zeros((num_frames, num_bins), dtype=bool)

    for n in range(num_frames):
        for t in range(sin_freqs.shape[1]):
            f = sin_freqs[n, t]
            if np.isnan(f) or f <= 0:
                continue
            k = int(np.argmin(np.abs(freq_bins - f)))
            stoch    = merged_mag[n, k] * np.exp(1j * merged_phase[n, k])
            sino     = sin_amps[n, t]   * np.exp(1j * sin_phases[n, t])
            combined = stoch + sino
            merged_mag[n, k]   = np.abs(combined)
            merged_phase[n, k] = np.angle(combined)
            sin_bin_map[n, k]  = True

    return merged_mag, merged_phase, sin_bin_map


def separate_sinusoidal_from_stft(out_mag, out_phase, sin_freqs, sin_amps,
                                   freq_bins, sin_bin_map):
    """Inverse of merge_sms_to_stft: re-extract sinusoidal params after quantum transform.

    Attributes the full transformed bin to the sinusoidal component and zeros the
    residual at those bins.  Used by 'full_stft' mode after the quantum transform.

    Returns
    -------
    new_sin_amps, new_sin_phases : ndarray (num_frames, max_tracks)
    new_res_mag, new_res_phase   : ndarray (num_frames, num_bins)
    """
    num_frames     = out_mag.shape[0]
    num_tracks     = sin_freqs.shape[1]
    new_sin_amps   = np.zeros_like(sin_amps)
    new_sin_phases = np.zeros_like(sin_amps)
    new_res_mag    = out_mag.copy()
    new_res_phase  = out_phase.copy()

    for n in range(num_frames):
        for t in range(num_tracks):
            f = sin_freqs[n, t]
            if np.isnan(f) or f <= 0:
                continue
            k = int(np.argmin(np.abs(freq_bins - f)))
            new_sin_amps[n, t]   = out_mag[n, k]
            new_sin_phases[n, t] = out_phase[n, k]
            new_res_mag[n, k]    = 0.0
            new_res_phase[n, k]  = 0.0

    return new_sin_amps, new_sin_phases, new_res_mag, new_res_phase


def get_nearest_sinusoidal_phases(sin_freqs, sin_phases, freq_bins,
                                   num_frames, num_bins):
    """For each STFT bin/frame, return the phase of the nearest active sinusoidal track.

    Used by 'stochastic_unraveler' mode: amplitudes stay from the residual STFT,
    but phases are sourced from the harmonic model to tie the stochastic noise to
    the tonal content's phase.  Falls back to 0.0 when no track is active.

    Returns
    -------
    nearest_phase : ndarray (num_frames, num_bins)
    """
    nearest_phase = np.zeros((num_frames, num_bins))

    for n in range(num_frames):
        active = (~np.isnan(sin_freqs[n])) & (sin_freqs[n] > 0)
        af     = sin_freqs[n, active]
        ap     = sin_phases[n, active]
        if len(af) == 0:
            continue
        for k in range(num_bins):
            nearest_phase[n, k] = ap[np.argmin(np.abs(af - freq_bins[k]))]

    return nearest_phase


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


def _rn_event(onset, duration, midi, root_pc):
    """Build one chord event from a Roman-numeral analysis entry."""
    midi_sorted = sorted(midi)
    root_idx, chord_type, _ = detect_chord_root(midi_sorted)
    for i, m in enumerate(midi_sorted):
        if m % 12 == root_pc % 12:
            root_idx = i
            break
    return dict(onset=onset, duration=duration, midi=midi_sorted,
                root_index=root_idx, chord_type=chord_type,
                root_name=midi_to_note_name(
                    next((m for m in midi_sorted if m % 12 == root_pc % 12),
                         midi_sorted[0])))


def score_informed_events_bwv6(tempo_bpm=58, fermata_extra=1.0):
    """Score-informed chord events for BWV 6 'Christus, der ist mein Leben'.

    Key: F major.  8 measures of 4/4.  Fermatas at mm. 2, 4, 6, 8.
    Analysis of mm. 1-4 from annotated score; mm. 5-8 estimated from
    standard Bach chorale practice (C major area returning to F).

    Parameters
    ----------
    tempo_bpm : float
        Quarter-note tempo (default 58, matching ~41 s total duration).
    fermata_extra : float
        Extra seconds added at each fermata hold.

    Returns
    -------
    events : list[dict]
        Same format as extract_chord_events().
    """
    beat = 60.0 / tempo_bpm

    # F major pitch classes: F=5 G=7 A=9 Bb=10 C=0 D=2 E=4
    # MIDI reference: C3=48 D3=50 E3=52 F3=53 G3=55 A3=57 Bb3=58 C4=60

    # fmt: (measure, beat_in_measure, Roman numeral, [MIDI], root_pc)
    # Beats are 1-indexed; each has duration = 1 beat unless fermata.
    raw = [
        # ── m.1 ──────────────────────────────────────────────────
        (1, 1.0, 'I',        [53, 57, 60],     5),   # F A C
        (1, 3.0, 'V6',       [52, 55, 60],     0),   # C/E: E G C
        (1, 4.0, 'V42/IV',   [53, 57, 63],     5),   # F7 (triad): F A Eb
        # ── m.2 (fermata at end) ─────────────────────────────────
        (2, 1.0, 'IV6',      [50, 58, 65],     10),  # Bb/D: D Bb F
        (2, 2.0, 'V',        [48, 52, 55],     0),   # C: C E G
        (2, 3.0, 'V-I cad',  [48, 52, 55],     0),   # C: C E G (cadential)
        (2, 4.0, 'I',        [53, 57, 60],     5),   # F (fermata)
        # ── m.3 ──────────────────────────────────────────────────
        (3, 1.0, 'IV',       [53, 58, 62],     10),  # Bb: F Bb D
        (3, 2.0, 'I6',       [57, 60, 65],     5),   # F/A: A C F
        (3, 3.0, 'ii7',      [55, 58, 62],     7),   # Gm: G Bb D (triad)
        (3, 4.0, 'viio6',    [55, 58, 64],     4),   # Edim/G: G Bb E
        # ── m.4 (fermata at end) ─────────────────────────────────
        (4, 1.0, 'I',        [53, 57, 60],     5),   # F
        (4, 2.0, 'I6',       [57, 60, 65],     5),   # F/A
        (4, 3.0, 'V',        [48, 52, 55],     0),   # C: C E G
        (4, 4.0, 'I',        [53, 57, 60],     5),   # F (fermata)
        # ── m.5 (begins second phrase pair, C major area) ────────
        (5, 1.0, 'C:V',      [55, 59, 62],     7),   # G: G B D
        (5, 2.0, 'C:I',      [48, 52, 55],     0),   # C: C E G
        (5, 3.0, 'C:vi',     [57, 60, 64],     9),   # Am: A C E
        (5, 4.0, 'C:IV',     [53, 57, 60],     5),   # F: F A C
        # ── m.6 (fermata at end) ─────────────────────────────────
        (6, 1.0, 'C:ii',     [50, 53, 57],     2),   # Dm: D F A
        (6, 2.0, 'C:V',      [55, 59, 62],     7),   # G: G B D
        (6, 3.5, 'C:I',      [48, 52, 55],     0),   # C (fermata)
        # ── m.7 (return to F) ────────────────────────────────────
        (7, 1.0, 'F:vi',     [50, 53, 57],     2),   # Dm: D F A
        (7, 2.0, 'F:V6',     [52, 55, 60],     0),   # C/E: E G C
        (7, 3.0, 'F:IV',     [50, 58, 65],     10),  # Bb: D Bb F
        (7, 4.0, 'F:V',      [48, 52, 55],     0),   # C: C E G
        # ── m.8 (final cadence, fermata) ─────────────────────────
        (8, 1.0, 'F:I',      [53, 57, 60],     5),   # F
        (8, 2.0, 'F:V',      [48, 52, 55],     0),   # C: C E G
        (8, 3.0, 'F:I',      [53, 57, 60],     5),   # F (fermata, final)
    ]

    fermata_measures = {2, 4, 6, 8}
    cumulative_fermata = 0.0
    events = []
    for meas, bt, label, midi, root_pc in raw:
        # absolute onset: (measure-1)*4 beats + (beat-1) beats, scaled
        abs_beat = (meas - 1) * 4.0 + (bt - 1.0)
        onset = abs_beat * beat + cumulative_fermata

        # duration = 1 beat by default (overridden by next event's onset)
        dur = beat

        events.append(_rn_event(onset, dur, midi, root_pc))
        events[-1]['_rn_label'] = label

        # accumulate fermata time after the last beat of a fermata measure
        if meas in fermata_measures and bt >= 3.5:
            cumulative_fermata += fermata_extra

    # fix durations: each event lasts until the next one starts
    for i in range(len(events) - 1):
        events[i]['duration'] = events[i+1]['onset'] - events[i]['onset']
    events[-1]['duration'] = max(beat, 1.0)

    return events


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
                        delay=0,
                        swap_angle_in=0.9, swap_angle_out=0.1,
                        sequence='ramsey',
                        method='sin_phi',
                        exp_feed_back=False, feedback_swap_angle=np.pi/2,
                        N_mod=10, delta_f=None, fast=False):
    """NV-center quantum sensing, single delay line.

    method='sin_phi': one Ramsey/Hahn circuit per sample; 'qpsd': N_mod
    sub-circuits per sample with IQ demodulation.

    With delay > 0 a persistent register S + D[0..d-1] is kept (never
    re-initialised). Per step: encode phi[n] on S; SWAP_in pushes it into the
    line; SWAP_out weakly couples S to the oldest qubit; read <sz> from D[0];
    trace it out (the fresh qubit becomes the newest line element).

    exp_feed_back (sin_phi only): read D[0] BEFORE any swap (a clean sample),
    then a strong feedback swap recirculates it into S -- an explicit reverb
    loop; feedback_swap_angle is the loop gain (pi/2 ~ lossless)."""
    gamma_eff = phase_scale / T_phi

    # ── build gates once ─────────────────────────────────────────────────
    U_sw_in  = gate_partial_SWAP(swap_angle_in)
    U_sw_out = gate_partial_SWAP(swap_angle_out)
    fresh_vac = fock_dm(2, 0)

    if method == 'sin_phi':
        n_cycles = max(int(len(B_ac_hires) / (hires_sr * T_seq)), 1)
        phases, pp, pm = _compute_phases_from_signal(
            B_ac_hires * gamma_eff, hires_sr, T_phi, T_seq, n_cycles, sequence)

        aux_sz = np.zeros(n_cycles)

        # persistent system state — |+⟩⟨+| ⊗ |0⟩^delay
        rho_S_init = 0.5 * Qobj([[1, 1], [1, 1]])
        if delay > 0:
            rho = tensor([rho_S_init] + [fresh_vac] * delay)
            n_ext  = delay + 2          # S + D[0..d-1] + f
            # pre-embed fixed-position gates (phase-encoding gate changes each step)
            Usw_in  = _embed_gate_2q(U_sw_in,  0, delay + 1, n_ext)
            Usw_out = _embed_gate_2q(U_sw_out, 0, 1,         n_ext)
            # explicit-feedback swap (strong, dedicated to recirculation)
            U_sw_fb  = gate_partial_SWAP(feedback_swap_angle)
            Usw_fb   = _embed_gate_2q(U_sw_fb, 0, 1, n_ext)
            if fast:   # exact: these embedded gates are identity on untouched qubits
                Usw_in, Usw_out, Usw_fb = (_sparsify_op(Usw_in),
                                           _sparsify_op(Usw_out), _sparsify_op(Usw_fb))
            Usw_in_d  = Usw_in.dag()
            Usw_out_d = Usw_out.dag()
            Usw_fb_d  = Usw_fb.dag()
            keep    = [0] + list(range(2, n_ext))   # drop D[0] at position 1
        else:
            rho = rho_S_init
            Usw_in_2q  = U_sw_in
            Usw_in_2q_d = U_sw_in.dag()

        for n in range(n_cycles):
            U_enc = (_ramsey_unitary_sinphi(phases[n]) if sequence == 'ramsey'
                     else _hahn_echo_unitary(pp[n], pm[n]))

            if delay == 0:
                rho_enc = U_enc * rho * U_enc.dag()
                rho_ext = tensor([rho_enc, fresh_vac])
                rho_ext = Usw_in_2q * rho_ext * Usw_in_2q_d
                aux_sz[n] = float(np.real(expect(sigmaz(), rho_ext.ptrace([1]))))
                rho = rho_ext.ptrace([0])
            else:
                R = _embed_1q_gate(U_enc, 0, n_ext)
                if fast:
                    R = _sparsify_op(R)
                rho_ext = tensor([rho, fresh_vac])
                rho_ext = R * rho_ext * R.dag()
                rho_ext = Usw_in  * rho_ext * Usw_in_d
                if exp_feed_back:
                    # read delayed qubit FIRST (clean output), THEN strong
                    # feedback swap → recirculate D[0] into S (reverb loop)
                    aux_sz[n] = float(np.real(expect(sigmaz(), rho_ext.ptrace([1]))))
                    rho_ext = Usw_fb * rho_ext * Usw_fb_d
                else:
                    # weak SWAP_out memory, then read (carries back-action)
                    rho_ext = Usw_out * rho_ext * Usw_out_d
                    aux_sz[n] = float(np.real(expect(sigmaz(), rho_ext.ptrace([1]))))
                rho = rho_ext.ptrace(keep)
                # layout after ptrace: [S, D[1]…D[d-1], f_enc] — already correct

        return dict(phases=phases, aux_sigma_z=aux_sz,
                    tlist=np.arange(n_cycles) * T_seq,
                    sensing_rate=1.0 / T_seq, method='sin_phi')

    elif method == 'qpsd':
        n_total  = int(len(B_ac_hires) / (hires_sr * T_seq))
        n_cycles = max(n_total // N_mod, 1)
        dt_sub   = 2 * T_seq
        if delta_f is None:
            delta_f = 1.0 / (N_mod * dt_sub)

        all_phases, all_pp, all_pm = _compute_phases_from_signal(
            B_ac_hires * gamma_eff, hires_sr, T_phi, T_seq,
            n_cycles * N_mod, sequence)

        qpsd_phases = np.zeros(n_cycles)
        qpsd_amps   = np.zeros(n_cycles)
        true_phases = np.zeros(n_cycles)

        rho_S_init = 0.5 * Qobj([[1, 1], [1, 1]])
        if delay > 0:
            rho = tensor([rho_S_init] + [fresh_vac] * delay)
            n_ext = delay + 2
            Usw_in    = _embed_gate_2q(U_sw_in,  0, delay + 1, n_ext)
            Usw_in_d  = Usw_in.dag()
            Usw_out   = _embed_gate_2q(U_sw_out, 0, 1,         n_ext)
            Usw_out_d = Usw_out.dag()
            keep      = [0] + list(range(2, n_ext))
        else:
            rho = rho_S_init
            Usw_in_2q   = U_sw_in
            Usw_in_2q_d = U_sw_in.dag()

        for N in range(n_cycles):
            subs = np.zeros(N_mod)
            for k in range(N_mod):
                si      = N * N_mod + k
                theta_k = 2 * np.pi * delta_f * k * dt_sub
                pp_v    = all_phases[si] if si < len(all_phases) else 0.0
                pp_val  = all_pp[si]     if si < len(all_pp)     else 0.0
                pm_val  = all_pm[si]     if si < len(all_pm)     else 0.0

                U_enc = _qpsd_subcircuit(pp_v if sequence == 'ramsey' else pp_val,
                                         pm_val, theta_k, sequence)

                if delay == 0:
                    rho_enc = U_enc * rho * U_enc.dag()
                    rho_ext = tensor([rho_enc, fresh_vac])
                    rho_ext = Usw_in_2q * rho_ext * Usw_in_2q_d
                    subs[k] = float(np.real(expect(sigmaz(), rho_ext.ptrace([1]))))
                    rho = rho_ext.ptrace([0])
                else:
                    R = _embed_1q_gate(U_enc, 0, n_ext)
                    rho_ext = tensor([rho, fresh_vac])
                    rho_ext = R * rho_ext * R.dag()
                    rho_ext = Usw_in  * rho_ext * Usw_in_d
                    rho_ext = Usw_out * rho_ext * Usw_out_d
                    subs[k] = float(np.real(expect(sigmaz(), rho_ext.ptrace([1]))))
                    rho = rho_ext.ptrace(keep)

            phi_d, amp_d   = _demodulate_qpsd(subs, delta_f, dt_sub)
            qpsd_phases[N] = phi_d
            qpsd_amps[N]   = amp_d
            i0, i1 = N * N_mod, min(N * N_mod + N_mod, len(all_phases))
            true_phases[N] = np.mean(all_phases[i0:i1]) if i1 > i0 else 0.0

        return dict(phases=true_phases, qpsd_phases=qpsd_phases,
                    qpsd_amplitudes=qpsd_amps,
                    tlist=np.arange(n_cycles) * N_mod * dt_sub,
                    sensing_rate=1.0 / T_seq,
                    phase_readout_rate=1.0 / (N_mod * dt_sub),
                    method='qpsd')
    else:
        raise ValueError(f"Unknown sensing method '{method}'")


# ---- 8a-ii. MODE Ia — Dual-channel NV sensing (QFDN-inspired) ----

def trotterize_sensing_dual(B_ac_hires, hires_sr, T_phi, T_seq,
                              phase_scale=0.5,
                              delay=1,
                              delay1=None, delay2=None,
                              swap_angle_in=0.9,
                              swap_angle_out=0.1,
                              joint_mode='xx',
                              xx_coupling=0.3,
                              sequence='ramsey',
                              summation_mode='rms',
                              exp_feed_back=False, feedback_swap_angle=np.pi/2,
                              verbose=True):
    """
    Two-channel NV-center quantum sensing with independent delay lines.

    ── Layout ────────────────────────────────────────────────────────────
    Maintained state (n_maint = 2+d1+d2 qubits, block layout):
      S1(0)  S2(1) | D1[0](2)…D1[d1-1](d1+1) | D2[0](d1+2)…D2[d2-1](d1+d2+1)

    S1 and S2 are NEVER re-initialised — quantum correlations across steps
    are preserved.  S1 feeds D1 only; S2 feeds D2 only.  The two lines
    couple exclusively via the joint unitary U acting on (S1, S2).

    ── Step sequence ─────────────────────────────────────────────────────
    1. Encode φ[n] on S1(0) and S2(1) via Ramsey/Hahn unitary.
    2. Joint coupling U_{S1,S2} on (0,1)  [all joint_modes].
    3. Tensor fresh vacuum qubits f1 at n_maint and f2 at n_maint+1.
    4. SWAP_in (angle_in≈0.9): S1(0)↔f1  and  S2(1)↔f2.
       Pushes most of the encoding into the newest delay slot.
    5. SWAP_out (angle_out≈0.1): S1(0)↔D1[0](2)  and  S2(1)↔D2[0](d1+2).
       Creates quantum memory: S receives a weak imprint of the oldest
       delay qubit, and vice versa.
    6. Read ⟨σz⟩ from D1[0](2) → r1  and  D2[0](d1+2) → r2
       (measured AFTER SWAP_out so they carry the memory back-action).
    7. Summation node: combine r1, r2 via summation_mode ('rms' or 'sum').
       No classical matrix mixing — coupling between channels is purely
       from the S1/S2 joint unitary chosen by joint_mode.
    8. Output sample recorded.
    9. Trace out D1[0](2) and D2[0](d1+2).
   10. Permute maintained state: bubble f1_enc into end of D1 block,
       restoring canonical block layout for the next step.

    ── Joint coupling ────────────────────────────────────────────────────
    'xx'          : U = exp(-i·θ·σx⊗σx),  θ = xx_coupling
    'hadamard_id' : U = H⊗I  (Hadamard on S1 only)
    'fdn_hadamard': QFDN circulant Hadamard (Rocchesso 2025 Eq. 2.3) on (S1, S2).
                    Maximally diffusive, A² = I.  Mixes both channels before
                    they enter the delay lines.

    ── Summation node ────────────────────────────────────────────────────
    'rms' : sign(r1+r2)·√((r1²+r2²)/2)
    'sum' : (r1+r2)/2
    'ch1' / 'ch2' : single-channel passthrough

    ── Hilbert-space layout (interleaved, 2d+4 qubits) ──────────────────
    Extended state per step:
      S1(0) ⊗ S2(1) ⊗ D1[0](2) ⊗ D2[0](3) ⊗ D1[1](4) ⊗ D2[1](5) ⊗ …
      ⊗ D1[d-1](2d) ⊗ D2[d-1](2d+1) ⊗ fresh_1(2d+2) ⊗ fresh_2(2d+3)

    ── Step sequence ────────────────────────────────────────────────────
    1. Embed Ramsey/Hahn unitary U_φ[n] on S1 (idx 0) and S2 (idx 1).
    2. Apply joint coupling U_joint on (S1, S2) = qubits (0, 1).
       This entangles the two channels before they enter the delay.
    3. Attach fresh_1 = |0⟩ (idx 2d+2) and fresh_2 = |0⟩ (idx 2d+3).
    4. Partial-SWAP S1(0) ↔ fresh_1(2d+2): encoded S1 migrates into delay.
    5. Partial-SWAP S2(1) ↔ fresh_2(2d+3): encoded S2 migrates into delay.
    6. Read ⟨σz⟩ from D1[0] (idx 2) → r1  (channel-1 readout).
    7. Read ⟨σz⟩ from D2[0] (idx 3) → r2  (channel-2 readout).
    8. Summation node → mono sample out[n].
    9. Trace out D1[0] (idx 2) and D2[0] (idx 3); keep rest → new state.

    ── Joint coupling modes ─────────────────────────────────────────────
    'xx'  :  U = exp(-iθ σx⊗σx),  θ = xx_coupling
             H_int = θ σx^{S1} ⊗ σx^{S2}   (transverse Ising / Bell-pair generator)
             Evolution: cos(θ)·𝟙 - i sin(θ)·(σx⊗σx)

    'fdn_hadamard':
             U = A  (QFDN circulant Hadamard, Rocchesso 2025 Eq. 2.3)
             A = ½[[-1,1,1,1],[1,-1,1,1],[1,1,-1,1],[1,1,1,-1]]
             Maximally diffusive, circulant, Householder reflector.
             Maps |00⟩ → -½|00⟩+½|01⟩+½|10⟩+½|11⟩.
             Unilossless (orthogonal, A² = I, det A = -1).

    'hadamard_id':
             U = H⊗I — Hadamard combgate on S1, identity on S2.
             |0⟩ → (|0⟩+|1⟩)/√2,  |1⟩ → (|0⟩-|1⟩)/√2  on S1 only.

    ── Summation node (QFDN Eq. 2.1) ───────────────────────────────────
    'rms': out = sign(r1+r2) × √((r1²+r2²)/2)   ← preserves QFDN formula
    'sum': out = (r1 + r2) / 2                   ← arithmetic mean
    'ch1' / 'ch2': single-channel passthrough for diagnostics

    Parameters
    ----------
    delay1, delay2 : int  delay lengths for D1 and D2 (default: `delay` for both)
                          Practical limit: d1+d2 ≤ 9  (dim = 2^{2+d1+d2})
    swap_angle_in  : float  SWAP angle S↔fresh  (default 0.9, ≈strong push into delay)
    swap_angle_out : float  SWAP angle S↔D[0]   (default 0.1, ≈weak quantum memory)
    exp_feed_back  : bool   explicit-feedback ("reverb") mode. Read D1[0]/D2[0]
                            FIRST (clean outputs; expectation is non-destructive),
                            THEN apply a strong feedback SWAP (feedback_swap_angle,
                            default π/2 full swap) S↔D[0] to recirculate the
                            delayed population instead of discarding it at
                            trace-out. swap_angle_out is ignored when True.
    feedback_swap_angle : float  reverb-time knob / loop gain (π/2 ≈ lossless).
    """
    d1 = delay1 if delay1 is not None else delay
    d2 = delay2 if delay2 is not None else delay
    assert 1 <= d1 and 1 <= d2 and d1 + d2 <= 9, (
        f"delay1={d1}, delay2={d2}: each ≥1, sum ≤9")

    n_maint  = 2 + d1 + d2      # maintained qubits: S1,S2,D1-block,D2-block
    n_ext    = n_maint + 2      # extended (+f1,f2 appended each step)
    idx_f1   = n_maint          # f1 position in extended state
    idx_f2   = n_maint + 1      # f2 position in extended state
    idx_D1_0 = 2                # oldest D1 qubit
    idx_D2_0 = d1 + 2           # oldest D2 qubit

    # keep everything except D1[0](2) and D2[0](d1+2) after each step
    keep = ([0, 1]
            + list(range(3, d1 + 2))           # D1[1..d1-1]
            + list(range(d1 + 3, d1 + d2 + 2)) # D2[1..d2-1]
            + [d1 + d2 + 2, d1 + d2 + 3])      # f1_enc, f2_enc

    if verbose:
        print("=" * 60)
        print("MODE Ia: DUAL-CHANNEL SENSING — Physics Summary")
        print("=" * 60)
        print(f"  Maintained state : {n_maint} qubits  (dim={2**n_maint})")
        print(f"  Extended / step  : {n_ext} qubits  (dim={2**n_ext})")
        print(f"  d1={d1}  d2={d2}  |  D1[0] @ idx {idx_D1_0},  D2[0] @ idx {idx_D2_0}")
        print(f"  Sequence         : {sequence}")
        print(f"  SWAP_in  angle   : {swap_angle_in:.3f} rad")
        print(f"  SWAP_out angle   : {swap_angle_out:.3f} rad  (quantum memory)")
        print()
        if joint_mode == 'xx':
            print(f"  Joint coupling: exp(-i·{xx_coupling:.3f}·σx⊗σx)  on (S1,S2)")
        elif joint_mode == 'fdn_hadamard':
            print("  Joint coupling: QFDN Householder A on (S1,S2)  [maximally diffusive, A²=I]")
        elif joint_mode == 'hadamard_id':
            print("  Joint coupling: H⊗I  on (S1,S2)")
        print(f"  Summation node   : '{summation_mode}'")
        print("=" * 60)

    # ── build gates ───────────────────────────────────────────────────────
    U_sw_in  = gate_partial_SWAP(swap_angle_in)
    U_sw_out = gate_partial_SWAP(swap_angle_out)
    U_full   = gate_partial_SWAP(np.pi / 2)   # for permutation step

    Usw_in1    = _embed_gate_2q(U_sw_in,  0, idx_f1,   n_ext)   # S1 ↔ f1
    Usw_in2    = _embed_gate_2q(U_sw_in,  1, idx_f2,   n_ext)   # S2 ↔ f2
    Usw_out1   = _embed_gate_2q(U_sw_out, 0, idx_D1_0, n_ext)   # S1 ↔ D1[0]
    Usw_out2   = _embed_gate_2q(U_sw_out, 1, idx_D2_0, n_ext)   # S2 ↔ D2[0]
    Usw_in1_d  = Usw_in1.dag()
    Usw_in2_d  = Usw_in2.dag()
    Usw_out1_d = Usw_out1.dag()
    Usw_out2_d = Usw_out2.dag()

    # explicit-feedback swaps (strong, dedicated to recirculation)
    U_sw_fb   = gate_partial_SWAP(feedback_swap_angle)
    Usw_fb1   = _embed_gate_2q(U_sw_fb, 0, idx_D1_0, n_ext)   # S1 ↔ D1[0]
    Usw_fb2   = _embed_gate_2q(U_sw_fb, 1, idx_D2_0, n_ext)   # S2 ↔ D2[0]
    Usw_fb1_d = Usw_fb1.dag()
    Usw_fb2_d = Usw_fb2.dag()

    U_joint     = _build_joint_sensing_unitary(joint_mode, xx_coupling)
    U_joint_ext = _embed_2q_gate_at_01(U_joint, n_ext)
    U_joint_d   = U_joint_ext.dag()

    # Permutation gates (on n_maint qubits) applied after ptrace:
    # After ptrace: [S1,S2, D1[1..d1-1], D2[1..d2-1], f1_enc, f2_enc]
    # Wanted      : [S1,S2, D1[1..d1-1], f1_enc, D2[1..d2-1], f2_enc]
    # Bubble f1_enc left from position d1+d2 to d1+1 using (d2-1) full SWAPs.
    perm_gates = []
    if d2 > 1:
        for k in range(d1 + d2, d1 + 1, -1):
            Up = _embed_gate_2q(U_full, k - 1, k, n_maint)
            perm_gates.append((Up, Up.dag()))

    # ── initial maintained state: |+⟩⟨+|⊗|+⟩⟨+|⊗|0⟩^{d1+d2} ───────────
    rho_S   = 0.5 * Qobj([[1, 1], [1, 1]])
    rho     = tensor([rho_S, rho_S] + [fock_dm(2, 0)] * (d1 + d2))
    fresh_vac = fock_dm(2, 0)

    # ── signal → phases ───────────────────────────────────────────────────
    gamma_eff = phase_scale / T_phi
    n_cycles  = max(int(len(B_ac_hires) / (hires_sr * T_seq)), 1)
    phases, pp, pm = _compute_phases_from_signal(
        B_ac_hires * gamma_eff, hires_sr, T_phi, T_seq, n_cycles, sequence)

    ch1_sz       = np.zeros(n_cycles)
    ch2_sz       = np.zeros(n_cycles)
    out_combined = np.zeros(n_cycles)

    for n in range(n_cycles):
        U_enc = (_ramsey_unitary_sinphi(phases[n]) if sequence == 'ramsey'
                 else _hahn_echo_unitary(pp[n], pm[n]))
        R1 = _embed_1q_gate(U_enc, 0, n_ext)
        R2 = _embed_1q_gate(U_enc, 1, n_ext)

        # Extend with fresh auxiliaries
        rho_ext = tensor([rho, fresh_vac, fresh_vac])

        # 1. Encode φ[n] on S1 and S2
        rho_ext = R1 * rho_ext * R1.dag()
        rho_ext = R2 * rho_ext * R2.dag()

        # 2. Joint coupling U_{S1,S2}
        rho_ext = U_joint_ext * rho_ext * U_joint_d

        # 3. SWAP_in: push encoding into newest delay slot
        rho_ext = Usw_in1 * rho_ext * Usw_in1_d
        rho_ext = Usw_in2 * rho_ext * Usw_in2_d

        if exp_feed_back:
            # 4'. Explicit feedback: read D1[0]/D2[0] FIRST (clean outputs),
            #     THEN strong feedback swaps → recirculate into S1/S2 (reverb).
            r1 = float(np.real(expect(sigmaz(), rho_ext.ptrace([idx_D1_0]))))
            r2 = float(np.real(expect(sigmaz(), rho_ext.ptrace([idx_D2_0]))))
            rho_ext = Usw_fb1 * rho_ext * Usw_fb1_d
            rho_ext = Usw_fb2 * rho_ext * Usw_fb2_d
        else:
            # 4. SWAP_out: weak coupling to oldest delay qubit → quantum memory
            rho_ext = Usw_out1 * rho_ext * Usw_out1_d
            rho_ext = Usw_out2 * rho_ext * Usw_out2_d
            # 5. Read ⟨σz⟩ from D1[0] and D2[0] after SWAP_out
            r1 = float(np.real(expect(sigmaz(), rho_ext.ptrace([idx_D1_0]))))
            r2 = float(np.real(expect(sigmaz(), rho_ext.ptrace([idx_D2_0]))))

        ch1_sz[n] = r1
        ch2_sz[n] = r2

        # 6. Summation node
        if summation_mode == 'rms':
            sgn = np.sign(r1 + r2) if abs(r1 + r2) > 1e-12 else 1.0
            out_combined[n] = float(sgn) * np.sqrt((r1**2 + r2**2) / 2.0)
        elif summation_mode == 'sum':
            out_combined[n] = (r1 + r2) / 1.0
        elif summation_mode == 'ch1':
            out_combined[n] = r1
        elif summation_mode == 'ch2':
            out_combined[n] = r2
        else:
            out_combined[n] = (r1 + r2) / 1.0

        # 7. Trace out D1[0](2) and D2[0](d1+2)
        rho = rho_ext.ptrace(keep)

        # 8. Permute: bubble f1_enc to end of D1 block
        for Up, Up_d in perm_gates:
            rho = Up * rho * Up_d

    return dict(
        phases=phases,
        ch1_sz=ch1_sz,
        ch2_sz=ch2_sz,
        output_combined=out_combined,
        tlist=np.arange(n_cycles) * T_seq,
        sensing_rate=1.0 / T_seq,
        joint_mode=joint_mode,
        summation_mode=summation_mode,
        delay=max(d1, d2),
        delay1=d1,
        delay2=d2,
        method='dual_sin_phi',
    )


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
    #omega_s = 0
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

    sys_sx, sys_sy, sys_sz, sys_pop = [], [], [], []
    sys_amp, sys_phase_arr  = [], []
    meas_record = []

    for n in range(runs):
        rho_s = rho.ptrace(0)
        sx  = float(np.real(expect(sigmax(), rho_s)))
        sz  = float(np.real(expect(sigmaz(), rho_s)))
        sy  = float(np.real(expect(sigmay(), rho_s)))
        pop = float(np.real(expect(Qobj([[0,0],[0,1]]), rho_s)))

        sys_sx.append(sx);  sys_sy.append(sy);  sys_sz.append(sz)
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
        p1 = max(float(np.real((P1 * rho_post).tr())), 0.0)   # B15: explicit tr(P1·ρ), no 1-p0 shortcut
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
        sys_sigmay=np.array(sys_sy),
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
                       sensing_rate=4000, T_phi=None, max_freq=2000,
                       phase_scale=0.5, method='sin_phi',
                       N_mod=10,
                       swap_angle_in=0.9, swap_angle_out=0.1,
                       sequence='ramsey',
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
        phase_scale=phase_scale, delay=delay,
        swap_angle_in=swap_angle_in, swap_angle_out=swap_angle_out,
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


def _pipeline_sensing_dual(audio_path, delay=1, delay1=None, delay2=None,
                            phase=np.pi,
                            sensing_rate=4000, T_phi=None, max_freq=2000,
                            phase_scale=0.5,
                            joint_mode='xx', xx_coupling=0.3,
                            swap_angle_in=0.9, swap_angle_out=0.1,
                            sequence='ramsey',
                            summation_mode='rms',
                            audio_sr=44100, verbose=True):
    """Mode Ia pipeline — dual-channel NV sensing with entangled systems."""
    if verbose:
        print("=== MODE Ia: Dual-Channel NV Quantum Sensing ===")

    data, sr = load_wav(audio_path, max_freq=max_freq)
    orig_dur = len(data) / sr

    stretch, _ = compute_stretch_factor(max_freq, sensing_rate)
    B_hires    = time_stretch(data, stretch) if stretch > 1 else data

    T_seq = 1.0 / sensing_rate
    if T_phi is None:
        T_phi = 0.8 * T_seq

    d1_eff = delay1 if delay1 is not None else delay
    d2_eff = delay2 if delay2 is not None else delay
    if verbose:
        print(f"  sensing_rate={sensing_rate} Hz  T_phi={T_phi:.5f}  "
              f"delay1={d1_eff}  delay2={d2_eff}  "
              f"stretch={stretch:.1f}  joint_mode={joint_mode}")

    res = trotterize_sensing_dual(
        B_hires, sr, T_phi, T_seq,
        phase_scale=phase_scale, delay=delay, delay1=delay1, delay2=delay2,
        swap_angle_in=swap_angle_in, swap_angle_out=swap_angle_out,
        joint_mode=joint_mode, xx_coupling=xx_coupling,
        sequence=sequence, summation_mode=summation_mode, verbose=verbose)

    out = res['output_combined']
    if stretch > 1:
        out = time_compress(out, stretch)
    out = out - np.mean(out)

    n_audio = int(orig_dur * audio_sr)
    audio   = resample(out, max(n_audio, 2))
    peak    = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.8

    out_path = _output_path(audio_path, 'sensing_dual')
    soundfile.write(out_path, audio.astype(np.float32), audio_sr)
    if verbose:
        print(f"  Saved: {out_path}")

    return dict(
        y_quantum=audio, y_original=data,
        fs=audio_sr, fs_original=sr,
        mode='sensing_dual', audio_path=audio_path,
        trotterize_result=res,
        output_paths=dict(sensing_dual=out_path),
        params=dict(sensing_rate=sensing_rate, T_phi=T_phi, delay=delay,
                    phase_scale=phase_scale, joint_mode=joint_mode,
                    xx_coupling=xx_coupling, summation_mode=summation_mode,
                    sequence=sequence),
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

    # ---------- mode-specific preprocessing --------------------------------
    # mode_sms controls which frequency bins are passed through the quantum kernel
    # and what amplitude/phase input they receive:
    #   'stochastic_only'    — bins NEVER occupied by a sinusoidal track (default)
    #   'reverse_sinusoidal' — bins occupied by a sinusoidal in SOME (not all) frames;
    #                          amplitude input tracks the sinusoidal's previous-frame bin
    #   'full_stft'          — every bin; sinusoidal and residual merged before transform
    #   'stochastic_unraveler' — same bins as stochastic_only, but phases replaced by
    #                            the nearest sinusoidal track's phase
    freq_bins  = ana['freq_bins']
    num_frames, num_bins = ana['Xr_mag'].shape

    if mode_sms == 'stochastic_only':
        mask        = get_stochastic_only_bins(ana['sin_freqs'], freq_bins, tolerance_hz)
        bin_indices = np.where(mask)[0]
        in_mag      = ana['Xr_mag']
        in_phase    = ana['Xr_phase']

    elif mode_sms == 'reverse_sinusoidal':
        remap, occupied_per_frame = get_reverse_sinusoidal_mapping(
            ana['sin_freqs'], freq_bins, tolerance_hz)
        # bins occupied in SOME frames (but not necessarily all) — complement of "always stochastic"
        bin_mask    = ~np.all(occupied_per_frame, axis=0)
        bin_indices = np.where(bin_mask)[0]
        # for each frame n, bin k', read magnitude from the bin the sinusoidal occupied
        # one frame earlier (continuity remapping); identity where no track moved there
        in_mag = ana['Xr_mag'].copy()
        for n in range(num_frames):
            for k_prime in bin_indices:
                k_src = remap[n].get(k_prime, k_prime)
                in_mag[n, k_prime] = ana['Xr_mag'][n, k_src]
        in_phase = ana['Xr_phase']

    elif mode_sms == 'full_stft':
        bin_indices = np.arange(num_bins)
        in_mag, in_phase, sin_bin_map = merge_sms_to_stft(
            ana['Xr_mag'], ana['Xr_phase'],
            ana['sin_freqs'], ana['sin_amps'], ana['sin_phases'],
            freq_bins, num_bins)

    else:  # 'stochastic_unraveler'
        mask        = get_stochastic_only_bins(ana['sin_freqs'], freq_bins, tolerance_hz)
        bin_indices = np.where(mask)[0]
        in_mag      = ana['Xr_mag']
        in_phase    = get_nearest_sinusoidal_phases(
            ana['sin_freqs'], ana['sin_phases'], freq_bins, num_frames, num_bins)

    if verbose:
        print(f"  Bins ({mode_sms}): {len(bin_indices)} / {num_bins} | "
              f"delay={delay}  phase={phase:.3f}")

    # ---------- normalise --------------------------------------------------
    if normperbin:
        mag_norm, scale_per_bin = normalize_amplitudes_per_bin(in_mag)
    else:
        mag_norm, sf = normalize_amplitudes(in_mag)
        scale_per_bin = np.full(num_bins, sf)

    # ---------- quantum transform ------------------------------------------
    out_mag, out_phase, diag = _quantum_colour_transform_per_bin(
        mag_norm, in_phase, freq_bins,
        bin_indices, delay, phase,
        DeltaT=DeltaT, lam=lam,
        omega_cont=omega_cont, omega_cont_z=omega_cont_z,
        gamma_x=gamma_x, gamma_z=gamma_z,
        headroom=headroom, N=[2], evuType=evuType, verbose=verbose)

    # ---------- mode-specific postprocessing + synthesis -------------------
    if mode_sms == 'full_stft':
        out_mag_scaled = out_mag * scale_per_bin[np.newaxis, :]
        new_sin_amps, _, res_mag, res_phase = separate_sinusoidal_from_stft(
            out_mag_scaled, out_phase,
            ana['sin_freqs'], ana['sin_amps'], freq_bins, sin_bin_map)
        synth = synthesise_sms(ana,
                               res_mag=res_mag, res_phase=res_phase,
                               sin_amps_mod=new_sin_amps,
                               use_quantum_amp=use_quantum_amp,
                               use_quantum_phase=use_quantum_phase,
                               softclip=softclip,
                               win_norm_dB_override=ana['win_norm_dB'])
    else:
        res_mag   = out_mag   * scale_per_bin[np.newaxis, :]
        res_phase = out_phase
        synth = synthesise_sms(ana,
                               res_mag=res_mag, res_phase=res_phase,
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
                    omega_cont=omega_cont, lam=lam, normperbin=normperbin,
                    mode_sms=mode_sms),
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
        print(f"  Bins (stochastic_only): {len(bin_indices)} / {len(ana['freq_bins'])} | "
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
                       score_events=None,
                       M=2501, N_fft=4096, H=1024, t=-80, nH=60,
                       minf0=50, maxf0=2000, f0et=5,
                       harmDevSlope=0.01, minSineDur=0.02,
                       Ns=2048, stocf=0.2,
                       verbose=True):
    if verbose:
        print("=== MODE IV: Quantum Tonal Brightness (Harmony) ===")

    # Load
    data, fs = load_wav_full(audio_path)

    # Chord events: score-informed (bypass basic_pitch) or auto-detected
    if score_events is not None:
        events = score_events
        if verbose:
            labels = [e.get('_rn_label', e.get('chord_type', '?'))
                      for e in events]
            print(f"  Score-informed: {len(events)} chord events")
            print(f"  Progression: {' → '.join(labels[:12])}"
                  + (' ...' if len(labels) > 12 else ''))
    else:
        events, _ = extract_chord_events(audio_path, chord_size=chord_size,
                                          onset_tolerance=onset_tolerance,
                                          min_velocity=min_velocity,
                                          verbose=verbose)
    if len(events) < 2:
        raise ValueError(f"Need ≥2 chord events, got {len(events)}. "
                         f"Pass score_events= or lower onset_tolerance.")

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
        ana['stocEnv'], H, ana['N_stoc'], ana.get('stoc_fs', 44100), melScale=1, phase_func=None)  # B14
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
        'sensing_dual':   _pipeline_sensing_dual,
        'sms_colour':     _pipeline_sms_colour,
        'sms_trajectory': _pipeline_sms_trajectory,
        'harmony':        _pipeline_harmony,
    }
    if mode not in dispatch:
        raise ValueError(f"Unknown mode '{mode}'. "
                         f"Choose from: {list(dispatch.keys())}")
    return dispatch[mode](audio_path, **kwargs)


# ============================================================
# SECTION 12 — PLOT FUNCTIONS
# ============================================================

def _mode_label(mode):
    return {'sensing':'I — Sensing', 'sensing_dual':'Ia — Dual Sensing',
            'sms_colour':'II — SMS Colour',
            'sms_trajectory':'III — SMS Trajectory', 'harmony':'IV — Harmony'}.get(mode, mode)

def _mode_short(mode):
    return {'sensing':'I', 'sensing_dual':'Ia',
            'sms_colour':'II', 'sms_trajectory':'III', 'harmony':'IV'}.get(mode, mode)

def _pick_representative_bins(bin_keys, n=4):
    if len(bin_keys) <= n:
        return bin_keys
    indices = np.linspace(0, len(bin_keys)-1, n, dtype=int)
    return [bin_keys[i] for i in indices]

def _make_plot_title(result):
    """Build a two-line title summarising mode and key parameters."""
    mode   = result['mode']
    params = result.get('params', {})
    phase  = params.get('phase', '?')
    if isinstance(phase, float) and abs(phase - np.pi) < 1e-6:
        phase_str = 'π'
    elif isinstance(phase, (int, float)):
        phase_str = f'{phase:.3f}'
    else:
        phase_str = str(phase)
    line1 = f"Quantum Audio  —  Mode {_mode_label(mode)}"
    line2 = f"delay={params.get('delay','?')}  φ={phase_str}"
    if mode in ('sensing', 'sensing_dual'):
        line2 += (f"  rate={params.get('sensing_rate','?')} Hz"
                  f"  seq={params.get('sequence','?')}")
        if mode == 'sensing_dual':
            line2 += (f"  joint={params.get('joint_mode','?')}"
                      f"  sum={params.get('summation_mode','?')}")
        else:
            line2 += f"  method={params.get('method','?')}"
    elif mode == 'sms_colour':
        line2 += (f"  bins={params.get('mode_sms','stochastic_only')}"
                  f"  γx={params.get('gamma_x','?')}  γz={params.get('gamma_z','?')}"
                  f"  ωcont={params.get('omega_cont','?')}  λ={params.get('lam','?')}")
    elif mode == 'sms_trajectory':
        line2 += (f"  bins={params.get('mode_sms','stochastic_only')}"
                  f"  basis={params.get('measurement_basis','?')}"
                  f"  γx={params.get('gamma_x','?')}  γz={params.get('gamma_z','?')}")
    elif mode == 'harmony':
        line2 += (f"  g={params.get('coupling_g','?')}"
                  f"  brightness={params.get('brightness','?')}"
                  f"  ωsys={params.get('omega_sys','?')}")
    return f"{line1}\n{line2}"

def _plot_evolution_row(ax_sys, ax_out, result):
    """Fill the bottom row (system state | quantum output) of the general overview plot."""
    mode = result['mode']

    if mode == 'sensing':
        res     = result['trotterize_result']
        meth    = res['method']
        tl      = res['tlist']
        phi     = res['phases']
        readout = res['qpsd_phases'] if meth == 'qpsd' else res['aux_sigma_z']
        lbl_out = 'QPSD phase (rad)' if meth == 'qpsd' else 'sin(φ) readout'
        ax_sys.plot(tl, phi, 'r-', lw=0.6)
        ax_sys.axhline( np.pi/2, color='grey', ls='--', alpha=0.4, lw=0.8)
        ax_sys.axhline(-np.pi/2, color='grey', ls='--', alpha=0.4, lw=0.8)
        ax_sys.set_ylabel('Phase (rad)'); ax_sys.set_xlabel('Time (s)')
        ax_sys.set_title('Accumulated phase φ')
        ax_out.plot(tl, readout, 'm-', lw=0.6)
        ax_out.set_ylabel(lbl_out); ax_out.set_xlabel('Time (s)')
        ax_out.set_title('Ancilla readout')

    elif mode == 'sensing_dual':
        res = result['trotterize_result']
        tl  = res['tlist']
        ax_sys.plot(tl, res['ch1_sz'], lw=0.6, color='steelblue', label='ch1 ⟨σz⟩')
        ax_sys.plot(tl, res['ch2_sz'], lw=0.6, color='tomato',    label='ch2 ⟨σz⟩')
        ax_sys.legend(fontsize=8)
        ax_sys.set_ylabel('⟨σz⟩'); ax_sys.set_xlabel('Time (s)')
        ax_sys.set_title(f'Per-channel readout  [joint={res["joint_mode"]}]')
        ax_out.plot(tl, res['output_combined'], lw=0.6, color='darkorchid')
        ax_out.set_ylabel('Combined out'); ax_out.set_xlabel('Time (s)')
        ax_out.set_title(f'Summation node  [{res["summation_mode"]}]')

    elif mode == 'sms_colour':
        diag = result['quantum_result']['diagnostics']
        if diag:
            pops = np.vstack([v['sys_pop']     for v in diag.values()])
            amps = np.vstack([v['aux_amp_out'] for v in diag.values()])
            fr   = np.arange(pops.shape[1])
            ax_sys.plot(fr, np.mean(pops, axis=0), lw=0.9, color='steelblue')
            ax_sys.fill_between(fr,
                np.percentile(pops, 10, axis=0), np.percentile(pops, 90, axis=0),
                alpha=0.2, color='steelblue', label='10–90%ile')
            ax_out.plot(fr, np.mean(amps, axis=0), lw=0.9, color='darkorchid')
            ax_out.fill_between(fr,
                np.percentile(amps, 10, axis=0), np.percentile(amps, 90, axis=0),
                alpha=0.2, color='darkorchid', label='10–90%ile')
            ax_sys.legend(fontsize=7); ax_out.legend(fontsize=7)
        ax_sys.set_ylabel('⟨ρ₁₁⟩'); ax_sys.set_xlabel('Frame')
        ax_sys.set_title('System population — mean across selected bins')
        ax_out.set_ylabel('Amplitude'); ax_out.set_xlabel('Frame')
        ax_out.set_title('Aux amplitude out — mean across selected bins')

    elif mode == 'sms_trajectory':
        diag = result.get('diagnostics', {})
        if diag:
            pops = np.vstack([v['sys_pop'] for v in diag.values()])
            amps = np.vstack([v['sys_amp'] for v in diag.values()])
            fr   = np.arange(pops.shape[1])
            ax_sys.plot(fr, np.mean(pops, axis=0), lw=0.9, color='steelblue')
            ax_sys.fill_between(fr,
                np.percentile(pops, 10, axis=0), np.percentile(pops, 90, axis=0),
                alpha=0.2, color='steelblue', label='10–90%ile')
            ax_out.plot(fr, np.mean(amps, axis=0), lw=0.9, color='darkorchid')
            ax_out.fill_between(fr,
                np.percentile(amps, 10, axis=0), np.percentile(amps, 90, axis=0),
                alpha=0.2, color='darkorchid', label='10–90%ile')
            ax_sys.legend(fontsize=7); ax_out.legend(fontsize=7)
        ax_sys.set_ylabel('⟨ρ₁₁⟩'); ax_sys.set_xlabel('Frame')
        ax_sys.set_title('System population — mean across selected bins')
        ax_out.set_ylabel('Amplitude'); ax_out.set_xlabel('Frame')
        ax_out.set_title('System amplitude — mean across selected bins')

    elif mode == 'harmony':
        diagonals = result['diagonals']
        winners   = result['winners']
        mod_log   = result['modulation_log']
        t  = np.arange(len(diagonals))
        vc = ['steelblue', 'darkorange', 'seagreen']
        ax_sys.stackplot(t, diagonals[:,0], diagonals[:,1], diagonals[:,2],
                         labels=['root','3rd','5th'], colors=vc, alpha=0.7)
        ax_sys.set_ylabel('Probability'); ax_sys.set_xlabel('Transition')
        ax_sys.set_title('Voice probabilities')
        ax_sys.legend(fontsize=7, loc='upper right')
        boosts = [e['boost_factor'] for e in mod_log]
        for i, (b, w) in enumerate(zip(boosts, winners)):
            ax_out.bar(i, b, color=vc[w], alpha=0.7, edgecolor='none')
        ax_out.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
        ax_out.set_ylabel('Boost ×'); ax_out.set_xlabel('Transition')
        ax_out.set_title('Brightness boost (colour = winning voice)')


def plot_quantum_audio(result, max_freq=5000, figsize=(14, 10)):
    """
    Universal 3×2 overview for all four modes.

    Row 0  input waveform          | output waveform
    Row 1  input spectrogram       | output spectrogram
    Row 2  system state evolution  | quantum output evolution

    Title summarises mode, delay, phase, and mode-specific key params.
    For per-bin / per-transition diagnostics call the matching detail function:
        plot_details_sensing(result)
        plot_details_sms_colour(result)
        plot_details_sms_trajectory(result)
        plot_details_harmony(result)
    """
    mode = result['mode']
    fs   = result['fs']
    y_q  = result['y_quantum']
    y_o  = result['y_original']

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(_make_plot_title(result), fontsize=10, y=1.01)

    # Row 0 — waveforms
    t_o = np.arange(len(y_o)) / fs
    t_q = np.arange(len(y_q)) / fs
    axes[0,0].plot(t_o, y_o, lw=0.4, color='steelblue')
    axes[0,0].set_title('Input signal'); axes[0,0].set_ylabel('Amplitude')
    axes[0,0].set_xlim([0, t_o[-1]])
    axes[0,1].plot(t_q, y_q, lw=0.4, color='darkorchid')
    axes[0,1].set_title(f'Output signal — quantum ({_mode_short(mode)})')
    axes[0,1].set_xlim([0, t_q[-1]])

    # Row 1 — spectrograms
    N_spect = 2048
    for ax, sig, ttl in [(axes[1,0], y_o, 'Input'),
                          (axes[1,1], y_q, f'Output — quantum ({_mode_short(mode)})')]:
        f_s, t_s, Sxx = _spectrogram(sig, fs, nperseg=N_spect, noverlap=N_spect//2)
        fm = f_s <= max_freq
        ax.pcolormesh(t_s, f_s[fm], 10*np.log10(Sxx[fm] + 1e-12),
                      shading='gouraud', cmap='inferno')
        ax.set_ylabel('Freq (Hz)'); ax.set_xlabel('Time (s)')
        ax.set_title(f'Spectrogram — {ttl}')

    # Row 2 — system and output evolution
    _plot_evolution_row(axes[2,0], axes[2,1], result)

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


# ---- Per-mode detail plots (call separately after plot_quantum_audio) ----

def plot_details_sensing(result, figsize=(14, 8)):
    """Mode I detail: phase trajectory, sin(φ) linearity, output spectrum."""
    res  = result['trotterize_result']
    meth = res['method']
    tl      = res['tlist']
    phi     = res['phases']
    readout = res['qpsd_phases'] if meth == 'qpsd' else res['aux_sigma_z']
    lbl     = 'QPSD phase' if meth == 'qpsd' else 'sin(φ)'

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(_make_plot_title(result) + '\nPhase & readout details', fontsize=10)

    axes[0,0].plot(tl, phi, 'r-', lw=0.6)
    axes[0,0].axhline( np.pi/2, color='grey', ls='--', alpha=0.4)
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


def plot_details_sms_colour(result, figsize=(14, 8)):
    """Mode II detail: per-bin system state, amplitude in/out, phase out."""
    diag      = result['quantum_result']['diagnostics']
    freq_bins = result['analysis']['freq_bins']
    Xr_mag    = result['analysis']['Xr_mag']

    bin_keys = sorted(diag.keys())
    if not bin_keys:
        print("No diagnostics to plot."); return

    show_keys = _pick_representative_bins(bin_keys, 4)
    n_cols    = len(show_keys)

    fig, axes = plt.subplots(3, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(_make_plot_title(result) + '\nPer-bin system & I/O', fontsize=10)

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
        ax.plot(np.arange(len(in_amp)), in_amp, lw=0.6, color='steelblue', label='in amp')
        ax.plot(np.arange(len(d['aux_amp_out'])), d['aux_amp_out'],
                lw=0.6, color='darkorchid', label='out amp')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Amplitude')

        ax = axes[2, col]
        ax.plot(steps, np.unwrap(d['aux_phase_out']), lw=0.5, color='seagreen')
        ax.set_xlabel('Frame'); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Output phase (rad)')

    plt.tight_layout(); plt.show()


def plot_details_sms_trajectory(result, figsize=(14, 8)):
    """Mode III detail: per-bin drive vs system amplitude, Bloch, measurement record."""
    diag       = result.get('diagnostics', {})
    freq_bins  = result['analysis']['freq_bins']
    Xr_mag     = result['analysis']['Xr_mag']
    num_frames = Xr_mag.shape[0]

    if not diag:
        print("No diagnostics to plot."); return

    normperbin = result.get('params', {}).get('normperbin', False)
    if normperbin:
        mag_norm, _ = normalize_amplitudes_per_bin(Xr_mag)
    else:
        mag_norm, _ = normalize_amplitudes(Xr_mag)

    bin_keys  = sorted(diag.keys())
    show_keys = _pick_representative_bins(bin_keys, 4)
    n_cols    = len(show_keys)

    fig, axes = plt.subplots(3, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(_make_plot_title(result) + '\nPer-bin drive, Bloch & measurement', fontsize=10)

    def _norm(arr):
        c = arr - np.mean(arr)
        p = np.max(np.abs(c)) or 1.0
        return c / p

    for col, k in enumerate(show_keys):
        d       = diag[k]
        sys_amp = d['sys_amp']
        drv_amp = mag_norm[:num_frames, k] if k < mag_norm.shape[1] else np.zeros(num_frames)
        steps   = np.arange(len(sys_amp))
        f_hz    = freq_bins[k] if k < len(freq_bins) else k

        ax = axes[0, col]
        ax.plot(np.arange(num_frames), _norm(drv_amp),
                lw=0.5, color='steelblue', alpha=0.8, label='drive')
        ax.plot(steps, _norm(sys_amp),
                lw=0.5, color='darkorchid', alpha=0.8, label='sys')
        ax.set_title(f'bin {k} ({f_hz:.0f} Hz)')
        ax.legend(fontsize=6); ax.set_ylim([-1.2, 1.2]); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Normalised amp')

        ax = axes[1, col]
        ax.plot(steps, d['sys_sigmax'], lw=0.6, label='⟨σx⟩')
        ax.plot(steps, d['sys_sigmaz'], lw=0.6, label='⟨σz⟩')
        ax.plot(steps, d['sys_pop'],    lw=0.6, ls='--', label='ρ₁₁')
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Bloch')

        ax = axes[2, col]
        meas = d['measurement_record']
        ax.fill_between(np.arange(len(meas)), 0, meas, step='mid',
                        alpha=0.5, color='orange')
        ax.set_ylim([-0.1, 1.1]); ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Outcome (0/1)')

    plt.tight_layout(); plt.show()


def plot_details_harmony(result, figsize=(14, 8)):
    """Mode IV detail: voice probability stack, winners, Woolhouse A, boost histogram."""
    diagonals = result['diagonals']
    winners   = result['winners']
    a_chain   = result['a_chain']
    mod_log   = result['modulation_log']

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(_make_plot_title(result) + '\nVoice selection & tonal attraction', fontsize=10)

    t  = np.arange(len(diagonals))
    vc = ['steelblue', 'darkorange', 'seagreen']

    axes[0,0].stackplot(t, diagonals[:,0], diagonals[:,1], diagonals[:,2],
                        labels=['root','3rd','5th'], colors=vc, alpha=0.7)
    axes[0,0].set_title('Voice probabilities per transition')
    axes[0,0].set_ylabel('Probability'); axes[0,0].legend(fontsize=8)
    axes[0,0].set_xlabel('Transition index')

    axes[0,1].bar(t, np.ones(len(winners)), color=[vc[w] for w in winners],
                  alpha=0.7, edgecolor='none')
    axes[0,1].set_title('Winning voice (colour = root/3rd/5th)')
    axes[0,1].set_yticks([])

    axes[1,0].plot(np.arange(len(a_chain)), a_chain, 'ko-', ms=4, lw=1)
    axes[1,0].set_title('Woolhouse attraction A')
    axes[1,0].set_ylabel('A'); axes[1,0].set_xlabel('Transition')

    boosts = [e['boost_factor'] for e in mod_log]
    axes[1,1].hist(boosts, bins=20, color='darkorchid', alpha=0.7, edgecolor='white')
    axes[1,1].set_title('Boost factor distribution')
    axes[1,1].set_xlabel('Boost ×'); axes[1,1].set_ylabel('Count')

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.show()


def plot_mode_diagrams(figsize=(16, 11)):
    """Circuit-diagram overview of all four quantum audio modes.

    Architecture (shared across modes)
    -----------------------------------
    • System qubit/qutrit [S] on the left — coupled to the delay line at entry.
    • Delay line: FIFO of aux qubits. Fresh aux enters at the left node; the
      oldest aux exits at the right node into the measurement block.
    • τ dashed rectangle marks the delay region — NOT a feedback loop.
    • Measurement block reads the EXITING auxiliary:
        – grey dashed box (⟨σ⟩ readout)  — Modes I & II (expectation value)
        – red solid box  (projective)      — Modes III & IV (stochastic collapse)
    • Green annotation box (top-left)  — explicit encoding equation.
    • Amber annotation box (top-right) — explicit decoding equation.
    • Amber horizontal arrow (below bus, ←) — classical audio-out path.
    """
    from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

    C_BUS  = '#1a2744'
    C_NODE = '#4a7a2e'
    C_SIG  = '#d4860a'
    C_RED  = '#c0392b'
    C_VT   = '#2980b9'
    C_ENC  = '#4a7a2e'
    C_DEC  = '#a05800'
    C_PUR  = '#7b3fa0'

    TITLES = [
        "Mode I — NV Quantum Sensing",
        "Mode II — SMS Spectral Colouring",
        "Mode III — SMS Quantum Trajectory",
        "Mode IV — Qutrit Tonal Brightness",
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(hspace=0.10, wspace=0.06)

    def _draw_mode(ax, idx):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7.0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('#fafafa')

        # ── layout constants ────────────────────────────────────────
        # Bus y-level. Encoding box top-left. Decoding box top-right.
        # Fresh aux always enters from ABOVE the first delay node.
        # Exiting aux leaves from the LAST delay node into measurement.
        # τ is a dashed rectangle around the delay-node region only.
        SYS_X  = 1.4     # system qubit/qutrit x
        BUS_Y  = 3.5     # bus / delay-line y
        NX     = [2.9, 4.1, 5.3]   # delay-node x positions
        MEAS_X = 6.6     # measurement block centre x
        SPK_X  = 7.8     # speaker x
        ENC    = (0.06, 4.75, 2.45, 2.0)   # enc box (x,y,w,h)
        DEC    = (6.1,  4.75, 3.84, 2.0)   # dec box (x,y,w,h)
        PAR    = (0.06, 0.08, 9.88, 1.4)   # params strip (x,y,w,h)

        # ── primitives ───────────────────────────────────────────────
        def _arr(x0, y0, x1, y1, color=C_BUS, lw=2.2, rad=0.0):
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle='->, head_width=0.22, head_length=0.18',
                            color=color, lw=lw,
                            connectionstyle=f'arc3,rad={rad}'),
                        zorder=8)

        def _node(x, y, s=0.26):
            ax.add_patch(Rectangle((x - s/2, y - s/2), s, s,
                                   color=C_NODE, zorder=5))

        def _sys_qubit(x, y, r=0.44):
            ax.add_patch(Circle((x, y), r, color=C_NODE, zorder=5))
            ax.text(x, y + 0.09, 'ω', color='white', ha='center', va='center',
                    fontsize=11, fontweight='bold', zorder=6)
            ax.plot([x - 0.18, x + 0.18], [y - 0.14, y - 0.14],
                    color='white', lw=1.3, zorder=6)

        def _sys_qutrit(x, y, r=0.44):
            ax.add_patch(Circle((x, y), r, color=C_NODE, zorder=5))
            for dy in (-0.14, 0.0, 0.14):
                ax.plot([x - 0.24, x + 0.24], [y + dy, y + dy],
                        color='white', lw=1.2, zorder=6)

        def _meas_expval(x, y, w=0.96, h=0.66):
            # grey dashed — expectation-value readout (Modes I & II)
            ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                       boxstyle='square,pad=0.03',
                                       edgecolor='#888', facecolor='white',
                                       linestyle='--', lw=1.6, zorder=4))
            xs2 = np.linspace(x - 0.30, x + 0.30, 50)
            ax.plot(xs2,
                    y + 0.13 * np.sin((xs2 - x) * 11)
                      * np.exp(-((xs2 - x)**2) / 0.10),
                    color='#555', lw=1.4, zorder=5)
            ax.text(x, y - h/2 - 0.17, '⟨σ⟩ readout',
                    ha='center', va='top', fontsize=6.0, color='#666')

        def _meas_proj(x, y, w=0.96, h=0.66):
            # red solid — projective measurement (Modes III & IV)
            ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                       boxstyle='square,pad=0.03',
                                       edgecolor=C_RED, facecolor='#fff0f0',
                                       linestyle='-', lw=1.8, zorder=4))
            lx = [x - 0.24, x - 0.06, x + 0.06, x + 0.24]
            ly = [y - 0.20, y + 0.20, y - 0.20, y + 0.20]
            ax.plot(lx, ly, color=C_RED, lw=1.8, zorder=6)
            ax.text(x, y - h/2 - 0.17, 'projective',
                    ha='center', va='top', fontsize=6.0, color=C_RED)

        def _speaker(x, y):
            ax.add_patch(Circle((x, y), 0.22, color='#888', zorder=4))
            ax.text(x, y, '♪', color='white', ha='center', va='center',
                    fontsize=9, zorder=5)

        def _box(x, y, w, h, lines, fc, ec, fs=7.0, tc='#222'):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                                       boxstyle='round,pad=0.09',
                                       edgecolor=ec, facecolor=fc,
                                       lw=1.2, zorder=6))
            n = len(lines)
            for i, ln in enumerate(lines):
                ax.text(x + w/2, y + h * (1 - (i + 0.5) / n),
                        ln, ha='center', va='center',
                        fontsize=fs, color=tc, zorder=7)

        def _enc(lines, fs=6.9):
            x, y, w, h = ENC
            _box(x, y, w, h, lines, fc='#e6f4e6', ec=C_ENC, fs=fs, tc='#0d2e0d')

        def _dec(lines, fs=6.9):
            x, y, w, h = DEC
            _box(x, y, w, h, lines, fc='#fff7e6', ec=C_DEC, fs=fs, tc='#3a1e00')

        def _par(lines, fs=7.0):
            x, y, w, h = PAR
            _box(x, y, w, h, lines, fc='#f0f6ff', ec='#5080a0', fs=fs, tc='#1a2744')

        # ── SHARED SKELETON ───────────────────────────────────────
        # τ dashed region around delay nodes (NOT a loop — FIFO shift register)
        ax.add_patch(Rectangle(
            (NX[0] - 0.42, BUS_Y - 0.55), NX[2] - NX[0] + 0.84, 1.1,
            edgecolor=C_RED, facecolor='#fff8f8',
            linestyle='--', lw=1.2, zorder=1, alpha=0.5))
        ax.text((NX[0] + NX[2]) / 2, BUS_Y + 0.62, 'τ  (delay FIFO)',
                ha='center', va='bottom', fontsize=7.5,
                color=C_RED, fontweight='bold')

        # Quantum bus: S → N0 → N1 → N2 → measurement (left to right)
        _arr(SYS_X + 0.44, BUS_Y, NX[0] - 0.13, BUS_Y)
        _arr(NX[0] + 0.13, BUS_Y, NX[1] - 0.13, BUS_Y)
        _arr(NX[1] + 0.13, BUS_Y, NX[2] - 0.13, BUS_Y)
        _arr(NX[2] + 0.13, BUS_Y, MEAS_X - 0.48, BUS_Y)

        # Delay nodes
        for xi in NX:
            _node(xi, BUS_Y)

        # Measurement → speaker
        _arr(MEAS_X + 0.48, BUS_Y, SPK_X - 0.22, BUS_Y)
        _speaker(SPK_X, BUS_Y)

        # Classical audio-out (amber ←, below bus)
        _arr(MEAS_X - 0.48, BUS_Y - 0.5, SYS_X + 0.44, BUS_Y - 0.5,
             color=C_SIG, lw=2.0)
        ax.text((SYS_X + MEAS_X) / 2, BUS_Y - 0.66,
                'audio out', ha='center', va='top',
                fontsize=6.0, color=C_SIG)

        # ── PER-MODE CONTENT ──────────────────────────────────────

        if idx == 0:  # ═══════ MODE I: NV SENSING ════════════════
            _sys_qubit(SYS_X, BUS_Y)
            _meas_expval(MEAS_X, BUS_Y)

            # Fresh qubit (white noise) enters from above first node each step
            _arr(NX[0], BUS_Y + 1.05, NX[0], BUS_Y + 0.13, color=C_BUS, lw=1.7)
            ax.text(NX[0], BUS_Y + 1.12, '|ψ⟩ fresh\nwhite-noise',
                    ha='center', va='bottom', fontsize=6.2, color='#333')

            # Encoding box
            _enc([
                'Carrier  Ω,  Envelope  Θ',
                'ω_S = λ · Ω',
                'V(t) = θ(Θ,Ω) · σ_x',
                '→  phase-encodes signal',
            ])
            ax.text(ENC[0] + 0.05, ENC[1] + ENC[3] + 0.05,
                    'audio → Hamiltonian',
                    fontsize=6.0, color=C_ENC, style='italic')
            _arr(ENC[0] + ENC[2], ENC[1] + ENC[3] * 0.55,
                 SYS_X - 0.44, BUS_Y + 0.1, color=C_ENC, lw=1.3, rad=-0.15)

            # Decoding box
            _dec([
                'exiting aux  →  ⟨σ_x⟩, ⟨σ_y⟩',
                'per Ramsey / Hahn step',
                'interpolate + normalise',
                '→  audio signal',
            ])
            ax.text(DEC[0] + DEC[2] / 2, DEC[1] + DEC[3] + 0.05,
                    'quantum → audio',
                    ha='center', fontsize=6.0, color=C_DEC, style='italic')
            _arr(MEAS_X + 0.48, BUS_Y + 0.1,
                 DEC[0], DEC[1] + DEC[3] * 0.6, color=C_DEC, lw=1.3, rad=-0.15)

            _par(['sensing_rate  ·  phase_scale  ·  delay  ·  phase',
                  'method:  sin_φ  |  qpsd     sequence:  Ramsey  |  Hahn-echo'])

        elif idx == 1:  # ═══════ MODE II: SMS COLOUR ══════════════
            _sys_qubit(SYS_X, BUS_Y)
            _meas_expval(MEAS_X, BUS_Y)

            # Fresh Bloch-encoded aux enters above first node
            _arr(NX[0], BUS_Y + 1.05, NX[0], BUS_Y + 0.13, color=C_BUS, lw=1.7)
            ax.text(NX[0], BUS_Y + 1.12,
                    '|θ,φ⟩  fresh aux\n(Bloch-encoded)',
                    ha='center', va='bottom', fontsize=6.2,
                    color='#333', fontweight='bold')

            # Encoding box — explicit Bloch encoding equations
            _enc([
                'SMS residual bin  (k, n)',
                'θ = 2·arcsin(√Xr_mag)',
                'φ = Xr_phase',
                '|θ,φ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩',
            ])
            ax.text(ENC[0] + 0.05, ENC[1] + ENC[3] + 0.05,
                    'audio → aux Bloch state',
                    fontsize=6.0, color=C_ENC, style='italic')
            _arr(ENC[0] + ENC[2], ENC[1] + ENC[3] * 0.7,
                 NX[0] - 0.13, BUS_Y + 1.05, color=C_ENC, lw=1.3, rad=-0.12)

            # Decoding box — explicit decode equations
            _dec([
                'exiting aux  ρ_out',
                'amp = ⟨|1⟩⟨1|⟩  =  ρ_11',
                'φ = atan2(⟨σ_y⟩, ⟨σ_x⟩)',
                '→  STFT bin  →  synthesis',
            ])
            ax.text(DEC[0] + DEC[2] / 2, DEC[1] + DEC[3] + 0.05,
                    'aux state → audio',
                    ha='center', fontsize=6.0, color=C_DEC, style='italic')
            _arr(MEAS_X + 0.48, BUS_Y + 0.1,
                 DEC[0], DEC[1] + DEC[3] * 0.6, color=C_DEC, lw=1.3, rad=-0.15)

            _par(['mode_sms:  stochastic_only  |  reverse_sinusoidal  |  full_stft  |  stochastic_unraveler',
                  'ω_cont  ·  γ_x  ·  γ_z  ·  lam  ·  normperbin  ·  delay  ·  phase'])

        elif idx == 2:  # ═══════ MODE III: SMS TRAJECTORY ══════════
            _sys_qubit(SYS_X, BUS_Y)
            _meas_proj(MEAS_X, BUS_Y)

            # V(t) drives SYSTEM from audio — waveform icon above system
            xs2 = np.linspace(SYS_X - 0.42, SYS_X + 0.42, 50)
            ax.plot(xs2, BUS_Y + 1.1 + 0.18 * np.sin((xs2 - SYS_X) * 16),
                    color=C_VT, lw=1.7, zorder=5)
            _arr(SYS_X, BUS_Y + 0.92, SYS_X, BUS_Y + 0.44, color=C_VT, lw=1.9)

            # Vacuum |0⟩ enters above first node each step
            _arr(NX[0], BUS_Y + 1.05, NX[0], BUS_Y + 0.13, color=C_BUS, lw=1.7)
            ax.text(NX[0], BUS_Y + 1.12, '|0⟩  vacuum\nfresh aux',
                    ha='center', va='bottom', fontsize=6.2, color='#333')

            # Projective result exits ONLY from last node (exiting aux is measured)
            _arr(NX[2], BUS_Y - 0.13, NX[2], BUS_Y - 0.55,
                 color=C_RED, lw=1.5)
            ax.text(NX[2], BUS_Y - 0.62,
                    '0 / 1', ha='center', va='top',
                    fontsize=6.5, color=C_RED, fontweight='bold')

            # Encoding box
            _enc([
                'SMS stochastic bins',
                'Xr_mag  →  ω_drive',
                'Xr_phase  →  drive phase',
                'H(t) = V(t) · σ_x  drives  S',
            ])
            ax.text(ENC[0] + 0.05, ENC[1] + ENC[3] + 0.05,
                    'audio → system Hamiltonian',
                    fontsize=6.0, color=C_ENC, style='italic')
            _arr(ENC[0] + ENC[2], ENC[1] + ENC[3] * 0.7,
                 SYS_X - 0.44, BUS_Y + 0.9, color=C_ENC, lw=1.3, rad=-0.1)

            # Decoding box
            _dec([
                'exiting aux  (last node)',
                'projective outcome  0 / 1',
                'quantum trajectory  →  STFT',
                '→  synthesis',
            ])
            ax.text(DEC[0] + DEC[2] / 2, DEC[1] + DEC[3] + 0.05,
                    'stochastic collapse → audio',
                    ha='center', fontsize=6.0, color=C_DEC, style='italic')
            _arr(MEAS_X + 0.48, BUS_Y + 0.1,
                 DEC[0], DEC[1] + DEC[3] * 0.6, color=C_DEC, lw=1.3, rad=-0.15)

            _par(['vacuum |0⟩ aux  ·  stochastic_only bins  ·  projective collapse on exiting aux',
                  'γ_x  ·  γ_z  ·  ω_drive_xy  ·  ω_drive_z  ·  basis: z|x|y  ·  seed  ·  delay  ·  phase'])

        else:  # ═══════ MODE IV: QUTRIT HARMONY ══════════════════
            _sys_qutrit(SYS_X, BUS_Y)
            _meas_proj(MEAS_X, BUS_Y)

            # Qutrit-level markers on delay nodes
            for xi in NX:
                for dy in (-0.07, 0.0, 0.07):
                    ax.plot([xi - 0.1, xi + 0.1], [BUS_Y + dy, BUS_Y + dy],
                            color='white', lw=1.1, zorder=7)

            # Fresh qutrit aux (TA→ρ) enters above first node
            _arr(NX[0], BUS_Y + 1.05, NX[0], BUS_Y + 0.13,
                 color=C_PUR, lw=1.7)
            ax.text(NX[0], BUS_Y + 1.12, 'ρ(TA)  fresh\nqutrit aux',
                    ha='center', va='bottom', fontsize=6.2,
                    color=C_PUR, fontweight='bold')

            # Projective winner exits last node (voice: root/3rd/5th)
            _arr(NX[2], BUS_Y - 0.13, NX[2], BUS_Y - 0.55,
                 color=C_RED, lw=1.5)
            ax.text(NX[2], BUS_Y - 0.62,
                    'root|3rd|5th', ha='center', va='top',
                    fontsize=6.0, color=C_RED, fontweight='bold')

            # Encoding box
            _enc([
                'basic-pitch  →  chord events',
                'Woolhouse  TA = IC×VL×RS×CD',
                'ta_to_density_matrix(TA)  →  ρ',
                'fresh aux  =  ρ  per chord step',
            ])
            ax.text(ENC[0] + 0.05, ENC[1] + ENC[3] + 0.05,
                    'audio → qutrit density matrix',
                    fontsize=6.0, color=C_ENC, style='italic')
            _arr(ENC[0] + ENC[2], ENC[1] + ENC[3] * 0.7,
                 NX[0] - 0.13, BUS_Y + 1.05, color=C_PUR, lw=1.3, rad=-0.12)

            # Decoding box
            _dec([
                'exiting qutrit aux',
                'projective  →  voice winner',
                'root|3rd|5th  harmonic bins',
                '× brightness  →  synthesis',
            ])
            ax.text(DEC[0] + DEC[2] / 2, DEC[1] + DEC[3] + 0.05,
                    'voice winner → spectral boost',
                    ha='center', fontsize=6.0, color=C_DEC, style='italic')
            _arr(MEAS_X + 0.48, BUS_Y + 0.1,
                 DEC[0], DEC[1] + DEC[3] * 0.6, color=C_DEC, lw=1.3, rad=-0.15)

            _par(['coupling_g  ·  brightness  ·  chord_size  ·  omega_sys  ·  Gell-Mann  λ₁,λ₄,λ₆',
                  'TA = IC × VL × RS × CD  ·  delay  ·  phase'])

        # ── TITLE ─────────────────────────────────────────────────
        ax.text(5.0, 6.82, TITLES[idx], ha='center', va='top',
                fontsize=10, fontweight='bold', color='#1a2744',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', lw=1.0))

    for i, ax in enumerate(axes.flat):
        _draw_mode(ax, i)

    plt.tight_layout(pad=0.5)
    plt.show()
    return fig


# ============================================================
# SECTION 13 — EXAMPLE EXECUTION
# ============================================================
# Run any of the four modes by changing `mode` below.
# Keep delay <= 9 for tractable run times (delay=3 is fast, delay=9 is slow).


# ============================================================
# SECTION 14 — DUAL SENSING: TESTS & DEDICATED PLOT
# ============================================================

def plot_dual_sensing(result_or_trot, fs=None, title_extra='', figsize=(14, 10)):
    """
    4-panel plot for dual-channel sensing results.

    Accepts either:
      • the dict returned by trotterize_sensing_dual  (pass fs= explicitly)
      • the dict returned by quantum_audio_pipeline(..., mode='sensing_dual')

    Panels
    ------
    Row 0 : input waveform (time)         |  input spectrum (FFT)
    Row 1 : ch1/ch2 readout (time)        |  combined output (time)
    Row 2 : combined output spectrum       |  cross-correlation ch1 vs ch2
    """
    # Accept both trotterize result and full pipeline result
    if 'trotterize_result' in result_or_trot:
        res      = result_or_trot['trotterize_result']
        y_in     = result_or_trot['y_original']
        fs_audio = result_or_trot['fs']
    else:
        res      = result_or_trot
        y_in     = None
        fs_audio = fs or res.get('sensing_rate', 1000)

    tl   = res['tlist']
    r1   = res['ch1_sz']
    r2   = res['ch2_sz']
    comb = res['output_combined']
    jm   = res['joint_mode']
    sm   = res['summation_mode']
    d    = res['delay']

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(
        f"Mode Ia — Dual Sensing  |  joint={jm}  delay={d}  sum={sm}"
        + (f"  {title_extra}" if title_extra else ''),
        fontsize=10, y=1.01)

    # Row 0 — input (or raw phases)
    if y_in is not None:
        t_in = np.arange(len(y_in)) / fs_audio
        axes[0, 0].plot(t_in, y_in, lw=0.5, color='steelblue')
        axes[0, 0].set_title('Input waveform'); axes[0, 0].set_ylabel('Amplitude')
        # FFT of input
        sp_in = np.abs(fft(y_in)); fr_in = fftfreq(len(y_in), 1/fs_audio)
        pos = fr_in >= 0
        axes[0, 1].plot(fr_in[pos], sp_in[pos] / (sp_in[pos].max() + 1e-12),
                        lw=0.6, color='steelblue')
        axes[0, 1].set_title('Input spectrum'); axes[0, 1].set_xlabel('Hz')
    else:
        phi = res['phases']
        axes[0, 0].plot(tl, phi, 'r-', lw=0.6)
        axes[0, 0].set_title('Accumulated phase φ'); axes[0, 0].set_ylabel('rad')
        sp_phi = np.abs(fft(phi - np.mean(phi)))
        fr_phi = fftfreq(len(phi), tl[1] - tl[0]) if len(tl) > 1 else np.arange(len(phi))
        axes[0, 1].plot(fr_phi[fr_phi >= 0],
                        sp_phi[fr_phi >= 0] / (sp_phi.max() + 1e-12),
                        lw=0.6, color='steelblue')
        axes[0, 1].set_title('Phase spectrum'); axes[0, 1].set_xlabel('Hz')

    # Row 1 — per-channel readouts and combined output
    axes[1, 0].plot(tl, r1, lw=0.5, color='steelblue', alpha=0.8, label='ch1 ⟨σz⟩')
    axes[1, 0].plot(tl, r2, lw=0.5, color='tomato',    alpha=0.8, label='ch2 ⟨σz⟩')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title('Per-channel ⟨σz⟩ readout'); axes[1, 0].set_ylabel('⟨σz⟩')
    axes[1, 0].set_xlabel('Time (s)')

    axes[1, 1].plot(tl, comb, lw=0.5, color='darkorchid')
    axes[1, 1].set_title(f'Combined output  [{sm} node]')
    axes[1, 1].set_ylabel('out[n]'); axes[1, 1].set_xlabel('Time (s)')

    # Row 2 — combined output spectrum + cross-correlation
    dt = (tl[1] - tl[0]) if len(tl) > 1 else 1.0
    comb_dc = comb - np.mean(comb)
    sp_out  = np.abs(fft(comb_dc))
    fr_out  = fftfreq(len(comb_dc), dt)
    pos_out = fr_out >= 0
    axes[2, 0].plot(fr_out[pos_out],
                    sp_out[pos_out] / (sp_out[pos_out].max() + 1e-12),
                    lw=0.6, color='darkorchid')
    axes[2, 0].set_title('Combined output spectrum')
    axes[2, 0].set_xlabel('Hz'); axes[2, 0].set_ylabel('Normalised |FFT|')

    # Cross-correlation between ch1 and ch2 readouts
    n_cc  = len(r1)
    r1_dc = r1 - np.mean(r1)
    r2_dc = r2 - np.mean(r2)
    xcorr = np.correlate(r1_dc, r2_dc, mode='full')
    lags  = np.arange(-(n_cc - 1), n_cc) * dt
    norm  = (np.linalg.norm(r1_dc) * np.linalg.norm(r2_dc) + 1e-12)
    axes[2, 1].plot(lags, xcorr / norm, lw=0.6, color='seagreen')
    axes[2, 1].axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    axes[2, 1].set_title('Cross-correlation ch1 ↔ ch2  (quantum correlations)')
    axes[2, 1].set_xlabel('Lag (s)'); axes[2, 1].set_ylabel('Normalised')

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def run_dual_sensing_tests(joint_mode='xx', xx_coupling=0.3, delay=1,
                            delay1=None, delay2=None,
                            sensing_rate=500,
                            swap_angle_in=0.9, swap_angle_out=0.1,
                            summation_mode='rms', sequence='ramsey'):
    """
    Self-contained tests for trotterize_sensing_dual — no audio file required.

    Test 1 — Impulse Response (IR)
    --------------------------------
    A single unit impulse at t ≈ 0.1 s in an otherwise silent 1-second signal.
    Low sensing_rate (default 500 Hz) keeps the simulation fast.
    This lets you inspect the system's IR: how a transient propagates through
    the two-qubit delay and joint unitary.

    Test 2 — Low-Frequency Oscillation
    ------------------------------------
    A pure sine at f0 = sensing_rate / 20  (e.g. 25 Hz at fs=500).
    Chosen so that several full cycles fit within the signal, revealing
    whether the frequency content survives the joint coupling and summation.

    Parameters
    ----------
    joint_mode    : 'xx' | 'fdn_hadamard' | 'hadamard_id'
    xx_coupling   : θ for 'xx' mode
    delay         : delay per channel  (d; total = 2d, recommend ≤ 2 for speed)
    sensing_rate  : Hz — sets T_seq = 1/sensing_rate  (also the output sample rate)
    swap_angle_in : S↔fresh SWAP angle (default 0.9)
    swap_angle_out: S↔D[0]  SWAP angle (default 0.1, quantum memory)
    summation_mode: 'rms' | 'sum' | 'ch1' | 'ch2'
    sequence      : 'ramsey' | 'hahn_echo'

    Returns
    -------
    dict with keys 'ir' and 'sine', each containing 'input', 'result', 'fs'.
    """
    print("\n" + "=" * 60)
    print("DUAL SENSING — Test Suite")
    print(f"  joint_mode={joint_mode}  xx_coupling={xx_coupling}")
    d1_eff = delay1 if delay1 is not None else delay
    d2_eff = delay2 if delay2 is not None else delay
    print(f"  delay1={d1_eff}  delay2={d2_eff}  sensing_rate={sensing_rate} Hz")
    print("=" * 60)

    T_seq = 1.0 / sensing_rate
    T_phi = 0.8 * T_seq
    hires_sr = sensing_rate          # no time-stretch needed for synthetic signals

    # ── Test 1: Impulse Response ──────────────────────────────────────────
    print("\n─── Test 1: Impulse Response (IR) ─────────────────────────")
    duration_ir = 1.0                            # 1-second window
    n_ir        = int(duration_ir * hires_sr)
    x_ir        = np.zeros(n_ir)
    x_ir[max(1, n_ir // 10)] = 1.0              # unit impulse at 10 % mark

    res_ir = trotterize_sensing_dual(
        x_ir, hires_sr, T_phi, T_seq,
        phase_scale=1.0, delay=delay, delay1=delay1, delay2=delay2,
        swap_angle_in=swap_angle_in, swap_angle_out=swap_angle_out,
        joint_mode=joint_mode, xx_coupling=xx_coupling,
        sequence=sequence, summation_mode=summation_mode, verbose=True)

    print(f"  IR output: n={len(res_ir['output_combined'])} samples  "
          f"peak={np.max(np.abs(res_ir['output_combined'])):.4f}")

    # ── Test 2: Low-Frequency Sinusoidal ─────────────────────────────────
    print("\n─── Test 2: Low-Frequency Oscillation ─────────────────────")
    f0          = sensing_rate / 20.0            # e.g. 25 Hz at fs=500
    duration_s  = 3.0
    n_s         = int(duration_s * hires_sr)
    t_s         = np.arange(n_s) / hires_sr
    x_sine      = np.sin(2 * np.pi * f0 * t_s)

    print(f"  Sine frequency: {f0:.1f} Hz  "
          f"({duration_s:.0f} s  ×  sensing_rate={sensing_rate} Hz)")

    res_sine = trotterize_sensing_dual(
        x_sine, hires_sr, T_phi, T_seq,
        phase_scale=0.5, delay=delay, delay1=delay1, delay2=delay2,
        swap_angle_in=swap_angle_in, swap_angle_out=swap_angle_out,
        joint_mode=joint_mode, xx_coupling=xx_coupling,
        sequence=sequence, summation_mode=summation_mode, verbose=False)

    print(f"  Sine output: n={len(res_sine['output_combined'])} samples  "
          f"peak={np.max(np.abs(res_sine['output_combined'])):.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_dual_sensing_tests(
        x_ir,   res_ir,   sensing_rate,
        x_sine, res_sine, sensing_rate, f0, joint_mode, delay)

    return dict(
        ir=dict(input=x_ir,   result=res_ir,   fs=sensing_rate),
        sine=dict(input=x_sine, result=res_sine, fs=sensing_rate),
    )


def _plot_dual_sensing_tests(x_ir, res_ir, fs_ir,
                               x_sine, res_sine, fs_sine, f0,
                               joint_mode, delay, figsize=(15, 12)):
    """
    4×2 summary figure for both dual-sensing tests.

    Row 0: IR — input impulse (time)          | IR output combined (time)
    Row 1: IR — ch1/ch2 readouts (time)       | IR output spectrum (FFT)
    Row 2: Sine — input sine (time)           | Sine output combined (time)
    Row 3: Sine — ch1/ch2 readouts (time)     | Sine output spectrum (FFT)
    """
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle(
        f"Dual-Channel Sensing Tests\n"
        f"joint_mode='{joint_mode}'   delay={delay}/ch   "
        f"sensing_rate={fs_ir} Hz",
        fontsize=10, y=1.01)

    def _fft_plot(ax, sig, dt, color, title):
        sig_dc = sig - np.mean(sig)
        sp     = np.abs(fft(sig_dc))
        fr     = fftfreq(len(sig_dc), dt)
        pos    = fr >= 0
        ax.plot(fr[pos], sp[pos] / (sp[pos].max() + 1e-12), lw=0.7, color=color)
        ax.set_title(title); ax.set_xlabel('Hz'); ax.set_ylabel('|FFT| norm.')

    # ── Row 0: IR input + combined output ────────────────────────────────
    tl_ir = res_ir['tlist']
    t_x_ir = np.arange(len(x_ir)) / fs_ir
    axes[0, 0].stem(t_x_ir, x_ir, linefmt='steelblue', markerfmt='o',
                    basefmt='grey')
    axes[0, 0].set_title('Test 1 — Input: unit impulse')
    axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('Amplitude')

    axes[0, 1].plot(tl_ir, res_ir['output_combined'], lw=0.7, color='darkorchid')
    axes[0, 1].set_title('Test 1 — Combined output (IR)')
    axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('out[n]')

    # ── Row 1: IR ch1/ch2 + spectrum ─────────────────────────────────────
    axes[1, 0].plot(tl_ir, res_ir['ch1_sz'], lw=0.6, color='steelblue',
                    alpha=0.8, label='ch1')
    axes[1, 0].plot(tl_ir, res_ir['ch2_sz'], lw=0.6, color='tomato',
                    alpha=0.8, label='ch2')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title('Test 1 — Per-channel ⟨σz⟩')
    axes[1, 0].set_xlabel('Time (s)'); axes[1, 0].set_ylabel('⟨σz⟩')

    dt_ir = (tl_ir[1] - tl_ir[0]) if len(tl_ir) > 1 else 1.0 / fs_ir
    _fft_plot(axes[1, 1], res_ir['output_combined'], dt_ir,
              'darkorchid', 'Test 1 — Output spectrum')

    # ── Row 2: Sine input + combined output ───────────────────────────────
    tl_s = res_sine['tlist']
    t_x_s = np.arange(len(x_sine)) / fs_sine
    axes[2, 0].plot(t_x_s, x_sine, lw=0.5, color='seagreen')
    axes[2, 0].set_title(f'Test 2 — Input: sine {f0:.1f} Hz')
    axes[2, 0].set_xlabel('Time (s)'); axes[2, 0].set_ylabel('Amplitude')

    axes[2, 1].plot(tl_s, res_sine['output_combined'], lw=0.5, color='darkorchid')
    axes[2, 1].set_title('Test 2 — Combined output (sine)')
    axes[2, 1].set_xlabel('Time (s)'); axes[2, 1].set_ylabel('out[n]')

    # ── Row 3: Sine ch1/ch2 + spectrum ───────────────────────────────────
    axes[3, 0].plot(tl_s, res_sine['ch1_sz'], lw=0.5, color='steelblue',
                    alpha=0.8, label='ch1')
    axes[3, 0].plot(tl_s, res_sine['ch2_sz'], lw=0.5, color='tomato',
                    alpha=0.8, label='ch2')
    axes[3, 0].legend(fontsize=8)
    axes[3, 0].set_title('Test 2 — Per-channel ⟨σz⟩')
    axes[3, 0].set_xlabel('Time (s)'); axes[3, 0].set_ylabel('⟨σz⟩')

    dt_s = (tl_s[1] - tl_s[0]) if len(tl_s) > 1 else 1.0 / fs_sine
    _fft_plot(axes[3, 1], res_sine['output_combined'], dt_s,
              'seagreen', 'Test 2 — Output spectrum')

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


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

    plot_quantum_audio(result, max_freq=5000)

    # Mode-specific diagnostic details:
    # plot_details_sensing(result)
    # plot_details_sms_colour(result)
    # plot_details_sms_trajectory(result)
    # plot_details_harmony(result)