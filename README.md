# Quantum Colour of Sound

Quantum-inspired audio processor. Four modes, one file: `quantum_audio_master.py`.

---

## Quick start

```python
import numpy as np
from quantum_audio_master import quantum_audio_pipeline, plot_quantum_audio

result = quantum_audio_pipeline('audio/satie.wav', mode='sensing', delay=3)
plot_quantum_audio(result)
# result['y_quantum']  — processed audio array
# result['output_paths'] — saved WAV paths
```

---

## Environment

```
conda activate basic_pitch_env    # Python 3.11
python quantum_audio_master.py
```

Key packages: `qutip`, `smstools` (editable at `../Projects/SMS/sms-tools`), `soundfile`, `scipy`, `numpy`, `matplotlib`, `basic_pitch` (Mode IV only).

---

## File structure

```
quantum_audio_master.py     ← single unified module
Quantum_audio_master_runner.ipynb  ← interactive notebook / test suite
audio/                      ← input and output WAV files
CLAUDE.md                   ← project instructions for Claude Code
README.md                   ← this file
```

---

## Code organisation (`quantum_audio_master.py`)

| Section | What lives here |
|---------|----------------|
| 1 | Imports |
| 2 | Shared quantum primitives |
| 3 | Hamiltonians |
| 4 | Audio I/O |
| 5 | SMS analysis & synthesis helpers |
| 6 | Woolhouse harmony functions (Mode IV) |
| 7 | NV sensing helpers (Mode I) |
| 8 | Trotterize kernels — one per mode |
| 9 | Per-bin quantum transforms (Modes II & III) |
| 10 | Mode-specific pipeline wrappers |
| 11 | Master dispatcher `quantum_audio_pipeline()` |
| 12 | Plot functions |
| 13 | `__main__` example block |

---

## Function call graph

```
quantum_audio_pipeline(audio_path, mode, **kwargs)
│
├─ mode='sensing'        → _pipeline_sensing()
│    ├── load_wav()                   # bandlimit + normalise
│    ├── compute_stretch_factor()     # sensing_rate vs sr
│    ├── time_stretch()               # resample to hires
│    └── trotterize_sensing()
│         ├── _compute_phases_from_signal()   # integrate B_ac → φ[n]
│         ├── _ramsey_unitary_sinphi()        # or _hahn_echo_unitary()
│         ├── gate_partial_SWAP()             # pSWAP gate
│         ├── _embed_gate_2q()               # embed in n-qubit space
│         └── expect(sigmaz(), …)            # readout ⟨σz⟩
│
├─ mode='sensing_dual'   → _pipeline_sensing_dual()
│    ├── load_wav()
│    ├── trotterize_sensing_dual()
│    │    ├── _compute_phases_from_signal()
│    │    ├── _ramsey_unitary_sinphi() / _hahn_echo_unitary()
│    │    ├── _build_joint_sensing_unitary()  # xx | hadamard_id | fdn_hadamard
│    │    ├── _embed_1q_gate()               # R1, R2 on S1, S2
│    │    ├── _embed_gate_2q()               # Usw_in1/2, Usw_out1/2
│    │    ├── expect(sigmaz(), ptrace([idx]))  # r1, r2
│    │    └── summation node (rms | sum | ch1 | ch2)
│    └── time_compress() → resample() → save_wav()
│
├─ mode='sms_colour'     → _pipeline_sms_colour()
│    ├── load_wav_full()
│    ├── sms_analyse()                 # Serra HPS: sin + stoc + STFT
│    ├── get_stochastic_only_bins()    # or other mode_sms selector
│    ├── normalize_amplitudes[_per_bin]()
│    ├── _quantum_colour_transform_per_bin()
│    │    └── [for each bin k]:
│    │         ├── _compute_sms_quantum_params()  # ω_s = lam × f_k
│    │         └── trotterize_colour()
│    │              ├── _get_SPbHam()           # H_free: ω_s σz + ω_cont σx + delays
│    │              ├── _get_interaction_ham()  # V: γx σx⊗σx + γz σz⊗σz + feedback
│    │              ├── U = exp(−i ΔT (H+V))   # precomputed once per bin
│    │              ├── encode_aux_state()      # (amp, phase) → Bloch state
│    │              ├── U · (ρ⊗ρ_aux) · U†
│    │              ├── decode_aux_state()      # ρ_aux_out → (amp', phase')
│    │              └── ptrace(indexlist)       # update maintained state
│    └── synthesise_sms()             # reconstruct waveform
│
├─ mode='sms_trajectory' → _pipeline_sms_trajectory()
│    ├── load_wav_full()
│    ├── sms_analyse()
│    ├── get_stochastic_only_bins()
│    ├── _quantum_trajectory_transform_per_bin()
│    │    └── [for each bin k]:
│    │         └── trotterize_trajectory()
│    │              ├── _get_SPbHam() + _get_interaction_ham() → U_static (precomputed)
│    │              ├── [per frame n]:
│    │              │    ├── H_drive(a_n, φ_n) = ω_xy(cos φ σx+sin φ σy) + a_n ω_z σz
│    │              │    ├── R_drive = exp(−i ΔT H_drive)
│    │              │    ├── ρ_post = U_static · (R_drive ρ R_drive† ⊗ |0⟩⟨0|) · U_static†
│    │              │    ├── p_m = Tr[P_m ρ_post]   (Born rule)
│    │              │    ├── outcome m ~ Bernoulli(p_0)
│    │              │    └── ρ_c = P_m ρ_post P_m / p_m  (collapse)
│    │              └── sys_amp = ⟨|1⟩⟨1|⟩, sys_phase = arctan2(⟨σy⟩,⟨σx⟩)
│    └── synthesise_sms()
│
└─ mode='harmony'        → _pipeline_harmony()
     ├── load_wav_full()
     ├── extract_chord_events()       # basic_pitch → MIDI note groups
     ├── score_to_density_chain()
     │    └── [for each chord pair i→i+1]:
     │         ├── tonal_attraction_matrix()
     │         │    ├── pitch_distance_matrix()
     │         │    ├── interval_cycle_matrix()
     │         │    ├── voice_leading_matrix()
     │         │    ├── root_salience_matrix()
     │         │    └── consonance_dissonance_scale()
     │         └── ta_to_density_matrix()   # TA → valid qutrit ρ
     ├── trotterize_harmony()
     │    ├── _build_qutrit_free_ham()      # H = ω·diag(0,1,2)
     │    ├── _build_qutrit_interaction()   # V = g Σ_j λ_j⊗λ_j  (Gell-Mann)
     │    ├── U = exp(−i ΔT (H+V))
     │    ├── [per chord transition n]:
     │    │    ├── rho_full = ρ_sys ⊗ ρ_TA[n]   (structured bath)
     │    │    ├── ρ_post = U · rho_full · U†
     │    │    └── diag(ρ_aux_out) → winner ∈ {root, 3rd, 5th}
     │    └── winners[], diagonals[]
     ├── sms_analyse()
     ├── apply_brightness_modulation()    # boost winner's harmonic bins
     └── synthesise_brightened_audio()
```

---

## Equations of motion by mode

### Shared: phase accumulation (Modes I and Ia)

Input signal x(t) is integrated over each sensing window T_φ:

```
φ[n] = (phase_scale / T_φ) ∫_{n T_seq}^{n T_seq + T_φ}  g(t) · x_stretched(t) dt
```

where `g(t) = +1` (Ramsey) or `±1` (Hahn echo, sign flip at T_φ/2).

---

### Mode I — NV Sensing

**Encoding unitary** (Ramsey, per sensing cycle n):

```
U_enc(φ) = Ry(−π/2) · Rz(φ) · Rx(π/2)
```

Action on |0⟩: Bloch vector (0,0,1) → (cos φ, 0, sin φ). Readout ⟨σz⟩ = sin(φ).

**State update** (delay d, one step):

```
ρ_ext = tensor(ρ^{(n)}, |0⟩⟨0|_f)
ρ_ext = R_enc · ρ_ext · R_enc†               [embed U_enc on qubit 0]
ρ_ext = U_in  · ρ_ext · U_in†                [pSWAP(θ_in) on (S, f)]
ρ_ext = U_out · ρ_ext · U_out†               [pSWAP(θ_out) on (S, D[0])]
readout: r[n] = ⟨σz⟩_{D[0]} = Tr[σz · Tr_{all\{D[0]}}(ρ_ext)]
ρ^{(n+1)} = Tr_{D[0]}(ρ_ext)                 [discard oldest delay qubit]
```

pSWAP(θ) = cos(θ)·I₄ + i sin(θ)·SWAP. Its action on (ρ_S ⊗ |0⟩⟨0|_f):

```
Bloch vector: x' = |cos θ| x,  y' = |cos θ| y,  z' = sin²θ + cos²θ · z
```

This is an **amplitude damping channel** toward |0⟩⟨0| with rate γ = sin²(θ).

**Non-Markovian memory**: delay d > 0 introduces a finite memory kernel of depth d steps. The maintained state carries correlations from steps n through n−d.

---

### Mode Ia — Dual-Channel NV Sensing

Hilbert space: `H_maint = C²_{S1} ⊗ C²_{S2} ⊗ (C²)^{d1} ⊗ (C²)^{d2}`.

**Full one-step unitary** (in extended space with fresh qubits f1, f2 appended):

```
U_total = U_out2 · U_out1 · U_in2 · U_in1 · U_J · R2 · R1

where:
  R1     = U_enc(φ[n]) ⊗ I_rest          [encode on S1]
  R2     = I_S1 ⊗ U_enc(φ[n]) ⊗ I_rest  [encode on S2]
  U_J    = U_joint^{S1,S2} ⊗ I_rest     [joint coupling]
  U_in1  = pSWAP(θ_in) on (S1, f1)      [push S1 encoding into delay]
  U_in2  = pSWAP(θ_in) on (S2, f2)
  U_out1 = pSWAP(θ_out) on (S1, D1[0])  [quantum memory feedback]
  U_out2 = pSWAP(θ_out) on (S2, D2[0])
```

**Joint coupling unitaries**:

| `joint_mode` | Unitary | Character |
|---|---|---|
| `xx` | exp(−i θ_xx σx⊗σx) = cos(θ)·I₄ − i sin(θ)·(σx⊗σx) | Transverse Ising; partial Bell-pair generator |
| `hadamard_id` | H⊗I | Hadamard on S1 only; no inter-channel entanglement |
| `fdn_hadamard` | Identity (no quantum gate) | Classical FDN mix M=[[-1,1],[1,-1]]/√2 applied to readouts after measurement |

**Readouts and summation**:

```
r1[n] = Tr[σz · Tr_{all\{D1[0]}}(ρ_ext^{post})]
r2[n] = Tr[σz · Tr_{all\{D2[0]}}(ρ_ext^{post})]

RMS output: out[n] = sign(r1+r2) · √((r1² + r2²)/2)
```

**State update**: ptrace over D1[0] and D2[0]; permute f1_enc, f2_enc into canonical delay positions.

---

### Mode II — SMS Colour (collision model)

**Hamiltonians** (per STFT bin k, precomputed once):

```
H_free = ω_s σz^S + ω_cont σx^S + ω_cont_z σz^S + Σ_j ω_r σz^{D_j}

V_int  = γx (σx^S ⊗ σx^{fresh}) + γz (σz^S ⊗ σz^{fresh})
       − e^{−iφ} γx (σx^S ⊗ σx^{D[0]}) − e^{−iφ} γz (σz^S ⊗ σz^{D[0]})

U = exp(−i ΔT (H_free + V_int))   [one matrix exponentiation per bin]
```

where `ω_s = lam × f_k`, `γx/z = gamma × ω_s`.

**Encoding / decoding** (Bloch sphere):

```
encode:  (amplitude a, phase φ) → |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩
          where θ = 2 arcsin(√a)
decode:  pop = ⟨|1⟩⟨1|⟩,  phase = arctan2(⟨σy⟩, ⟨σx⟩)
```

**Per-step map**:

```
ρ^{(n+1)} = Tr_{D[0]} [ U · (ρ^{(n)} ⊗ ρ_aux^{(n)}) · U† ]
```

Continuum limit (ΔT→0, γ²ΔT→Γ): recovers Lindblad ME with jump operators ∝ σx, σz — dephasing + relaxation bath. With delay: non-Markovian collision model (NMCM).

---

### Mode III — SMS Trajectory (quantum jumps)

**Time-dependent drive** (per bin k, per frame n):

```
H_drive(n) = ΔT [ ω_xy (cos φ_n σx + sin φ_n σy) + a_n ω_z σz ]
```

where `a_n = normalised Xr_mag[n,k]` and `φ_n = phase of nearest sinusoidal track`.

**Static map** (precomputed, identical to Mode II with ω_s_static = 0):

```
U_static = exp(−i ΔT (V_int + H_delay))
```

**Quantum jump unravelling** (per frame n):

```
1. R_drive(n) = exp(−i H_drive(n))
2. ρ_driven   = R_drive · ρ^{(n)} · R_drive†
3. ρ_full     = ρ_driven ⊗ |0⟩⟨0|_f
4. ρ_post     = U_static · ρ_full · U_static†
5. p_m        = Tr[P_m · ρ_post],  m ∈ {0,1}   (Born rule)
6. outcome m  ~ Bernoulli(p_0)
7. ρ_c        = P_m · ρ_post · P_m / p_m         (state collapse)
8. ρ^{(n+1)} = Tr_{D[0]}(ρ_c)
```

Output: `sys_amp[n] = ⟨|1⟩⟨1|⟩`, `sys_phase[n] = arctan2(⟨σy⟩, ⟨σx⟩)` measured *before* collapse. Energy per bin preserved by rescaling: `∑_n |out[n]|² = ∑_n |in[n]|²`.

The measurement record `m[n] ∈ {0,1}` is a delta-sigma modulation of the spectral envelope.

---

### Mode IV — Harmony (qutrit SU(3))

**Free Hamiltonian** (3-level system, equally spaced):

```
H_sys = ω_sys · diag(0, 1, 2)
```

**Interaction** (off-diagonal Gell-Mann generators λ₁, λ₄, λ₆):

```
V = g Σ_{j∈{1,4,6}} (λ_j^{sys} ⊗ λ_j^{fresh})  +  delay feedback
U = exp(−i ΔT (H_sys + V))
```

**Bath state**: the fresh auxiliary carries the Woolhouse tonal attraction density matrix:

```
ρ_TA = ta_to_density_matrix( IC × VL × RS × CD )
```

where IC = interval cycle, VL = voice leading, RS = root salience, CD = consonance/dissonance.

**Per-step collision**:

```
ρ_sys^{(n+1)} = Tr_{fresh}[ U · (ρ_sys^{(n)} ⊗ ρ_TA^{(n)}) · U† ]
```

Output: `diag(ρ_aux_out) = (p_root, p_3rd, p_5th)` → winner voice selected by argmax.

Brightness modulation: `boost = 1 + brightness × (p_winner − 1/3)` applied to the winner voice's harmonic partials across the chord's time span.

---

## Parameter quick-reference

### Universal

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_path` | str | — | Input WAV (resolved relative to script dir) |
| `mode` | str | `'sensing'` | `'sensing'` \| `'sensing_dual'` \| `'sms_colour'` \| `'sms_trajectory'` \| `'harmony'` |
| `delay` | int | 3 | Delay-line depth; Hilbert dim = 2^(delay+2) or 3^(delay+2); **keep ≤ 9** |
| `phase` | float | π | Feedback phase φ in delay-line coupling [rad] |

### Mode I / Ia

| Parameter | Default | Description |
|---|---|---|
| `sensing_rate` | 4000 | Sensing cycles/s → effective Nyquist |
| `max_freq` | 2000 | Low-pass cutoff before encoding [Hz] |
| `phase_scale` | 0.5 | φ_max ≈ phase_scale at unit input amplitude |
| `sequence` | `'ramsey'` | `'ramsey'` or `'hahn_echo'` |
| `method` | `'sin_phi'` | `'sin_phi'` (one circuit/sample) or `'qpsd'` (IQ demod) |
| `swap_angle_in` | 0.9 | θ_in: S↔fresh SWAP angle; γ = sin²(θ) |
| `swap_angle_out` | 0.1 | θ_out: S↔D[0] feedback angle |
| `summation_mode` | `'rms'` | `'rms'` \| `'sum'` \| `'ch1'` \| `'ch2'` (Ia only) |
| `joint_mode` | `'xx'` | `'xx'` \| `'hadamard_id'` \| `'fdn_hadamard'` (Ia only) |
| `xx_coupling` | 0.3 | θ_xx [rad] for `joint_mode='xx'`; π/4 = max entanglement |
| `delay1`, `delay2` | 1,1 | Independent delay depths for D1 and D2 (Ia only; d1+d2 ≤ 9) |

### Mode II

| Parameter | Default | Description |
|---|---|---|
| `omega_cont` | 50.0 | σx transverse drive; controls phase rotation of output |
| `omega_cont_z` | 0.0 | σz longitudinal drive; amplitude asymmetry |
| `gamma_x` | None (auto) | σx⊗σx coupling strength |
| `gamma_z` | 1.0 | σz⊗σz coupling strength; dephasing rate |
| `lam` | 1/100 | ω_s = lam × f_bin; sets frequency scale |
| `mode_sms` | `'stochastic_only'` | Bin selection strategy (see below) |
| `normperbin` | False | Per-bin amplitude normalisation before transform |
| `headroom` | None | Clip output to headroom × input per bin |

**`mode_sms` options:**

| Value | Bins used | Input source |
|---|---|---|
| `'stochastic_only'` | Bins never containing a sinusoidal | Residual STFT |
| `'reverse_sinusoidal'` | Bins occupied by tracked partials | Sinusoidal from previous frame |
| `'full_stft'` | All bins | Sinusoidal + residual merged |
| `'stochastic_unraveler'` | Same as `stochastic_only` | Residual amp, sinusoidal phase |

### Mode III

| Parameter | Default | Description |
|---|---|---|
| `gamma_x` | 0.1 | σx⊗σx coupling |
| `gamma_z` | 1.0 | σz⊗σz coupling |
| `omega_drive_xy` | 1.0 | In-plane Rabi drive strength |
| `omega_drive_z` | 0.0 | Longitudinal drive |
| `measurement_basis` | `'z'` | `'z'` \| `'x'` \| `'y'` — projective collapse basis |
| `seed` | None | RNG seed for reproducibility |
| `normperbin` | False | Per-bin normalisation of drive amplitude |

### Mode IV

| Parameter | Default | Description |
|---|---|---|
| `coupling_g` | 0.5 | Gell-Mann coupling strength g |
| `omega_sys` | 10.0 | Qutrit level spacing ω (free Hamiltonian) |
| `brightness` | 1.5 | Harmonic boost factor |
| `n_harmonics` | 8 | Harmonic partials boosted per winner voice |
| `alpha` | 50.0 | Voice-leading weight in TA matrix |
| `chord_size` | 3 | Minimum notes per chord event |

---

## Output files

All outputs are written to the `audio/` directory beside the input file.

| Mode | Suffixes |
|---|---|
| I sensing | `_I_sensing.wav` |
| Ia sensing_dual | `_Ia_sensing_dual.wav` |
| II sms_colour | `_II_colour_combined.wav`, `_II_colour_original.wav`, `_II_colour_harmonic_q.wav`, `_II_colour_stoc_q.wav`, `_II_colour_stoc_orig.wav` |
| III sms_trajectory | `_III_trajectory_combined.wav`, `_III_trajectory_original.wav`, `_III_trajectory_harmonic.wav`, `_III_trajectory_stoc_q.wav`, `_III_trajectory_stoc_orig.wav` |
| IV harmony | `_IV_harmony_quantum.wav`, `_IV_harmony_baseline.wav` |

---

## Known constraints and bug fixes

- `delay ≤ 9`: Hilbert space dimension 2^(d+2) or 3^(d+2); beyond d=9 runtime is prohibitive
- `soundfile` is **never** aliased as `sf` anywhere in the codebase (B4)
- `omega_s_static = 0` by default in Mode III — all precession from drive terms (B5)
- `save_wav_with_resampling` is defined before first use (B3)
- Mode IV calls `sms_analyse()` — does not inline analysis (B13)
- Audio path resolution: tries path as given → relative to script dir → `audio/<basename>` — notebooks from any working directory work without path adjustment
