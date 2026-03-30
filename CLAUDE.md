# Quantum Colour of Sound — Project Brief

## Environment
- **Conda env**: `basic_pitch_env` (Python 3.11, miniconda3)
- **Full Python path**: `/opt/miniconda3/envs/basic_pitch_env/bin/python`
- **Run scripts as**: `conda activate basic_pitch_env && python quantum_audio_master.py`
- **sms-tools**: installed editable at `/Users/xxxxx/Documents/Projects/SMS/sms-tools`
  - Installed via `pip install -e .` inside that folder — do NOT add sys.path manually
  - Import as `from smstools.models import ...` (not `from smstools.smstools...`)

## Key Dependencies
```
qutip          # quantum simulation
smstools       # Serra's SMS audio analysis/synthesis (local editable install)
basic_pitch    # chord extraction for Mode IV only
soundfile      # wav I/O (imported as `soundfile`, NEVER aliased as `sf` — B4)
scipy, numpy, matplotlib
```

## Project File
**`quantum_audio_master.py`** — single unified file, all four modes. Structure:

| Section | Contents |
|---------|----------|
| 1  | Imports |
| 2  | Shared quantum primitives (gates, encoding, Gell-Mann λ, TA matrices) |
| 3  | Hamiltonians |
| 4  | Audio I/O (`load_wav`, `save_wav`, `save_wav_with_resampling`, etc.) |
| 5  | SMS analysis & synthesis (`sms_analyse`, `synthesise_sms`, bin helpers) |
| 6  | Woolhouse harmony functions (Mode IV only) |
| 7  | NV sensing helpers (Mode I only) |
| 8  | Trotterize kernels — one per mode (8a–8d) |
| 9  | Per-bin quantum transforms (Modes II & III) |
| 10 | Mode-specific pipeline wrappers (`_pipeline_*`) |
| 11 | Master dispatcher `quantum_audio_pipeline()` |
| 12 | Plot function `plot_quantum_audio()` |
| 13 | `if __name__ == '__main__'` example block |

## The Four Modes

### Mode I — `'sensing'` (NV-center quantum sensing)
Audio → phase-encoded signal → qubit Ramsey/Hahn circuit → ancilla readout

Key params: `sensing_rate=2000`, `phase_scale=0.5`, `method='sin_phi'|'qpsd'`, `sequence='ramsey'|'hahn_echo'`

### Mode II — `'sms_colour'` (deterministic spectral colouring)
SMS residual STFT bins → encode into fresh auxiliary qubits → collision model → decode output

Key params: `omega_cont`, `omega_cont_z`, `gamma_x`, `gamma_z`, `lam=1/100`, `normperbin`, `headroom`

Bin selection via `mode_sms='stochastic_only'` (default) or `'all'`

### Mode III — `'sms_trajectory'` (stochastic quantum trajectory)
SMS spectral data drives system; vacuum aux bath; projective collapse at each step

Key params: `gamma_x`, `gamma_z`, `omega_drive_xy`, `omega_drive_z`, `measurement_basis='z'|'x'|'y'`, `seed`

### Mode IV — `'harmony'` (qutrit tonal brightness)
`basic_pitch` chord extraction → Woolhouse (2009) TA matrices → qutrit density matrices → qutrit collision model → spectral brightness modulation

Key params: `coupling_g=0.5`, `brightness=1.5`, `chord_size=3`, `omega_sys=10.0`
**Requires `basic_pitch` installed AND ≥2 chord events in the audio file.**

## Universal API

```python
result = quantum_audio_pipeline(
    'audio/satie.wav',
    mode='sensing',   # 'sensing' | 'sms_colour' | 'sms_trajectory' | 'harmony'
    delay=3,          # int, keep ≤9 (runtime explodes beyond)
    phase=np.pi,
    **mode_kwargs
)
plot_quantum_audio(result, show_details=True)
```

`result` always contains:
- `result['y_quantum']` — processed audio array
- `result['y_original']` — original audio array
- `result['fs']` — sample rate
- `result['output_paths']` — dict of saved `.wav` paths
- `result['params']` — run parameters

Output filenames follow: `<stem>_I_sensing.wav`, `<stem>_II_colour_combined.wav`, etc.

## Critical Constraints
- `delay ≤ 9` — Hilbert space dimension = 2^(1+delay+1), runtime explodes beyond delay=9
- Mode IV needs `basic_pitch` installed separately and ≥2 detectable chord events
- `soundfile` is NEVER aliased as `sf` anywhere in this codebase (B4 bug fix)
- `omega_s_static=0` by default in trajectory mode — all precession from drive terms (B5)
- `save_wav_with_resampling` is defined before first use (B3 — do not reorganise)

## Known Bug Fixes (B1–B13)
All documented in the module docstring. Key ones when editing:
- **B2**: `trotterize_trajectory` delay=0 projector must use index 1 (fresh only), not index 0
- **B4**: never `import soundfile as sf` — use `soundfile.write(...)` directly
- **B5**: `omega_s_static` explicit param (default 0) in `trotterize_trajectory`
- **B7**: `sin_phases` must be included in all inline analysis dicts
- **B12**: no spurious `tensor()` wrapping of single `Qobj`
- **B13**: Mode IV calls `sms_analyse()` function, does NOT inline the analysis

## Woolhouse Harmony Details (Mode IV)
- TA matrix = IC × VL × RS × CD (interval cycle × voice leading × root salience × consonance/dissonance)
- TA matrix converted to valid qutrit density matrix via `ta_to_density_matrix()` (symmetrize or gram method)
- Gell-Mann λ₁, λ₄, λ₆ (off-diagonal) used for SU(3) coupling in `_build_qutrit_interaction()`
- Winners (root / 3rd / 5th) select which voice's harmonic bins get brightness boost

## SMS Analysis Parameters (defaults)
```python
M=2501, N=4096, H=1024, t=-80, nH=60,
minf0=100, maxf0=400, f0et=5,
harmDevSlope=0.01, minSineDur=0.02,
Ns=2048, stocf=0.2
```
Mode IV uses `minf0=50, maxf0=2000` to capture full chord range.

## Typical Workflow
```python
import numpy as np
from quantum_audio_master import quantum_audio_pipeline, plot_quantum_audio

result = quantum_audio_pipeline('satie.wav', mode='sms_colour',
                                 delay=3, phase=np.pi,
                                 gamma_x=0, gamma_z=1,
                                 omega_cont=0, omega_cont_z=0.1,
                                 normperbin=True)
plot_quantum_audio(result)
```
