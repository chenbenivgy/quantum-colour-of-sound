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

### ⛔ QUANTUM-AS-CARRIER IS INVIOLABLE (read before proposing ANY fix)
The audio output must be **decoded from the quantum channel and from nothing else**.
The signal is represented as a quantum state, evolved, measured, and decoded back —
the whole scientific claim (that the output carries correlations no classical
measurement channel could produce) dies the moment any part of the output comes
from somewhere other than the decoded quantum state.

**NEVER propose or implement, under any name or justification:**
- routing cells/bins/frames *around* the channel (gates, thresholds, bypass, "silence
  is untouched anyway", passthrough of near-empty cells)
- mixing, blending, or crossfading the input into the output (wet/dry, "only 1% dry")
- subtracting the input from the output, or deriving the output as a *modification of
  the input* rather than as the decoded channel state
- substituting classically-generated content (noise, interpolation, smoothing) for any
  output cell — including "repairing" artifacts this way
- any per-cell conditional that selects between "quantum result" and "input value"

This class of bug has been introduced repeatedly and is expensive to catch. **If an
artifact appears, the only admissible fixes act INSIDE the physics**: the initial
state, the couplings (gamma_x/gamma_z), the delay, the feedback phase, the
measurement basis, the encode/decode companding (must be invertible and applied to
ALL cells identically), omega_s_res, the STFT parameters, or the number of
trajectories. If none of those fix it, say so plainly and leave the artifact in.

Exception, clearly labelled: the `USE_QUANTUM_AMP` / `USE_QUANTUM_PHASE` A/B flags,
which are diagnostic comparisons, never the deliverable output.

### Other constraints
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

## Encoding target flag (`encode_target`, added 2026-09-02)
Both main routes can now encode into either the persistent **system** qubit or the
fresh **auxiliary** probe. Defaults reproduce the pre-flag code bit-for-bit.
- Sensing: `trotterize_sensing(..., encode_target='system')` (also on `_pipeline_sensing`
  → `quantum_audio_pipeline(mode='sensing', encode_target=...)`). `'aux'` prepares the
  fresh probe as `U_enc|0⟩` (carries `⟨σz⟩ = sin φ` exactly) and never rotates S.
- Dependent: `dependent_colour_transform_per_bin(..., encode_target='aux')` and
  `_dependent_colour_kernel(...)`. `'system'` keeps probes as vacuum and applies
  `qam.encode_unitary(a, φ) = Rz(φ)·Ry(2·asin√a)` to S (`R|0⟩` = `encode_aux_state`).
- Companding/decoding/measurement are unchanged for both targets. Not wired into
  `trotterize_sensing_dual` or the Mode-III `trotterize_trajectory` route.
- Comparison cells: `Sensing_summary.ipynb` §13, `Dependent_trajectory_summary.ipynb` §15.

## Discontinuous Sound Modulator (VCV Rack module, added 2026-09-03)
Records `t_l` of input, runs the dependent-trajectory channel (`encode_target='system'`,
summary-notebook operating point) and loops the render; recompute + crossfade (t_l/2) on
parameter change; TS1 = varispeed with a t_l/4 glide; Off clears memory; no declick.
- `dsm_engine.py` — pure-NumPy blueprint of the kernel/transform, certified vs qutip
  (`test_dsm_engine.py`, run after ANY change to `dependent_trajectory.py`).
- `dsm_chain.py` — decimation, periodic STFT + warm-up prefix, circular OLA, upsampling
  (`test_dsm_chain.py`). `dsm_server.py` — engine A (unchanged Python code) job server.
- `dsm/` — the Rack plugin. `dsm/src/dsm_engine.hpp` = engine B (C++), certified by
  `test_dsm_cpp.py` (same-STFT decoded matrices ≤1e-13 vs qutip, identical measurement
  records); `dsm/src/DsmCore.hpp` = module logic (`dsm/tools/core_test.cpp`).
  Build: `cd dsm && make install` (Makefile forces IEEE-strict float flags).
- `dsm_verify.py <export_dir>` — re-runs the original code on a buffer the module
  recorded (right-click → Export) and reports per-stage differences.
- `DSM_coupling_sweep.ipynb` — coupling sweep for the system encoding (tables A/B, audio).
- Numerically-silent bins: the system encoding applies Rz(φ) to S even at zero amplitude,
  so FFT round-off phases leak into the output at ~1e-8; compare engines on the SAME STFT.
- v2 (2026-09-08): PAST/FUTURE material modes (rolling 10.5 s tap; PAST = the t_l before On,
  computed at once); physics jobs (abort + new seed on any physics change) vs synthesis jobs
  (render selector undep/dep0/dep1/twin, trajectory count down: no quantum compute);
  TRAJECTORIES knob 1..8 appends trajectories incrementally (own seed each, exact); layered
  crossfades aligned on material time; engine A returns the trajectory store
  (`dsm_engine.trajectories(kernel_impl='qutip')`), server heartbeat + cancel, fallback to B
  with ERR light. `dsm_engine.render_from_store` / `twin` / `twin_bin` are the render
  definitions; `test_dsm_store.py` certifies them against the unchanged transform.
- v2.1 (2026-09-20): progressive publication of appended trajectories; Off clears the tap;
  SYNC switch (matched-filter start on the input's repeat, 1.5 t_l fallback,
  `dsm/tools/sync_test.cpp`); colour-grouped panel with custom knob SVGs.
  `Core::log()` locks the worker mutex — never call it while holding `mu`.

## Known Bug Fixes — audit additions (B15–B21)
- **B15**: `p1` must be `tr(P1·ρ)` explicitly — never the `1 - p0` shortcut (it silently
  assumes trace 1 and biased first-step outcome sampling)
- **B16**: dependent transform compensates delay-line latency internally (input padded by
  `delay`, decoded arrays sliced `[delay : delay+num_frames]`) — output is time-aligned
- **B17**: `diag[k]['fallback_frac']` records cond0/cond1 empty-class fallbacks
- **B18**: notebook "amp OFF" cell must actually pass `uqa=False` and use the §4 render route
- **B19**: §15 fast/standard cell must mirror the §4 call exactly (all companding params)
- **B21**: `_initial_system_state('Pur.Deph')` returns **ground |0⟩⟨0|**, NOT |+⟩⟨+|.
  |+⟩ carries population 0.5 = real energy in an excitation-preserving channel, so it
  injects half a quantum into EVERY bin including silent ones, and with `phase=pi` it
  recirculates rather than draining (+28 dB noise-floor lift on tonal material).
  Ground init also restores the effective dynamics of the pre-B15 code.
