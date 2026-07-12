# Quantum Sensing vs Rocchesso's QFDN

Head-to-head comparison of a **qubit delay-line quantum sensing** audio processor against
**Rocchesso's Quantum Feedback Delay Network** (QFDN, *Quantum Music* Ch. 2, 2025).

## Files

| file | what it is |
|---|---|
| `QFDN_vs_Sensing_comparison.ipynb` | the main notebook (8 runs + summary) |
| `QFDN_vs_Sensing_comparison-Copy1.ipynb` | same notebook with the full executed results (single delay d=9) |
| `quantum_audio_master.py` | the engine module the notebook imports (sensing kernels, gates, SMS/audio I/O) |
| `audio/` | sample inputs (`bachchoral.wav`, `vcv_dry.wav`) |

`quantum_audio_master.py` is the **only local dependency** — everything else is pip/conda packages.

## Dependencies

```bash
conda activate basic_pitch_env       # Python 3.11
pip install qutip qiskit qiskit-aer soundfile scipy numpy matplotlib sounddevice
# sms-tools (Serra), installed editable:
#   git clone https://github.com/MTG/sms-tools && cd sms-tools && pip install -e .
```

## Running

Open `QFDN_vs_Sensing_comparison.ipynb` and run top-to-bottom.

Structure:
- **Util cells** — params, QFDN engines (Qiskit), sensing helpers, single-channel + amplitude
  stretch, post-processing/presentation.
- **8 run cells** — `{Impulse, Sinusoid 3 s, Bass notes, Audio input} x {single, dual}`.
  Each shows the input, the matched Rocchesso QFDN, and the sensing **standard** read
  (DC-filter + empirical stretch) and **+FB** read (DC-filter + matched to input).
- **Summary cell** — raw output of all 16 sensing runs.

### Cost warning (read before changing `SINGLE_D`)

The sensing simulates a full density matrix, so the cost per step is exponential in the delay:

| | Hilbert dim | relative cost |
|---|---|---|
| single, `d = 3` | `2^5 = 32` | 1x |
| single, `d = 5` | `2^7 = 128` | ~16x |
| single, `d = 8` | `2^10 = 1024` | ~250x |
| single, `d = 9` | `2^11 = 2048` | ~1000x |

dual is `2^(2 + d1 + d2)`. A ~10 s clip at `FS = 1400` is ~13,000 steps — at `d = 9` that is
**hours**. Use `SINGLE_D <= 5` (and/or a short `AUDIO_SEC`) for quick passes.

## Audio input

`AUDIO_PATH` in the params cell selects the input sound. The clip used for the paper figures
(`nightfall.wav`, 106 MB) is **not committed** (over GitHub's 100 MB file limit) — point
`AUDIO_PATH` at your own file, or use the included `audio/bachchoral.wav`.

Note `FS` sets the Nyquist limit: at `FS = 1400` only content below 700 Hz survives the
downsample, so bright/percussive sources (bells, cymbals) are gutted before the sensing sees
them. Raise `FS` for those.

## Key parameters (params cell)

- `phase_scale` (lambda) — modulation depth; sample -> phase `phi_n = lambda * s_n`, readout `<sz> = sin(phi_n)`.
- `SINGLE_D`, `D1`, `D2` — delay-line lengths (the memory / loop delay).
- `SWAP_IN`, `SWAP_OUT`, `FB_ANGLE` — swap **angles** in radians; the swapped fraction is `sin^2(theta)`.
- `exp_feed_back` — read the delayed qubit *before* the swap and recirculate it (the reverb mode).
- `XX_COUPLING` — dual-only inter-channel coupling, `U = exp(-i * theta * sx (x) sx)`.
- `QFDN_FCOEF`, `QFDN_FB_MATRIX` — Rocchesso's loop coefficient and feedback matrix.
