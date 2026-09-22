"""
Discontinuous Sound Modulator -- engine blueprint (pure NumPy).

A re-implementation of the dependent-trajectory kernel
(`dependent_trajectory._dependent_colour_kernel`) and of the per-bin transform
(`dependent_trajectory.dependent_colour_transform_per_bin`) with NO qutip, in
exactly the arithmetic a C++ port will use:

  * the collision unitary is built ONCE per bin as the local 3-qubit operator
    U_A on (S, D_0, F) times commuting single-qubit phases on the spectator
    delay qubits (exact factorisation: H_A and the spectator terms act on
    disjoint qubits, so exp(-i dt (H_A + sum_j h_j)) = U_A (x) prod_j exp(-i dt h_j));
  * per tick, U_A is applied to the three local axes of the register tensor,
    the exiting probe's 2x2 reduced state is contracted directly, the outcome
    is drawn with the same numpy RandomState call sequence as the original,
    and the collapse + trace-out of index 1 are one contraction.

Physics, measurement, trace and RNG are untouched: the certificate is the
regression test `test_dsm_engine.py` (max |diff| <= 1e-12 against the qutip
kernel and identical measurement records, across delays 0..5, both encoding
targets, bases x/z/none, with and without carrier clock / resonator).

Qubit order (big-endian, as qutip.tensor): index 0 = S, 1 = D_0 (the probe
read this tick), 2..d = D_1..D_{d-1}, d+1 = F (fresh probe). After the tick,
index 1 is traced out; the kept order (S, D_1..D_{d-1}, F) is next tick's
(S, D_0..D_{d-1}) -- the line shifts by dropping index 1 and appending.
"""
import numpy as np
from scipy.linalg import expm
try:
    from threadpoolctl import threadpool_limits as _tp_limits
except ImportError:                       # pragma: no cover
    import contextlib
    def _tp_limits(limits=1):
        return contextlib.nullcontext()

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2, dtype=complex)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], complex)


def _rz(phi):
    return np.array([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]], complex)


def encode_unitary(a, phi):
    """R(a, phi) = Rz(phi) Ry(2 asin sqrt a)  (== qam.encode_unitary)."""
    a = min(max(float(a), 0.0), 1.0)
    return _rz(float(phi)) @ _ry(2.0 * np.arcsin(np.sqrt(a)))


def encode_state(a, phi):
    """|psi(a,phi)><psi|, psi = cos(th/2)|0> + e^{i phi} sin(th/2)|1>  (== qam.encode_aux_state)."""
    a = min(max(float(a), 0.0), 1.0)
    th = 2.0 * np.arcsin(np.sqrt(a))
    psi = np.array([np.cos(th / 2), np.exp(1j * float(phi)) * np.sin(th / 2)], complex)
    return np.outer(psi, psi.conj())


def basis_vectors(measurement_basis):
    """Outcome kets (v0, v1) of the projective measurement on the exiting probe."""
    if measurement_basis == 'none':
        return None
    if measurement_basis == 'z':
        return np.array([1, 0], complex), np.array([0, 1], complex)
    if measurement_basis == 'x':
        return (np.array([1, 1], complex) / np.sqrt(2), np.array([1, -1], complex) / np.sqrt(2))
    if measurement_basis == 'y':
        return (np.array([1, 1j], complex) / np.sqrt(2), np.array([1, -1j], complex) / np.sqrt(2))
    raise ValueError(measurement_basis)


# ----------------------------------------------------------------------------
# the collision unitary, factorised
# ----------------------------------------------------------------------------
def build_collision(gx, gz, omega_s, omega_cont, omega_cont_z, omega_r, delay, phase,
                    DeltaT, interaction='beamsplitter', theta_clock=0.0):
    """Returns (U_A, local_axes, spec_phase) with
         U_A        : 2^L x 2^L unitary on the local axes (L = 3 for delay>=1, 2 for delay=0)
         local_axes : register indices U_A acts on, in U_A's own order
         spec_phase : diagonal single-qubit phases for spectator axes {j: 2-vector}
    such that the full U = exp(-i dt (H + V)) of the original kernel equals
    U_A on local_axes tensored with diag(spec_phase[j]) on every spectator j.

    H (system + delay free terms, tensored with I on F):
        omega_s sz_S + omega_cont sx_S + omega_cont_z sz_S + omega_r sum_k sz_{D_k}
        + (theta_clock/2) sum_{all qubits} sz
    V ('beamsplitter'): gx (sx sx + sy sy) + gz sz sz  on (S,F)
                        + fb [gx (sx sx + sy sy) + gz sz sz] on (S,D_0),  fb = -exp(-i phase)
      ('spin')        : gx sx sx + gz sz sz  (same placement)
    Note: for phase not in {0, pi}, fb is complex and V is non-Hermitian; the
    original renormalises the trace per tick, and so do we. The exponential is
    the general dense expm in that case.
    """
    n = delay + 2
    fb = -np.exp(-1j * phase)
    if delay == 0:
        local = [0, 1]                     # S, F  (index 1 IS the fresh probe)
    else:
        local = [0, 1, n - 1]              # S, D_0, F
    L = len(local)
    pos = {q: i for i, q in enumerate(local)}

    def op1(q, m):                          # m on local qubit q, identity elsewhere
        ops = [I2] * L
        ops[pos[q]] = m
        return _kron(*ops)

    def op2(q1, m1, q2, m2):
        ops = [I2] * L
        ops[pos[q1]] = m1
        ops[pos[q2]] = m2
        return _kron(*ops)

    S, F = 0, n - 1
    H = (omega_s + omega_cont_z) * op1(S, SZ) + omega_cont * op1(S, SX)
    for q in local:
        H = H + 0.5 * theta_clock * op1(q, SZ)
    if delay > 0:
        H = H + omega_r * op1(1, SZ)      # D_0 is local; other D_k are spectators
    if interaction == 'beamsplitter':
        V = gx * (op2(S, SX, F, SX) + op2(S, SY, F, SY)) + gz * op2(S, SZ, F, SZ)
        if delay > 0:
            V = V + fb * (gx * (op2(S, SX, 1, SX) + op2(S, SY, 1, SY)) + gz * op2(S, SZ, 1, SZ))
    elif interaction == 'spin':
        V = gx * op2(S, SX, F, SX) + gz * op2(S, SZ, F, SZ)
        if delay > 0:
            V = V + fb * (gx * op2(S, SX, 1, SX) + gz * op2(S, SZ, 1, SZ))
    else:
        raise ValueError(interaction)
    U_A = expm(-1j * DeltaT * (H + V))
    spec = {}
    for j in range(n):
        if j in local:
            continue
        e = 0.5 * theta_clock + omega_r      # spectator D_k: clock + resonator, both sz
        spec[j] = np.array([np.exp(-1j * DeltaT * e), np.exp(1j * DeltaT * e)], complex)
    return U_A, local, spec


def _apply_local(rho_t, U, axes, n):
    """rho_t: tensor (2,)*n + (2,)*n.  Returns U rho U^dag with U acting on `axes`."""
    L = len(axes)
    # rows
    src = list(axes)
    dst = list(range(L))
    r = np.moveaxis(rho_t, src, dst)
    shp = r.shape
    r = U @ r.reshape(2 ** L, -1)
    r = np.moveaxis(r.reshape(shp), dst, src)
    # cols
    src_c = [n + a for a in axes]
    r = np.moveaxis(r, src_c, dst)
    shp = r.shape
    r = U.conj() @ r.reshape(2 ** L, -1)
    r = np.moveaxis(r.reshape(shp), dst, src_c)
    return r


def _apply_spectator_phases(rho_t, spec, n):
    for j, ph in spec.items():
        shape = [1] * (2 * n)
        shape[j] = 2
        rho_t = rho_t * ph.reshape(shape)
        shape = [1] * (2 * n)
        shape[n + j] = 2
        rho_t = rho_t * ph.conj().reshape(shape)
    return rho_t


# ----------------------------------------------------------------------------
# the kernel: one trajectory of one bin
# ----------------------------------------------------------------------------
def kernel(*args, **kw):
    """Mirror of dependent_trajectory._dependent_colour_kernel (dim=2, fast irrelevant).
    Returns SX, SY, POP, MREC (MREC = -1 when measurement_basis == 'none').

    Runs with ONE BLAS thread: OpenBLAS multi-threads the (8 x 8)·(8 x 2^(n-3))
    complex products from 2^(n-3) >= 1024 columns on and becomes ~30x slower
    (measured: 3.7 ms vs 0.12 ms per product at delay 5). Bin-level parallelism
    belongs in separate processes, exactly as the C++ engine will thread it."""
    with _tp_limits(limits=1):
        return _kernel(*args, **kw)


def _kernel(aux_amps, aux_phases, gx, gz, omega_s, omega_cont, omega_cont_z, omega_r,
            delay, phase, DeltaT, measurement_basis, seed, interaction='beamsplitter',
            theta_clock=0.0, encode_target='aux', evuType='Pur.Deph', probs_out=None):
    """probs_out (diagnostic only): list that receives the probability of the
    outcome actually drawn at every tick -- a tiny value flags a step where the
    collapse renormalisation divides by ~0 and amplifies round-off."""
    if evuType != 'Pur.Deph':
        raise NotImplementedError("engine blueprint supports evuType='Pur.Deph' only")
    if encode_target not in ('aux', 'system'):
        raise ValueError(encode_target)
    enc_on_sys = (encode_target == 'system')
    n = delay + 2
    U_A, local, spec = build_collision(gx, gz, omega_s, omega_cont, omega_cont_z, omega_r,
                                       delay, phase, DeltaT, interaction, theta_clock)
    vecs = basis_vectors(measurement_basis)
    no_meas = vecs is None
    rng = np.random.RandomState(seed)
    runs = len(aux_amps)
    SXo = np.empty(runs); SYo = np.empty(runs); POP = np.empty(runs); MREC = np.empty(runs, int)

    # register (S, D_0..D_{d-1}) as a (2,)*(n-1) x (2,)*(n-1) tensor; ground + vacuum
    m = n - 1
    rho = np.zeros((2,) * (2 * m), complex)
    rho[(0,) * (2 * m)] = 1.0
    vac = np.array([[1, 0], [0, 0]], complex)
    for t in range(runs):
        if enc_on_sys:
            R = encode_unitary(aux_amps[t], aux_phases[t])
            rho = _apply_local(rho, R, [0], m)
            aux = vac
        else:
            aux = encode_state(aux_amps[t], aux_phases[t])
        # rp = rho (x) aux  as an n-qubit tensor
        rp = np.multiply.outer(rho, aux)                       # axes: rows_m, cols_m, a, a'
        rp = np.moveaxis(rp, 2 * m, m)                          # -> rows_m, a, cols_m, a'
        rp = _apply_local(rp, U_A, local, n)
        rp = _apply_spectator_phases(rp, spec, n)
        trp = float(np.real(np.einsum(rp.reshape(2 ** n, 2 ** n), [0, 0])))
        if trp > 1e-12:
            rp = rp / trp
        # reduced state of index 1 (D_0, or F when delay == 0)
        R6 = rp.reshape(2, 2, 2 ** (n - 2), 2, 2, 2 ** (n - 2))
        rex = np.einsum('saxsbx->ab', R6)
        SXo[t] = float(np.real(rex[0, 1] + rex[1, 0]))
        SYo[t] = float(np.real(1j * rex[0, 1] - 1j * rex[1, 0]))
        POP[t] = float(np.real(rex[1, 1]))
        if no_meas:
            MREC[t] = -1
            rho = np.einsum('saxtay->sxty', R6).reshape((2,) * (2 * m))
            continue
        v0, v1 = vecs
        p0 = max(float(np.real(v0.conj() @ rex @ v0)), 0.0)
        p1 = max(float(np.real(v1.conj() @ rex @ v1)), 0.0)
        s = p0 + p1
        p0, p1 = (0.5, 0.5) if s < 1e-12 else (p0 / s, p1 / s)
        o = rng.choice([0, 1], p=[p0, p1]); MREC[t] = o
        if probs_out is not None:
            probs_out.append(p0 if o == 0 else p1)
        v = v0 if o == 0 else v1
        rc = np.einsum('a,saxtby,b->sxty', v.conj(), R6, v)
        nr = float(np.real(np.einsum('sxsx->', rc)))
        if nr > 1e-12:
            rc = rc / nr
        rho = rc.reshape((2,) * (2 * m))
    return SXo, SYo, POP, MREC


# ----------------------------------------------------------------------------
# the per-bin transform: mirror of dependent_colour_transform_per_bin
# ----------------------------------------------------------------------------
_VERSIONS = ['uncond', 'cond0', 'cond1']


def carrier_clock_rates(in_mag, in_phase, freq_bins, stft_hop, fs, mode=True):
    """theta_k per bin: grid rate 2 pi f_k H / fs ('bin') or the energy-weighted
    phase-vocoder instantaneous rate of the input (True)."""
    th_bin = 2.0 * np.pi * np.asarray(freq_bins, float) * float(stft_hop) / float(fs)
    if mode == 'bin':
        return th_bin
    w = in_mag[1:, :] ** 2
    zc = np.sum(w * np.exp(1j * (np.diff(in_phase, axis=0) - th_bin[None, :])), axis=0)
    return th_bin + np.angle(zc + 1e-30)


def bin_setup(in_mag, in_phase, freq_bins, delay, gamma_x, gamma_z, omega_s_res, fs,
              carrier_clock, stft_hop, amp_pow, freq_scale=False, omega_s=0.0, omega_cont=0.0,
              omega_cont_z=0.0, omega_r=0.0):
    """Everything the per-bin kernel needs, exactly as the transform derives it:
    global scale, companded amplitudes, per-bin (gx, gz, omega_s, theta_clock).
    Returns a dict; per_bin(k) gives (amps, phs, gx, gz, os_, oc, ocz, orr, th)."""
    num_frames, num_bins = in_mag.shape
    scale_f = float(np.max(np.abs(in_mag)))
    if scale_f < 1e-12:
        mag_norm, scale_f = np.zeros_like(in_mag), 1.0
    else:
        mag_norm = in_mag / scale_f
    th_clock_arr = None
    if carrier_clock:
        if not stft_hop:
            raise ValueError('carrier_clock requires stft_hop')
        _fsc = fs if fs else 2.0 * float(np.max(freq_bins))
        th_clock_arr = carrier_clock_rates(in_mag, in_phase, freq_bins, stft_hop, _fsc, carrier_clock)
    _fs = fs if fs else 2.0 * float(np.max(freq_bins))

    def per_bin(k):
        sc = (freq_bins[k] / 100.0) if freq_scale else 1.0
        gx, gz = gamma_x * sc, gamma_z * sc
        oc, ocz, os_, orr = omega_cont * sc, omega_cont_z * sc, omega_s * sc, omega_r * sc
        if omega_s_res:
            os_ = os_ + omega_s_res * (freq_bins[k] / _fs) * gz
        th = float(th_clock_arr[k]) if th_clock_arr is not None else 0.0
        amps = np.concatenate([np.clip(mag_norm[:, k], 0.0, 1.0) ** amp_pow, np.zeros(delay)])
        phs = np.concatenate([in_phase[:, k], np.zeros(delay)])
        return amps, phs, gx, gz, os_, oc, ocz, orr, th
    return dict(scale=scale_f, mag_norm=mag_norm, theta=th_clock_arr, per_bin=per_bin,
                T=num_frames + delay, num_frames=num_frames, num_bins=num_bins)


def trajectories(in_mag, in_phase, freq_bins, bin_indices, delay, phase, traj_range=(0, 1),
                 base_seed=0, DeltaT=1.0, gamma_x=0.1, gamma_z=0.1, omega_cont=0.0,
                 omega_cont_z=0.0, omega_s=0.0, omega_r=0.0, freq_scale=False,
                 interaction='beamsplitter', measurement_basis='x', amp_pow=1.0,
                 omega_s_res=0.0, fs=None, carrier_clock=False, stft_hop=None,
                 encode_target='aux', evuType='Pur.Deph', kernel_impl='numpy', store=None,
                 progress=None):
    """The trajectory STORE: raw kernel outputs for trajectories ti in
    [traj_range[0], traj_range[1]) of every bin in bin_indices, seed
    base_seed + 10007 k + ti. Because each trajectory has its own seed, computing
    trajectories incrementally (append) is exact. Pass an existing `store` to
    append to it (its arrays are extended). kernel_impl 'numpy' (certified
    blueprint) or 'qutip' (the unchanged dependent_trajectory kernel).

    Returns dict(SX, SY, POP: float [nb, ntraj, T]; MREC: int8 [nb, ntraj, T];
    scale, T, num_frames, theta, ntraj, bins)."""
    su = bin_setup(in_mag, in_phase, freq_bins, delay, gamma_x, gamma_z, omega_s_res, fs,
                   carrier_clock, stft_hop, amp_pow, freq_scale, omega_s, omega_cont, omega_cont_z, omega_r)
    T, nb = su['T'], su['num_bins']
    t0, t1 = int(traj_range[0]), int(traj_range[1])
    if measurement_basis == 'none':
        t1 = min(t1, 1)                      # deterministic: one trajectory says everything
    if store is None:
        store = dict(SX=np.zeros((nb, 0, T)), SY=np.zeros((nb, 0, T)), POP=np.zeros((nb, 0, T)),
                     MREC=np.full((nb, 0, T), -1, np.int8), scale=su['scale'], T=T,
                     num_frames=su['num_frames'], theta=su['theta'], ntraj=0, bins=np.asarray(bin_indices))
    assert store['ntraj'] == t0, (store['ntraj'], t0)
    n_new = max(t1 - t0, 0)
    if n_new == 0:
        return store
    SX = np.zeros((nb, n_new, T)); SY = np.zeros((nb, n_new, T)); POP = np.zeros((nb, n_new, T))
    MR = np.full((nb, n_new, T), -1, np.int8)
    if kernel_impl == 'qutip':
        import dependent_trajectory as dt
    for count, k in enumerate(bin_indices):
        amps, phs, gx, gz, os_, oc, ocz, orr, th = su['per_bin'](k)
        for j, ti in enumerate(range(t0, t1)):
            seed = base_seed + 10007 * int(k) + ti
            if kernel_impl == 'qutip':
                sx, sy, pop, mr = dt._dependent_colour_kernel(
                    amps, phs, gx, gz, os_, oc, ocz, orr, delay, phase, DeltaT, measurement_basis,
                    evuType, seed, interaction=interaction, fast=False, theta_clock=th,
                    encode_target=encode_target)
            else:
                sx, sy, pop, mr = kernel(amps, phs, gx, gz, os_, oc, ocz, orr, delay, phase, DeltaT,
                                         measurement_basis, seed, interaction=interaction, theta_clock=th,
                                         encode_target=encode_target, evuType=evuType)
            SX[k, j], SY[k, j], POP[k, j], MR[k, j] = sx, sy, pop, mr
        if progress and not progress(count + 1, len(bin_indices)):
            raise RuntimeError('aborted')
    for key, arr in (('SX', SX), ('SY', SY), ('POP', POP), ('MREC', MR)):
        store[key] = np.concatenate([store[key], arr], axis=1)
    store['ntraj'] = t1
    return store


def render_from_store(store, version, n_use, delay, amp_pow=1.0, in_phase=None, bins=None):
    """Decoded (magnitude, phase) matrices for 'uncond' | 'cond0' | 'cond1' from the
    first n_use trajectories of the store: the transform's averaging rule
    (retrodictive conditioning on the outcome at frame n + delay, fallback to the
    unconditional mean when the class is empty), B16 slicing, inverse companding.
    Magnitudes are returned SCALED (multiplied by the global scale)."""
    T, nf = store['T'], store['num_frames']
    nb = store['SX'].shape[0]
    n_use = max(1, min(int(n_use), store['ntraj']))
    bins = store['bins'] if bins is None else bins
    out_mag = np.zeros((nf, nb)); out_phase = np.zeros((nf, nb)) if in_phase is None else in_phase.copy()
    for k in bins:
        SX, SY, POP, MR = (store[key][k, :n_use] for key in ('SX', 'SY', 'POP', 'MREC'))
        amp = np.empty(T); ph = np.empty(T)
        for t in range(T):
            sel = slice(None)
            if version != 'uncond':
                a = 0 if version == 'cond0' else 1
                cf = t + delay
                if cf < T:
                    mask = (MR[:, cf] == a)
                    if mask.any():
                        sel = mask
            amp[t] = POP[sel, t].mean()
            ph[t] = np.arctan2(SY[sel, t].mean(), SX[sel, t].mean())
        out_mag[:, k] = np.clip(amp[delay:delay + nf], 0.0, 1.0) ** (1.0 / amp_pow) * store['scale']
        out_phase[:, k] = ph[delay:delay + nf]
    return out_mag, out_phase


def twin(in_mag, in_phase, freq_bins, bin_indices, delay, phase, gamma_x=0.1, gamma_z=0.1,
         omega_s_res=0.0, fs=None, carrier_clock=False, stft_hop=None, amp_pow=1.0,
         encode_target='system', DeltaT=1.0):
    """The CLASSICAL TWIN of the channel: the exact one-excitation (linear) model of
    the same delay line -- a per-bin feedback comb with the channel's own taps,
    driven by the coherent amplitude sqrt(a) e^{i phi} of each frame, including the
    carrier clock and the resonator. It is the weak-drive limit of the quantum map;
    what it lacks is qubit saturation and measurement backaction. Same output
    contract as render_from_store (scaled magnitudes, phases). No free parameter."""
    su = bin_setup(in_mag, in_phase, freq_bins, delay, gamma_x, gamma_z, omega_s_res, fs,
                   carrier_clock, stft_hop, amp_pow)
    nf, nb = su['num_frames'], su['num_bins']
    out_mag = np.zeros((nf, nb)); out_phase = in_phase.copy()
    for k in bin_indices:
        amps, phs, gx, gz, os_, oc, ocz, orr, th = su['per_bin'](k)
        pop, ang = twin_bin(amps, phs, gx, gz, os_, th, delay, phase, encode_target, DeltaT)
        out_mag[:, k] = np.clip(pop[delay:delay + nf], 0.0, 1.0) ** (1.0 / amp_pow) * su['scale']
        out_phase[:, k] = ang[delay:delay + nf]
    return out_mag, out_phase


def twin_bin(amps, phs, gx, gz, omega_s, theta, delay, phase, encode_target='system', DeltaT=1.0):
    """One bin of the classical twin: decoded population and phase per tick of the
    linear one-excitation model driven by sqrt(a) e^{i phi}. Site energies are
    RELATIVE TO THE VACUUM branch (the decoded phase is the coherence between
    vacuum and one excitation): sigma_z sigma_z terms, the resonator omega_s
    sigma_z on S, and the clock (theta/2) sum_j sigma_z_j on every qubit."""
    from scipy.linalg import expm
    n = delay + 2; S, D0, F = 0, 1, delay + 1
    fb = -np.exp(-1j * phase); J = 2 * gx
    Hm = np.zeros((n, n), complex)
    Hm[S, F] = Hm[F, S] = J
    Hm[S, D0] = fb * J; Hm[D0, S] = np.conj(fb) * J
    E = np.zeros(n)
    E[S] = -4 * gz - 2 * omega_s - theta
    E[F] = -2 * gz - theta
    E[D0] = -2 * gz - theta
    for j in range(2, delay + 1):
        E[j] = -theta
    U1 = expm(-1j * DeltaT * (Hm + np.diag(E)))
    T = len(amps)
    c = np.zeros(n, complex); pop = np.zeros(T); ang = np.zeros(T)
    for t in range(T):
        if encode_target == 'system':
            c[S] = np.exp(1j * phs[t]) * (c[S] + np.sqrt(amps[t]))   # Rz(phi) Ry(theta) linearised on S
        else:
            c[F] = np.sqrt(amps[t]) * np.exp(1j * phs[t])           # probe prepared in |psi(a, phi)>
        c = U1 @ c
        pop[t] = abs(c[D0]) ** 2; ang[t] = np.angle(c[D0])
        c = np.concatenate([c[:1], c[2:], [0.0]])
    return pop, ang


def pink_field(nf, nb, p, seed=0):
    """1/f^p noise field over frames, standardised per bin (summary notebook, pink_rival)."""
    rng = np.random.default_rng(seed); w = rng.standard_normal((nf, nb))
    f = np.fft.rfftfreq(nf).copy(); f[0] = f[1] if len(f) > 1 else 1.0
    x = np.fft.irfft(np.fft.rfft(w, axis=0) / (f[:, None] ** (p / 2)), n=nf, axis=0)
    return (x - x.mean(0)) / (x.std(0) + 1e-9)


def _cacf(f, W, L):
    out = np.empty(L); out[0] = 1.0
    for l in range(1, L):
        wl = W[l:] * W[:-l]
        out[l] = float((wl * np.cos(f[l:] - f[:-l])).sum() / (wl.sum() + 1e-30))
    return out


def pink_rival(in_mag, in_phase, out_mag, out_phase, seed=0, return_fit=False):
    """The PINK NOISE render (Dependent_trajectory_summary.ipynb, pink_rival): the input's
    own carrier (magnitudes and phases) dressed with 1/f^P noise whose phase depth, memory
    exponent P and AM depth are fitted to the quantum output (out_mag, out_phase). All
    statistics energy-weighted. A classical comparison, not channel output.
    Returns (cl_mag, cl_phase) [, fit dict]."""
    nf, nb = in_mag.shape
    dev_q = np.angle(np.exp(1j * (out_phase - in_phase)))
    W = in_mag ** 2
    depth = float(np.sqrt((W * dev_q ** 2).sum() / (W.sum() + 1e-30)))
    L = min(25, nf - 1)
    target = _cacf(dev_q, W, L)
    grid = np.linspace(0.3, 6.0, 20)
    cost = [float(np.sum((_cacf(depth * pink_field(nf, nb, p, seed + 3), W, L) - target) ** 2)) for p in grid]
    P = float(grid[int(np.argmin(cost))])
    cl_phase = in_phase + depth * pink_field(nf, nb, P, seed + 7)
    en = in_mag > 0.1 * in_mag.max()
    am = float(np.clip(np.std((out_mag / np.maximum(in_mag, 1e-12))[en] - 1.0), 0.05, 1.0)) if en.any() else 0.3
    cl_mag = np.clip(in_mag * (1.0 + am * pink_field(nf, nb, P, seed + 8)), 0.0, None)
    if return_fit:
        return cl_mag, cl_phase, dict(depth=depth, P=P, am=am, acf_residual=float(np.sqrt(min(cost) / L)))
    return cl_mag, cl_phase


def transform(in_mag, in_phase, freq_bins, bin_indices, delay, phase, n_traj=1, base_seed=0,
              DeltaT=1.0, gamma_x=0.1, gamma_z=0.1, omega_cont=0.0, omega_cont_z=0.0,
              omega_s=0.0, omega_r=0.0, freq_scale=False, interaction='beamsplitter',
              measurement_basis='x', amp_pow=1.0, fast=None, omega_s_res=0.0, fs=None,
              carrier_clock=False, stft_hop=None, encode_target='aux', verbose=False,
              amp_mode='power', amp_gain=1.0, evuType='Pur.Deph', versions=None,
              kernel_impl='numpy'):
    """Same contract as dependent_trajectory.dependent_colour_transform_per_bin
    (power companding; B16 latency compensation; retrodictive cond0/cond1),
    implemented as trajectories() + render_from_store(). `versions` lets a caller
    skip cond0/cond1. Returns out_mag as 0-1 tables (unscaled) like the original."""
    if amp_mode != 'power':
        raise NotImplementedError('engine blueprint implements amp_mode="power" only')
    versions = versions or _VERSIONS
    no_meas = (measurement_basis == 'none')
    if no_meas:
        n_traj = 1
    st = trajectories(in_mag, in_phase, freq_bins, bin_indices, delay, phase, (0, n_traj), base_seed,
                      DeltaT, gamma_x, gamma_z, omega_cont, omega_cont_z, omega_s, omega_r, freq_scale,
                      interaction, measurement_basis, amp_pow, omega_s_res, fs, carrier_clock, stft_hop,
                      encode_target, evuType, kernel_impl)
    num_frames, num_bins = in_mag.shape
    scale_per_bin = np.full(num_bins, st['scale'])
    out_mag = {v: np.clip(st['mag_norm'] if 'mag_norm' in st else in_mag / st['scale'], 0.0, 1.0) for v in versions}
    out_phase = {}
    for v in versions:
        m, p = render_from_store(st, 'uncond' if no_meas else v, n_traj, delay, amp_pow, in_phase, bin_indices)
        for k in bin_indices:
            out_mag[v][:, k] = m[:, k] / st['scale']
        out_phase[v] = p
    diag = {}
    T = st['T']
    for k in bin_indices:
        MR = st['MREC'][k, :n_traj]
        both = np.mean([(MR[:, t] == 0).any() and (MR[:, t] == 1).any() for t in range(T)])
        n_cond = 0; n_fb = 0
        if not no_meas:
            for v in versions:
                if v == 'uncond':
                    continue
                a = 0 if v == 'cond0' else 1
                for t in range(T - delay):
                    n_cond += 1; n_fb += int(not (MR[:, t + delay] == a).any())
        diag[k] = dict(fallback_frac=n_fb / max(n_cond, 1), both_outcomes_frac=float(both))
    return out_mag, out_phase, scale_per_bin, diag
