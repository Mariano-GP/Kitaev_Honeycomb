#!/usr/bin/env python3
"""
CAMPS-only calculation for the 2D Kitaev honeycomb model on a hexagonal lattice.

Hamiltonian conventions
-----------------------
This script represents the Hamiltonian as Pauli strings on a 1D snake/row-major MPS ordering.

By default, --normalization spin is used, i.e.

    H = J sum_<ij> S_i . S_j
      = (J/4) sum_<ij> (X_i X_j + Y_i Y_j + Z_i Z_j),

because the reference quoted in the prompt, E0/N = -0.603522 for 6x6, is the usual
spin-1/2 Heisenberg normalization. If you literally want Pauli operators with no 1/4,
use --normalization pauli; then the comparable reference energy is four times larger.

Example
-------
Run this code as: 

python CAMPS_2d_Kitaev_Honeycomb_v3.py \
  --J 1.0 \
  --normalization pauli \
  --Dlist_camps 60 \
  --n_camps_iters 3 \
  --n_cliff_sweeps 2 \
  --n_cliff_use 720 \
  --outdir heis_6x6_camps
"""

import os
import time
import argparse
import warnings
from collections import defaultdict

import numpy as np
warnings.filterwarnings("ignore")

try:
    from scipy.linalg import svdvals as _scipy_svdvals
except Exception:
    _scipy_svdvals = None

from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.linalg import np_conserved as npc


_PAULI_OP_NAME = {0: "Id", 1: "Sigmax", 2: "Sigmay", 3: "Sigmaz"}
_PAULI_MATS = {
    0: np.eye(2, dtype=complex),
    1: np.array([[0, 1], [1, 0]], dtype=complex),
    2: np.array([[0, -1j], [1j, 0]], dtype=complex),
    3: np.array([[1, 0], [0, -1]], dtype=complex),
}


N_SITES = 22

Z_BONDS = [(1,18),(3,4),(5,6),(7,8),(10,22),(12,21),(14,15),(16,17),(19,20)]
X_BONDS = [(1,2),(4,19),(6,22),(8,9),(10,11),(12,13),(15,16),(17,18),(20,21)]
Y_BONDS = [(2,3),(4,5),(6,7),(9,10),(11,12),(13,14),(16,21),(18,19),(20,22)]

# N_SITES = 13

# Z_BONDS = [(3,4),(5,6),(10,11),(1,12),(13,8)]
# X_BONDS = [(1,2),(13,4),(7,6),(8,9),(11,12)]
# Y_BONDS = [(2,3),(4,5),(7,8),(9,10),(12,13)]


def Kitaev_Honeycomb_Pauli_terms(K=-1.0, Kx=None, Ky=None, Kz=None, V=-0.3, h=0.1, normalization="spin"):
    """Return [(coeff, {site: pauli_label})] for the fixed N_SITES-site Kitaev honeycomb cluster.

    H = H_K + H_V, where:
      H_K = sum_bonds K^alpha sigma_i^alpha sigma_j^alpha        (pure Kitaev bonds,
                                                                   K^alpha = Kx/Ky/Kz)
      H_V = V * sum_{adjacent bond pairs} (three-body term)
          + h * sum_{boundary sites} sigma_j^alpha                (alpha = that site's
                                                                    MISSING bond direction)

    Bonds are the ones given by X_BONDS/Y_BONDS/Z_BONDS (1-indexed site labels,
    converted here to 0-indexed).

    Three-body term convention (Kitaev's original perturbative magnetic-field
    expansion): at a vertex j connected to neighbor i via an alpha-bond and to
    neighbor k via a beta-bond (alpha != beta), the term is
    sigma_i^alpha sigma_j^gamma sigma_k^beta, where gamma is the THIRD Pauli
    type (!= alpha, != beta). A bulk (degree-3) site contributes all three such
    pairs (xy, xz, yz); a boundary (degree-2) site contributes exactly one.

    Boundary field: h * sigma_j^alpha for every degree-2 site j, where alpha is
    whichever bond direction j is missing ENTIRELY (not one of the two bonds
    forming the three-body term above). This is the term that lifts unphysical
    ground-state degeneracies associated with "dangling" Majorana fermions at
    open boundaries (Kitaev, Annals of Physics 321, 2 (2006); see also the
    discussion in arXiv:2507.08939 Eq. 3).

    Pauli labels: X=1, Y=2, Z=3. For normalization='spin', an n-site term's
    coefficient is divided by 2^n (matching S=sigma/2 for each site in the
    term); for normalization='pauli', coefficients are used as given.
    """
    if normalization not in ("spin", "pauli"):
        raise ValueError("normalization must be spin or pauli")
    Kx = K if Kx is None else Kx
    Ky = K if Ky is None else Ky
    Kz = K if Kz is None else Kz

    def scale(val, n_sites):
        return float(val) / (2.0 ** n_sites) if normalization == "spin" else float(val)

    terms = []

    def add_bonds(bonds, coeff, pauli):
        for i, j in bonds:
            a, b = sorted((i - 1, j - 1))
            terms.append((coeff, {a: pauli, b: pauli}))

    add_bonds(X_BONDS, scale(Kx, 2), 1)
    add_bonds(Y_BONDS, scale(Ky, 2), 2)
    add_bonds(Z_BONDS, scale(Kz, 2), 3)

    # --- H_V: three-body effective-field terms + boundary field ---------------
    TYPE_LABEL = {"X": 1, "Y": 2, "Z": 3}
    ALL_TYPES = ("X", "Y", "Z")

    # bonds_at[site][type] = neighbor, for whichever bond directions actually
    # touch that site (0-indexed sites, matching the rest of this function).
    #This function is used to find the three-body terms and the boundary field terms.
    bonds_at = [dict() for _ in range(N_SITES)]
    for type_name, bonds in (("X", X_BONDS), ("Y", Y_BONDS), ("Z", Z_BONDS)):
        for i, j in bonds:
            a, b = i - 1, j - 1
            bonds_at[a][type_name] = b
            bonds_at[b][type_name] = a

    v_coeff = scale(V, 3)
    h_coeff = scale(h, 1)
    for j in range(N_SITES):
        present = [t for t in ALL_TYPES if t in bonds_at[j]]

        for a_idx in range(len(present)):
            for b_idx in range(a_idx + 1, len(present)):
                alpha, beta = present[a_idx], present[b_idx]
                gamma = (set(ALL_TYPES) - {alpha, beta}).pop()
                i_site = bonds_at[j][alpha]
                k_site = bonds_at[j][beta]
                terms.append((v_coeff, {i_site: TYPE_LABEL[alpha],
                                         j: TYPE_LABEL[gamma],
                                         k_site: TYPE_LABEL[beta]}))

        if len(present) == 2:
            missing = (set(ALL_TYPES) - set(present)).pop()
            terms.append((h_coeff, {j: TYPE_LABEL[missing]}))

    return combine_pauli_terms(terms)


class QubitPauliTermsModel(CouplingMPOModel):
    """Generic qubit-chain MPO from Pauli-string terms."""
    default_lattice = "Chain"
    force_default_lattice = True

    def init_sites(self, model_params):
        return SpinHalfSite(conserve=None)

    def init_lattice(self, model_params):
        N = model_params.get("N", None)
        if N is None:
            raise ValueError("Need model parameter N.")
        site = self.init_sites(model_params)
        return Chain(N, site, bc="open", bc_MPS="finite")

    def init_terms(self, model_params):
        terms = model_params.get("pauli_terms", [])
        drop_tol = model_params.get("drop_tol", 1e-13)
        for coeff, site_ops in terms:
            if abs(coeff) < drop_tol:
                continue
            sites = sorted(site_ops.keys())
            if len(sites) == 0:
                self.add_onsite_term(coeff, 0, "Id")
            elif len(sites) == 1:
                s = sites[0]
                self.add_onsite_term(coeff, s, _PAULI_OP_NAME[site_ops[s]])
            elif len(sites) == 2:
                s0, s1 = sites
                self.add_coupling_term(coeff, s0, s1,
                                       _PAULI_OP_NAME[site_ops[s0]],
                                       _PAULI_OP_NAME[site_ops[s1]])
            else:
                ops = [_PAULI_OP_NAME[site_ops[s]] for s in sites]
                self.add_multi_coupling_term(coeff, sites, ops, "Id")


def build_tenpy_model_from_terms(n_qubits, pauli_terms):
    return QubitPauliTermsModel({"N": n_qubits, "pauli_terms": pauli_terms,
                                 "bc_MPS": "finite", "conserve": None})


def combine_pauli_terms(terms, drop_tol=1e-13):
    accum = defaultdict(complex)
    for coeff, ops in terms:
        key = tuple(sorted((int(s), int(p)) for s, p in ops.items() if p != 0))
        accum[key] += complex(coeff)
    out = []
    for key, coeff in accum.items():
        if abs(coeff) > drop_tol:
            coeff = np.real_if_close(coeff, tol=1000)
            if np.ndim(coeff) == 0:
                coeff = coeff.item()
            out.append((coeff, dict(key)))
    return out


def product_state_pattern(N, pattern="neel"):
    if pattern == "up":
        return ["up"] * N
    if pattern == "down":
        return ["down"] * N
    return ["up" if i % 2 == 0 else "down" for i in range(N)]


def run_dmrg(model, chi_max, psi_init=None, max_sweeps=30, min_sweeps=4,
             mixer=True, verbose=0, init_state="neel"):
    N = model.lat.N_sites
    if psi_init is None:
        psi = MPS.from_product_state(model.lat.mps_sites(), product_state_pattern(N, init_state), bc="finite")
    else:
        psi = psi_init.copy()

    dmrg_params = {
        "trunc_params": {"chi_max": int(chi_max), "svd_min": 1e-14},
        "max_sweeps": int(max_sweeps),
        "min_sweeps": int(min_sweeps),
        "mixer": mixer,
        "mixer_params": {"amplitude": 1e-5, "decay": 1.2, "disable_after": 15},
        "max_trunc_err": 1.0,
        "max_E_err": 1e-10,
    }
    if verbose < 2:
        dmrg_params["verbose"] = 0
    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, _ = eng.run()
    return float(np.real(E)), psi


def pauli_2q_matrix(p1, p2):
    return np.kron(_PAULI_MATS[p1], _PAULI_MATS[p2])


def enumerate_cliffords_2q_fast(verbose=False, quotient_phases=True):
    """Enumerate two-qubit Clifford gates by BFS.

    Parameters
    ----------
    quotient_phases:
        True  -> keep one representative for each unsigned symplectic action
                 on Pauli labels.  This gives 720 gates, matching the CAMPS
                 paper: the extra 2^4 sign/phase choices do not change the
                 two-site Schmidt singular values and are redundant for the
                 truncation-loss minimization.
        False -> keep distinct signed Pauli-generator actions, i.e. the usual
                 two-qubit Clifford group modulo global phase.  This gives
                 11520 gates.

    Returns
    -------
    np.ndarray with shape (720,4,4) by default, or (11520,4,4) if
    quotient_phases=False.
    """
    I2 = np.eye(2, dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                     [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    gen_mats = [np.kron(H, I2), np.kron(I2, H), np.kron(S, I2), np.kron(I2, S), CNOT]
    test_paulis = [pauli_2q_matrix(1, 0), pauli_2q_matrix(3, 0),
                   pauli_2q_matrix(0, 1), pauli_2q_matrix(0, 3)]

    def signed_key(U):
        """Images of X1,Z1,X2,Z2 as signed Paulis: (phase,p1,p2)."""
        key = []
        for P in test_paulis:
            img = U.conj().T @ P @ U
            found = None
            for p1 in range(4):
                for p2 in range(4):
                    M = pauli_2q_matrix(p1, p2)
                    for ph in range(4):
                        if np.allclose(img, (1j ** ph) * M, atol=1e-10):
                            found = (ph, p1, p2)
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is None:
                return None
            key.append(found)
        return tuple(key)

    def unsigned_key_from_signed(key):
        """Drop the phase/sign part of each generator image.

        This quotients out the 2^4 redundant Pauli/sign choices.  For CAMPS
        SVD minimization, Clifford gates that differ only by these signs have
        identical two-site singular values, so one representative is enough.
        """
        return tuple((p1, p2) for (_ph, p1, p2) in key)

    I4 = np.eye(4, dtype=complex)
    k0 = signed_key(I4)
    if k0 is None:
        raise RuntimeError("identity was not recognized as a Clifford")

    # We must BFS over the full signed Clifford graph to be able to reach all
    # unsigned symplectic classes.  However, we store only the first matrix for
    # each unsigned class when quotient_phases=True.
    seen_signed = {k0}
    seen_store = {unsigned_key_from_signed(k0) if quotient_phases else k0}
    queue = [I4.copy()]
    all_unitaries = [I4.copy()]

    while queue:
        U = queue.pop(0)
        for G in gen_mats:
            U_new = G @ U
            k_signed = signed_key(U_new)
            if k_signed is None:
                continue
            if k_signed not in seen_signed:
                seen_signed.add(k_signed)
                queue.append(U_new.copy())

                k_store = unsigned_key_from_signed(k_signed) if quotient_phases else k_signed
                if k_store not in seen_store:
                    seen_store.add(k_store)
                    all_unitaries.append(U_new.copy())

    if verbose:
        if quotient_phases:
            print(f"Enumerated {len(all_unitaries)} two-qubit Clifford representatives modulo Pauli/sign phases")
            print(f"  Full signed Clifford actions visited: {len(seen_signed)}")
        else:
            print(f"Enumerated {len(all_unitaries)} two-qubit Clifford gates modulo global phase")

    expected = 720 if quotient_phases else 11520
    if len(all_unitaries) != expected:
        raise RuntimeError(f"Expected {expected} Clifford representatives, got {len(all_unitaries)}")
    return np.array(all_unitaries)

def load_or_build_cliffords(outdir, n_use=720, seed=42, verbose=1, quotient_phases=True):
    """Load/build the two-qubit Clifford search set.

    By default this uses 720 representatives, i.e. one representative for each
    element of Sp(4,F_2).  This is the search set used in the CAMPS paper for
    SVD/truncation-loss minimization.  The omitted 2^4 signed variants give the
    same two-site singular values and are therefore redundant.
    """
    os.makedirs(outdir, exist_ok=True)
    cache_name = "clifford_gates_2q_720_symplectic_reps.npy" if quotient_phases else "clifford_gates_2q_11520_full.npy"
    cache = os.path.join(outdir, cache_name)
    expected = 720 if quotient_phases else 11520

    if os.path.exists(cache):
        U = np.load(cache)
        if len(U) != expected:
            if verbose:
                print(f"Ignoring stale Clifford cache {cache}: found {len(U)}, expected {expected}")
            U = None
        elif verbose:
            print(f"Loaded {len(U)} Clifford gates from {cache}")
    else:
        U = None

    if U is None:
        t0 = time.time()
        U = enumerate_cliffords_2q_fast(verbose=verbose >= 2, quotient_phases=quotient_phases)
        np.save(cache, U)
        if verbose:
            print(f"Built {len(U)} Clifford gates in {time.time()-t0:.1f}s; cached to {cache}")

    n_use = min(int(n_use), len(U))
    if n_use < len(U):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([[0], rng.choice(len(U), size=n_use - 1, replace=False)])
        U = U[idx]
        if verbose:
            print(f"Using a reproducible subset of {len(U)} Clifford gates")
    else:
        if verbose:
            print(f"Using all {len(U)} Clifford representatives")
    return U

def _svdvals_fast(mat):
    """Singular values with scipy fast path when available."""
    if _scipy_svdvals is not None:
        return _scipy_svdvals(mat, check_finite=False)
    return np.linalg.svd(mat, compute_uv=False)


def _entropy_from_svals(s):
    s = np.abs(np.asarray(s))
    s = s[s > 1e-16]
    if len(s) == 0:
        return 0.0
    p = s ** 2
    z = p.sum()
    if z <= 0:
        return 0.0
    p /= z
    return float(-np.sum(p * np.log(p + 1e-30)))


def compute_bond_entropy_from_theta(theta):
    arr = theta.to_ndarray() if hasattr(theta, "to_ndarray") else np.asarray(theta)
    DL, d1, d2, DR = arr.shape
    mat = arr.reshape(DL * d1, d2 * DR)
    return _entropy_from_svals(_svdvals_fast(mat))


def _truncate_theta_for_eval(theta_np, max_eval_chi=None):
    """Cheap approximate theta used only to rank Clifford candidates.

    If the MPS is near canonical, the virtual indices are approximately ordered by
    Schmidt weight.  Keeping the leading virtual slices gives a much smaller SVD
    problem while preserving the dominant entanglement structure.  The final
    accepted gate is still checked on the full theta, so this is a speed/quality
    knob rather than a physics approximation to the final state.
    """
    if max_eval_chi is None or max_eval_chi <= 0:
        return theta_np
    m = int(max_eval_chi)
    DL, d1, d2, DR = theta_np.shape
    return theta_np[:min(DL, m), :, :, :min(DR, m)]


def _entropy_after_gate_numpy(theta_np, U):
    DL, d1, d2, DR = theta_np.shape
    flat = theta_np.reshape(DL, d1 * d2, DR)
    new = np.einsum("ij,ajb->aib", U, flat, optimize=True).reshape(DL, d1, d2, DR)
    mat = new.reshape(DL * d1, d2 * DR)
    return _entropy_from_svals(_svdvals_fast(mat))


def all_bond_entropies(psi):
    if psi.L <= 1:
        return np.array([], dtype=float)
    return np.array([compute_bond_entropy_from_theta(psi.get_theta(i, 2))
                     for i in range(psi.L - 1)], dtype=float)


def entropy_summary(psi):
    vals = all_bond_entropies(psi)
    if len(vals) == 0:
        return {"EE_mid": 0.0, "EE_max": 0.0, "EE_sum": 0.0, "EE_mean": 0.0}
    mid = psi.L // 2 - 1
    mid = min(max(mid, 0), len(vals) - 1)
    return {"EE_mid": float(vals[mid]), "EE_max": float(np.max(vals)),
            "EE_sum": float(np.sum(vals)), "EE_mean": float(np.mean(vals))}


def clifford_entropy_diagnostics(before, after, prefix="clifford"):
    out = {}
    for key in ["EE_mid", "EE_max", "EE_sum", "EE_mean"]:
        b = float(before.get(key, np.nan))
        a = float(after.get(key, np.nan))
        rem = b - a if np.isfinite(b) and np.isfinite(a) else np.nan
        frac = rem / b if np.isfinite(rem) and abs(b) > 1e-14 else 0.0
        out[f"{prefix}_{key}_before"] = b
        out[f"{prefix}_{key}_after"] = a
        out[f"{prefix}_{key}_removed"] = rem
        out[f"{prefix}_{key}_frac"] = frac
    return out


def apply_twosite_gate(psi, i, U, chi_max=None, canonicalize=True):
    theta = psi.get_theta(i, 2)
    theta_np = theta.to_ndarray()
    DL, d1, d2, DR = theta_np.shape
    flat = theta_np.reshape(DL, d1 * d2, DR)
    new = np.einsum("ij,ajb->aib", U, flat, optimize=True).reshape(DL, d1, d2, DR)
    mat = new.reshape(DL * d1, d2 * DR)
    A, S, Vh = np.linalg.svd(mat, full_matrices=False)
    if chi_max is not None and len(S) > chi_max:
        A = A[:, :chi_max]
        S = S[:chi_max]
        Vh = Vh[:chi_max, :]
    norm = np.linalg.norm(S)
    if norm > 0:
        S = S / norm
    Dnew = len(S)
    A_i = A.reshape(DL, d1, Dnew)
    B_j = (np.diag(S) @ Vh).reshape(Dnew, d2, DR)
    psi._B[i] = npc.Array.from_ndarray_trivial(A_i, labels=["vL", "p", "vR"], dtype=complex)
    psi._B[i + 1] = npc.Array.from_ndarray_trivial(B_j, labels=["vL", "p", "vR"], dtype=complex)
    if canonicalize:
        psi.canonical_form()
    return S


def disentangle_sweep(psi, clifford_unitaries, chi_max=None, verbose=0,
                      entropy_tol=1e-12, canonicalize_at_end=True,
                      safe_canonicalize=True, max_eval_chi=64,
                      exact_top_k=8, canonicalize_every=4):
    """One Clifford disentangling sweep with a faster two-stage search.

    Speed knobs
    -----------
    max_eval_chi:
        Use a truncated theta only to rank Clifford candidates.  The accepted
        candidate is checked on the full theta before applying it.  Set 0 or a
        negative value to disable this approximation.
    exact_top_k:
        Number of best approximate candidates to recheck with the full theta.
        Larger is safer; 4--16 is usually enough.
    safe_canonicalize:
        If True, repair TenPy canonical data regularly.  Instead of the older
        very slow behavior of canonicalizing after every accepted gate, this
        version canonicalizes every `canonicalize_every` accepted gates and
        retries immediately if TenPy reports stale Schmidt arrays.
    """
    N = psi.L
    circuit = []
    total_reduction = 0.0
    U_stack = np.asarray(clifford_unitaries)
    n_gates = len(U_stack)
    exact_top_k = max(1, min(int(exact_top_k), n_gates))
    accepted_since_canon = 0

    if safe_canonicalize:
        psi.canonical_form()

    for i in range(N - 1):
        try:
            theta = psi.get_theta(i, 2).to_ndarray()
        except ValueError:
            psi.canonical_form()
            accepted_since_canon = 0
            theta = psi.get_theta(i, 2).to_ndarray()

        current_S = compute_bond_entropy_from_theta(theta)

        # Stage 1: cheap approximate ranking on a truncated theta.
        theta_eval = _truncate_theta_for_eval(theta, max_eval_chi=max_eval_chi)
        DL, d1, d2, DR = theta_eval.shape
        flat = theta_eval.reshape(DL, d1 * d2, DR)
        theta_all = np.einsum("gij,ajb->gaib", U_stack, flat, optimize=True)
        mats = theta_all.reshape(n_gates, DL * d1, d2 * DR)
        approx_S = np.empty(n_gates, dtype=float)
        for g in range(n_gates):
            approx_S[g] = _entropy_from_svals(_svdvals_fast(mats[g]))

        # Only recheck the top few candidates exactly on the full theta.
        cand = np.argpartition(approx_S, exact_top_k - 1)[:exact_top_k]
        best_S = current_S
        best_idx = -1
        for g in cand:
            Sg = _entropy_after_gate_numpy(theta, U_stack[g])
            if Sg < best_S - entropy_tol:
                best_S = float(Sg)
                best_idx = int(g)

        if best_idx >= 0:
            # Avoid canonical_form after every accepted gate when possible.  If
            # TenPy's cached Schmidt arrays become stale, the next get_theta()
            # triggers the fallback canonicalization above.
            do_canon = bool(safe_canonicalize and canonicalize_every > 0 and
                            (accepted_since_canon + 1) >= canonicalize_every)
            apply_twosite_gate(psi, i, U_stack[best_idx], chi_max=chi_max,
                               canonicalize=do_canon)
            accepted_since_canon = 0 if do_canon else accepted_since_canon + 1
            circuit.append((i, best_idx))
            total_reduction += current_S - best_S

    if circuit and canonicalize_at_end:
        psi.canonical_form()
    if verbose:
        print(f"    Clifford sweep: {len(circuit)} gates, entropy reduction={total_reduction:.6f}")
    return circuit, total_reduction

def _precompute_pauli_transfer(U):
    table = {}
    for p1 in range(4):
        for p2 in range(4):
            P_in = pauli_2q_matrix(p1, p2)
            P_out = U @ P_in @ U.conj().T
            found = False
            for q1 in range(4):
                for q2 in range(4):
                    M = pauli_2q_matrix(q1, q2)
                    for ph in range(4):
                        if np.allclose(P_out, (1j ** ph) * M, atol=1e-10):
                            table[(p1, p2)] = (ph, q1, q2)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                raise ValueError(f"UPU† not Pauli for input {(p1, p2)}")
    return table


def _conjugate_terms_by_gate(terms, site_a, ptable):
    site_b = site_a + 1
    out = []
    for coeff, site_ops in terms:
        pa = site_ops.get(site_a, 0)
        pb = site_ops.get(site_b, 0)
        if pa == 0 and pb == 0:
            out.append((coeff, site_ops))
            continue
        ph, pa2, pb2 = ptable[(pa, pb)]
        new_ops = {s: p for s, p in site_ops.items() if s != site_a and s != site_b}
        if pa2:
            new_ops[site_a] = pa2
        if pb2:
            new_ops[site_b] = pb2
        c = coeff * (1j ** ph)
        c = np.real_if_close(c, tol=1000)
        if np.ndim(c) == 0:
            c = c.item()
        out.append((c, new_ops))
    return out


def transform_hamiltonian(pauli_terms, circuit, clifford_unitaries, combine=True, transfer_cache=None):
    terms = [(c, dict(ops)) for c, ops in pauli_terms]
    if transfer_cache is None:
        transfer_cache = {}
    for site_i, gate_idx in circuit:
        if gate_idx not in transfer_cache:
            transfer_cache[gate_idx] = _precompute_pauli_transfer(clifford_unitaries[gate_idx])
        terms = _conjugate_terms_by_gate(terms, site_i, transfer_cache[gate_idx])
    if combine:
        terms = combine_pauli_terms(terms)
    return terms


def run_camps(n_qubits, chi_max, pauli_terms_orig, clifford_unitaries,
              n_cliff_sweeps=2, n_camps_iters=3, max_dmrg_sweeps=30,
              init_state="neel", verbose=1, entropy_tol=1e-12,
              max_eval_chi=64, exact_top_k=8, canonicalize_every=4):
    if verbose:
        print(f"  CAMPS D={chi_max}: initial DMRG in current Clifford frame")
    model0 = build_tenpy_model_from_terms(n_qubits, pauli_terms_orig)
    E, psi = run_dmrg(model0, chi_max, max_sweeps=max_dmrg_sweeps,
                      init_state=init_state, verbose=verbose)
    current_terms = pauli_terms_orig
    circuits = []
    transfer_cache = {}
    clifford_stats = {
        "clifford_total_EE_reduction": 0.0,
        "clifford_total_gates": 0,
        # Diagnostics for the last CAMPS Clifford block, measured
        # immediately before and immediately after the Clifford sweeps,
        # before the following DMRG optimization changes the entropy again.
        "clifford_mid_before_last": 0.0,
        "clifford_mid_after_last": 0.0,
        "clifford_mid_removed_last": 0.0,
        "clifford_mid_frac_last": 0.0,
        "clifford_max_before_last": 0.0,
        "clifford_max_after_last": 0.0,
        "clifford_sum_before_last": 0.0,
        "clifford_sum_after_last": 0.0,
        "clifford_sum_removed_last": 0.0,
        "clifford_sum_frac_last": 0.0,
        "clifford_mean_before_last": 0.0,
        "clifford_mean_after_last": 0.0,
    }

    for it in range(n_camps_iters):
        if verbose:
            print(f"    CAMPS iteration {it+1}/{n_camps_iters}: E={E:.12f}")
        iter_circuit = []
        before_cliff = entropy_summary(psi)
        sweep_reduction = 0.0
        for _ in range(n_cliff_sweeps):
            circ, red = disentangle_sweep(psi, clifford_unitaries, chi_max=chi_max,
                                          verbose=verbose, entropy_tol=entropy_tol,
                                          safe_canonicalize=True,
                                          max_eval_chi=max_eval_chi,
                                          exact_top_k=exact_top_k,
                                          canonicalize_every=canonicalize_every)
            iter_circuit.extend(circ)
            sweep_reduction += red
        after_cliff = entropy_summary(psi)
        diag = clifford_entropy_diagnostics(before_cliff, after_cliff)
        clifford_stats["clifford_total_EE_reduction"] += float(sweep_reduction)
        clifford_stats["clifford_total_gates"] += len(iter_circuit)
        clifford_stats["clifford_mid_before_last"] = diag["clifford_EE_mid_before"]
        clifford_stats["clifford_mid_after_last"] = diag["clifford_EE_mid_after"]
        clifford_stats["clifford_mid_removed_last"] = diag["clifford_EE_mid_removed"]
        clifford_stats["clifford_mid_frac_last"] = diag["clifford_EE_mid_frac"]
        clifford_stats["clifford_max_before_last"] = diag["clifford_EE_max_before"]
        clifford_stats["clifford_max_after_last"] = diag["clifford_EE_max_after"]
        clifford_stats["clifford_sum_before_last"] = diag["clifford_EE_sum_before"]
        clifford_stats["clifford_sum_after_last"] = diag["clifford_EE_sum_after"]
        clifford_stats["clifford_sum_removed_last"] = diag["clifford_EE_sum_removed"]
        clifford_stats["clifford_sum_frac_last"] = diag["clifford_EE_sum_frac"]
        clifford_stats["clifford_mean_before_last"] = diag["clifford_EE_mean_before"]
        clifford_stats["clifford_mean_after_last"] = diag["clifford_EE_mean_after"]
        if verbose:
            print(
                "    Clifford EE_mid: "
                f"before={clifford_stats['clifford_mid_before_last']:.6f}, "
                f"after={clifford_stats['clifford_mid_after_last']:.6f}, "
                f"removed={clifford_stats['clifford_mid_removed_last']:.6f} "
                f"({100.0 * clifford_stats['clifford_mid_frac_last']:.2f}%)"
            )
        if not iter_circuit:
            if verbose:
                print("    No accepted Clifford gates; stopping CAMPS loop")
            break
        circuits.append(iter_circuit)
        current_terms = transform_hamiltonian(current_terms, iter_circuit, clifford_unitaries,
                                              combine=True, transfer_cache=transfer_cache)
        if verbose:
            max_k = max((len(ops) for _, ops in current_terms), default=0)
            print(f"    transformed H: {len(current_terms)} Pauli terms, max locality={max_k}")
        model_new = build_tenpy_model_from_terms(n_qubits, current_terms)
        E, psi = run_dmrg(model_new, chi_max, psi_init=psi,
                          max_sweeps=max_dmrg_sweeps, init_state=init_state, verbose=verbose)
    return float(E), psi, circuits, clifford_stats


def export_camps_state_for_julia(psi, terms, outdir, tag=f"N={N_SITES}_camps_state"):
    """Export an MPS ground state and its Hamiltonian for use in Julia/ITensors.

    `psi` and `terms` must be in the *same* frame (e.g. pass the final `psi`
    and `current_terms` returned by `run_camps`, which are both expressed in
    the Clifford-disentangled CAMPS frame, not the original physical frame).

    Writes two files:
      {tag}_mps.npz   -- one dense array per site, key "B{i}" (0-indexed),
                         shape (chi_left, d, chi_right), TeNPy's right-
                         canonical ('B') form transposed to (vL, p, vR).
                         Because this is a finite MPS with open boundaries
                         (bond dimension 1 at both ends), sequentially
                         contracting B_0 B_1 ... B_{L-1} on the shared bond
                         legs reproduces the state exactly -- no singular
                         values need to be reinserted.
      {tag}_terms.csv -- one Hamiltonian term per line:
                         coeff_re,coeff_im,site,pauli,site,pauli,...
                         site is 0-indexed (as in `terms`); pauli labels are
                         X=1, Y=2, Z=3 (matches _PAULI_OP_NAME).
    """
    os.makedirs(outdir, exist_ok=True)
    L = psi.L

    tensors = {}
    for i in range(L):
        B = psi.get_B(i, form='B')
        B = B.itranspose(['vL', 'p', 'vR'])
        tensors[f"B{i}"] = B.to_ndarray()
    mps_path = os.path.join(outdir, f"{tag}_mps.npz")
    np.savez(mps_path, **tensors)

    terms_path = os.path.join(outdir, f"{tag}_terms.csv")
    with open(terms_path, "w") as f:
        for coeff, site_ops in terms:
            c = complex(coeff)
            row = [repr(c.real), repr(c.imag)]
            for site in sorted(site_ops):
                row.append(str(site))
                row.append(str(site_ops[site]))
            f.write(",".join(row) + "\n")

    print(f"Saved {L}-site MPS to {mps_path}")
    print(f"Saved {len(terms)} Hamiltonian terms to {terms_path}")
    return mps_path, terms_path


def export_rotated_single_site_ops(N, all_gates, clifford_unitaries, outdir, tag):
    """Export, for every site and every Pauli type, the single-site operator
    conjugated by the SAME Clifford circuit used to rotate the Hamiltonian
    (transform_hamiltonian), so a Julia-side correlator can build its W(0)/V_j
    operators in the same rotated frame as the exported Hamiltonian and MPS
    ground state.

    Using a plain, un-rotated Pauli in the disentangled frame measures a
    DIFFERENT physical quantity than the original operator: if H' = C H C^dag
    and |phi> = C|psi>, then <phi|Z_i|phi> = <psi|(C^dag Z_i C)|psi>, not
    <psi|Z_i|psi>, unless Z_i is also rotated forward the same way (Z_i' =
    C Z_i C^dag) before being used in the disentangled frame.

    Writes one file:
      {tag}_rotated_ops.csv -- one Pauli term per line:
                         origin_site,origin_pauli,coeff_re,coeff_im,site,pauli,site,pauli,...
                         origin_site/site are 0-indexed; pauli labels are
                         X=1, Y=2, Z=3 (matches _PAULI_OP_NAME). origin_site/
                         origin_pauli identify which single-site operator
                         (before rotation) this term belongs to -- there are
                         generally several term lines per (origin_site,
                         origin_pauli) pair, since rotation can spread a
                         single-site operator across multiple sites.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{tag}_rotated_ops.csv")
    transfer_cache = {}
    n_terms = 0
    with open(path, "w") as f:
        for site in range(N):
            for label in (1, 2, 3):  # X, Y, Z
                seed = [(1.0, {site: label})]
                rotated = transform_hamiltonian(seed, all_gates, clifford_unitaries,
                                                 combine=True, transfer_cache=transfer_cache)
                for coeff, site_ops in rotated:
                    c = complex(coeff)
                    row = [str(site), str(label), repr(c.real), repr(c.imag)]
                    for s in sorted(site_ops):
                        row.append(str(s))
                        row.append(str(site_ops[s]))
                    f.write(",".join(row) + "\n")
                    n_terms += 1
    print(f"Saved {n_terms} rotated single-site-operator terms ({N} sites x 3 Pauli types) to {path}")
    return path


def run_camps_and_export_for_julia(n_qubits, chi_max, pauli_terms_orig, clifford_unitaries,
                                    outdir, tag=f"N={N_SITES}", **run_camps_kwargs):
    """Run CAMPS, apply the Hamiltonian rotation, and export for Julia.

    "Apply the Hamiltonian rotation" means: `psi` comes back from `run_camps`
    expressed in the Clifford-disentangled frame (each accepted Clifford gate
    changed the MPS), so the *original* Hamiltonian no longer matches it. To
    keep working in that frame (rather than un-rotating `psi`), the original
    Hamiltonian must be conjugated by the exact same sequence of gates. This
    conjugation is composable -- doing it once with every gate from every
    CAMPS iteration concatenated in order gives the identical result as the
    per-iteration conjugation `run_camps` already does internally to build
    each iteration's `current_terms` -- so flattening `circuits` (which *is*
    returned) and calling `transform_hamiltonian` once reconstructs the exact
    Hamiltonian matching the final `psi`, without needing `run_camps` itself
    to return that intermediate value.
    """
    E, psi, circuits, clifford_stats = run_camps(n_qubits, chi_max, pauli_terms_orig,
                                                  clifford_unitaries, **run_camps_kwargs)
    all_gates = [gate for iter_circuit in circuits for gate in iter_circuit]
    current_terms = transform_hamiltonian(pauli_terms_orig, all_gates, clifford_unitaries,
                                          combine=True)
    if tag is None:
        tag = f"camps_D{chi_max}"
    mps_path, terms_path = export_camps_state_for_julia(psi, current_terms, outdir, tag=tag)
    return E, psi, current_terms, clifford_stats, mps_path, terms_path


def parse_int_list(s):
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def prepare_output_dir(path):
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    return path


def save_results_csv(rows, path):
    import csv
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def save_plot(rows, outdir, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rr = sorted(rows, key=lambda x: x["D"])
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot([r["D"] for r in rr], [r["E_per_site"] for r in rr], marker="s", label="CAMPS")
        if all(np.isfinite(r.get("MPS_E_per_site_same_D", np.nan)) for r in rr):
            ax.plot([r["D"] for r in rr], [r["MPS_E_per_site_same_D"] for r in rr], marker="o", label="MPS same D")
        # Eref_per_site removed: the old reference value was for a 16x16 periodic
        # cluster, not this 22-site open cluster, so it wasn't a meaningful target.
        # ax.axhline(rr[0]["Eref_per_site"], ls="--", label="reference")
        ax.set_xlabel("CAMPS max bond dimension D")
        ax.set_ylabel("E/N")
        ax.set_title(tag)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(outdir, f"plot_{tag}.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"Saved {path}")
    except Exception as exc:
        print(f"Plotting skipped/failed: {exc}")


def build_argparser():
    p = argparse.ArgumentParser(description="CAMPS-only Kitaev honeycomb model on a fixed 22-site cluster")
    p.add_argument("--K", type=float, default=-1.0, help="Kitaev bond coupling K^alpha (all directions, unless overridden below).")
    p.add_argument("--Kx", type=float, default=None, help="Overrides --K for x-bonds if given.")
    p.add_argument("--Ky", type=float, default=None, help="Overrides --K for y-bonds if given.")
    p.add_argument("--Kz", type=float, default=None, help="Overrides --K for z-bonds if given.")
    p.add_argument("--V", type=float, default=0.3, help="Effective-field three-body term strength (H_V).")
    p.add_argument("--h_field", type=float, default=0.1, help="Boundary field strength (H_V); applied along each boundary site's missing bond direction.")
    p.add_argument("--normalization", choices=["spin", "pauli"], default="pauli",
                   help="spin: H=(J/4)sum XX+YY+ZZ per bond. pauli: H=J sum XX+YY+ZZ per bond")
    # Eref_per_site removed: it was a 16x16-periodic-cluster reference value, not a
    # meaningful target for this fixed 22-site open cluster (finite-size/BC mismatch).
    # p.add_argument("--Eref_per_site", type=float, default=-0.194)
    p.add_argument("--Dlist_camps", type=str, default="100")
    p.add_argument("--skip_mps_same_D", action="store_true",
                   help="Skip ordinary MPS baselines at the same D. If not set, the summary reports MPS_EE_same_D and MPS_E - CAMPS_E.")
    p.add_argument("--max_dmrg_sweeps", type=int, default=30)
    p.add_argument("--n_camps_iters", type=int, default=3)
    p.add_argument("--n_cliff_sweeps", type=int, default=2)
    p.add_argument("--n_cliff_use", type=int, default=720)
    p.add_argument("--use_full_11520_cliffords", action="store_true",
                   help="Use full signed two-qubit Clifford group modulo global phase. Default uses 720 Sp(4,F2) representatives, as in CAMPS paper.")
    p.add_argument("--entropy_tol", type=float, default=1e-10)
    p.add_argument("--max_eval_chi", type=int, default=64,
                   help="Truncate virtual legs to this size only while ranking Clifford gates; <=0 disables.")
    p.add_argument("--exact_top_k", type=int, default=8,
                   help="Number of best approximate Clifford candidates rechecked exactly on the full theta.")
    p.add_argument("--canonicalize_every", type=int, default=4,
                   help="Canonicalize after this many accepted Clifford gates; 1 is safest, 4-8 is faster.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_state", choices=["neel", "up", "down"], default="neel")
    p.add_argument("--outdir", type=str, default="./Kitaev_Honeycomb_camps_outputs")
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--export_for_julia", action="store_true",
                   help="Also apply the Hamiltonian rotation (conjugate the original "
                        "Hamiltonian by the full CAMPS Clifford circuit) and export the "
                        "ground state + rotated Hamiltonian for Correlations/camps_state_bridge.jl.")
    return p


def main():
    args = build_argparser().parse_args()
    args.outdir = prepare_output_dir(args.outdir)
    N = N_SITES
    tag = f"Kitaev_{N}site_{args.normalization}"

    print("\n" + "=" * 80)
    print(f"CAMPS-only Kitaev honeycomb: fixed 22-site cluster, N={N}, K={args.K}")
    print(f"normalization={args.normalization}")
    # print(f"reference E/N = {args.Eref_per_site:.12f}; reference E = {args.Eref_per_site * N:.12f}")
    if args.export_for_julia:
        print(f"Julia export: ENABLED -- will write camps_N{N}_D<D>_V{args.V}_h{args.h_field}_mps.npz/_terms.csv/_rotated_ops.csv to {args.outdir}")
    else:
        print("Julia export: disabled (pass --export_for_julia on the command line to enable; "
              "the VS Code Run button does not pass flags, use a terminal instead)")
    print("=" * 80)

    terms = Kitaev_Honeycomb_Pauli_terms(K=args.K, Kx=args.Kx, Ky=args.Ky, Kz=args.Kz,
                                      V=args.V, h=args.h_field,
                                      normalization=args.normalization)
    max_k = max((len(ops) for _, ops in terms), default=0)
    print(f"Initial Pauli terms={len(terms)}; max locality={max_k}")

    cliffords = load_or_build_cliffords(args.outdir, n_use=args.n_cliff_use,
                                        seed=args.seed, verbose=args.verbose,
                                        quotient_phases=not args.use_full_11520_cliffords)

    #Dlist = parse_int_list(args.Dlist_camps)
    Dlist_camps = parse_int_list(args.Dlist_camps)

    # Use exactly the same bond dimensions for the ordinary MPS baseline.

    Dlist_mps_baseline = list(Dlist_camps)

    # ------------------------------------------------------------------
    # Ordinary MPS baselines at the same bond dimensions as CAMPS.
    # These are needed to judge whether CAMPS actually reduces the MPS
    # burden at fixed D.  We warm-start increasing-D MPS runs from the
    # previous optimized MPS, exactly as one usually does for a D sweep.
    # ------------------------------------------------------------------
    mps_same_D = {}
    if not args.skip_mps_same_D:
        print("\n" + "=" * 80)
        print("MPS baselines at the same D values")
        print("=" * 80)
        model_mps = build_tenpy_model_from_terms(N, terms)
        psi_prev_mps = None
        #for D in Dlist:
        for D in Dlist_mps_baseline:
            print(f"\nMPS D={D}")
            t0_mps = time.time()
            E_mps, psi_mps = run_dmrg(model_mps, D, psi_init=psi_prev_mps,
                                       max_sweeps=args.max_dmrg_sweeps,
                                       init_state=args.init_state,
                                       verbose=args.verbose)
            dt_mps = time.time() - t0_mps
            es_mps = entropy_summary(psi_mps)
            mps_same_D[D] = {
                "E": E_mps,
                "E_per_site": E_mps / N,
                "EE_mid": es_mps["EE_mid"],
                "EE_max": es_mps["EE_max"],
                "EE_sum": es_mps["EE_sum"],
                "EE_mean": es_mps["EE_mean"],
                "time_sec": dt_mps,
            }
            print(f"  MPS E={E_mps:.12f}, E/N={E_mps/N:.12f}, "
                  f"EE_mid={es_mps['EE_mid']:.6f}, time={dt_mps:.1f}s")
            psi_prev_mps = psi_mps
    else:
        print("\nSkipping MPS same-D baselines; comparison columns will be NaN.")

    rows = []
    #for D in Dlist:
    for D in Dlist_camps:
        print(f"\nCAMPS D={D}")
        t0 = time.time()
        E, psi, circuits, cstats = run_camps(N, D, terms, cliffords,
                                             n_cliff_sweeps=args.n_cliff_sweeps,
                                             n_camps_iters=args.n_camps_iters,
                                             max_dmrg_sweeps=args.max_dmrg_sweeps,
                                             init_state=args.init_state,
                                             verbose=args.verbose,
                                             entropy_tol=args.entropy_tol,
                                             max_eval_chi=args.max_eval_chi,
                                             exact_top_k=args.exact_top_k,
                                             canonicalize_every=args.canonicalize_every)
        dt = time.time() - t0
        es = entropy_summary(psi)
        ng = sum(len(c) for c in circuits)

        if args.export_for_julia:
            # Apply the Hamiltonian rotation: conjugate the original Hamiltonian
            # by the full CAMPS Clifford circuit so it matches psi's frame, then
            # export both for Correlations/kitaev_correlation.jl.
            all_gates = [gate for iter_circuit in circuits for gate in iter_circuit]
            current_terms = transform_hamiltonian(terms, all_gates, cliffords, combine=True)
            # V/h_field are included here (not just N/D) so runs at different
            # effective-field strengths -- e.g. pure Kitaev V=0 vs. the
            # perturbed V=0.3 case -- can't silently overwrite each other's
            # exports, the same class of bug already fixed once for i_site.
            julia_tag = f"camps_N{N}_D{D}_V{args.V}_h{args.h_field}"
            export_camps_state_for_julia(psi, current_terms, args.outdir, tag=julia_tag)
            # Also rotate every single-site X/Y/Z operator by the SAME circuit,
            # so the Julia-side correlator's W(0)/V_j operators are expressed in
            # the same disentangled frame as psi and current_terms above -- a
            # plain un-rotated single-site Pauli would measure a different
            # (generally unphysical) quantity in this frame. See
            # export_rotated_single_site_ops's docstring for the derivation.
            export_rotated_single_site_ops(N, all_gates, cliffords, args.outdir, tag=julia_tag)

        Eps = E / N
        # err_ps = Eps - args.Eref_per_site  # Eref_per_site removed, see build_argparser note
        print(f"  E={E:.12f}, E/N={Eps:.12f}")
        print(f"  EE_mid={es['EE_mid']:.6f}, EE_max={es['EE_max']:.6f}, gates={ng}, time={dt:.1f}s")
        print(
            f"  Clifford EE_mid before={cstats['clifford_mid_before_last']:.6f}, "
            f"after={cstats['clifford_mid_after_last']:.6f}, "
            f"removed={cstats['clifford_mid_removed_last']:.6f} "
            f"({100*cstats['clifford_mid_frac_last']:.2f}%)"
        )
        if D in mps_same_D:
            print(
                f"  Same-D comparison: MPS_EE={mps_same_D[D]['EE_mid']:.6f}, "
                f"CAMPS_EE={es['EE_mid']:.6f}, "
                f"EE_gain={mps_same_D[D]['EE_mid'] - es['EE_mid']:.6f}, "
                f"MPS_E-CAMPS_E={mps_same_D[D]['E'] - E:.6e}"
            )
        rows.append({
            "case": tag,
            "method": "CAMPS",
            "D": D,
            "E": E,
            "E_per_site": Eps,
            # Eref_per_site / Eref_total / E_per_site_minus_ref / E_minus_ref removed,
            # see build_argparser note.
            # "Eref_per_site": args.Eref_per_site,
            # "Eref_total": args.Eref_per_site * N,
            # "E_per_site_minus_ref": err_ps,
            # "E_minus_ref": E - args.Eref_per_site * N,
            "EE_mid": es["EE_mid"],
            "EE_max": es["EE_max"],
            "EE_sum": es["EE_sum"],
            "EE_mean": es["EE_mean"],
            "MPS_E_same_D": mps_same_D.get(D, {}).get("E", np.nan),
            "MPS_E_per_site_same_D": mps_same_D.get(D, {}).get("E_per_site", np.nan),
            "MPS_EE_same_D": mps_same_D.get(D, {}).get("EE_mid", np.nan),
            "MPS_EE_max_same_D": mps_same_D.get(D, {}).get("EE_max", np.nan),
            "MPS_time_same_D_sec": mps_same_D.get(D, {}).get("time_sec", np.nan),
            "CAMPS_EE_final": es["EE_mid"],
            "EE_gain": mps_same_D.get(D, {}).get("EE_mid", np.nan) - es["EE_mid"],
            "MPS_E_minus_CAMPS_E": mps_same_D.get(D, {}).get("E", np.nan) - E,
            "MPS_E_per_site_minus_CAMPS_E_per_site": mps_same_D.get(D, {}).get("E_per_site", np.nan) - Eps,
            "clifford_EE_mid_before": cstats["clifford_mid_before_last"],
            "clifford_EE_mid_after": cstats["clifford_mid_after_last"],
            "clifford_EE_mid_removed": cstats["clifford_mid_removed_last"],
            "clifford_EE_mid_frac": cstats["clifford_mid_frac_last"],
            "clifford_EE_max_before": cstats["clifford_max_before_last"],
            "clifford_EE_max_after": cstats["clifford_max_after_last"],
            "clifford_EE_sum_before": cstats["clifford_sum_before_last"],
            "clifford_EE_sum_after": cstats["clifford_sum_after_last"],
            "clifford_EE_sum_removed": cstats["clifford_sum_removed_last"],
            "clifford_EE_sum_frac": cstats["clifford_sum_frac_last"],
            "clifford_EE_mean_before": cstats["clifford_mean_before_last"],
            "clifford_EE_mean_after": cstats["clifford_mean_after_last"],
            "clifford_total_EE_reduction": cstats["clifford_total_EE_reduction"],
            "n_gates": ng,
            "time_sec": dt,
        })
        save_results_csv(rows, os.path.join(args.outdir, f"results_{tag}.csv"))

    save_plot(rows, args.outdir, tag)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        # "E/N-ref" column removed along with Eref_per_site, see build_argparser note.
        f"{'D':>5} {'E':>18} {'E/N':>16} "
        f"{'MPS_EE_same_D':>14} {'CAMPS_EE_final':>15} {'EE_gain':>10} "
        f"{'MPS_E-CAMPS_E':>16} {'EE_before_C':>12} {'EE_after_C':>11} "
        f"{'C_removed':>10} {'Cliff%':>8} {'gates':>7}"
    )
    for r in rows:
        print(
            f"{r['D']:>5} {r['E']:>18.10f} {r['E_per_site']:>16.10f} "
            f"{r['MPS_EE_same_D']:>14.5f} {r['CAMPS_EE_final']:>15.5f} "
            f"{r['EE_gain']:>10.5f} {r['MPS_E_minus_CAMPS_E']:>16.6e} "
            f"{r['clifford_EE_mid_before']:>12.5f} {r['clifford_EE_mid_after']:>11.5f} "
            f"{r['clifford_EE_mid_removed']:>10.5f} "
            f"{100*r['clifford_EE_mid_frac']:>7.2f}% {r['n_gates']:>7}"
        )
    print(f"\nAll data saved under: {args.outdir}")


if __name__ == "__main__":
    main()

