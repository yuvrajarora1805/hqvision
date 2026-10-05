"""Qiskit quantum processing engine: feature map, Gram matrices, classifier.

Built directly on Qiskit primitives rather than qiskit-machine-learning, which
publishes no cp314 wheel (see requirements.txt). The classifier is a
scikit-learn SVC over a *precomputed* quantum kernel, so no qiskit
qiskit-ml dependency is needed at all.

Performance note
----------------
The naive way to build a Gram matrix runs |A| x |B| separate circuits in a
Python loop -- 1.6 M simulations for this dataset. Because the Gram entry is
K(a_i, b_j) = |<psi(a_i)|psi(b_j)>|^2, the whole matrix factorises: simulate
|A| + |B| statevectors once, then take one complex matmul. This reduces the
cost from quadratic Python-level circuit evaluations to a single BLAS call.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from . import config


# --- Circuit construction ----------------------------------------------------
def build_zz_feature_map(
    n_qubits: int = config.N_QUBITS,
    reps: int = config.N_REPS,
    entanglement: str = config.ENTANGLEMENT,
) -> QuantumCircuit:
    """ZZ feature map: H + P(x) layers interleaved with ZZ entanglers.

    Qiskit renamed the blueprint class to a function in 2.1 (the class is
    deprecated and removed in 3.0). Both spellings are handled so the report can
    cite either without the code breaking on a version bump.
    """
    try:
        from qiskit.circuit.library import zz_feature_map  # Qiskit >= 2.1

        return zz_feature_map(
            feature_dimension=n_qubits, reps=reps, entanglement=entanglement
        )
    except ImportError:
        from qiskit.circuit.library import ZZFeatureMap  # Qiskit < 2.1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return ZZFeatureMap(
                feature_dimension=n_qubits, reps=reps, entanglement=entanglement
            )


def build_overlap_feature_map(n_qubits: int = config.OVERLAP_N_QUBITS) -> QuantumCircuit:
    """Angle-encoding feature map used by the legacy PennyLane baseline.

    The legacy ``train_hybrid_model.py`` used ``AngleEmbedding`` sandwiched
    around ``StronglyEntanglingLayers`` with an all-zero weight tensor. A
    zero-weight entangler is the identity, so that circuit collapses to
    ``AngleEmbedding(x1 - x2)`` and the returned ``probs()[0]`` is the |0000>
    amplitude of the *difference* state, i.e. ``prod_i cos^2((x1_i - x2_i)/2)`` --
    a product-cosine kernel.

    Angle-encoding is realised as ``RY(x)`` then ``RZ(x)``:
    ``RY(x)|0> = cos(x/2)|0> + sin(x/2)|1>`` and ``RZ(x)`` supplies the
    ``e^{ix}`` relative phase, matching PennyLane's convention up to a global
    phase. (``H`` + ``P(2x)`` is a *different* state and is not used here.)

    Reproducing this in Qiskit rather than installing PennyLane gives the report
    a faithful second baseline with a genuinely different inductive bias from the
    ZZ map, at zero dependency cost.
    """
    x = ParameterVector("x", n_qubits)
    circuit = QuantumCircuit(n_qubits)
    for qubit, param in zip(range(n_qubits), x):
        circuit.ry(param, qubit)
        circuit.rz(param, qubit)
    return circuit


def circuit_summary(circuit: QuantumCircuit) -> dict[str, int]:
    """Gate/depth statistics for the circuit-analysis section of the report."""
    counts: dict[str, int] = {}
    for instruction in circuit.data:
        name = instruction.operation.name
        counts[name] = counts.get(name, 0) + 1
    return {
        "n_qubits": circuit.num_qubits,
        "n_parameters": circuit.num_parameters,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "two_qubit_gates": sum(
            c for g, c in counts.items() if g in {"cx", "cz", "rzz", "cp"}
        ),
        **{f"gate_{k}": v for k, v in counts.items()},
    }


# --- Statevector batching ----------------------------------------------------
def statevectors(circuit: QuantumCircuit, X: np.ndarray) -> np.ndarray:
    """Simulate one statevector per row of ``X``. Returns ``(n, 2**n)`` complex."""
    X = np.asarray(X, dtype=float)
    parameters = list(circuit.parameters)
    if len(parameters) != X.shape[1]:
        raise ValueError(
            f"Circuit expects {len(parameters)} parameters, data has {X.shape[1]}"
        )
    out = np.empty((X.shape[0], 2**circuit.num_qubits), dtype=complex)
    for i, row in enumerate(X):
        out[i] = Statevector.from_instruction(
            circuit.assign_parameters(dict(zip(parameters, row)))
        ).data
    return out


def fidelity_gram(
    circuit: QuantumCircuit, A: np.ndarray, B: np.ndarray | None = None
) -> np.ndarray:
    """Gram matrix of the fidelity kernel K(a,b) = |<psi(a)|psi(b)>|^2.

    Symmetry and unit diagonal are enforced: K_ij = K_ji and K_ii = 1.
    """
    psi_a = statevectors(circuit, A)
    psi_b = psi_a if B is None else statevectors(circuit, B)
    gram = np.abs(psi_a @ psi_b.conj().T) ** 2
    if B is None:
        gram = 0.5 * (gram + gram.T)  # numerical symmetry
        np.fill_diagonal(gram, 1.0)
    return gram


def overlap_gram(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Legacy product-cosine kernel: ``prod_i cos^2((a_i - b_i) / 2)``.

    This is the |0000> amplitude of ``AngleEmbedding(a - b)``, i.e. exactly the
    ``probs()[0]`` that ``train_hybrid_model.py`` returned. The closed form is
    used instead of a circuit because the difference statevector factorises, and
    ``unit tests`` assert agreement with the explicit circuit.
    """
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    # cos^2(d/2) = (1 + cos d) / 2 keeps the result in [0, 1] without overflow.
    return np.prod(0.5 * (1.0 + np.cos(A[:, None, :] - B[None, :, :])), axis=-1)


# --- Classifier --------------------------------------------------------------
@dataclass
class EncodingParams:
    """Parameters needed to encode a raw 4x4 patch into the angle domain."""

    pca: PCA
    scaler: MinMaxScaler


class QuantumKernelClassifier:
    """PCA -> angle scaling -> quantum kernel -> precomputed-kernel SVC.

    Every transform is fitted on the training split only. Fitting the scaler on
    the full dataset (as the original script did) leaks test-set statistics into
    training and inflates reported accuracy.
    """

    def __init__(
        self,
        kernel: str = "zz_fidelity",
        pca_dims: int = config.PCA_DIMS,
        n_qubits: int = config.N_QUBITS,
        reps: int = config.N_REPS,
        class_weight: str | None = config.SVC_CLASS_WEIGHT,
        C: float = 1.0,
        angle_range: tuple[float, float] = (config.ANGLE_MIN, config.ANGLE_MAX),
    ) -> None:
        self.kernel = kernel
        self.pca_dims = pca_dims
        self.n_qubits = n_qubits
        self.reps = reps
        self.class_weight = class_weight
        self.C = C
        self.angle_range = angle_range
        self.encoding: EncodingParams | None = None
        self.svc: SVC | None = None
        self.circuit: QuantumCircuit | None = None
        self.train_gram: np.ndarray | None = None
        self._train_A: np.ndarray | None = None

    # --- Encoding ------------------------------------------------------------
    def _build_circuit(self) -> QuantumCircuit | None:
        if self.kernel == "zz_fidelity":
            return build_zz_feature_map(self.n_qubits, self.reps, config.ENTANGLEMENT)
        if self.kernel in {"overlap", "pennylane_overlap"}:
            return None  # closed form
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit_encode(self, X_raw: np.ndarray) -> np.ndarray:
        """Fit PCA + MinMax on training data only, then transform it."""
        X_raw = np.asarray(X_raw, dtype=float)
        pca = PCA(n_components=self.pca_dims, random_state=config.SEED)
        reduced = pca.fit_transform(X_raw)
        scaler = MinMaxScaler(feature_range=self.angle_range)
        scaled = scaler.fit_transform(reduced)
        self.encoding = EncodingParams(pca=pca, scaler=scaler)
        return scaled

    def encode(self, X_raw: np.ndarray) -> np.ndarray:
        """Transform new data with the *fitted* training parameters."""
        if self.encoding is None:
            raise RuntimeError("Call fit_encode() before encode().")
        return self.encoding.scaler.transform(self.encoding.pca.transform(X_raw))

    def kernel_matrix(self, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        if self.kernel in {"overlap", "pennylane_overlap"}:
            return overlap_gram(A, B)
        if self.circuit is None:
            self.circuit = self._build_circuit()
        return fidelity_gram(self.circuit, A, B)

    # --- Training / inference ------------------------------------------------
    def fit(self, X_raw: np.ndarray, y: np.ndarray) -> "QuantumKernelClassifier":
        """Fit the encoder on training data, then train the SVC on its kernel."""
        A = self.fit_encode(X_raw)
        self._train_A = A  # cached so predict() reuses the training encodings
        self.circuit = self._build_circuit()
        self.train_gram = self.kernel_matrix(A)
        self.svc = SVC(
            kernel="precomputed",
            C=self.C,
            class_weight=self.class_weight,
        )
        self.svc.fit(self.train_gram, y)
        return self

    def test_gram(self, X_raw: np.ndarray) -> np.ndarray:
        """Cross-kernel between unseen data and the cached training encodings."""
        return self.kernel_matrix(self.encode(X_raw), self._train_A)

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        return self.svc.predict(self.test_gram(X_raw))

    def decision_function(self, X_raw: np.ndarray) -> np.ndarray:
        """Signed margin, required for ROC curves and AUC."""
        return self.svc.decision_function(self.test_gram(X_raw))

    def predict_from_gram(self, gram_test: np.ndarray) -> np.ndarray:
        return self.svc.predict(gram_test)

    def decision_from_gram(self, gram_test: np.ndarray) -> np.ndarray:
        return self.svc.decision_function(gram_test)


# --- Finite-shot / noisy inference (Experiment B) ----------------------------
def noisy_fidelity_gram(
    circuit: QuantumCircuit,
    A: np.ndarray,
    B: np.ndarray,
    error_rate: float,
    shots: int = config.NOISE_SHOTS,
    seed: int = config.SEED,
) -> np.ndarray:
    """Fidelity Gram estimated on a depolarising AerSimulator with finite shots.

    Each entry is the fidelity |<psi(a)|psi(b)>|^2, estimated the way it has to be
    on hardware: prepare U(a), then apply U(b)^dagger, and read the probability
    of landing back on |0...0>.  Measuring a single bound state instead would give
    P(|0...0>) of psi(b) only -- a different quantity whose rows do not depend on
    ``a`` at all.

    The transpiled circuit and its dagger are built once (the gate *structure* is
    parameter-independent) and only the angles are re-bound per sample pair, so
    the pair loop pays for binding and simulation, not for two transpiles each.
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit import transpile

    basis = ["rz", "sx", "x", "cx"]
    compiled = transpile(circuit, basis_gates=basis, optimization_level=1, seed_transpiler=seed)
    compiled_dagger = transpile(
        circuit.inverse(), basis_gates=basis, optimization_level=1, seed_transpiler=seed
    )

    noise = NoiseModel()
    present = {inst.operation.name for inst in compiled.data}
    for gate in ("rz", "sx", "x"):
        if gate in present:
            noise.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), [gate])
    if "cx" in present:
        noise.add_all_qubit_quantum_error(depolarizing_error(error_rate, 2), ["cx"])

    simulator = AerSimulator(noise_model=noise)
    simulator.set_options(seed_simulator=seed)

    parameters = list(circuit.parameters)
    zero_key = "0" * circuit.num_qubits

    circuits = []
    for a in A:
        psi_a = compiled.assign_parameters(dict(zip(parameters, a)))
        for b in B:
            psi_b = compiled_dagger.assign_parameters(dict(zip(parameters, b)))
            interference = psi_a.compose(psi_b)
            interference.measure_all()
            circuits.append(interference)

    result = simulator.run(circuits, shots=shots).result()
    counts = result.get_counts()
    if isinstance(counts, dict):  # a single circuit returns one dict, not a list
        counts = [counts]
    # Strip register separators so both "0000" and "0 0 0 0" style keys match.
    probs = np.array(
        [
            sum(v for key, v in counts[i].items() if key.replace(" ", "") == zero_key) / shots
            for i in range(len(circuits))
        ]
    )
    return probs.reshape(len(A), len(B))
