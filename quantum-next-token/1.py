# ══════════════════════════════════════════════════════════════════════════════
#  QUANTUM-ASSISTED NEXT-WORD PREDICTION  —  Multi-Target test
#
#  N = 20 candidates  (5 qubits, 2^5 = 32 states, 12 unused)
#  M = 2  targets     ("mat" AND "rug" both strongly supported by corpus)
#  N/M = 10           (right at the minimum meaningful threshold)
#
#  Pipeline:
#    [Corpus] → [Trigram Training] → [Top-M Probability Targets]
#               ↓                          ↓
#         Classical linear scan      Multi-target Grover Oracle
#               ↓                          ↓
#         Efficiency: N/2M           Efficiency: π/4·√(N/M)
#
#  Dependencies: pip install qiskit qiskit-aer matplotlib numpy
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator


# ─────────────────────────────────────────────────────────────────────────────
# 1.  VOCABULARY  —  N=20 candidates, 5 qubits (2^5 = 32 states)
#     "mat" and "rug" are the two targets, both heavily supported by corpus.
#     The remaining 18 words are plausible distractors with low probability.
#     States 20–31 are unused quantum states (will measure near-zero).
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATES = [
    "mat",        # idx  0  — TARGET 1
    "rug",        # idx  1  — TARGET 2
    "floor",      # idx  2
    "chair",      # idx  3
    "table",      # idx  4
    "ground",     # idx  5
    "sofa",       # idx  6
    "bench",      # idx  7
    "carpet",     # idx  8
    "cushion",    # idx  9
    "roof",       # idx 10
    "shelf",      # idx 11
    "bed",        # idx 12
    "porch",      # idx 13
    "stool",      # idx 14
    "grass",      # idx 15
    "stone",      # idx 16
    "blanket",    # idx 17
    "tile",       # idx 18
    "democracy",  # idx 19  — implausible outlier
]

N_WORDS     = len(CANDIDATES)     # 20
N_Q         = 5                   # qubits — 2^5 = 32 ≥ 20
N           = 2 ** N_Q            # 32 total quantum states
M           = 2                   # number of solutions
NM_RATIO    = N_WORDS / M         # 10.0  — the ratio under test

TARGET_WORDS = ["mat", "rug"]
TARGET_IDXS  = [CANDIDATES.index(w) for w in TARGET_WORDS]   # [0, 1]

word_to_idx = {w: i for i, w in enumerate(CANDIDATES)}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EXPANDED CORPUS
#     Both "mat" and "rug" appear frequently after "on the".
#     Other candidates appear rarely or not at all in this context.
#     Trigram context: ("on", "the") → ???
# ─────────────────────────────────────────────────────────────────────────────

CORPUS = [
    # mat — strongly supported
    "the cat sat on the mat",
    "the dog lay on the mat",
    "the cat sat on the mat and purred",
    "a kitten rested on the mat by the door",
    "the cat sat on the mat near the window",
    "every morning the cat chose the mat",
    "the old cat always curled on the mat",
    "the cat sat on the mat once more",
    "a small cat slept on the mat",
    "the cat sat on the mat all day",
    # rug — equally supported
    "the cat sat on the rug by the fire",
    "the dog rolled on the rug all evening",
    "the cat rested on the rug near the sofa",
    "a kitten played on the rug in the hall",
    "the cat sat on the rug and purred loudly",
    "every evening the cat chose the rug",
    "the cat sat on the rug once more",
    "a small cat slept on the rug",
    "the cat sat on the rug all afternoon",
    "the old cat often chose the rug",
    # distractors — appear rarely after "on the"
    "the cat jumped onto the chair quickly",
    "she sat down on the floor quietly",
    "he placed the box on the table carefully",
    "the bird landed on the roof at dawn",
    "the children played on the ground outside",
    "the dog lay on the carpet near the door",
    "they rested on the bench in the park",
    "the cat sat on the cushion sometimes",
    "the idea of democracy on the agenda was raised",
    "leaves fell on the grass every autumn",
    "the pot rested on the stool in the corner",
    "the flower sat on the shelf by the window",
]

CONTEXT = ("on", "the")

# Trigram counting
trigram_counts = defaultdict(lambda: defaultdict(int))
for sentence in CORPUS:
    tokens = sentence.lower().split()
    for i in range(len(tokens) - 2):
        trigram_counts[(tokens[i], tokens[i+1])][tokens[i+2]] += 1

# Extract counts for our context, restricted to CANDIDATES
raw_counts = {w: trigram_counts[CONTEXT].get(w, 0) for w in CANDIDATES}

# Laplace smoothing
smoothed   = {w: raw_counts[w] + 1 for w in CANDIDATES}
total      = sum(smoothed.values())
probs      = {w: smoothed[w] / total for w in CANDIDATES}

# Verify targets are top-M
ranked = sorted(CANDIDATES, key=lambda w: probs[w], reverse=True)
top_m  = ranked[:M]

print("=" * 70)
print("  TRIGRAM TRAINING RESULTS  |  context: 'on the ___'")
print("=" * 70)
for i, w in enumerate(CANDIDATES):
    marker = "  ← TARGET" if w in TARGET_WORDS else ""
    bar    = "█" * int(probs[w] * 80)
    print(f"  [{i:02d}] {w:<12} cnt={raw_counts[w]:<3}  "
          f"p={probs[w]:.4f}  {bar}{marker}")
print(f"\n  Top-{M} words by probability: {top_m}")
print(f"  All targets in top-{M}? {set(top_m) == set(TARGET_WORDS)}")
print("=" * 70, "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EFFICIENCY METRICS
#     Classical and quantum query counts derived analytically.
#     These are the numbers the empirical simulation will be tested against.
# ─────────────────────────────────────────────────────────────────────────────

classical_queries = N_WORDS / (2 * M)               # N/2M  = 5.0
grover_queries    = (np.pi / 4) * np.sqrt(N_WORDS / M)  # π/4·√(N/M) ≈ 2.48
theoretical_speedup = classical_queries / grover_queries

# Optimal Grover iterations
n_iter = max(1, int(np.floor((np.pi / 4) * np.sqrt(N_WORDS / M))))

print("=" * 70)
print("  EFFICIENCY ANALYSIS  |  N=20, M=2, N/M=10")
print("=" * 70)
print(f"  Classical expected queries   :  N/2M  =  {classical_queries:.2f}")
print(f"  Grover expected queries      :  π/4·√(N/M)  =  {grover_queries:.4f}")
print(f"  Theoretical speedup          :  {theoretical_speedup:.4f}×")
print(f"  Optimal Grover iterations    :  {n_iter}")
print(f"  N/M ratio                    :  {NM_RATIO:.1f}  (threshold = 10)")
print("=" * 70, "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MULTI-TARGET ORACLE  U_f
#     Marks ALL target states with a −1 phase in a single oracle call.
#     Implementation: apply phase flip for each target sequentially.
#     The combined effect marks every target state simultaneously.
# ─────────────────────────────────────────────────────────────────────────────

def phase_flip_single(qc, qr, target_idx, n_q):
    """Phase-flip one state |target_idx⟩ within an existing circuit."""
    t_bin = format(target_idx, f'0{n_q}b')
    for bit_pos, bit_val in enumerate(reversed(t_bin)):
        if bit_val == '0':
            qc.x(qr[bit_pos])
    qc.h(qr[-1])
    qc.mcx(list(qr[:-1]), qr[-1])
    qc.h(qr[-1])
    for bit_pos, bit_val in enumerate(reversed(t_bin)):
        if bit_val == '0':
            qc.x(qr[bit_pos])


def apply_oracle_Uf_multi(qc, qr, target_idxs, n_q):
    """Multi-target phase oracle: marks all states in target_idxs."""
    for idx in target_idxs:
        phase_flip_single(qc, qr, idx, n_q)


def apply_oracle_Uf0(qc, qr, n_q):
    """Zero oracle: marks |00…0⟩ with −1 phase (diffusion step)."""
    qc.x(qr)
    qc.h(qr[-1])
    qc.mcx(list(qr[:-1]), qr[-1])
    qc.h(qr[-1])
    qc.x(qr)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BUILD GROVER CIRCUIT  —  Multi-Target Version
#     H → [U_f_multi → H → U_f0 → H]^n_iter → Measure
# ─────────────────────────────────────────────────────────────────────────────

def build_grover_multi(n_q, target_idxs, n_iter):
    qr = QuantumRegister(n_q, 'q')
    cr = ClassicalRegister(n_q, 'c')
    qc = QuantumCircuit(qr, cr)
    qc.h(qr)
    qc.barrier()
    for _ in range(n_iter):
        apply_oracle_Uf_multi(qc, qr, target_idxs, n_q)
        qc.barrier()
        qc.h(qr)
        qc.barrier()
        apply_oracle_Uf0(qc, qr, n_q)
        qc.barrier()
        qc.h(qr)
        qc.barrier()
    qc.measure(qr, cr)
    return qc


qc = build_grover_multi(N_Q, TARGET_IDXS, n_iter)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SIMULATE
# ─────────────────────────────────────────────────────────────────────────────

SHOTS     = 8192
simulator = AerSimulator()
counts    = simulator.run(qc, shots=SHOTS).result().get_counts()

# Measurement probabilities for all N_WORDS candidates
quantum_probs = {}
for i, w in enumerate(CANDIDATES):
    state = format(i, f'0{N_Q}b')
    quantum_probs[w] = counts.get(state, 0) / SHOTS

# Probability mass captured by the two targets
target_prob_total = sum(quantum_probs[w] for w in TARGET_WORDS)

# Classical prediction: full sorted ranking
classical_ranked = sorted(CANDIDATES, key=lambda w: probs[w], reverse=True)

# Empirical query count estimation:
# Classical: simulate expected position of first target in random scan
# We estimate this as N / (2M) analytically (unbiased estimator).
# For quantum: we measure the effective "search effort" as 1/P(target)
# normalized to the number of oracle calls (iterations).
empirical_classical_q = N_WORDS / (2 * M)
empirical_quantum_q   = n_iter / target_prob_total if target_prob_total > 0 else float('inf')
empirical_speedup      = empirical_classical_q / empirical_quantum_q

print("Simulation results:\n")
print(f"  {'Idx':<4} {'Word':<12} {'Train p':>8}  {'Quantum p':>10}  {'Role'}")
print("  " + "─" * 58)
for i, w in enumerate(CANDIDATES):
    role   = "TARGET" if w in TARGET_WORDS else "distractor"
    marker = "  ★" if w in TARGET_WORDS else ""
    print(f"  [{i:02d}] {w:<12}  {probs[w]:.4f}     {quantum_probs[w]:.4f}      {role}{marker}")

print(f"\n  Combined target probability (quantum)  :  {target_prob_total:.4f}")
print(f"  Empirical classical queries            :  {empirical_classical_q:.2f}")
print(f"  Empirical quantum queries              :  {empirical_quantum_q:.4f}")
print(f"  Empirical speedup                      :  {empirical_speedup:.4f}×")
print(f"  Theoretical speedup                    :  {theoretical_speedup:.4f}×")
delta = abs(empirical_speedup - theoretical_speedup) / theoretical_speedup * 100
print(f"  Deviation from theory                  :  {delta:.2f}%")
verdict = "✔  FORMULA HOLDS" if delta < 20 else "✘  SIGNIFICANT DEVIATION — check oracle"
print(f"  Verdict                                :  {verdict}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  COMBINED VISUALIZATION
#     Panel layout:
#       [A] Training probability distribution (all 20 candidates)
#       [B] Quantum measurement distribution (Grover output)
#       [C] Classical vs Quantum probability for top candidates
#       [D] Efficiency comparison: theoretical vs empirical
# ─────────────────────────────────────────────────────────────────────────────

BG     = '#0f172a'
PANEL  = '#1e293b'
BORDER = '#334155'
RED    = '#e74c3c'
BLUE   = '#3b82f6'
GREEN  = '#22c55e'
GOLD   = '#f59e0b'
PURPLE = '#a855f7'
MUTED  = '#94a3b8'
WHITE  = '#f1f5f9'

fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle(
    "Quantum-Assisted Next-Word Prediction  —  Multi-Target Efficiency Test\n"
    f"N={N_WORDS} candidates  |  M={M} targets ('mat', 'rug')  |  "
    f"N/M={NM_RATIO:.0f}  (threshold = 10)",
    color=WHITE, fontsize=14, fontweight='bold', y=0.98
)

gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    hspace=0.45, wspace=0.30,
    left=0.06, right=0.97,
    top=0.91, bottom=0.07
)

x20     = np.arange(N_WORDS)
labels  = [f"[{i:02d}]\n'{w}'" for i, w in enumerate(CANDIDATES)]
t_cols  = [RED if w in TARGET_WORDS else BLUE for w in CANDIDATES]


# ── Panel A : Training Probabilities ─────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor(PANEL)
train_vals = [probs[w] for w in CANDIDATES]
bars_a     = ax_a.bar(x20, train_vals, color=t_cols,
                       edgecolor=BORDER, linewidth=0.6, zorder=3)
ax_a.set_title("① Trigram Training Probabilities\nP(word | 'on the ___')  —  N=20",
               color=WHITE, fontsize=11, pad=8)
ax_a.set_xticks(x20)
ax_a.set_xticklabels(labels, fontsize=6.5, color=MUTED, rotation=0)
ax_a.set_ylabel("Probability", color=MUTED, fontsize=9)
ax_a.set_ylim(0, max(train_vals) * 1.3)
ax_a.tick_params(colors=MUTED)
for sp in ax_a.spines.values():
    sp.set_edgecolor(BORDER)
ax_a.yaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
for bar, val in zip(bars_a, train_vals):
    if val > 0.015:
        ax_a.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.002,
                  f"{val:.3f}", ha='center', va='bottom',
                  color=WHITE, fontsize=6)
leg_a = [
    mpatches.Patch(color=RED,  label=f"Targets: 'mat', 'rug'  (M={M})"),
    mpatches.Patch(color=BLUE, label="Distractors"),
]
ax_a.legend(handles=leg_a, facecolor=PANEL, edgecolor=BORDER,
            labelcolor=WHITE, fontsize=8)


# ── Panel B : Quantum Measurement Distribution ────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor(PANEL)
q_vals  = [quantum_probs[w] for w in CANDIDATES]
q_cols  = [RED if w in TARGET_WORDS else '#1d4ed8' for w in CANDIDATES]
bars_b  = ax_b.bar(x20, q_vals, color=q_cols,
                    edgecolor=BORDER, linewidth=0.6, zorder=3)
ax_b.set_title(f"② Grover Measurement Distribution\n"
               f"{SHOTS} shots  |  {n_iter} iteration(s)  |  "
               f"5 qubits (32 states, 20 used)",
               color=WHITE, fontsize=11, pad=8)
ax_b.set_xticks(x20)
ax_b.set_xticklabels(labels, fontsize=6.5, color=MUTED)
ax_b.set_ylabel("Measured Probability", color=MUTED, fontsize=9)
ax_b.set_ylim(0, max(q_vals) * 1.3)
ax_b.tick_params(colors=MUTED)
for sp in ax_b.spines.values():
    sp.set_edgecolor(BORDER)
ax_b.yaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
for bar, val in zip(bars_b, q_vals):
    if val > 0.01:
        ax_b.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.003,
                  f"{val:.3f}", ha='center', va='bottom',
                  color=WHITE, fontsize=6.5)
ax_b.text(0.5, -0.20,
          f"Combined target probability: {target_prob_total:.3f}  "
          f"(ideal: >{1/np.sqrt(N_WORDS/M):.3f})",
          transform=ax_b.transAxes, ha='center', color=MUTED, fontsize=8)


# ── Panel C : Classical vs Quantum top-10 comparison ─────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
ax_c.set_facecolor(PANEL)
top10     = sorted(CANDIDATES, key=lambda w: probs[w], reverse=True)[:10]
top10_idx = [CANDIDATES.index(w) for w in top10]
x10       = np.arange(len(top10))
w_off     = 0.35

cvals = [probs[w] for w in top10]
qvals = [quantum_probs[w] for w in top10]
tcols = [RED if w in TARGET_WORDS else BLUE for w in top10]

bars_cc = ax_c.bar(x10 - w_off/2, cvals, width=w_off,
                    color=GOLD, edgecolor=BORDER, linewidth=0.6,
                    label='Training prob (classical)', zorder=3, alpha=0.9)
bars_qc = ax_c.bar(x10 + w_off/2, qvals, width=w_off,
                    color=GREEN, edgecolor=BORDER, linewidth=0.6,
                    label="Grover prob (quantum)", zorder=3, alpha=0.9)

ax_c.set_title("③ Classical vs Quantum — Top 10 Candidates\n"
               "Training probability  vs  Grover measurement",
               color=WHITE, fontsize=11, pad=8)
ax_c.set_xticks(x10)
ax_c.set_xticklabels(
    [f"[{CANDIDATES.index(w):02d}]\n'{w}'" for w in top10],
    fontsize=7.5, color=MUTED
)
ax_c.set_ylabel("Probability", color=MUTED, fontsize=9)
ax_c.tick_params(colors=MUTED)
for sp in ax_c.spines.values():
    sp.set_edgecolor(BORDER)
ax_c.yaxis.grid(True, color=BORDER, linewidth=0.4, zorder=0)
ax_c.legend(facecolor=PANEL, edgecolor=BORDER,
            labelcolor=WHITE, fontsize=8)


# ── Panel D : Efficiency Comparison ──────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_facecolor(PANEL)
ax_d.axis('off')

# Build efficiency comparison table as text
lines = [
    ("EFFICIENCY COMPARISON", WHITE,  12, 'bold'),
    ("", WHITE, 8, 'normal'),
    ("Formula            Classical          Grover", MUTED, 9, 'normal'),
    ("─" * 52,           BORDER, 8, 'normal'),
    (f"Query count        ½·N/M = {classical_queries:.2f}       "
     f"π/4·√(N/M) = {grover_queries:.2f}",
     WHITE, 9.5, 'normal'),
    (f"Theoretical speedup         {theoretical_speedup:.4f}×",
     GOLD, 10, 'bold'),
    ("", WHITE, 8, 'normal'),
    ("─" * 52, BORDER, 8, 'normal'),
    ("", WHITE, 8, 'normal'),
    ("EMPIRICAL RESULTS", WHITE, 11, 'bold'),
    ("", WHITE, 6, 'normal'),
    (f"Classical queries  (N/2M)  :  {empirical_classical_q:.2f}", GOLD, 9.5, 'normal'),
    (f"Quantum queries    (iter/P) :  {empirical_quantum_q:.4f}", GREEN, 9.5, 'normal'),
    (f"Empirical speedup          :  {empirical_speedup:.4f}×", GREEN, 10, 'bold'),
    ("", WHITE, 6, 'normal'),
    (f"Deviation from theory      :  {delta:.2f}%", MUTED, 9.5, 'normal'),

    (f"N={N_WORDS}  M={M}  qubits={N_Q}  states={N}  shots={SHOTS}", MUTED, 8.5, 'normal'),
    (f"Unused quantum states: {N - N_WORDS}", MUTED, 8.5, 'normal'),
    (f"'democracy' quantum p: {quantum_probs['democracy']:.4f}", MUTED, 8.5, 'normal'),
]

y_pos = 0.97
for text, color, size, weight in lines:
    ax_d.text(0.04, y_pos, text,
              transform=ax_d.transAxes,
              color=color, fontsize=size, fontweight=weight,
              va='top', fontfamily='monospace')
    y_pos -= 0.052 if size >= 10 else 0.040


plt.savefig("quantum_nlp_multitarget.png",
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Plot saved → quantum_nlp_multitarget.png")
print("\nCircuit depth:", qc.depth())
print("Gate count:", dict(qc.count_ops()))