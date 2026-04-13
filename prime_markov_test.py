"""
Prime Propagation Markov Chain Test
=====================================
Tests whether prime structure can be reproduced from purely local
conditional rules (last digit → gap distribution → next last digit),
with no global assumptions.

Stages:
  1. Generate primes, build empirical conditional distributions
  2. Run Markov chain generator from those distributions
  3. Compare synthetic vs actual on density, gap shape, digit transitions, drift

Usage:
  pip install numpy matplotlib sympy
  python prime_markov_test.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import random
import math

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SIEVE_LIMIT      = 10_000_000   # generate primes up to this
MARKOV_STEPS     = 50_000       # how many synthetic primes to generate
DRIFT_ANCHORS    = [0, 100, 500, 1000, 5000, 10000, 20000]  # anchor indices for drift measurement
DRIFT_TRIALS     = 50           # independent Markov runs per anchor for drift stats
RANDOM_SEED      = 42

DIGITS = [1, 3, 7, 9]
DIGIT_COLORS = {1: '#4C72B0', 3: '#DD8452', 7: '#55A868', 9: '#C44E52'}

# ─── STAGE 0: Sieve ───────────────────────────────────────────────────────────

def sieve(limit):
    print(f"Sieving primes up to {limit:,}...")
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    print(f"  Found {len(primes):,} primes")
    return primes

# ─── STAGE 1: Empirical distributions ─────────────────────────────────────────

def build_distributions(primes):
    """
    For each prime P with last digit D:
      - gap to next prime
      - last digit of next prime D'
    Returns:
      gap_dist[D]        = list of gaps
      transition[D][D']  = count
    """
    print("Building empirical conditional distributions...")

    gap_dist   = defaultdict(list)
    transition = {d: defaultdict(int) for d in DIGITS}

    # skip 2 and 5 (only even prime / only prime ending in 5)
    relevant = [p for p in primes if p % 10 in DIGITS]

    for i in range(len(relevant) - 1):
        p     = relevant[i]
        p_nxt = relevant[i + 1]
        d     = p % 10
        d_nxt = p_nxt % 10
        gap   = p_nxt - p

        gap_dist[d].append(gap)
        transition[d][d_nxt] += 1

    # normalise transition to probabilities
    trans_prob = {}
    for d in DIGITS:
        total = sum(transition[d].values())
        trans_prob[d] = {d2: transition[d][d2] / total for d2 in DIGITS}

    # gap stats per digit
    print("\n  Conditional gap statistics (mean | std):")
    for d in DIGITS:
        gaps = gap_dist[d]
        print(f"    last digit {d}: mean={np.mean(gaps):.2f}  std={np.std(gaps):.2f}  n={len(gaps):,}")

    print("\n  Digit transition matrix (rows=from, cols=to):")
    header = "         " + "".join(f"  →{d}" for d in DIGITS)
    print(header)
    for d in DIGITS:
        row = f"  from {d}:  " + "  ".join(f"{trans_prob[d].get(d2,0):.3f}" for d2 in DIGITS)
        print(row)

    return gap_dist, trans_prob, relevant

# ─── STAGE 2: Markov chain generator ──────────────────────────────────────────

def markov_chain(anchor_prime, steps, gap_dist, trans_prob, rng):
    """
    Starting from anchor_prime, generate `steps` synthetic primes
    using only the local conditional distributions.
    Returns list of synthetic 'primes' (positions, not verified).
    """
    current = anchor_prime
    d = current % 10
    sequence = [current]

    # precompute gap arrays per digit for fast sampling
    gap_arrays = {dig: np.array(gaps) for dig, gaps in gap_dist.items()}

    for _ in range(steps):
        # sample gap from empirical distribution for current last digit
        gap = rng.choice(gap_arrays[d])
        current += gap

        # sample next last digit from transition distribution
        d_vals = DIGITS
        d_probs = [trans_prob[d].get(d2, 0) for d2 in d_vals]
        d = rng.choice(d_vals, p=d_probs)

        # adjust current to end in d (small correction, ±1..±9)
        current_last = current % 10
        if current_last != d:
            # find nearest offset to make last digit == d
            offsets = [(d - current_last) % 10, -((current_last - d) % 10)]
            offset = min(offsets, key=abs)
            current += offset

        sequence.append(current)

    return sequence

# ─── STAGE 3: Comparison metrics ──────────────────────────────────────────────

def density_decay(primes, synthetic):
    """Compare 1/ln(N) density prediction against actual and synthetic."""
    print("\nComputing density decay...")

    def windowed_density(seq, window=500):
        densities, midpoints = [], []
        for i in range(0, len(seq) - window, window // 2):
            chunk = seq[i:i+window]
            span = chunk[-1] - chunk[0]
            if span > 0:
                densities.append(window / span)
                midpoints.append(chunk[len(chunk)//2])
        return np.array(midpoints), np.array(densities)

    actual_x, actual_d    = windowed_density(primes[:len(synthetic)])
    synth_x,  synth_d     = windowed_density(synthetic)
    predicted_d = 1.0 / np.log(actual_x)

    return actual_x, actual_d, synth_x, synth_d, predicted_d

def gap_distribution(primes, synthetic):
    """Compare gap distributions."""
    actual_gaps = np.diff(primes[:len(synthetic)+1])
    synth_gaps  = np.diff(synthetic)
    return actual_gaps, synth_gaps

def digit_transitions(sequence):
    """Measure digit transition frequencies in a sequence."""
    trans = {d: defaultdict(int) for d in DIGITS}
    for i in range(len(sequence) - 1):
        d     = sequence[i] % 10
        d_nxt = sequence[i+1] % 10
        if d in DIGITS and d_nxt in DIGITS:
            trans[d][d_nxt] += 1
    # normalise
    for d in DIGITS:
        total = sum(trans[d].values())
        if total:
            trans[d] = {d2: trans[d][d2]/total for d2 in DIGITS}
    return trans

def drift_analysis(actual_primes, gap_dist, trans_prob, anchors, trials):
    """
    For each anchor index, run `trials` Markov chains and measure
    mean absolute drift from actual prime sequence over next 1000 steps.
    """
    print("\nRunning drift analysis...")
    drift_results = {}

    for anchor_idx in anchors:
        if anchor_idx >= len(actual_primes) - 1000:
            continue
        anchor_prime = actual_primes[anchor_idx]
        actual_slice = actual_primes[anchor_idx:anchor_idx+1000]
        drifts = []

        for t in range(trials):
            rng = np.random.default_rng(RANDOM_SEED + t)
            synth = markov_chain(anchor_prime, 999, gap_dist, trans_prob, rng)
            if len(synth) < len(actual_slice):
                continue
            # mean absolute difference at each step
            n = min(len(synth), len(actual_slice))
            drift = np.mean(np.abs(np.array(synth[:n]) - np.array(actual_slice[:n])))
            drifts.append(drift)

        drift_results[anchor_idx] = {
            'anchor_prime': anchor_prime,
            'mean_drift': np.mean(drifts),
            'std_drift': np.std(drifts)
        }
        print(f"  Anchor index {anchor_idx:6d} (prime {anchor_prime:12,}): "
              f"mean drift = {np.mean(drifts):12.1f}  std = {np.std(drifts):10.1f}")

    return drift_results

# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_results(actual_x, actual_d, synth_x, synth_d, predicted_d,
                 actual_gaps, synth_gaps,
                 actual_trans, synth_trans,
                 drift_results):

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#FAFAFA')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    # ── Plot 1: Density decay ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(actual_x, actual_d,    color='#4C72B0', lw=1.5, label='Actual primes',    alpha=0.8)
    ax1.plot(synth_x,  synth_d,     color='#DD8452', lw=1.5, label='Markov synthetic', alpha=0.8, ls='--')
    ax1.plot(actual_x, predicted_d, color='#55A868', lw=1.2, label='1/ln(N) predicted', alpha=0.7, ls=':')
    ax1.set_xlabel('Value on number line')
    ax1.set_ylabel('Local density (primes / unit)')
    ax1.set_title('Density Decay: Actual vs Markov Chain vs 1/ln(N)')
    ax1.legend(fontsize=9)
    ax1.set_facecolor('white')
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Gap distributions ──
    ax2 = fig.add_subplot(gs[0, 2])
    max_gap = min(int(np.percentile(actual_gaps, 99)), 200)
    bins = range(0, max_gap + 2, 2)
    ax2.hist(actual_gaps, bins=bins, density=True, alpha=0.6, color='#4C72B0', label='Actual')
    ax2.hist(synth_gaps,  bins=bins, density=True, alpha=0.6, color='#DD8452', label='Markov')
    ax2.set_xlabel('Gap size')
    ax2.set_ylabel('Density')
    ax2.set_title('Gap Distribution')
    ax2.legend(fontsize=9)
    ax2.set_facecolor('white')
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Transition matrices ──
    ax3 = fig.add_subplot(gs[1, 0])
    mat_actual = np.array([[actual_trans[d].get(d2, 0) for d2 in DIGITS] for d in DIGITS])
    im = ax3.imshow(mat_actual, cmap='Blues', vmin=0, vmax=0.5)
    ax3.set_xticks(range(4)); ax3.set_xticklabels(DIGITS)
    ax3.set_yticks(range(4)); ax3.set_yticklabels(DIGITS)
    ax3.set_xlabel('Next last digit')
    ax3.set_ylabel('Current last digit')
    ax3.set_title('Actual Digit Transitions')
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f'{mat_actual[i,j]:.2f}', ha='center', va='center', fontsize=9,
                     color='white' if mat_actual[i,j] > 0.3 else 'black')

    ax4 = fig.add_subplot(gs[1, 1])
    mat_synth = np.array([[synth_trans[d].get(d2, 0) for d2 in DIGITS] for d in DIGITS])
    ax4.imshow(mat_synth, cmap='Oranges', vmin=0, vmax=0.5)
    ax4.set_xticks(range(4)); ax4.set_xticklabels(DIGITS)
    ax4.set_yticks(range(4)); ax4.set_yticklabels(DIGITS)
    ax4.set_xlabel('Next last digit')
    ax4.set_ylabel('Current last digit')
    ax4.set_title('Markov Digit Transitions')
    for i in range(4):
        for j in range(4):
            ax4.text(j, i, f'{mat_synth[i,j]:.2f}', ha='center', va='center', fontsize=9,
                     color='white' if mat_synth[i,j] > 0.3 else 'black')

    # ── Plot 4: Drift growth ──
    ax5 = fig.add_subplot(gs[1, 2])
    anchor_idxs  = sorted(drift_results.keys())
    anchor_primes = [drift_results[i]['anchor_prime'] for i in anchor_idxs]
    mean_drifts  = [drift_results[i]['mean_drift'] for i in anchor_idxs]
    std_drifts   = [drift_results[i]['std_drift'] for i in anchor_idxs]

    ax5.errorbar(anchor_idxs, mean_drifts, yerr=std_drifts,
                 fmt='o-', color='#C44E52', capsize=4, lw=1.5, label='Mean drift ± 1σ')
    ax5.set_xlabel('Anchor index (prime count from start)')
    ax5.set_ylabel('Mean absolute drift after 1000 steps')
    ax5.set_title('Drift Growth from Anchor')
    ax5.legend(fontsize=9)
    ax5.set_facecolor('white')
    ax5.grid(True, alpha=0.3)

    fig.suptitle('Prime Propagation Markov Chain Test\n'
                 'Can local conditional rules reproduce global prime structure?',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.savefig('/mnt/user-data/outputs/prime_markov_results.png', dpi=150,
                bbox_inches='tight', facecolor='#FAFAFA')
    print("\nPlot saved to prime_markov_results.png")
    plt.show()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Stage 0
    primes = sieve(SIEVE_LIMIT)

    # Stage 1
    gap_dist, trans_prob, relevant_primes = build_distributions(primes)

    # Stage 2 — generate synthetic sequence from first relevant prime as anchor
    print(f"\nGenerating Markov chain ({MARKOV_STEPS:,} steps)...")
    anchor = relevant_primes[0]
    synthetic = markov_chain(anchor, MARKOV_STEPS - 1, gap_dist, trans_prob, rng)
    print(f"  Synthetic range: {synthetic[0]:,} → {synthetic[-1]:,}")
    print(f"  Actual range:    {relevant_primes[0]:,} → {relevant_primes[MARKOV_STEPS-1]:,}")

    # Stage 3 — comparisons
    actual_x, actual_d, synth_x, synth_d, predicted_d = density_decay(
        relevant_primes, synthetic)

    actual_gaps, synth_gaps = gap_distribution(relevant_primes, synthetic)

    actual_trans = digit_transitions(relevant_primes[:MARKOV_STEPS])
    synth_trans  = digit_transitions(synthetic)

    drift_results = drift_analysis(
        relevant_primes, gap_dist, trans_prob, DRIFT_ANCHORS, DRIFT_TRIALS)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nGap distribution:")
    print(f"  Actual mean gap:    {np.mean(actual_gaps):.2f}")
    print(f"  Synthetic mean gap: {np.mean(synth_gaps):.2f}")
    print(f"\nDensity correlation (actual vs synthetic): "
          f"{np.corrcoef(actual_d[:min(len(actual_d),len(synth_d))], synth_d[:min(len(actual_d),len(synth_d))])[0,1]:.4f}")
    print(f"\nDrift growth: {'increasing' if list(drift_results.values())[-1]['mean_drift'] > list(drift_results.values())[0]['mean_drift'] else 'stable'}")
    print("\nKey question: if drift grows with anchor distance, the local model")
    print("is missing non-local information. If drift is bounded, local rules suffice.")

    plot_results(actual_x, actual_d, synth_x, synth_d, predicted_d,
                 actual_gaps, synth_gaps,
                 actual_trans, synth_trans,
                 drift_results)

if __name__ == '__main__':
    main()
