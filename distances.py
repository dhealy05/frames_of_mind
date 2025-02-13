import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SequenceMetrics:
    """Aggregate metrics for analyzing patterns in sequences of distances"""
    avg_max_index_pct: float  # Average position of maximum value as percentage
    avg_min_index: float  # Average position of minimum value
    mean_first_value: float  # Average of first values across sequences
    mean_last_value: float  # Average of last values across sequences
    avg_sequence_length: float  # Average length of sequences
    avg_range: float  # Average range (max-min) within each sequence
    avg_volatility: float  # Average absolute step-to-step change
    front_loading_score: float  # Ratio of avg first third vs last third
    peak_distribution: Dict[int, int]  # Distribution of where peaks occur
    trend_direction: float  # Average slope across sequences
    normalized_avg_sequence: List[float]  # Average sequence normalized to 100 points

def normalize_sequence(seq: List[float], target_length: int = 100) -> List[float]:
    """
    Interpolate a sequence to a target length using linear interpolation.
    """
    if len(seq) == 1:
        return [seq[0]] * target_length

    # Create evenly spaced points for the target length
    x_orig = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, target_length)

    # Interpolate to new length
    return list(np.interp(x_new, x_orig, seq))

def analyze_sequences(sequences: List[List[float]], normalize_length: int = 100) -> SequenceMetrics:
    """
    Analyze multiple sequences of distances to extract aggregate patterns.

    Args:
        sequences: List of variable-length sequences containing distance values
        normalize_length: Number of points to use in normalized average sequence

    Returns:
        SequenceMetrics object containing aggregate statistics
    """
    if not sequences:
        raise ValueError("Must provide at least one sequence")

    # Initialize collectors
    max_indices_pct = []  # New percentage-based collector
    min_indices = []
    first_values = []
    last_values = []
    sequence_lengths = []
    ranges = []
    volatilities = []
    front_vs_back_ratios = []
    peak_positions = []
    sequence_slopes = []
    normalized_sequences = []

    for seq in sequences:
        if not seq:
            continue

        # Basic sequence properties
        seq_length = len(seq)
        sequence_lengths.append(seq_length)

        # Calculate max index as percentage (0 to 1)
        max_idx = np.argmax(seq)
        max_idx_pct = max_idx / (seq_length - 1) if seq_length > 1 else 0
        max_indices_pct.append(max_idx_pct)

        # Normalize sequence to common length
        normalized_seq = normalize_sequence(seq, normalize_length)
        normalized_sequences.append(normalized_seq)

        # Rest of the calculations
        first_values.append(seq[0])
        last_values.append(seq[-1])
        min_indices.append(np.argmin(seq))
        ranges.append(max(seq) - min(seq))

        if len(seq) > 1:
            volatility = np.mean(np.abs(np.diff(seq)))
            volatilities.append(volatility)

        if len(seq) >= 3:
            third_size = len(seq) // 3
            first_third = np.mean(seq[:third_size])
            last_third = np.mean(seq[-third_size:])
            if last_third != 0:
                front_vs_back_ratios.append(first_third / last_third)

        peaks = []
        for i in range(1, len(seq) - 1):
            if seq[i] > seq[i-1] and seq[i] > seq[i+1]:
                peaks.append(i)
        if peaks:
            peak_positions.extend(peaks)

        if len(seq) > 1:
            x = np.arange(len(seq))
            slope, _ = np.polyfit(x, seq, 1)
            sequence_slopes.append(slope)

    # Calculate average normalized sequence
    avg_normalized_sequence = list(np.mean(normalized_sequences, axis=0))

    # Compile peak distribution
    peak_dist = {}
    if peak_positions:
        for pos in peak_positions:
            peak_dist[pos] = peak_dist.get(pos, 0) + 1

    return SequenceMetrics(
        avg_max_index_pct=np.mean(max_indices_pct),
        avg_min_index=np.mean(min_indices),
        mean_first_value=np.mean(first_values),
        mean_last_value=np.mean(last_values),
        avg_sequence_length=np.mean(sequence_lengths),
        avg_range=np.mean(ranges),
        avg_volatility=np.mean(volatilities) if volatilities else 0.0,
        front_loading_score=np.mean(front_vs_back_ratios) if front_vs_back_ratios else 1.0,
        peak_distribution=peak_dist,
        trend_direction=np.mean(sequence_slopes) if sequence_slopes else 0.0,
        normalized_avg_sequence=avg_normalized_sequence
    )

def plot_normalized_sequences(sequences: List[List[float]],
                           figsize: Tuple[int, int] = (12, 6),
                           show_individual: bool = True,
                           normalize_length: int = 100,
                           output_path: str = "img/normalized_sequences.png") -> None:
    """
    Plot normalized sequences and their average.

    Args:
        sequences: List of sequences to plot
        figsize: Figure size (width, height)
        show_individual: Whether to show individual sequences
        normalize_length: Number of points to normalize sequences to
    """
    plt.figure(figsize=figsize)

    # Normalize sequences and calculate average
    normalized_sequences = [normalize_sequence(seq, normalize_length) for seq in sequences]
    avg_sequence = np.mean(normalized_sequences, axis=0)

    # Create x-axis as percentages
    x_vals = np.linspace(0, 100, normalize_length)

    # Plot individual sequences if requested
    if show_individual:
        for i, seq in enumerate(normalized_sequences):
            plt.plot(x_vals, seq, alpha=0.2, color='gray', label='Individual sequences' if i == 0 else None)

    # Plot average sequence
    plt.plot(x_vals, avg_sequence, 'b-', linewidth=2, label='Average sequence')

    # Add labels and title
    plt.xlabel('Position (%)')
    plt.ylabel('Value')
    plt.title('Normalized Sequences Analysis')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend if showing individual sequences
    if show_individual:
        plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig(output_path)

# Example sequences of different lengths
#test_sequences = [
#    [0.5, 0.8, 0.3],  # length 3
#    [0.6, 0.9, 0.7, 0.5],  # length 4
#    [0.4, 0.8, 0.6, 0.3, 0.2, 0.1, 0.05]  # length 7
#]

# Analyze sequences
#metrics = analyze_sequences(test_sequences)
#print(f"Average max index (as percentage): {metrics.avg_max_index_pct:.2f}")
#print(f"Average sequence length: {metrics.avg_sequence_length:.2f}")

# Plot the sequences
#plot_normalized_sequences(test_sequences, show_individual=True)
