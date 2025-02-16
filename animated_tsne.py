import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Tuple, Dict, Optional, Union
import imageio
from pathlib import Path
from textwrap import wrap

class AnimatedTSNEVisualizer:

####### Initialization

    def __init__(self):
        self.cached_tsne_coords = None
        self.cached_vectors = None
        self.cached_labels = None

    def from_dict(self,
                 vector_dict: Dict[Union[str, int], np.ndarray],
                 perplexity: int = 5,
                 n_iter: int = 2000,
                 random_state: int = 42) -> None:
        vectors = []
        labels = []
        for label, vector in vector_dict.items():
            vectors.append(vector)
            labels.append(label)

        self.compute_tsne_embedding(
            vectors=vectors,
            labels=labels,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state
        )

####### Compute

    def compute_tsne_embedding(self,
                             vectors: List[np.ndarray],
                             labels: Optional[List[Union[str, int]]] = None,
                             perplexity: int = 5,
                             n_iter: int = 2000,
                             random_state: int = 42) -> None:
        vectors_array = np.array(vectors)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            early_exaggeration=4,
            learning_rate=200,
            n_iter=n_iter,
            random_state=random_state,
            metric='cosine'
        )

        self.cached_tsne_coords = tsne.fit_transform(vectors_array)
        self.cached_vectors = vectors_array
        self.cached_labels = labels if labels is not None else list(range(len(vectors)))

####### Animation

    def create_frame(self,
                        displayed_indices: List[int],
                        figsize: Tuple[int, int] = (12, 8),
                        show_line: bool = False) -> plt.Figure:
            if self.cached_tsne_coords is None:
                raise ValueError("Must call compute_tsne_embedding() or from_dict() first")

            # Create figure with fixed size and margins
            fig = plt.figure(figsize=figsize)

            # Create a gridspec with three rows for title and main content
            gs = plt.GridSpec(3, 2,
                             width_ratios=[2, 1],
                             height_ratios=[0.3, 0.85, 0.85],
                             hspace=0.3)

            # Title text area (spans full width)
            ax_title = fig.add_subplot(gs[0, :])
            ax_title.axis('off')

            # Main plot for t-SNE
            ax_main = fig.add_subplot(gs[1:, 0])
            # Legend plot
            ax_legend = fig.add_subplot(gs[1:, 1])

            # Get displayed points
            displayed_coords = self.cached_tsne_coords[displayed_indices]
            displayed_labels = [self.cached_labels[i] for i in displayed_indices]

            # Set consistent axis limits with padding for main plot
            all_coords = self.cached_tsne_coords
            padding = 0.1
            x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
            y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min

            ax_main.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
            ax_main.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

            # Draw connecting line if enabled
            if show_line and len(displayed_coords) > 1:
                ax_main.plot(displayed_coords[:, 0], displayed_coords[:, 1],
                            '-', color='gray', alpha=0.5, linewidth=1)

            # Color points based on their order of appearance
            colors = plt.cm.viridis(np.linspace(0, 1, len(displayed_indices)))
            scatter = ax_main.scatter(displayed_coords[:, 0], displayed_coords[:, 1],
                                    c=colors, s=100, zorder=2)  # Increased zorder to keep points on top

            # Add numerical labels to points with better positioning
            for i in range(len(displayed_indices)):
                # Calculate offset direction based on point position
                x, y = displayed_coords[i]
                x_center = (x_max + x_min) / 2
                y_center = (y_max + y_min) / 2

                # Adjust offset direction to avoid overlapping
                x_offset = -15 if x > x_center else 15
                y_offset = -15 if y > y_center else 15

                ax_main.annotate(
                    str(i+1),
                    (x, y),
                    xytext=(x_offset * 0.3, y_offset * 0.3),
                    textcoords='offset points',
                    fontsize=9,
                    color='black',
                    bbox=dict(
                        facecolor='white',
                        edgecolor='none',
                        alpha=0.7,
                        pad=0.5
                    ),
                    zorder=3,
                    ha='center',
                    va='center'
                )

            # Enhanced legend formatting
            ax_legend.axis('off')
            legend_text = []

            for i, label in enumerate(displayed_labels):
                # Format with consistent spacing and show more text
                if len(str(label)) > 35:
                    wrapped_label = str(label)[:32] + "..."
                else:
                    wrapped_label = str(label)
                legend_text.append(f"{str(i+1).rjust(2)}. {wrapped_label}")

            # Enhanced legend text display
            ax_legend.text(
                0.02, 0.98,
                '\n'.join(legend_text),
                transform=ax_legend.transAxes,
                verticalalignment='top',
                fontsize=12,
                fontfamily='monospace',
                linespacing=1.2
            )

            # Enhanced title with text wrapping
            current_label = displayed_labels[-1]

            # Wrap text to 3 lines with larger width
            wrapped_text = wrap(current_label, width=100)
            if len(wrapped_text) > 5:
                title_text = '\n'.join(wrapped_text[:5]) + '...'
            else:
                title_text = '\n'.join(wrapped_text)

            ax_title.text(0.5, 0.8,
                f'{title_text}',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14,
                fontweight='bold',
                transform=ax_title.transAxes,
                linespacing=1.2
            )

            # Adjust layout to prevent overlapping
            plt.tight_layout()
            return fig

    def create_animation(self,
                        output_path: str,
                        duration: float = 1.0,
                        figsize: Tuple[int, int] = (12, 8),
                        show_line: bool = False) -> None:
        if self.cached_tsne_coords is None:
            raise ValueError("Must call compute_tsne_embedding() or from_dict() first")

        frames = []
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)

        # Create frames
        for i in range(len(self.cached_vectors)):
            # Create frame showing vectors up to index i
            fig = self.create_frame(list(range(i + 1)), figsize, show_line)

            # Save frame to temporary file with fixed dpi and no bbox_inches='tight'
            temp_path = temp_dir / f"frame_{i:03d}.png"
            fig.savefig(temp_path, dpi=100)
            plt.close(fig)

            # Read frame back and append to frames list
            frames.append(imageio.imread(str(temp_path)))

        # Save animation
        fps = 1.0 / duration
        imageio.mimsave(output_path, frames, fps=fps)

        # Cleanup temporary files
        for file in temp_dir.glob("*.png"):
            file.unlink()
        temp_dir.rmdir()

######### Distance

    def calculate_consecutive_distances(self,
                                   metric: str = 'cosine',
                                   normalization: str = 'maxunit') -> List[Dict[str, Union[float, str, str]]]:
        """
        Calculate distances between consecutive points.

        Args:
            metric: 'euclidean' for t-SNE space distances, 'cosine' for similarity in original space
            normalization: Optional normalization method ['minmax', 'standard', 'maxunit', 'meanunit']
        """
        """
        Calculate distances between consecutive points in the t-SNE embedding.

        Returns:
            List of dictionaries containing:
                - distance: Euclidean distance between consecutive points
                - from_label: Label of the starting point
                - to_label: Label of the ending point
        """
        if self.cached_tsne_coords is None:
            raise ValueError("Must call compute_tsne_embedding() or from_dict() first")

        distances = []

        # Calculate distances between consecutive points
        for i in range(len(self.cached_tsne_coords) - 1):
            point1 = self.cached_tsne_coords[i]
            point2 = self.cached_tsne_coords[i + 1]

            if metric == 'euclidean':
                # Calculate Euclidean distance in t-SNE space
                distance = np.sqrt(np.sum((point2 - point1) ** 2))
            elif metric == 'cosine':
                # Calculate cosine similarity in original embedding space
                vec1 = self.cached_vectors[i]
                vec2 = self.cached_vectors[i + 1]
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                # Convert similarity to distance (1 - similarity)
                distance = 1.0 - similarity

            distances.append({
                'distance': float(distance),  # Convert to float for better serialization
                'from_label': str(self.cached_labels[i]),
                'to_label': str(self.cached_labels[i + 1])
            })

        # Apply normalization if specified
        if normalization:
            all_distances = [d['distance'] for d in distances]

            if normalization == 'minmax':
                # Scale to range [0,1]
                min_dist = min(all_distances)
                max_dist = max(all_distances)
                range_dist = max_dist - min_dist
                if range_dist != 0:
                    for d in distances:
                        d['distance'] = (d['distance'] - min_dist) / range_dist

            elif normalization == 'standard':
                # Scale to mean=0, std=1
                mean_dist = np.mean(all_distances)
                std_dist = np.std(all_distances)
                if std_dist != 0:
                    for d in distances:
                        d['distance'] = (d['distance'] - mean_dist) / std_dist

            elif normalization == 'maxunit':
                # Scale by maximum to get range [0,1]
                max_dist = max(all_distances)
                if max_dist != 0:
                    for d in distances:
                        d['distance'] = d['distance'] / max_dist

            elif normalization == 'meanunit':
                # Scale by mean to get mean=1
                mean_dist = np.mean(all_distances)
                if mean_dist != 0:
                    for d in distances:
                        d['distance'] = d['distance'] / mean_dist

        return distances

    def create_distance_animation(self,
                                output_path: str,
                                duration: float = 1.0,
                                figsize: Tuple[int, int] = (8, 6),
                                metric: str = 'cosine',
                                normalization: str = 'standard') -> None:
        """
        Create an animation of consecutive distances as a growing bar chart.

        Args:
            output_path: Path to save the animation
            duration: Duration between frames in seconds
            figsize: Figure size (width, height)
            metric: Distance metric to use ('cosine' or 'euclidean')
            normalization: Normalization method for distances
        """
        if self.cached_vectors is None:
            raise ValueError("Must call compute_tsne_embedding() or from_dict() first")

        # Calculate all distances upfront
        distances = self.calculate_consecutive_distances(metric=metric, normalization=normalization)
        distance_values = [d['distance'] for d in distances]

        frames = []
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)

        # Create frames
        for i in range(len(distances) + 1):  # +1 to show final state
            fig, ax = plt.subplots(figsize=figsize)

            if i > 0:  # Skip first frame as it has no distances yet
                # Plot bars up to current index
                current_distances = distance_values[:i]
                bars = ax.bar(range(1, i+1), current_distances)

                # Color bars based on distance value
                normalize = plt.Normalize(min(distance_values), max(distance_values))
                colors = plt.cm.RdYlBu_r(normalize(current_distances))  # Red for high distance, blue for low
                for bar, color in zip(bars, colors):
                    bar.set_color(color)

            # Set consistent axis limits
            ax.set_xlim(0, len(distances) + 1)
            y_min = min(min(distance_values) * 1.1, 0)  # Include 0 and add 10% padding
            y_max = max(distance_values) * 1.1  # Add 10% padding
            ax.set_ylim(y_min, y_max)

            # Labels and title
            ax.set_xlabel('Transition Number')
            ax.set_ylabel(f'Normalized {metric.capitalize()} Distance')
            ax.set_title(f'Consecutive {metric.capitalize()} Distances\n(showing first {i} transitions)')

            # Add grid for better readability
            ax.grid(True, alpha=0.3)

            # Save frame
            temp_path = temp_dir / f"dist_frame_{i:03d}.png"
            fig.savefig(temp_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            frames.append(imageio.imread(str(temp_path)))

        # Save animation
        fps = 1.0 / duration
        imageio.mimsave(output_path, frames, fps=fps)

        # Cleanup temporary files
        for file in temp_dir.glob("*.png"):
            file.unlink()
        temp_dir.rmdir()

########## Combined

    def create_combined_frame(self,
        displayed_indices: List[int],
        distances: List[Dict[str, Union[float, str, str]]],
        figsize: Tuple[int, int] = (15, 8),
        show_line: bool = False) -> plt.Figure:
        """
        Create a single frame showing both t-SNE and distance visualizations with improved layout
        """
        if self.cached_tsne_coords is None:
            raise ValueError("Must call compute_tsne_embedding() or from_dict() first")

        # Create figure with revised layout including text area
        fig = plt.figure(figsize=figsize)

        # Create gridspec with three rows:
        # Row 1: Title text (spans full width)
        # Row 2: t-SNE plot and legend
        # Row 3: t-SNE plot and distance plot
        gs = plt.GridSpec(3, 2,
                         width_ratios=[2, 1],
                         height_ratios=[0.3, 0.85, 0.85],
                         hspace=0.3)

        # Title text area (spans full width)
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')

        # Main t-SNE plot (spans rows 2 and 3 on left)
        ax_tsne = fig.add_subplot(gs[1:, 0])

        # Legend (top right)
        ax_legend = fig.add_subplot(gs[1, 1])

        # Distance plot (bottom right)
        ax_dist = fig.add_subplot(gs[2, 1])

        # Get displayed points
        displayed_coords = self.cached_tsne_coords[displayed_indices]
        displayed_labels = [self.cached_labels[i] for i in displayed_indices]

        # Set consistent axis limits with padding for t-SNE plot
        all_coords = self.cached_tsne_coords
        padding = 0.1
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min

        ax_tsne.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax_tsne.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

        # Draw connecting line if enabled
        if show_line and len(displayed_coords) > 1:
            ax_tsne.plot(displayed_coords[:, 0], displayed_coords[:, 1],
                        '-', color='gray', alpha=0.5, linewidth=1)

        # Color points based on their order of appearance
        colors = plt.cm.viridis(np.linspace(0, 1, len(displayed_indices)))
        scatter = ax_tsne.scatter(displayed_coords[:, 0], displayed_coords[:, 1],
                                c=colors, s=100, zorder=2)

        # Add numerical labels to points with better positioning
        for i in range(len(displayed_indices)):
            # Calculate offset direction based on point position
            x, y = displayed_coords[i]
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2

            # Adjust offset direction to avoid overlapping with other points
            x_offset = -15 if x > x_center else 15
            y_offset = -15 if y > y_center else 15

            ax_tsne.annotate(
                str(i+1),
                (x, y),
                xytext=(x_offset * 0.3, y_offset * 0.3),
                textcoords='offset points',
                fontsize=9,
                color='black',
                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.7,
                    pad=0.5
                ),
                zorder=3,
                ha='center',
                va='center'
            )

        # Enhanced legend formatting
        ax_legend.axis('off')
        legend_text = []
        max_label_length = max(len(str(label)) for label in displayed_labels)

        for i, label in enumerate(displayed_labels):
            # Format with consistent spacing and show more text
            if len(str(label)) > 35:
                wrapped_label = str(label)[:32] + "..."
            else:
                wrapped_label = str(label)
            legend_text.append(f"{str(i+1).rjust(2)}. {wrapped_label}")

        # Enhanced legend text display
        ax_legend.text(
            0.02, 0.98,
            '\n'.join(legend_text),
            transform=ax_legend.transAxes,
            verticalalignment='top',
            fontsize=12,
            fontfamily='monospace',
            linespacing=1.2
        )

        # Distance visualization
        if len(displayed_indices) > 1:
            current_distances = [d['distance'] for d in distances[:len(displayed_indices)-1]]
            bars = ax_dist.bar(range(1, len(current_distances) + 1), current_distances)

            if current_distances:
                all_distances = [d['distance'] for d in distances]
                normalize = plt.Normalize(min(all_distances), max(all_distances))
                colors = plt.cm.RdYlBu_r(normalize(current_distances))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)

        # Set consistent axis limits for distance plot
        ax_dist.set_xlim(0, len(distances) + 1)
        all_distances = [d['distance'] for d in distances]
        y_min = min(min(all_distances) * 1.1, 0)
        y_max = max(all_distances) * 1.1
        ax_dist.set_ylim(y_min, y_max)

        # Improved labels and grid for distance plot
        ax_dist.set_xlabel('Transition', fontsize=9)
        ax_dist.set_ylabel('Distance', fontsize=9)
        ax_dist.tick_params(axis='both', which='major', labelsize=8)
        ax_dist.grid(True, alpha=0.2, linestyle='--')

        # Enhanced title with text wrapping and full width
        current_label = displayed_labels[-1]

        # Wrap text to 5 lines
        wrapped_text = wrap(current_label, width=100)
        if len(wrapped_text) > 5:
            title_text = '\n'.join(wrapped_text[:5]) + '...'
        else:
            title_text = '\n'.join(wrapped_text)

        ax_title.text(0.5, 0.8,
            f'{title_text}',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=14,
            fontweight='bold',
            transform=ax_title.transAxes,
            linespacing=1.2
        )

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        return fig

    def create_combined_animation(self,
                                output_path: str,
                                duration: float = 1.0,
                                figsize: Tuple[int, int] = (12, 12),
                                show_line: bool = True,
                                metric: str = 'cosine',
                                normalization: str = 'maxunit') -> None:
        """
        Create an animation combining both t-SNE and distance visualizations
        """
        if self.cached_tsne_coords is None:
            raise ValueError("Must call compute_tsne_embedding() or from_dict() first")

        # Calculate all distances upfront
        distances = self.calculate_consecutive_distances(metric=metric, normalization=normalization)

        frames = []
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)

        # Create frames
        for i in range(len(self.cached_vectors)):
            # Create combined frame showing both visualizations
            fig = self.create_combined_frame(
                displayed_indices=list(range(i + 1)),
                distances=distances,
                figsize=figsize,
                show_line=show_line
            )

            # Save frame
            temp_path = temp_dir / f"frame_{i:03d}.png"
            fig.savefig(temp_path, dpi=100)
            plt.close(fig)

            frames.append(imageio.imread(str(temp_path)))

        # Save animation
        fps = 1.0 / duration
        imageio.mimsave(output_path, frames, fps=fps)

        # Cleanup temporary files
        for file in temp_dir.glob("*.png"):
            file.unlink()
        temp_dir.rmdir()
