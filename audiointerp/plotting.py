"""Shared figure style: one fixed color per model, light axes, and a save helper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed per model so colors stay stable across figures with different model subsets.
MODEL_COLORS = {
    "wav2vec2-base": "#2a78d6",
    "wav2vec2-base-960h": "#4a3aa7",
    "wav2vec2-large-emotion": "#008300",
    "whisper-small": "#eda100",
}
FALLBACK_COLOR = "#898781"
GRID_COLOR = "#e1e0d9"


def model_color(model):
    return MODEL_COLORS.get(model, FALLBACK_COLOR)


def style_axes(ax, grid="y"):
    """Light grid behind the data, no top or right spine."""
    ax.grid(axis=grid, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved → {path}")
