import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit as st
import time


def zygote_simulator(
    max_generations: int = 6,
    division_delay: int = 8,
    growth_factor: float = 0.9,
    drift_strength: float = 0.02,
    differentiation_depth: float = 0.6,
    frames: int = 220,
):
    """Run the simulation and stream frames to Streamlit."""
    colors = ["gold", "skyblue", "salmon", "lightgreen"]
    labels = ["Zygote / Early", "Ectoderm", "Mesoderm", "Endoderm"]

    cells = [{"pos": np.array([0.0, 0.0]), "gen": 0, "type": 0}]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    title = ax.set_title("Zygote → Embryo Growth Simulation", fontsize=14, fontweight="bold")
    stage_text = ax.text(-9, 9, "Stage: Zygote", fontsize=12, fontweight="bold", color="black")

    scatter = ax.scatter([], [], s=150, edgecolors="k")
    for label, color in zip(labels, colors):
        ax.scatter([], [], c=color, label=label, edgecolors="k")
    ax.legend(frameon=False, loc="lower right", title="Cell Types")

    placeholder = st.empty()
    for frame in range(frames):
        # Division events
        if frame % division_delay == 0 and len(cells) < 2 ** max_generations:
            new_cells = []
            for cell in cells:
                if cell["gen"] < max_generations:
                    angle1 = np.random.uniform(0, 2 * np.pi)
                    angle2 = angle1 + np.pi
                    offset1 = growth_factor * np.array([np.cos(angle1), np.sin(angle1)])
                    offset2 = growth_factor * np.array([np.cos(angle2), np.sin(angle2)])
                    new_cells.append({"pos": cell["pos"] + offset1, "gen": cell["gen"] + 1, "type": 0})
                    new_cells.append({"pos": cell["pos"] + offset2, "gen": cell["gen"] + 1, "type": 0})
            cells.extend(new_cells)

            # Differentiate by generation depth
            for c in cells:
                g = c["gen"]
                if g > max_generations * differentiation_depth:
                    c["type"] = 3
                elif g > max_generations * 0.6:
                    c["type"] = 2
                elif g > max_generations * 0.3:
                    c["type"] = 1
                else:
                    c["type"] = 0

        # Random drift
        for c in cells:
            c["pos"] += np.random.normal(0, drift_strength, 2)

        positions = np.array([c["pos"] for c in cells])
        color_list = [colors[c["type"]] for c in cells]

        scatter.set_offsets(positions)
        scatter.set_color(color_list)
        scatter.set_sizes([150 - c["gen"] * 10 for c in cells])

        num_cells = len(cells)
        if num_cells < 8:
            stage = "Zygote"
        elif num_cells < 40:
            stage = "Blastula"
        elif num_cells < 150:
            stage = "Gastrula"
        else:
            stage = "Fetus"

        title.set_text(f"Zygote → Embryo Simulation | Cells: {num_cells}")
        stage_text.set_text(f"Stage: {stage}")

        # Stream current frame to the app
        placeholder.pyplot(fig, clear_figure=False)
        time.sleep(0.05)

    # Close the figure to free memory
    plt.close(fig)


def main():
    st.set_page_config(page_title="Zygote-to-Baby Evolution Simulator", layout="wide")
    st.title("🧬 Interactive Zygote-to-Baby Evolution Simulator")
    st.write(
        "This app simulates embryonic development from a single zygote to a multi-layered embryo using simple rules."
    )

    with st.sidebar:
        st.header("Controls")
        max_generations = st.slider("Max Generations", 3, 9, 6, 1)
        division_delay = st.slider("Division Speed (lower = faster)", 3, 15, 8, 1)
        growth_factor = st.slider("Growth Factor", 0.5, 1.5, 0.9, 0.1)
        drift_strength = st.slider("Cell Drift", 0.0, 0.05, 0.02, 0.005)
        differentiation_depth = st.slider("Diff. Depth", 0.3, 0.9, 0.6, 0.05)
        frames = st.slider("Frames", 60, 300, 220, 20)
        run = st.button("Run Simulation")

    if run or "_ran_once" not in st.session_state:
        st.session_state["_ran_once"] = True
        with st.spinner("Running simulation..."):
            zygote_simulator(
                max_generations=max_generations,
                division_delay=division_delay,
                growth_factor=growth_factor,
                drift_strength=drift_strength,
                differentiation_depth=differentiation_depth,
                frames=frames,
            )
    else:
        st.info("Adjust parameters on the left and click 'Run Simulation'.")


if __name__ == "__main__":
    main()


