import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import time


def generate_shape(stage, num_points):
    """Return target coordinates for each stage based on realistic developmental shapes."""
    
    if stage == "germinal":
        # Simple circular morula-like structure
        theta = np.linspace(0, 2*np.pi, num_points)
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        
    elif stage == "embryonic":
        # C-shaped embryo with limb buds - more realistic proportions
        points = []
        
        # Main C-shaped body
        t_body = np.linspace(np.pi/3, 5*np.pi/3, num_points//2)
        x_body = 3 * np.cos(t_body)
        y_body = 2 * np.sin(t_body)
        points.extend(zip(x_body, y_body))
        
        # Head region (enlarged)
        t_head = np.linspace(0, 2*np.pi, num_points//4)
        x_head = 1.8 * np.cos(t_head) + 2.5
        y_head = 1.8 * np.sin(t_head) + 1.5
        points.extend(zip(x_head, y_head))
        
        # Limb buds - smaller and more realistic
        # Left arm bud
        t_arm_l = np.linspace(0, np.pi, num_points//8)
        x_arm_l = 0.6 * np.cos(t_arm_l) - 1.5
        y_arm_l = 0.6 * np.sin(t_arm_l) + 1.2
        points.extend(zip(x_arm_l, y_arm_l))
        
        # Right arm bud
        x_arm_r = 0.6 * np.cos(t_arm_l) + 1.5
        y_arm_r = 0.6 * np.sin(t_arm_l) + 1.2
        points.extend(zip(x_arm_r, y_arm_r))
        
        # Left leg bud
        x_leg_l = 0.5 * np.cos(t_arm_l) - 0.8
        y_leg_l = 0.5 * np.sin(t_arm_l) - 1.8
        points.extend(zip(x_leg_l, y_leg_l))
        
        # Right leg bud
        x_leg_r = 0.5 * np.cos(t_arm_l) + 0.8
        y_leg_r = 0.5 * np.sin(t_arm_l) - 1.8
        points.extend(zip(x_leg_r, y_leg_r))
        
        x, y = zip(*points)
        x, y = np.array(x), np.array(y)
        
    else:  # fetal
        # Human-like fetus with proper proportions
        points = []
        
        # Head (large and rounded)
        t_head = np.linspace(0, 2*np.pi, num_points//4)
        x_head = 2.2 * np.cos(t_head)
        y_head = 2.2 * np.sin(t_head) + 2.8
        points.extend(zip(x_head, y_head))
        
        # Body/torso
        t_body = np.linspace(0, 2*np.pi, num_points//4)
        x_body = 1.8 * np.cos(t_body)
        y_body = 1.2 * np.sin(t_body) + 0.5
        points.extend(zip(x_body, y_body))
        
        # Left arm
        t_arm = np.linspace(0, np.pi, num_points//8)
        x_arm_l = 1.0 * np.cos(t_arm + np.pi/3) - 1.2
        y_arm_l = 1.0 * np.sin(t_arm + np.pi/3) + 1.5
        points.extend(zip(x_arm_l, y_arm_l))
        
        # Right arm
        x_arm_r = 1.0 * np.cos(t_arm - np.pi/3) + 1.2
        y_arm_r = 1.0 * np.sin(t_arm - np.pi/3) + 1.5
        points.extend(zip(x_arm_r, y_arm_r))
        
        # Left leg
        x_leg_l = 1.2 * np.cos(t_arm + np.pi) - 0.6
        y_leg_l = 1.2 * np.sin(t_arm + np.pi) - 1.2
        points.extend(zip(x_leg_l, y_leg_l))
        
        # Right leg
        x_leg_r = 1.2 * np.cos(t_arm) + 0.6
        y_leg_r = 1.2 * np.sin(t_arm) - 1.2
        points.extend(zip(x_leg_r, y_leg_r))
        
        x, y = zip(*points)
        x, y = np.array(x), np.array(y)
    
    return np.vstack((x, y)).T


def zygote_simulator_with_morphing(
    max_generations=6, 
    division_delay=8, 
    growth_factor=0.9, 
    drift_strength=0.02, 
    morph_speed=0.04, 
    differentiation_depth=0.6,
    frames=220
):
    """Run the zygote simulation with shape morphing and stream to Streamlit."""
    
    colors = ['gold', 'skyblue', 'salmon', 'lightgreen']
    labels = ['Zygote', 'Ectoderm', 'Mesoderm', 'Endoderm']
    cells = [{'pos': np.array([0.0, 0.0]), 'gen': 0, 'type': 0}]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    title = ax.set_title("Zygote → Embryo → Baby Morphing Simulation", fontsize=14, fontweight='bold')
    stage_text = ax.text(-9, 9, "Stage: Germinal", fontsize=12, fontweight='bold', color='black')
    scatter = ax.scatter([], [], s=150, edgecolors='k')

    for label, color in zip(labels, colors):
        ax.scatter([], [], c=color, label=label, edgecolors='k')
    ax.legend(frameon=False, loc='lower right', title="Cell Types")

    # Generate target shapes for morphing
    target_germinal = generate_shape("germinal", 100)
    target_embryo = generate_shape("embryonic", 100)
    target_fetus = generate_shape("fetal", 100)

    def get_stage(num_cells):
        if num_cells < 20:
            return "germinal"
        elif num_cells < 60:
            return "embryonic"
        else:
            return "fetal"

    placeholder = st.empty()
    
    for frame in range(frames):
        # Division events - more aggressive cell division
        if frame % division_delay == 0 and len(cells) < 2**max_generations:
            new_cells = []
            for cell in cells:
                if cell['gen'] < max_generations:
                    # Create 2 daughter cells for each division
                    angle1 = np.random.uniform(0, 2*np.pi)
                    angle2 = angle1 + np.pi + np.random.uniform(-0.5, 0.5)  # Slight variation
                    offset1 = growth_factor * np.array([np.cos(angle1), np.sin(angle1)])
                    offset2 = growth_factor * np.array([np.cos(angle2), np.sin(angle2)])
                    new_cells.append({'pos': cell['pos'] + offset1, 'gen': cell['gen']+1, 'type': 0})
                    new_cells.append({'pos': cell['pos'] + offset2, 'gen': cell['gen']+1, 'type': 0})
            cells.extend(new_cells)

            # Cell differentiation based on generation
            for c in cells:
                g = c['gen']
                if g > max_generations * differentiation_depth:
                    c['type'] = 3
                elif g > max_generations * 0.6:
                    c['type'] = 2
                elif g > max_generations * 0.3:
                    c['type'] = 1
                else:
                    c['type'] = 0

        # Random drift
        for c in cells:
            c['pos'] += np.random.normal(0, drift_strength, 2)

        # Shape morphing based on stage
        num_cells = len(cells)
        stage = get_stage(num_cells)
        
        if stage == "germinal":
            target = target_germinal
            alpha = morph_speed * 0.8  # Stronger morphing for germinal
        elif stage == "embryonic":
            target = target_embryo
            alpha = morph_speed * 1.2  # Strong morphing for embryonic
        else:
            target = target_fetus
            alpha = morph_speed * 1.5  # Strongest morphing for fetal

        # Apply morphing to cell positions with better target assignment
        for i, c in enumerate(cells):
            # Find closest target point for this cell
            if len(target) > 0:
                distances = np.linalg.norm(target - c['pos'], axis=1)
                closest_idx = np.argmin(distances)
                tpos = target[closest_idx]
                
                # Stronger morphing for cells that are far from target
                distance_factor = min(1.0, distances[closest_idx] / 5.0)
                effective_alpha = alpha * (0.5 + 0.5 * distance_factor)
                
                c['pos'] = (1 - effective_alpha) * c['pos'] + effective_alpha * tpos

        # Update visualization
        positions = np.array([c['pos'] for c in cells])
        color_list = [colors[c['type']] for c in cells]
        
        scatter.set_offsets(positions)
        scatter.set_color(color_list)
        scatter.set_sizes([150 - c['gen']*10 for c in cells])

        # Draw target shape outline
        if len(target) > 0:
            ax.plot(target[:, 0], target[:, 1], 'k--', alpha=0.3, linewidth=1)
            ax.plot(target[0, 0], target[0, 1], 'ko', markersize=3, alpha=0.5)

        stage_text.set_text(f"Stage: {stage.title()} ({num_cells} cells)")
        title.set_text(f"Zygote → Embryo → Baby Morphing | Cells: {num_cells} | Frame: {frame+1}/{frames}")
        
        # Add progress indicator
        progress = min(1.0, frame / frames)
        stage_progress = ""
        if stage == "germinal":
            stage_progress = f"Germinal: {min(100, int(100 * frame / (frames/3)))}%"
        elif stage == "embryonic":
            stage_progress = f"Embryonic: {min(100, int(100 * (frame - frames/3) / (frames/3)))}%"
        else:
            stage_progress = f"Fetal: {min(100, int(100 * (frame - 2*frames/3) / (frames/3)))}%"
        
        ax.text(-9, -8, stage_progress, fontsize=10, color='gray')

        # Stream current frame to Streamlit
        placeholder.pyplot(fig, clear_figure=False)
        time.sleep(0.05)

    plt.close(fig)


def main():
    st.set_page_config(
        page_title="Zygote-to-Baby Evolution Simulator (Morphing)", 
        layout="wide"
    )
    
    st.title("🧬 Interactive Zygote-to-Baby Evolution Simulator (Shape Morphing)")
    st.write("This upgraded simulation morphs the embryo shape through germinal → embryonic → fetal stages. You can adjust morphing speed, growth rate, and drift interactively.")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Interactive Controls")
        
        max_generations = st.slider("Max Generations", 3, 9, 6, 1)
        division_delay = st.slider("Division Delay (lower = faster)", 3, 15, 8, 1)
        growth_factor = st.slider("Growth Factor", 0.5, 1.5, 0.9, 0.1)
        drift_strength = st.slider("Cell Drift", 0.0, 0.05, 0.02, 0.005)
        morph_speed = st.slider("Morph Speed", 0.01, 0.1, 0.04, 0.01)
        differentiation_depth = st.slider("Differentiation Depth", 0.3, 0.9, 0.6, 0.05)
        frames = st.slider("Animation Frames", 60, 300, 220, 20)
        
        st.markdown("---")
        st.markdown("**Stage Information:**")
        st.markdown("- **Germinal**: < 20 cells (morula-like)")
        st.markdown("- **Embryonic**: 20-60 cells (C-shaped with limb buds)")
        st.markdown("- **Fetal**: > 60 cells (human-like fetus)")
        
        run_button = st.button("🚀 Run Simulation", type="primary")
    
    # Main simulation area
    if run_button or "_ran_once" not in st.session_state:
        st.session_state["_ran_once"] = True
        
        with st.spinner("Running morphing simulation..."):
            zygote_simulator_with_morphing(
                max_generations=max_generations,
                division_delay=division_delay,
                growth_factor=growth_factor,
                drift_strength=drift_strength,
                morph_speed=morph_speed,
                differentiation_depth=differentiation_depth,
                frames=frames
            )
    else:
        st.info("👆 Adjust parameters in the sidebar and click 'Run Simulation' to start!")
        
        # Show shape targets
        st.subheader("🎯 Target Developmental Shapes")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Germinal Stage (0-2 weeks)**")
            st.markdown("*Morula-like circular structure*")
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            target = generate_shape("germinal", 100)
            ax1.plot(target[:, 0], target[:, 1], 'b-', linewidth=3, alpha=0.8)
            ax1.fill(target[:, 0], target[:, 1], 'lightblue', alpha=0.3)
            ax1.set_xlim(-6, 6)
            ax1.set_ylim(-6, 6)
            ax1.set_aspect('equal')
            ax1.axis('off')
            ax1.set_title("Morula-like")
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col2:
            st.markdown("**Embryonic Stage (3-8 weeks)**")
            st.markdown("*C-shaped with limb buds*")
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            target = generate_shape("embryonic", 100)
            ax2.plot(target[:, 0], target[:, 1], 'g-', linewidth=3, alpha=0.8)
            ax2.fill(target[:, 0], target[:, 1], 'lightgreen', alpha=0.3)
            ax2.set_xlim(-6, 6)
            ax2.set_ylim(-6, 6)
            ax2.set_aspect('equal')
            ax2.axis('off')
            ax2.set_title("C-shaped Embryo")
            st.pyplot(fig2)
            plt.close(fig2)
        
        with col3:
            st.markdown("**Fetal Stage (9+ weeks)**")
            st.markdown("*Human-like with defined limbs*")
            fig3, ax3 = plt.subplots(figsize=(3, 3))
            target = generate_shape("fetal", 100)
            ax3.plot(target[:, 0], target[:, 1], 'r-', linewidth=3, alpha=0.8)
            ax3.fill(target[:, 0], target[:, 1], 'lightcoral', alpha=0.3)
            ax3.set_xlim(-6, 6)
            ax3.set_ylim(-6, 6)
            ax3.set_aspect('equal')
            ax3.axis('off')
            ax3.set_title("Human-like Fetus")
            st.pyplot(fig3)
            plt.close(fig3)


if __name__ == "__main__":
    main()


