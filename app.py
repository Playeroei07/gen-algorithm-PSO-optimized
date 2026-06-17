import streamlit as st
import pandas as pd
import time
import altair as alt
from genetic_algorithm import run_ga_experiment
from pso_tuner import (
    run_pso_tuner_generator, 
    BOUNDS, 
    snap_pop_size, 
    snap_mut_rate, 
    snap_crossover_prob
)

# Page config
st.set_page_config(
    page_title="PSO-Tuned Genetic Algorithm Visualizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for dashboard
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Sleek card container for metrics */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s, border-color 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    
    /* Header design elements */
    .dashboard-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    /* Customize Streamlit sidebar */
    .css-1d391kg {
        background-color: #0f172a;
    }
    
    /* Customize buttons */
    .stButton>button {
        background: linear-gradient(to right, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        padding: 12px 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #6366f1, #8b5cf6);
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
        color: white;
    }
    .stButton>button:active {
        transform: translateY(1px);
    }
    
    /* Style headers */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    
    /* Code container border styling */
    .stCodeBlock {
        border: 1px solid #1e293b;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title and Description
st.markdown('<h1 class="dashboard-title">🧬 PSO-Tuned Genetic Algorithm</h1>', unsafe_allow_html=True)
st.write("Optimize GA parameters (Population Size, Mutation Rate, Crossover Probability) using Particle Swarm Optimization (PSO) to find the target word in fewest generations.")

# ==========================================
# SIDEBAR - CONFIGURATIONS
# ==========================================
st.sidebar.markdown("### ⚙️ Target Settings")

# Target word input
target_word = st.sidebar.text_input("Target Word (Letters Only):", value="PRABOWO").upper()

# GA limitations
max_gens = st.sidebar.slider("GA Max Generations:", min_value=50, max_value=1000, value=500, step=50)
num_runs_per_eval = 3

# Optimal Swarm Dynamics parameters (hidden from UI to simplify user experience)
num_particles = 10
max_iterations = 12
w_inertia = 0.7
c1_cognitive = 2.0
c2_social = 1.0

# Input validation
run_allowed = True
if not target_word.isalpha():
    st.sidebar.error("⚠️ Target word must only contain letters (A-Z) without spaces, numbers, or symbols!")
    run_allowed = False

start_optimization = st.sidebar.button("Optimize hyperparameters 🚀")

# ==========================================
# MAIN PAGE - TUNING PROCESS
# ==========================================
if start_optimization and run_allowed:
    target_num = [ord(c) - 64 for c in target_word]
    
    st.subheader(f"Optimizing Hyperparameters for Target String: `{target_word}`")
    
    # Progress UI Components
    col_status, col_progress = st.columns([1, 4])
    with col_status:
        status_text = st.empty()
    with col_progress:
        progress_bar = st.progress(0.0)
        
    # Placeholders for live components (5 Metrics columns now to show Cache statistics)
    metric_cols = st.columns(5)
    best_pop_metric = metric_cols[0].empty()
    best_mut_metric = metric_cols[1].empty()
    best_cross_metric = metric_cols[2].empty()
    best_fit_metric = metric_cols[3].empty()
    cache_metric = metric_cols[4].empty()
    
    st.write("---")
    
    # 2D Plot and Table Layout
    plot_col, table_col = st.columns([3, 2])
    with plot_col:
        chart_placeholder = st.empty()
    with table_col:
        table_placeholder = st.empty()
        
    # Variables to track convergence history
    convergence_history = []
    
    # Start PSO Loop generator
    pso_gen = run_pso_tuner_generator(
        target_num=target_num,
        num_particles=num_particles,
        max_iterations=max_iterations,
        num_runs_per_eval=num_runs_per_eval,
        max_gens_limit=max_gens,
        w=w_inertia,
        c1=c1_cognitive,
        c2=c2_social
    )
    
    for state in pso_gen:
        it = state['iteration']
        particles_list = state['particles']
        gbest_pos = state['gbest_pos']
        gbest_fitness = state['gbest_fitness']
        cache_hits = state.get('cache_hits', 0)
        total_evals = state.get('total_evaluations', 1)
        
        # Log convergence
        convergence_history.append((it, gbest_fitness))
        
        # Update progress bar
        pct = it / max_iterations
        progress_bar.progress(min(pct, 1.0))
        status_text.markdown(f"**Iteration {it} / {max_iterations}**")
        
        # Snap parameters for display metrics
        best_pop_val = snap_pop_size(gbest_pos['pop_size'])
        best_mut_val = snap_mut_rate(gbest_pos['mut_rate'])
        best_cross_val = snap_crossover_prob(gbest_pos['crossover_prob'])
        
        # Update metrics
        best_pop_metric.metric("Best Pop Size (gbest)", f"{best_pop_val}")
        best_mut_metric.metric("Best Mutation Rate", f"{best_mut_val:.2f}")
        best_cross_metric.metric("Best Crossover Prob", f"{best_cross_val:.1f}")
        best_fit_metric.metric("Min Avg Generations", f"{gbest_fitness:.2f} gen")
        
        # Caching status metric
        hit_rate = (cache_hits / total_evals) * 100 if total_evals > 0 else 0.0
        cache_metric.metric("Cache Hit Rate", f"{hit_rate:.1f}%", f"{cache_hits} runs saved")
        
        # Create particles DataFrame
        df_particles = pd.DataFrame(particles_list)
        df_particles_numeric = df_particles.copy()
        
        # Filter for numeric columns
        df_particles_numeric['Population Size'] = df_particles_numeric['Population Size'].astype(float)
        df_particles_numeric['Mutation Rate'] = df_particles_numeric['Mutation Rate'].astype(float)
        df_particles_numeric['Crossover Prob'] = df_particles_numeric['Crossover Prob'].astype(float)
        
        # Create a layered Altair chart (Capped at 550 on the X-axis, 0.35 on the Y-axis)
        particles_chart = alt.Chart(df_particles_numeric).mark_circle(size=180, opacity=0.85).encode(
            x=alt.X('Population Size:Q', scale=alt.Scale(domain=[0, 550]), title='Population Size'),
            y=alt.Y('Mutation Rate:Q', scale=alt.Scale(domain=[0.0, 0.35]), title='Mutation Rate'),
            color=alt.Color('Fitness (Avg Gen):Q', scale=alt.Scale(scheme='viridis', reverse=True), title='Avg Gen (Lower is Better)'),
            size=alt.Size('Crossover Prob:Q', scale=alt.Scale(domain=[0.4, 1.1]), title='Crossover Prob'),
            tooltip=['Particle ID', 'Population Size', 'Mutation Rate', 'Crossover Prob', 'Fitness (Avg Gen)']
        )
        
        # 2. Glowing star marking the global best coordinate
        gbest_df = pd.DataFrame([{
            'Population Size': float(best_pop_val),
            'Mutation Rate': float(best_mut_val),
            'Crossover Prob': float(best_cross_val),
            'Fitness (Avg Gen)': float(round(gbest_fitness, 2))
        }])
        
        gbest_chart = alt.Chart(gbest_df).mark_point(
            size=400, shape='star', color='#f43f5e', filled=True
        ).encode(
            x='Population Size:Q',
            y='Mutation Rate:Q',
            tooltip=[
                alt.Tooltip('Population Size:Q', title='Best Pop Size'),
                alt.Tooltip('Mutation Rate:N', title='Best Mutation Rate'),
                alt.Tooltip('Crossover Prob:N', title='Best Crossover Prob'),
                alt.Tooltip('Fitness (Avg Gen):Q', title='Best Avg Gen')
            ]
        )
        
        layered_chart = (particles_chart + gbest_chart).properties(
            title=f"Swarm Search Space - Iteration {it}",
            height=400
        ).interactive()
        
        chart_placeholder.altair_chart(layered_chart, width='stretch')
        
        # Update table
        table_placeholder.write("### Particle Swarm Details")
        table_placeholder.dataframe(
            df_particles[['Particle ID', 'Population Size', 'Mutation Rate', 'Crossover Prob', 'Fitness (Avg Gen)']]
            .sort_values(by='Fitness (Avg Gen)', ascending=True),
            width='stretch'
        )
        
        # Sleep slightly for visual animation pacing
        time.sleep(0.1)
        
    st.success("🎉 **Swarm Optimization Complete!** PSO has converged to the optimal hyperparameter values.")
    
    # ------------------------------------------
    # CONVERGENCE CHART & POST-OPTIMIZATION ANALYSIS
    # ------------------------------------------
    st.write("## 📊 Optimization Analysis")
    col_anal1, col_anal2 = st.columns(2)
    
    with col_anal1:
        # Plot convergence curve
        df_conv = pd.DataFrame(convergence_history, columns=['Iteration', 'Best Generations'])
        conv_chart = alt.Chart(df_conv).mark_line(point=True, color='#818cf8').encode(
            x=alt.X('Iteration:O', title='PSO Iteration'),
            y=alt.Y('Best Generations:Q', title='Minimum Generations (Avg)'),
            tooltip=['Iteration', 'Best Generations']
        ).properties(
            title="PSO Convergence Curve (Fitness improvement over time)",
            height=300
        )
        st.altair_chart(conv_chart, width='stretch')
        
    with col_anal2:
        final_pop = snap_pop_size(gbest_pos['pop_size'])
        final_mut = snap_mut_rate(gbest_pos['mut_rate'])
        final_cross = snap_crossover_prob(gbest_pos['crossover_prob'])
        
        st.markdown(f"""
        ### 🏆 Swarm Best Configurations Found
        The particles have explored the parameter space and discovered that the following Genetic Algorithm hyperparameter values are optimal:
        
        * **Population Size**: `{final_pop}` (multiple of 50)
        * **Mutation Rate**: `{final_mut:.2f}` (multiple of 0.05)
        * **Crossover Probability**: `{final_cross:.1f}` (multiple of 0.1)
        * **Expected Search Effort**: `{gbest_fitness:.2f} generations`
        
        **Tuning Efficiency Metrics**:
        * **Total Evaluations**: `{total_evals}` GA configurations checked.
        * **Cache Hits**: `{cache_hits}` evaluations retrieved instantly from memory.
        * **GA runs saved**: `{cache_hits * num_runs_per_eval}` total runs avoided!
        """)

    # ------------------------------------------
    # GA VALIDATION RUN
    # ------------------------------------------
    st.write("## 📜 Final GA Run using Optimal Parameters")
    
    with st.spinner("Executing final run with optimal parameters..."):
        # Ensure snapped pop size for validation
        validated_pop_size = snap_pop_size(gbest_pos['pop_size'])
        validated_mut_rate = snap_mut_rate(gbest_pos['mut_rate'])
        validated_crossover_prob = snap_crossover_prob(gbest_pos['crossover_prob'])
        
        validation_gens, word_log = run_ga_experiment(
            pop_size=validated_pop_size,
            mut_rate=validated_mut_rate,
            target_num=target_num,
            crossover_probability=validated_crossover_prob,
            max_generations=max_gens
        )
        
    st.info(f"Target word `'{target_word}'` matched successfully in **{validation_gens}** generations!")
    
    with st.expander(f"View Full Evolution Log (Matched in {validation_gens} Generations)", expanded=True):
        log_text = "\n".join(word_log)
        st.code(log_text, language="text")
        
else:
    # Landing page state
    st.info("👈 Enter the target word and configure PSO in the sidebar, then click **Optimize hyperparameters 🚀** to start.")
