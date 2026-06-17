# gen-algorithm-string-matcher-enhanced
This project simulates the process of natural selection (Darwinian Theory) to match a target word, but elevates the search for optimal hyperparameters by replacing Grid Search with **Particle Swarm Optimization (PSO)**. 

PSO acts as a meta-optimizer, finding the best combination of **Population Size**, **Mutation Rate**, and **Crossover Probability** in a highly efficient manner.

---

## Key Enhancements
* **Swarm Intelligence Hyperparameter Tuning**: Instead of testing every combination on a grid, a swarm of particles flies through the parameter space, guided by cognitive and social forces to converge on the best settings.
* **Continuous Search Space**: Population size, mutation rate, and crossover probability are searched in continuous spaces, allowing for precise parameter values.
* **Stochastic Noise Reduction**: The tuner evaluates parameter configurations by running the GA multiple times and taking the average generations to target, ensuring robust parameter ranking.
* **Premium Interactive Visualizations**: Real-time 2D scattering plot (Altair) showing particles moving in search space, color-coded by their fitness, alongside a live star representing the Global Best (`gbest`).
* **Detailed Convergence Analysis**: A post-optimization chart of the best fitness across iterations to visualize swarm intelligence in action.

---

## Scientific Concept Overview

### 1. Genetic Algorithm (GA)
GA mimics Darwin's theory of natural selection:
* **Chromosomes (Genotype)**: Array of numbers $[1, 26]$ representing letters $[A, Z]$.
* **Fitness Calculation**: Abs difference sum subtracted from max possible difference.
* **Roulette Wheel Selection**: Higher fitness gives a higher probability of being chosen for reproduction.
* **Single-Point Crossover**: Recombines two parents at a random split.
* **Mutation**: Introduces random letter variations.
* **Survivor Selection ($\mu + \lambda$)**: Combine parents and children, sort, and retain the top $N$ individuals.

### 2. Particle Swarm Optimization (PSO)
PSO mimics the social behavior of bird flocking or fish schooling:
* **Position ($\mathbf{x}$)**: A vector representing a GA configuration: $[P_{\text{size}}, P_{\text{mut}}, P_{\text{cross}}]$.
* **Velocity ($\mathbf{v}$)**: The speed and direction of parameter search.
* **Personal Best ($\mathbf{pbest}$)**: The best parameter set found by a single particle.
* **Global Best ($\mathbf{gbest}$)**: The best parameter set found by any particle in the swarm.
* **Updating Equation**:
  $$v_{i, d}(t+1) = w \cdot v_{i, d}(t) + c_1 \cdot r_1 \cdot (pbest_{i, d} - x_{i, d}(t)) + c_2 \cdot r_2 \cdot (gbest_{d} - x_{i, d}(t))$$
  $$x_{i, d}(t+1) = x_{i, d}(t) + v_{i, d}(t)$$

---

## Running The Program

### Quick Windows Run
Double-click `autorun.bat` to automatically install dependencies and launch the Streamlit app.

### Manual Launch
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
