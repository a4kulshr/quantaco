import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read historical prices from CSV
df = pd.read_csv('historical_prices.csv')
prices = df['Price'].values

# Calculate daily returns
returns = np.diff(prices) / prices[:-1]

def get_state(return_val):
    if return_val < -0.02:
        return 0  # Large negative return
    elif return_val < 0:
        return 1  # Small negative return
    elif return_val < 0.02:
        return 2  # Small positive return
    else:
        return 3  # Large positive return

# Create transition matrix
n_states = 4
transition_matrix = np.zeros((n_states, n_states))
state_counts = np.zeros(n_states)

# Calculate transition probabilities
for i in range(len(returns)-1):
    current_state = get_state(returns[i])
    next_state = get_state(returns[i+1])
    transition_matrix[current_state][next_state] += 1
    state_counts[current_state] += 1

# Normalize transition matrix
for i in range(n_states):
    if state_counts[i] > 0:
        transition_matrix[i] /= state_counts[i]

# Simulate future prices
n_simulations = 100
n_steps = 10
simulated_prices = np.zeros((n_simulations, n_steps + 1))
simulated_prices[:, 0] = prices[-1]  # Start from last known price

# Run simulations
for sim in range(n_simulations):
    current_price = prices[-1]
    current_state = get_state(returns[-1])
    
    for step in range(n_steps):
        # Get next state based on transition probabilities
        next_state = np.random.choice(n_states, p=transition_matrix[current_state])
        
        # Generate return based on state
        if next_state == 0:
            return_val = np.random.normal(-0.03, 0.01)  # Large negative
        elif next_state == 1:
            return_val = np.random.normal(-0.01, 0.005)  # Small negative
        elif next_state == 2:
            return_val = np.random.normal(0.01, 0.005)   # Small positive
        else:
            return_val = np.random.normal(0.03, 0.01)    # Large positive
            
        # Update price
        current_price *= (1 + return_val)
        simulated_prices[sim, step + 1] = current_price
        current_state = next_state

# --- Enhanced Plotting Section ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    "text.usetex": False,  # Use mathtext, not LaTeX
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22
})

fig, ax = plt.subplots(figsize=(15, 8))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#ffffff')

# Use a colormap for the simulation paths
cmap = plt.get_cmap('viridis')
colors = [cmap(i / n_simulations) for i in range(n_simulations)]

# Plot simulated paths with colors
for i in range(n_simulations):
    ax.plot(
        range(len(prices)-1, len(prices) + n_steps),
        simulated_prices[i],
        color=colors[i], alpha=0.5, linewidth=1
    )

# Plot mean path
mean_path = np.mean(simulated_prices, axis=0)
ax.plot(
    range(len(prices)-1, len(prices) + n_steps),
    mean_path,
    color="#222222", linestyle="--", linewidth=4, label=r"Mean Path"
)

# Formatting
ax.set_title(r"Markov Chain Simulation Paths for AMT Stock", pad=18)
ax.set_xlabel(r"Time Steps $t$", labelpad=12)
ax.set_ylabel(r"Stock Price $S_t$ (\$)", labelpad=12)
ax.grid(True, alpha=0.25)

# Legend
ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", shadow=True)

# Border styling
for spine in ax.spines.values():
    spine.set_color('#dcdcdc')

plt.tight_layout()
plt.savefig('markov_chain_simulation.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

# --- Histogram of final simulated prices (unchanged) ---
plt.figure(figsize=(10, 6))
final_prices = simulated_prices[:, -1]
sns.histplot(final_prices, bins=30, kde=True)
plt.title('Distribution of Final Simulated Prices')
plt.xlabel('Final Price ($)')
plt.ylabel('Frequency')
plt.savefig('markov_chain_final_prices.png', dpi=300, bbox_inches='tight')
plt.close()