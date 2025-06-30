import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import norm

# --- Simulation Section (from markov_chain_sim.py) ---

# Read historical prices from CSV
try:
    df = pd.read_csv('historical_prices.csv')
    prices = df['Price'].values
except FileNotFoundError:
    print("historical_prices.csv not found. Using dummy data.")
    prices = np.cumprod(1 + np.random.randn(100) * 0.01) * 100

# Calculate daily returns
returns = np.diff(prices) / prices[:-1]

def get_state(return_val):
    if return_val < -0.015: return 0  # Volatile down
    elif return_val < 0: return 1      # Slight down
    elif return_val < 0.015: return 2   # Slight up
    else: return 3                     # Volatile up

# Create transition matrix
n_states = 4
transition_matrix = np.zeros((n_states, n_states))
state_counts = np.zeros(n_states)

# Calculate transition probabilities
for i in range(len(returns) - 1):
    current_state = get_state(returns[i])
    next_state = get_state(returns[i+1])
    transition_matrix[current_state][next_state] += 1
    state_counts[current_state] += 1

# Normalize transition matrix
for i in range(n_states):
    if state_counts[i] > 0:
        transition_matrix[i] /= state_counts[i]
    else:
        # If a state was never visited, assume equal probability of transition
        transition_matrix[i] = 1.0 / n_states

# Simulate future prices
n_simulations = 500
n_steps = 100
simulated_prices = np.zeros((n_simulations, n_steps + 1))
start_price = prices[-1]
simulated_prices[:, 0] = start_price

# Define return characteristics for each state (mu, sigma)
state_returns = {
    0: (-0.02, 0.02),
    1: (-0.005, 0.01),
    2: (0.005, 0.01),
    3: (0.02, 0.02)
}

# Run simulations
for sim in range(n_simulations):
    current_price = start_price
    # Start from the state of the last known return
    current_state = get_state(returns[-1])
    
    for step in range(n_steps):
        # Get next state based on transition probabilities
        next_state = np.random.choice(n_states, p=transition_matrix[current_state])
        
        # Generate return based on state
        mu, sigma = state_returns[next_state]
        return_val = np.random.normal(mu, sigma)
            
        # Update price
        current_price *= (1 + return_val)
        simulated_prices[sim, step + 1] = current_price
        current_state = next_state

# --- Professional Plotting Section ---

# Use a professional style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 22,
})

# Create figure and grid layout
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
ax_main = plt.subplot(gs[0])
ax_hist = plt.subplot(gs[1], sharey=ax_main)

# --- Main Plot: Simulation Paths ---

# Colormap for paths based on final price
final_prices = simulated_prices[:, -1]
cmap = plt.get_cmap('cividis')
norm_colors = plt.Normalize(vmin=final_prices.min(), vmax=final_prices.max())
colors = cmap(norm_colors(final_prices))

# Sort paths by final price for nice plotting order
sorted_indices = np.argsort(final_prices)

for i in sorted_indices:
    ax_main.plot(simulated_prices[i, :], color=colors[i], lw=0.7, alpha=0.6)

# Calculate and plot mean path and confidence interval
mean_path = np.mean(simulated_prices, axis=0)
std_dev = np.std(simulated_prices, axis=0)
ax_main.plot(mean_path, color='white', lw=2, ls='--', label=r'$E[S_t]$')

ax_main.fill_between(
    range(n_steps + 1),
    mean_path - std_dev,
    mean_path + std_dev,
    color='grey', alpha=0.3, zorder=0
)

# Main plot formatting
ax_main.set_title('Markov Chain Simulation', fontsize=20, pad=35, fontweight='bold')
fig.suptitle(r'$dS_t = \mu(S_t, t) dt + \sigma(S_t, t) dW_t$', fontsize=16, y=0.93)
ax_main.set_xlabel(r'Time Steps $t$', fontsize=14)
ax_main.set_ylabel(r'Stock Price $S_t$ (\$)', fontsize=14)
ax_main.legend(loc='upper left')
ax_main.tick_params(axis='y', labelleft=True)
ax_main.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- Side Plot: Final Price Distribution ---
n_bins = 30
hist, bin_edges = np.histogram(final_prices, bins=n_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Bar colors from colormap
bar_colors = cmap(norm_colors(bin_centers))

ax_hist.barh(bin_centers, hist, height=bin_edges[1]-bin_edges[0], color=bar_colors, edgecolor='white', alpha=0.8)
ax_hist.tick_params(axis='y', labelleft=False)
ax_hist.tick_params(axis='x', rotation=45)
ax_hist.set_xlabel(r'Frequency', fontsize=14)
ax_hist.set_title('Final Price Dist.', fontsize=16, pad=20, fontweight='bold')
ax_hist.grid(False)

# Overlay KDE plot
sns.kdeplot(y=final_prices, ax=ax_hist, color='white', lw=1.5)

# Mean line
mean_final_price = np.mean(final_prices)
ax_hist.axhline(mean_final_price, color='red', ls='--', lw=1.5, label=r'$E[S_T]$')
ax_hist.legend()

# Final adjustments
plt.tight_layout(rect=[0, 0, 1, 0.92]) # Adjust layout to make space for suptitle
plt.savefig('professional_markov_plot.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print("Professional plot 'professional_markov_plot.png' has been generated.") 