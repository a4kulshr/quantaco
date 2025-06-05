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

# Plotting
plt.figure(figsize=(12, 6))
plt.style.use('seaborn-v0_8')

# Plot historical prices
plt.plot(range(len(prices)), prices, 'b-', label='Historical Prices', linewidth=2)

# Plot simulated paths
for i in range(n_simulations):
    plt.plot(range(len(prices)-1, len(prices) + n_steps), 
             simulated_prices[i], 'r-', alpha=0.1)

# Calculate and plot mean path
mean_path = np.mean(simulated_prices, axis=0)
plt.plot(range(len(prices)-1, len(prices) + n_steps), 
         mean_path, 'g--', label='Mean Simulated Path', linewidth=2)

plt.title('Markov Chain Price Simulation for AMT Stock')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('markov_chain_simulation.png', dpi=300, bbox_inches='tight')
plt.close()

# Create histogram of final simulated prices
plt.figure(figsize=(10, 6))
final_prices = simulated_prices[:, -1]
sns.histplot(final_prices, bins=30, kde=True)
plt.title('Distribution of Final Simulated Prices')
plt.xlabel('Final Price ($)')
plt.ylabel('Frequency')
plt.savefig('markov_chain_final_prices.png', dpi=300, bbox_inches='tight')
plt.close()