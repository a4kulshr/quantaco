import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class MarkovChainSimulator:
    def __init__(self, n_states=10):
        self.n_states = n_states
        self.transition_matrix = None
        self.state_bounds = None
        self.state_means = None
        
    def get_historical_data(self):
        """Get historical data from our previous simulation"""
        # Using the data from our previous simulation
        prices = np.array([
            183.98, 182.577, 182.097, 183.847, 184.26, 185.774, 185.801, 184.758, 186.826, 184.93,
            183.98, 183.186, 183.239, 184.845, 184.922, 183.897, 183.612, 184.877, 185.5, 184.73,
            183.98, 184.544, 183.836, 184.635, 182.661, 181.569, 181.018, 180.391, 180.319, 180.501,
            183.98, 183.411, 182.97, 181.701, 183.586, 184.535, 182.837, 183.437, 182.379, 181.321,
            183.98, 183.101, 183.348, 183.108, 181.412, 182.836, 184.953, 186.326, 187.47, 186.651
        ])
        return prices
    
    def create_states(self, prices):
        """Create price states based on historical data"""
        min_price = np.min(prices)
        max_price = np.max(prices)
        self.state_bounds = np.linspace(min_price, max_price, self.n_states + 1)
        self.state_means = (self.state_bounds[:-1] + self.state_bounds[1:]) / 2
        
    def get_state(self, price):
        """Get the state index for a given price"""
        return np.searchsorted(self.state_bounds, price) - 1
    
    def build_transition_matrix(self, prices):
        """Build the transition matrix from historical data"""
        n = len(prices)
        transitions = np.zeros((self.n_states, self.n_states))
        
        for i in range(n-1):
            current_state = self.get_state(prices[i])
            next_state = self.get_state(prices[i+1])
            if current_state >= 0 and next_state >= 0:  # Ensure valid states
                transitions[current_state, next_state] += 1
        
        # Normalize rows to get probabilities
        row_sums = transitions.sum(axis=1)
        self.transition_matrix = np.divide(transitions, row_sums[:, np.newaxis],
                                         where=row_sums[:, np.newaxis] != 0)
        
    def simulate(self, initial_price, n_steps=100, n_paths=10):
        """Simulate multiple price paths using the Markov Chain"""
        initial_state = self.get_state(initial_price)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = initial_price
        
        for path in range(n_paths):
            current_state = initial_state
            for step in range(n_steps):
                # Get next state based on transition probabilities
                next_state = np.random.choice(self.n_states, 
                                            p=self.transition_matrix[current_state])
                # Use the mean price of the next state
                paths[path, step + 1] = self.state_means[next_state]
                current_state = next_state
                
        return paths
    
    def plot_simulation(self, paths, title="Markov Chain Simulation"):
        """Plot the simulated paths"""
        plt.figure(figsize=(12, 6))
        time_steps = np.arange(paths.shape[1])
        
        # Plot individual paths
        for i in range(paths.shape[0]):
            plt.plot(time_steps, paths[i], alpha=0.3, linewidth=1)
        
        # Plot mean path
        mean_path = np.mean(paths, axis=0)
        plt.plot(time_steps, mean_path, 'k--', linewidth=2, label='Mean Path')
        
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Stock Price ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('markov_simulation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot final price distribution
        plt.figure(figsize=(10, 6))
        final_prices = paths[:, -1]
        sns.histplot(final_prices, bins=30, kde=True)
        plt.title('Distribution of Final Prices (Markov Chain)')
        plt.xlabel('Final Price ($)')
        plt.ylabel('Frequency')
        plt.savefig('markov_final_prices.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Create simulator
    simulator = MarkovChainSimulator(n_states=10)
    
    # Get historical data
    print("Getting historical data...")
    prices = simulator.get_historical_data()
    
    # Create states and build transition matrix
    print("Building transition matrix...")
    simulator.create_states(prices)
    simulator.build_transition_matrix(prices)
    
    # Get initial price (most recent price)
    initial_price = prices[-1]
    print(f"Initial price: ${initial_price:.2f}")
    
    # Run simulation
    print("Running simulation...")
    paths = simulator.simulate(initial_price, n_steps=100, n_paths=50)
    
    # Plot results
    print("Generating plots...")
    simulator.plot_simulation(paths)
    
    # Print transition matrix
    print("\nTransition Matrix:")
    print(simulator.transition_matrix)
    
    print("\nSimulation complete! Check markov_simulation.png and markov_final_prices.png for results.")

if __name__ == "__main__":
    main() 