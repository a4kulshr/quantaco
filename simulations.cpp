#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// GBM class for simulation
class GBM {
public:
    double mu;  // Drift
    double sigma;  // Volatility
    double S0;  // Initial stock price
    double T;  // Time to maturity
    int N;  // Number of time steps
    int M;  // Number of paths

    // Constructor
    GBM(double mu, double sigma, double S0, double T, int N, int M)
        : mu(mu), sigma(sigma), S0(S0), T(T), N(N), M(M) {}

    // Simulate GBM paths
    std::vector<std::vector<double>> simulate() {
        std::vector<std::vector<double>> paths(M, std::vector<double>(N));
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 1.0);
        double dt = T / N;

        for (int i = 0; i < M; ++i) {
            paths[i][0] = S0;
            for (int t = 1; t < N; ++t) {
                double Z = distribution(generator);
                paths[i][t] = paths[i][t - 1] * exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
            }
        }
        return paths;
    }
};

int main() {
    // Parameters for AMT stock
    double mu = -0.001514;  // Drift
    double sigma = 0.0190993;  // Volatility
    double S0 = 183.98;  // Initial price
    double T = 1.0;  // Time horizon (1 year)
    int N = 10;  // Number of time steps
    int M = 100;  // Number of paths

    // Run GBM simulation
    std::cout << "Time,Path,Price" << std::endl;  // CSV header
    GBM gbm(mu, sigma, S0, T, N, M);
    auto paths = gbm.simulate();

    // Print simulation paths in CSV format
    for (int i = 0; i < paths.size(); i++) {
        for (int t = 0; t < paths[i].size(); t++) {
            std::cout << t << "," << i + 1 << "," << paths[i][t] << std::endl;
        }
    }

    return 0;
}




