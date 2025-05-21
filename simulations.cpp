#include <iostream>
#include <string>
#include <curl/curl.h>
#include "json.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using json = nlohmann::json;

// Callback function for libcurl
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t totalSize = size * nmemb;
    s->append((char*)contents, totalSize);
    return totalSize;
}

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

// log returns
std::vector<double> logreturns(const std::vector<std::vector<double>>& paths) {
    std::vector<double> logReturns;
    for (const auto& path : paths) {
        for (size_t i = 1; i < path.size(); ++i) {
            double logReturn = log(path[i] / path[i - 1]);
            logReturns.push_back(logReturn);
        }
    }
    return logReturns;
}

//  mu
double calculatemean(const std::vector<double>& logReturns) {
    double sum = 0.0;
    for (const auto& logReturn : logReturns) {
        sum += logReturn;
    }
    return sum / logReturns.size();
}

// sigma
double calculatestandardeviation(const std::vector<double>& logReturns, double mean) {
    double sum = 0.0;
    for (const auto& logReturn : logReturns) {
        sum += (logReturn - mean) * (logReturn - mean);
    }
    return sqrt(sum / (logReturns.size() - 1));
}

int main() {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;
    std::string ticker = "AMT";
    std::string apiKey = "F8Y8DSU6J7URM9JO";
    std::string url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + ticker + "&apikey=" + apiKey;

    std::cout << "Requesting URL: " << url << std::endl;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0");

        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
            return 1;
        }

        try {
            json j = json::parse(readBuffer);

            // Print the parsed JSON object 
            std::cout << "Parsed JSON Response: " << j.dump(4) << std::endl;

            auto prices = j["Time Series (Daily)"];
            if (!prices.empty()) {
                auto latestEntry = prices.begin();
                std::string date = latestEntry.key();
                std::string closePrice = latestEntry.value()["4. close"];
                std::cout << "Latest date: " << date << std::endl;
                std::cout << "Latest close price: " << closePrice << std::endl;

                // Reverse price to chronological order
                std::vector<double> pricesVector;
                for (auto it = prices.rbegin(); it != prices.rend(); ++it) {
                    double price = std::stod(static_cast<std::string>(it.value()["4. close"]));
                    pricesVector.push_back(price);
                }

                // Calculate log returns, mu, and sigma
                std::vector<std::vector<double>> wrappedPricesVector = {pricesVector};
                auto logReturns = logreturns(wrappedPricesVector);
                double mu = calculatemean(logReturns);
                double sigma = calculatestandardeviation(logReturns, mu);

                // Get latest price
                double latestPrice = std::stod(closePrice);

                std::cout << "Mu (Drift): " << mu << std::endl;
                std::cout << "Sigma (Volatility): " << sigma << std::endl;
                std::cout << "Latest price: " << latestPrice << std::endl;

                // Run GBM simulation
                std::cout << "-------- Starting GBM Simulation --------" << std::endl;
                GBM gbm(mu, sigma, latestPrice, 1.0, 10, 100);
                auto paths = gbm.simulate();

                // Print simulation paths
                for (int i = 0; i < paths.size(); i++) {
                    std::cout << "Path " << i + 1 << ": ";
                    for (double price : paths[i]) {
                        std::cout << price << " ";
                    }
                    std::cout << std::endl;
                }
            } else {
                std::cerr << "No data found for the given ticker." << std::endl;
            }
        } catch (json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            std::cerr << "Raw response: " << readBuffer << std::endl;
        } catch (std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Failed to initialize CURL." << std::endl;
    }

    return 0;
}

class orderbook {
 public:
    std::string ticker;
    double price;
    int quantity;
    std::string side;

    orderbook(std::string ticker, double price, int quantity, std::string side)
        : ticker(ticker), price(price), quantity(quantity), side(side) {}
}


PYBIND11_MODULE(GBM, m) {
    m.doc() = "Python bindings for GBM simulation and order book";
    py::class_<GBM>(m, "GBM")
        .def(py::init<double, double, double, double, int, int>())
        .def("simulate", &GBM::simulate);

    m.def("logreturns", &logreturns);
    m.def("calculatemean", &calculatemean);
    m.def("calculatestandardeviation", &calculatestandardeviation);
}




