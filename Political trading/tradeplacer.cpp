// set orderbook for stocks in metadata
// Set all inputs for the orderbook for stocks in metadata
// (Assuming a C++ structure for orderbook and metadata, and that you have access to a list/vector of stocks and their metadata)

#include <map>
#include <string>
#include <vector>
// set alphavantage api key an parse info into json
#include <json/json.h>
#include <curl/curl.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
// Set Alpha Vantage API key and provide a function to fetch stock data using Alpha Vantage

void symbol { std::META; 
    std; FLL; 
    std::NFLX; 
    std::NOW}

const std::string ALPHA_VANTAGE_API_KEY = "F8Y8DSU6J7URM9JO"; // <-- Replace with your real key

// Helper function for CURL write callback
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Fetch stock data from Alpha Vantage and parse JSON
Json::Value fetchStockDataAlphaVantage(const std::string& symbol, const std::string& function = "TIME_SERIES_DAILY") {
    std::string url = "https://www.alphavantage.co/query?function=" + function +
                      "&symbol=" + symbol +
                      "&apikey=" + ALPHA_VANTAGE_API_KEY +
                      "&outputsize=compact";
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }

    Json::Value jsonData;
    Json::CharReaderBuilder builder;
    std::string errs;
    std::istringstream s(readBuffer);
    if (!Json::parseFromStream(builder, s, &jsonData, &errs)) {
        std::cerr << "Failed to parse Alpha Vantage JSON: " << errs << std::endl;
    }
    return jsonData;
}



