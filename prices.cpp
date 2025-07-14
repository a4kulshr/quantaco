#include <iostream>
#include <string>
#include <curl/curl.h>
#include "json.hpp"

using json = nlohmann::json;

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::string* s = static_cast<std::string*>(userp);
    size_t totalSize = size * nmemb;
    s->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

int main() {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;
    std::string ticker = "AMT";
    std::string apiKey = "F8Y8DSU6J7URM9JO";
    std::string url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + ticker + "&apikey=" + apiKey;

    std::cout << "Requesting URL: " << url << std::endl;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0");

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
        } else {
            try {
                json j = json::parse(readBuffer);
                std::cout << j.dump(4) << std::endl;
            } catch (const json::parse_error& e) {
                std::cerr << "JSON parse error: " << e.what() << std::endl;
                std::cerr << "Raw response: " << readBuffer << std::endl;
            }
        }

        curl_easy_cleanup(curl);
    } else {
        std::cerr << "Failed to initialize CURL" << std::endl;
    }

    curl_global_cleanup();
    return 0;
}

// choose high price if close price is below 200 otherwise choose close price use json parse to read the data

double getPriceBasedOnCondition(const json& j) {
    double selectedPrice = 0.0;
    if (j.contains("Time Series (Daily)")) {
        for (const auto& day : j["Time Series (Daily)"].items()) {
            const auto& dailyData = day.value();
            if (dailyData.contains("2. high") && dailyData.contains("4. close")) {
                double highPrice = std::stod(dailyData["2. high"].get<std::string>());
                double closePrice = std::stod(dailyData["4. close"].get<std::string>());
                if (closePrice < 200) {
                    selectedPrice = std::max(selectedPrice, highPrice);
                } else {
                    selectedPrice = std::max(selectedPrice, closePrice);
                }
            }
        }
    }
    return selectedPrice;
}