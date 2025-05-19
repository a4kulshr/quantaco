#include <iostream>
#include <curl/curl.h>
#include "json.hpp"
#include <string>
using json = nlohmann::json;
// 252 trading days setting GBM path 
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t newLength = size * nmemb;
    s->append((char*)contents, newLength);
    return newLength;
}

int main() {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;
    std::string ticker = "AMT";
    std::string url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=F8Y8DSU6J7URM9JO";


    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        json j = json::parse(readBuffer);
        std::cout << j.dump(4) << std::endl;
    }

    return 0;
}

// set public class for stock


// Set GBM parameters and muh sigma and logs
//Plot trajectories check asscoaited features