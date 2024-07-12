#include "TradeSimulator.h"
#include <iostream>
#include <fstream>
#include <sstream>

TradeSimulator::TradeSimulator(const std::string& filename, double initial_balance)
    : filename(filename), initial_balance(initial_balance), finalBalance(0.0), totalTrades(0), successfulTrades(0), successRate(0.0) {}

void TradeSimulator::simulate(const std::vector<int>& targets, const std::vector<int>& predictions) {
    // Ensure vectors are of the same size
    if (targets.size() != predictions.size()) {
        std::cerr << "Error: Target and prediction vectors must be of the same size." << std::endl;
        return;
    }

    int numTrades = targets.size();
    int successfulTradesCount = 0;
    double balance = initial_balance;

    for (int i = 0; i < numTrades; ++i) {
        int target = targets[i];
        int prediction = predictions[i];

        // Assuming $100 per trade for simplicity
        balance -= 100.0;

        if (prediction == target) {
            balance += 200.0; // Profit $100 if prediction correct
            ++successfulTradesCount;
        }
    }

    finalBalance = balance;
    totalTrades = numTrades;
    successfulTrades = successfulTradesCount;

    // Calculate success rate
    if (totalTrades > 0) {
        successRate = static_cast<double>(successfulTrades) / totalTrades * 100.0;
    } else {
        successRate = 0.0;
    }
}

void TradeSimulator::printResults() const {
    std::cout << "Initial Balance: $" << initial_balance << std::endl;
    std::cout << "Final Balance: $" << finalBalance << std::endl;
    std::cout << "Total Trades: " << totalTrades << std::endl;
    std::cout << "Successful Trades: " << successfulTrades << std::endl;
    std::cout << "Success Rate: " << successRate << "%" << std::endl;
}
