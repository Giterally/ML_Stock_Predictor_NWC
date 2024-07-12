#ifndef TRADE_SIMULATOR_H
#define TRADE_SIMULATOR_H

#include <string>
#include <vector>

class TradeSimulator {
public:
    TradeSimulator(const std::string& filename, double initial_balance);

    void simulate(const std::vector<int>& targets, const std::vector<int>& predictions);
    void printResults() const;

private:
    std::string filename;
    double initial_balance;
    double finalBalance;
    int totalTrades;
    int successfulTrades;
    double successRate;
};

#endif // TRADE_SIMULATOR_H
