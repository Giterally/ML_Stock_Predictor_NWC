#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "TradeSimulator.h"

// Function to read a specific column from CSV
std::vector<int> readCSVColumn(const std::string& filename, int colIndex) {
    std::vector<int> result;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return result; // Empty vector on error
    }

    std::string line, val;
    // Skip header if present
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        for (int i = 0; i <= colIndex; ++i) {
            std::getline(ss, val, ',');
        }
        result.push_back(std::stoi(val)); // Assuming integer values in the CSV
    }

    file.close();
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <file1.csv> <file2.csv>" << std::endl;
        return 1;
    }

    // Load targets and predictions from both files
    std::vector<int> target1 = readCSVColumn(argv[1], 0);
    std::vector<int> predictions1 = readCSVColumn(argv[1], 1);
    std::vector<int> target2 = readCSVColumn(argv[2], 0);
    std::vector<int> predictions2 = readCSVColumn(argv[2], 1);

    // Create TradeSimulator instances for both datasets
    TradeSimulator simulator1(argv[1], 10000); // Initial balance $10,000 for dataset 1
    TradeSimulator simulator2(argv[2], 10000); // Initial balance $10,000 for dataset 2

    // Simulate trading for both datasets
    simulator1.simulate(target1, predictions1);
    simulator2.simulate(target2, predictions2);

    // Output results for both simulations
    std::cout << "Simulation 1 Results:" << std::endl;
    simulator1.printResults();
    std::cout << std::endl;

    std::cout << "Simulation 2 Results:" << std::endl;
    simulator2.printResults();
    std::cout << std::endl;

    return 0;
}
