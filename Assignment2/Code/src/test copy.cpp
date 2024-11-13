#include <xtensor/xarray.hpp>
#include <iostream>

int main() {
    xt::xarray<double> T_hard_onehot = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; // (N, K) one-hot -> hard-label
    xt::xarray<double> T_soft = {{0.8, 0.2, 0.0}, {0.0, 0.9, 0.1}, {0.3, 0.7, 0.0}}; // (N, K) soft-label
    
    // Lambda to check if each row is a hard label (one-hot)
    auto is_hard_label_onehot = [](const xt::xarray<double>& T) {
        return std::all_of(T.begin(), T.end(), [&](auto row) {
            int count_ones = 0;
            for (std::size_t i = 0; i < row.size(); ++i) {
                if (row[i] == 1) count_ones++;
                else if (row[i] != 0) return false;  // If there's a value other than 0 or 1
            }
            return count_ones == 1;  // Must have exactly one '1'
        });
    };

    std::cout << "T_hard_onehot is " 
              << (is_hard_label_onehot(T_hard_onehot) ? "hard-label (one-hot)" : "soft-label") 
              << std::endl;

    std::cout << "T_soft is " 
              << (is_hard_label_onehot(T_soft) ? "hard-label" : "soft-label") 
              << std::endl;

    return 0;
}