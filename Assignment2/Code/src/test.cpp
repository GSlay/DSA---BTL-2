#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
using namespace std;

#include <list/DLinkedListDemo.h>
#include <list/XArrayListDemo.h>
#include <Test2 - BTL 2.h>

int main(int argc, char* argv[]){
    // cout << "\ndlistDemo1\n";
    // xlistDemo1();
    // cout << "\n\ndlistDemo2\n";
    // dlistDemo2();
    // cout << "\n\nxlistDemo2\n";
    // xlistDemo2();
    // cout << "\n\ndlistDemo3\n";
    // dlistDemo3();
    // cout << "\n\nxlistDemo3\n";
    // xlistDemo3();
    // cout << "\n\ndlistDemo4\n";
    // dlistDemo4();
    // cout << "\n\nxlistDemo4\n";
    // xlistDemo4();
    // cout << "\n\ndlistDemo5\n";
    // dlistDemo5();
    // cout << "\n\ndlistDemo6\n";
    // dlistDemo6();
    // xt::random::seed(42);
    // DSFactory factory("./config.txt");
    // xmap<string, TensorDataset<double, double>*>* pMap = factory.get_datasets_2cc();
    // TensorDataset<double, double>* train_ds = pMap->get("train_ds");
    // TensorDataset<double, double>* valid_ds = pMap->get("valid_ds");
    // TensorDataset<double, double>* test_ds = pMap->get("test_ds");
    // DataLoader<double, double> train_loader(train_ds, 50, true, false);
    // DataLoader<double, double> valid_loader(valid_ds, 50, false, false);
    // DataLoader<double, double> test_loader(test_ds, 50, false, false);
        xt::xarray<double> A = xt::random::randn<double>({50, 2});  // Shape (50, 2)
    xt::xarray<double> B = xt::random::randn<double>({2, 50});  // Shape (2, 50)

    // Use tensordot with axes configured to produce shape (2, 50)
    xt::xarray<double> C = xt::linalg::tensordot(A, B, {}, {1});

    // Print the resulting shape
    std::cout << "Shape of C: " << shape2str(C.shape()) << std::endl;

    runAll();
    return 0;
}