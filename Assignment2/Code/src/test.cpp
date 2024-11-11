#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
using namespace std;

#include <list/DLinkedListDemo.h>
#include <list/XArrayListDemo.h>
#include <Test1.cpp>

int main(){
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
    DLinkedList<int> * a;
    a = new DLinkedList<int> [10];
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            a[i].add(j*i);
        }
    }
    
    // cout << a[5].get(3);
    a[0] = a[5];
    runAll();
}