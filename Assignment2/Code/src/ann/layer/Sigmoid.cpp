/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this template
 */

/* 
 * File:   Sigmoid.cpp
 * Author: ltsach
 * 
 * Created on August 25, 2024, 2:45 PM
 */

#include "layer/Sigmoid.h"
#include "sformat/fmt_lib.h"
#include "ann/functions.h"

Sigmoid::Sigmoid(string name) {
    if(trim(name).size() != 0) m_sName = name;
    else m_sName = "Sigmoid_" + to_string(++m_unLayer_idx);
}

Sigmoid::Sigmoid(const Sigmoid& orig) {
    m_sName = "Sigmoid_" + to_string(++m_unLayer_idx);
}

Sigmoid::~Sigmoid() {
}
xt::xarray<double> Sigmoid::forward(xt::xarray<double> X) {
    //YOUR CODE IS HERE
    // Tính toán đầu ra Y của Sigmoid theo công thức Y = 1 / (1 + exp(-X))
    m_aCached_Y = 1.0 / (1.0 + xt::exp(-X));  // Áp dụng hàm Sigmoid

    return m_aCached_Y;

}
xt::xarray<double> Sigmoid::backward(xt::xarray<double> DY) {
    //YOUR CODE IS HERE
    xt::xarray<double> dX = DY * m_aCached_Y * (1.0 - m_aCached_Y);  // Áp dụng đạo hàm Sigmoid

    return dX;
}

string Sigmoid::get_desc(){
    string desc = fmt::format("{:<10s}, {:<15s}:",
                    "Sigmoid", this->getname());
    return desc;
}