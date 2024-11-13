/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this template
 */

/* 
 * File:   Softmax.cpp
 * Author: ltsach
 * 
 * Created on August 25, 2024, 2:46 PM
 */

#include "layer/Softmax.h"
#include "ann/functions.h"
#include "sformat/fmt_lib.h"
#include <filesystem> //require C++17
namespace fs = std::filesystem;

Softmax::Softmax(int axis, string name): m_nAxis(axis) {
    if(trim(name).size() != 0) m_sName = name;
    else m_sName = "Softmax_" + to_string(++m_unLayer_idx);
}

Softmax::Softmax(const Softmax& orig) {
}

Softmax::~Softmax() {
}

xt::xarray<double> Softmax::forward(xt::xarray<double> X) {
    //YOUR CODE IS HERE
    // Tính toán softmax cho mỗi hàng trong X
    // Đầu tiên, trừ giá trị lớn nhất trong mỗi hàng để tránh vấn đề tràn số học
    xt::xarray<double> exp_X = xt::exp(X - xt::amax(X, 1));  // {1} là axis=1 (theo hàng)
    xt::xarray<double> sum_exp_X = xt::sum(exp_X, 1);  // Tính tổng exp(X) theo mỗi hàng
    m_aCached_Y = exp_X / sum_exp_X;  // Chuẩn hóa các giá trị

    return m_aCached_Y;
}
xt::xarray<double> Softmax::backward(xt::xarray<double> DY) {
    //YOUR CODE IS HERE
    // Tính toán đạo hàm của Softmax
    xt::xarray<double> dX = DY * (xt::diag(m_aCached_Y) - (m_aCached_Y * xt::transpose(m_aCached_Y)));

    return dX;
}

string Softmax::get_desc(){
    string desc = fmt::format("{:<10s}, {:<15s}: {:4d}",
                    "Softmax", this->getname(), m_nAxis);
    return desc;
}
