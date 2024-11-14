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
    cout << 1 << endl;
    xt::xarray<double> exp_X = xt::exp(X - xt::expand_dims(xt::amax(X, 1), 1));
// {1} là axis=1 (theo hàng)
    cout << 2 << endl;
    xt::xarray<double> sum_exp_X = xt::expand_dims(xt::sum(exp_X, 1), 1);  // Tính tổng exp(X) theo mỗi hàng
    cout << 3 << endl;
    m_aCached_Y = exp_X / sum_exp_X;  // Chuẩn hóa các giá trị
    cout << 4 << endl;

    return m_aCached_Y;
}
xt::xarray<double> Softmax::backward(xt::xarray<double> DY) {
    //YOUR CODE IS HERE
    // Tính toán đạo hàm của Softmax
    // Compute diagonal matrix of softmax output
    auto diag_y = xt::diag(m_aCached_Y);

    // Compute the outer product of the softmax output vector
    auto y_outer = xt::linalg::outer(m_aCached_Y, m_aCached_Y);

    // Jacobian matrix: DIAG(y) - y ⊗ y^T
    auto jacobian = diag_y - y_outer;

    // Multiply the Jacobian by delta_y to get delta_z
    xt::xarray<double> delta_z = xt::linalg::dot(jacobian, DY);
    return delta_z;
}

string Softmax::get_desc(){
    string desc = fmt::format("{:<10s}, {:<15s}: {:4d}",
                    "Softmax", this->getname(), m_nAxis);
    return desc;
}
