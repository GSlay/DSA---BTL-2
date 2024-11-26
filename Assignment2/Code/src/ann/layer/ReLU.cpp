/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this template
 */

/* 
 * File:   ReLU.cpp
 * Author: ltsach
 * 
 * Created on August 25, 2024, 2:44 PM
 */

#include "layer/ReLU.h"
#include "sformat/fmt_lib.h"
#include "ann/functions.h"

ReLU::ReLU(string name) {
    if(trim(name).size() != 0) m_sName = name;
    else m_sName = "ReLU_" + to_string(++m_unLayer_idx);
}

ReLU::ReLU(const ReLU& orig) {
    m_sName = "ReLU_" + to_string(++m_unLayer_idx);
}

ReLU::~ReLU() {
}

xt::xarray<double> ReLU::forward(xt::xarray<double> X) {
    //YOUR CODE IS HERE
    m_aMask = xt::cast<double>(X >= 0);

    // Tính đầu ra Y = M ⊙ X (nhân phần tử tương ứng)
    xt::xarray<double> Y = m_aMask * X;

    return Y;
}
xt::xarray<double> ReLU::backward(xt::xarray<double> DY) {
    //YOUR CODE IS HERE
    // Đạo hàm của ReLU là 1 nếu X >= 0, 0 nếu X < 0. Tính đạo hàm đối với DY
    xt::xarray<double> dX = m_aMask * DY; // Nhân phần tử với mặt nạ M_cache
    return dX;
}

string ReLU::get_desc(){
    string desc = fmt::format("{:<10s}, {:<15s}:",
                    "ReLU", this->getname());
    return desc;
}