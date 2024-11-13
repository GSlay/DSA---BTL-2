/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this template
 */

/* 
 * File:   CrossEntropy.cpp
 * Author: ltsach
 * 
 * Created on August 25, 2024, 2:47 PM
 */

#include "loss/CrossEntropy.h"
#include "ann/functions.h"

CrossEntropy::CrossEntropy(LossReduction reduction): ILossLayer(reduction){
    
}

CrossEntropy::CrossEntropy(const CrossEntropy& orig):
ILossLayer(orig){
}

CrossEntropy::~CrossEntropy() {
}

double CrossEntropy::forward(xt::xarray<double> X, xt::xarray<double> t){
    //YOUR CODE IS HERE
    // Y: Dự đoán của mô hình, T: Nhãn thực tế (soft-label hoặc hard-label)
    
    // Tính toán Cross-Entropy cho từng mẫu
    xt::xarray<double> ce = -t * xt::log(X);  // EPSILON tránh chia cho 0, đảm bảo không bị log(0)

    // Tổng hoặc trung bình các giá trị Cross-Entropy (reduce_mean=true là trung bình, false là tổng)
    double loss = m_eReduction ? xt::sum(ce)() / X.shape(0) : xt::sum(ce)();

    return loss;
}
xt::xarray<double> CrossEntropy::backward() {
    //YOUR CODE IS HERE
    const int EPSILON = 10e-7;
    int N_norm = m_eReduction ? 1 : m_aCached_Ypred.shape(0);
    // Tính gradient của hàm Cross-Entropy theo công thức (28)
    xt::xarray<double> dY =  - 1/N_norm * (m_aYtarget / (m_aCached_Ypred + EPSILON));  // Đảm bảo không chia cho 0 nhờ EPSILON

    // Trả về gradient dY
    return dY;
}