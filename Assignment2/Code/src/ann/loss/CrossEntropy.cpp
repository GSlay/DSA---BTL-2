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
    const double EPSILON = 1e-7;
    int N_norm = (m_eReduction !=  REDUCE_MEAN) ? 1 : X.shape(0);
    double loss = 0.0;
    bool is_soft_label = t.dimension() == 2 and t.shape(1) > 1;

    // Lưu các biến cache cho forward
    m_aCached_Ypred = X;
    m_aYtarget = t;

    // Y: Dự đoán của mô hình, T: Nhãn thực tế (soft-label hoặc hard-label)
    xt::xarray<double> ce;
    // Tính toán Cross-Entropy cho từng mẫu
    if ((is_soft_label)) {
        ce = -t * xt::log(X + EPSILON);  // EPSILON tránh chia cho 0, đảm bảo không bị log(0)
        loss = xt::sum(ce)();
    }
    else {
        for (int i = 0; i < X.shape(0); ++i) {
            int class_index = t(i);  // Lớp đúng của mẫu thứ i
            loss -= std::log(X(i, class_index) + EPSILON);  // Log của xác suất lớp đúng
        }
    }
    // Tổng hoặc trung bình các giá trị Cross-Entropy (reduce_mean lấy trung bình)
    loss *= 1/N_norm;

    return loss;
}
xt::xarray<double> CrossEntropy::backward() {
    //YOUR CODE IS HERE
    const double EPSILON = 1e-7;
    int N_norm = (m_eReduction !=  REDUCE_MEAN) ? 1 : m_aCached_Ypred.shape(0);
    // Tính gradient của hàm Cross-Entropy theo công thức (28)
    xt::xarray<double> dY =  - 1/N_norm * (m_aYtarget / (m_aCached_Ypred + EPSILON));  // Đảm bảo không chia cho 0 nhờ EPSILON

    // Trả về gradient dY
    return dY;
}