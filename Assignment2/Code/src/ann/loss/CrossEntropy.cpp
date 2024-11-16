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
    const double EPSILON = 1e-17;
    int N_norm = (m_eReduction !=  REDUCE_MEAN) ? 1 : X.shape(0);
    // cout << m_eReduction << " " << N_norm << endl;
    double loss = 0.0;
    bool is_soft_label = false;
    for (size_t i = 0; i < t.shape(0); ++i) {
        // Lấy hàng thứ i trong ma trận T
        auto row = xt::view(t, i, xt::all());
        // Kiểm tra nếu tổng các phần tử trong hàng là 1 và tất cả phần tử là 0 hoặc 1
        if (std::abs(xt::sum(row)() - 1.0) > EPSILON || xt::any(row < 0.0 || row > 1.0)) {
            is_soft_label = true;
            break;
        }
    }

    bool is_binary_classification = true;  // Kiểm tra BCE cho phân loại hai lớp
    for (size_t i = 0; i < X.shape(0); ++i) {
        // Lấy hàng thứ i trong ma trận X
        auto row = xt::view(X, i, xt::all());
        // Kiểm tra nếu tổng các phần tử trong hàng bằng 1
        if (xt::sum(row)() == 1.0) {
            // cout << row << endl;
            is_binary_classification = false;
            break;
        }
    }

    // Lưu các biến cache cho forward
    m_aCached_Ypred = X;
    m_aYtarget = t;

    // Y: Dự đoán của mô hình, T: Nhãn thực tế (soft-label hoặc hard-label)
    xt::xarray<double> ce;
    // Tính toán Cross-Entropy cho từng mẫu
    if (is_binary_classification) {
        // Tính toán Binary Cross-Entropy
        // cout << 1 << endl;
        loss = -xt::sum(t * xt::log(X + EPSILON) + (1 - t) * xt::log(1 - X + EPSILON))();
    }
    else if (is_soft_label) {
        // cout << 2 << endl;
        // Duyệt qua các mẫu dữ liệu
        loss = -xt::sum(t * xt::log(X + EPSILON))();
    }
    else {
        // cout << 3 << endl;
        for (int i = 0;i < X.shape(0); i++) {
            auto class_index = xt::argmax(xt::view(t, i, xt::all()))();
            loss -= std::log(X(i, class_index) + EPSILON);
        }
    }
    // cout << "loss = " << loss << endl;
    // Tổng hoặc trung bình các giá trị Cross-Entropy (reduce_mean lấy trung bình)
    loss *= 1.0/N_norm;
    // cout << "loss_2 = " << loss << endl;

    return loss;
}
xt::xarray<double> CrossEntropy::backward() {
    //YOUR CODE IS HERE
    const double EPSILON = 1e-17;
    int N_norm = (m_eReduction !=  REDUCE_MEAN) ? 1 : m_aCached_Ypred.shape(0);

    xt::xarray<double> dY;

    if (m_aCached_Ypred.dimension() == 1) {
        // Tính gradient Binary Cross-Entropy
        dY = -1.0 / N_norm * (m_aYtarget / (m_aCached_Ypred + EPSILON) - (1 - m_aYtarget) / (1 - m_aCached_Ypred + EPSILON));
    }
    else {
        // Tính gradient của hàm Cross-Entropy theo công thức (28)
        dY =  - 1.0/N_norm * (m_aYtarget / (m_aCached_Ypred + EPSILON));  // Đảm bảo không chia cho 0 nhờ EPSILON
    }
    return dY;
}