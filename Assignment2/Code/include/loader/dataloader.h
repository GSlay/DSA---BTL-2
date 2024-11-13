/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   dataloader.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 4:01 PM
 */

#ifndef DATALOADER_H
#define DATALOADER_H
#include "tensor/xtensor_lib.h"
#include "loader/dataset.h"

using namespace std;

template<typename DType, typename LType>
class DataLoader{
public:
    class Iterator; //forward declaration for class Iterator
    
private:
    Dataset<DType, LType>* ptr_dataset;
    int batch_size;
    bool shuffle;
    bool drop_last;
    int nbatch;
    ulong_tensor item_indices;
    int m_seed;
    
public:
    DataLoader(Dataset<DType, LType>* ptr_dataset, 
            int batch_size, bool shuffle=true, 
            bool drop_last=false, int seed=-1)
                : ptr_dataset(ptr_dataset), 
                batch_size(batch_size), 
                shuffle(shuffle),
                m_seed(seed){
            nbatch = ptr_dataset->len()/batch_size;
            item_indices = xt::arange(0, ptr_dataset->len());
    }
    virtual ~DataLoader(){}
    
    //New method: from V2: begin
    int get_batch_size(){ return batch_size; }
    int get_sample_count(){ return ptr_dataset->len(); }
    int get_total_batch(){return int(ptr_dataset->len()/batch_size); }
    
    //New method: from V2: end
    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// START: Section                                                     //
    /////////////////////////////////////////////////////////////////////////
public:
    Iterator begin(){
        //YOUR CODE IS HERE
        int dataset_size = ptr_dataset->len();
        if (dataset_size < batch_size) return Iterator(this, 0, 0);
        int batch_end = (drop_last and dataset_size % batch_size != 0) 
            ? (dataset_size / batch_size) * batch_size : dataset_size;
        return Iterator(this, 0, batch_end);
    }
    Iterator end(){
        //YOUR CODE IS HERE
        int dataset_size = ptr_dataset->len();
        if (dataset_size < batch_size) return Iterator(this, 0, 0);
        int batch_end = (drop_last and dataset_size % batch_size != 0) 
            ? (dataset_size / batch_size) * batch_size : dataset_size;
        return Iterator(this, batch_end, batch_end);
    }
    
    //BEGIN of Iterator

    //YOUR CODE IS HERE: to define iterator
        class Iterator{
    private:
        DataLoader *data_loader;
        int batch_index;//batch hiện tại đang load
        int end_index;//index cuối cùng của dataset
    public:
        Iterator(DataLoader *_data_loader, int _batch_index, int _end_index){
            data_loader = _data_loader;
            batch_index = _batch_index;
            end_index = _end_index;
        }
        // trả về batch hiện tại
        Batch<DType, LType> operator*(){
            int batch_end = (batch_index + data_loader->batch_size*2 > end_index) 
                ? end_index : batch_index + data_loader->batch_size;
            // cout << "b_i: " << batch_index << " b_e: " << batch_end << endl; //debug
            // if (batch_index >= end_index) throw_with_nested("nahhh"); //debug
            xt::xarray<DType> batch_data;
            xt::xarray<LType> batch_label;
            for (int i = batch_index; i < batch_end; i++){
                int index = data_loader->item_indices[i];
                auto data_label = data_loader->ptr_dataset->getitem(index);
                if (i == batch_index){
                    batch_data = xt::expand_dims(data_label.getData(), 0);
                    batch_label = xt::expand_dims(data_label.getLabel(), 0);
                } 
                else{
                    xt::xarray<DType> nbatch_data = xt::concatenate(xt::xtuple(batch_data, xt::expand_dims(data_label.getData(), 0) ));
                    xt::xarray<LType> nbatch_label;
                    if (data_loader->ptr_dataset->get_label_shape().size() > 0)
                        nbatch_label = xt::concatenate(xt::xtuple(batch_label, xt::expand_dims(data_label.getLabel(), 0) ));
                    else nbatch_label = 0;
                    batch_data = nbatch_data;
                    batch_label = nbatch_label;
                }
            }
            return Batch<DType, LType>(batch_data, batch_label);
        }

        bool operator!=(const Iterator &it){
            return (batch_index != it.batch_index);
        }
        //prefix overload ++
        Iterator &operator++(){
            if (batch_index + data_loader->batch_size*2 > end_index)
                batch_index = end_index;
            else batch_index += data_loader->batch_size;
            return *this;
        }
        //postfix overload ++
        Iterator operator++(int){
            Iterator it = *this;
            ++*this;
            return it;
        }

    };
    //END of Iterator
    
    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// END: Section                                                       //
    /////////////////////////////////////////////////////////////////////////
};


#endif /* DATALOADER_H */

