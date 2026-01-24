#pragma once

#include "../tensor/tensor.hpp"
#include "../core/llaisys_core.hpp"
#include "llaisys/models/qwen2.h"
#include "../llaisys/llaisys_tensor.hpp"

#include <vector>
#include <memory>
namespace llaisys::models{
    class Modelqwen2{
    private :
    // 模型配置  
    LlaisysQwen2Meta meta_;  
      
    // 权重张量  
    std::unique_ptr<LlaisysQwen2Weights> weights_;  
      
    // KV-Cache管理  
    std::vector<tensor_t> key_cache_;  // [nlayer] 每层的key cache  
    std::vector<tensor_t> value_cache_; // [nlayer] 每层的value cache  
      
    // 运行时状态  
    llaisysDeviceType_t device_type_;  
    int device_id_;
    size_t seq_len_;        // 当前序列长度  
    
    // 运行时张量  
    tensor_t input_ids;  
    tensor_t positions;  
    tensor_t hidden_states;  
    tensor_t residual;  
    tensor_t norm_output;  
    tensor_t attn_output;  
    tensor_t mlp_output;
    tensor_t _q;  
    tensor_t _k;  
    tensor_t _v;  
    tensor_t gate;  
    tensor_t up;  
    tensor_t logits;

    void _allocate_weights();
    void _allocate_tensor();
    tensor_t _forward_layer(tensor_t hidden_states, size_t layer_idx, size_t seq_len, size_t start_pos);
    int64_t _sample_token(tensor_t logits);
    
    public:
    Modelqwen2(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device_type, int device_id);
    ~Modelqwen2();

    // 获取权重结构
    LlaisysQwen2Weights* weights() { return weights_.get(); }

    // 推理一个token
    int64_t infer(int64_t* token_ids, size_t ntoken);
    };

}