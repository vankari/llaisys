#include "model_qwen2.hpp"
#include "../ops/add/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/argmax/op.hpp"
#include <cmath>
namespace llaisys::models {
void Modelqwen2::_allocate_weights() {
    weights_ = std::make_unique<LlaisysQwen2Weights>();

    // 分配主要权重
    weights_->in_embed = nullptr;     // 输入embedding
    weights_->out_embed = nullptr;    // 输出embedding
    weights_->out_norm_w = nullptr;   // 最终层归一化

    // 分配每层权重数组
    weights_->attn_norm_w = new llaisysTensor_t[meta_.nlayer];
    weights_->attn_q_w = new llaisysTensor_t[meta_.nlayer];
    weights_->attn_q_b = new llaisysTensor_t[meta_.nlayer];
    weights_->attn_k_w = new llaisysTensor_t[meta_.nlayer];
    weights_->attn_k_b = new llaisysTensor_t[meta_.nlayer];
    weights_->attn_v_w = new llaisysTensor_t[meta_.nlayer];
    weights_->attn_v_b = new llaisysTensor_t[meta_.nlayer];
    weights_->attn_o_w = new llaisysTensor_t[meta_.nlayer];
    weights_->mlp_norm_w = new llaisysTensor_t[meta_.nlayer];
    weights_->mlp_gate_w = new llaisysTensor_t[meta_.nlayer];
    weights_->mlp_up_w = new llaisysTensor_t[meta_.nlayer];
    weights_->mlp_down_w = new llaisysTensor_t[meta_.nlayer];

    // 初始化为nullptr
    for (size_t i = 0; i < meta_.nlayer; i++) {
        weights_->attn_norm_w[i] = nullptr;
        weights_->attn_q_w[i] = nullptr;
        weights_->attn_q_b[i] = nullptr;
        weights_->attn_k_w[i] = nullptr;
        weights_->attn_k_b[i] = nullptr;
        weights_->attn_v_w[i] = nullptr;
        weights_->attn_v_b[i] = nullptr;
        weights_->attn_o_w[i] = nullptr;
        weights_->mlp_norm_w[i] = nullptr;
        weights_->mlp_gate_w[i] = nullptr;
        weights_->mlp_up_w[i] = nullptr;
        weights_->mlp_down_w[i] = nullptr;
    }
}
void Modelqwen2::_allocate_tensor() {
    using namespace llaisys;
    input_ids = Tensor::create({meta_.maxseq}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    positions = Tensor::create({meta_.maxseq}, LLAISYS_DTYPE_I64, device_type_, device_id_);

    // 隐藏状态张量 (2D)
    hidden_states = Tensor::create({meta_.maxseq, meta_.hs}, meta_.dtype, device_type_, device_id_);
    residual = Tensor::create({meta_.maxseq, meta_.hs}, meta_.dtype, device_type_, device_id_);
    norm_output = Tensor::create({meta_.maxseq, meta_.hs}, meta_.dtype, device_type_, device_id_);
    
    // Attention张量 (3D: [maxseq, nh, dh])
    _q = Tensor::create({meta_.maxseq, meta_.nh, meta_.dh}, meta_.dtype, device_type_, device_id_);
    attn_output = Tensor::create({meta_.maxseq, meta_.nh, meta_.dh}, meta_.dtype, device_type_, device_id_);

    // MLP张量 (2D: [maxseq, di])
    gate = Tensor::create({meta_.maxseq, meta_.di}, meta_.dtype, device_type_, device_id_);
    up = Tensor::create({meta_.maxseq, meta_.di}, meta_.dtype, device_type_, device_id_);
    mlp_output = Tensor::create({meta_.maxseq, meta_.di}, meta_.dtype, device_type_, device_id_);

    // 输出张量
    logits = Tensor::create({meta_.voc}, meta_.dtype, device_type_, device_id_);

    // KV缓存 (3D: [maxseq, nkvh, dh])
    key_cache_.resize(meta_.nlayer);
    value_cache_.resize(meta_.nlayer);
    for (size_t i = 0; i < meta_.nlayer; i++) {
        key_cache_[i] = Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, meta_.dtype, device_type_, device_id_);
        value_cache_[i] = Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, meta_.dtype, device_type_, device_id_);
    }
}
Modelqwen2::Modelqwen2(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device_type, int device_id)
    : meta_(*meta), device_type_(device_type), device_id_(device_id), seq_len_(0) {
    core::context().setDevice(device_type_, device_id_);
    _allocate_weights();
    _allocate_tensor();  
}
Modelqwen2::~Modelqwen2() {
    //由于weight类型没有实现析构函数，这里手动释放内存
    delete[] weights_->attn_norm_w;
    delete[] weights_->attn_q_w;
    delete[] weights_->attn_q_b;
    delete[] weights_->attn_k_w;
    delete[] weights_->attn_k_b;
    delete[] weights_->attn_v_w;
    delete[] weights_->attn_v_b;
    delete[] weights_->attn_o_w;
    delete[] weights_->mlp_norm_w;
    delete[] weights_->mlp_gate_w;
    delete[] weights_->mlp_up_w;
    delete[] weights_->mlp_down_w;
}

int64_t Modelqwen2::infer(int64_t* token_ids, size_t ntoken) {
    using namespace llaisys;
    core::context().setDevice(device_type_, device_id_);

    for (size_t i = 0; i < ntoken; i++) {
        // 加载输入 token
        auto input_id_slice = input_ids->slice(0, seq_len_, seq_len_ + 1);
        core::context().runtime().api()->memcpy_sync(
            input_id_slice->data(),
            &token_ids[i],
            sizeof(int64_t),
            LLAISYS_MEMCPY_H2D
        );

        // 设置位置 ID
        int64_t pos_id = static_cast<int64_t>(seq_len_);
        auto pos_slice = positions->slice(0, seq_len_, seq_len_ + 1);
        core::context().runtime().api()->memcpy_sync(
            pos_slice->data(),
            &pos_id,
            sizeof(int64_t),
            LLAISYS_MEMCPY_H2D
        );

        // Embedding 层
        auto embed_weights = weights_->in_embed ? ((LlaisysTensor*)weights_->in_embed)->tensor : nullptr;
        if (embed_weights) {
            auto hidden_slice = hidden_states->slice(0, seq_len_, seq_len_ + 1);
            ops::embedding(hidden_slice, input_id_slice, embed_weights);
        }

        // 所有 transformer 层
        for (size_t layer_idx = 0; layer_idx < meta_.nlayer; layer_idx++) {
            _forward_layer(layer_idx, seq_len_, 1);
        }

        seq_len_ += 1;
    }

    // 最终层归一化
    auto out_norm_w = weights_->out_norm_w ? ((LlaisysTensor*)weights_->out_norm_w)->tensor : nullptr;
    auto final_hidden = hidden_states->slice(0, seq_len_ - 1, seq_len_);  // [1, hs]
    auto normed_final = norm_output->slice(0, seq_len_ - 1, seq_len_);    // [1, hs]
    
    if (out_norm_w) {
        ops::rms_norm(normed_final, final_hidden, out_norm_w, meta_.epsilon);
    } else {
        ops::add(normed_final, final_hidden, final_hidden);
    }

    // 投影到 logits [1, voc]
    auto out_embed = weights_->out_embed ? ((LlaisysTensor*)weights_->out_embed)->tensor : nullptr;
    if (out_embed) {
        ops::linear(logits, normed_final, out_embed, nullptr);
    }
    
    int64_t sampled_token = _sample_token(logits);
    return sampled_token;
}

void Modelqwen2::_forward_layer(size_t layer_idx, size_t start_pos, size_t seq_len) {
    using namespace llaisys;
    
    auto hidden_slice = hidden_states->slice(0, start_pos, start_pos + seq_len);  // [seq_len, hs]
    auto pos_slice = positions->slice(0, start_pos, start_pos + seq_len);         // [seq_len]
    auto norm_out = norm_output->slice(0, start_pos, start_pos + seq_len);        // [seq_len, hs]

    // ===== Attention Block =====
    // Layer norm
    auto attn_norm_w = weights_->attn_norm_w[layer_idx] ? ((LlaisysTensor*)weights_->attn_norm_w[layer_idx])->tensor : nullptr;
    if (attn_norm_w) {
        ops::rms_norm(norm_out, hidden_slice, attn_norm_w, meta_.epsilon);
    } else {
        ops::add(norm_out, hidden_slice, hidden_slice);
    }

    // Q, K, V projections
    auto attn_q_w = weights_->attn_q_w[layer_idx] ? ((LlaisysTensor*)weights_->attn_q_w[layer_idx])->tensor : nullptr;
    auto attn_q_b = weights_->attn_q_b[layer_idx] ? ((LlaisysTensor*)weights_->attn_q_b[layer_idx])->tensor : nullptr;
    auto attn_k_w = weights_->attn_k_w[layer_idx] ? ((LlaisysTensor*)weights_->attn_k_w[layer_idx])->tensor : nullptr;
    auto attn_k_b = weights_->attn_k_b[layer_idx] ? ((LlaisysTensor*)weights_->attn_k_b[layer_idx])->tensor : nullptr;
    auto attn_v_w = weights_->attn_v_w[layer_idx] ? ((LlaisysTensor*)weights_->attn_v_w[layer_idx])->tensor : nullptr;
    auto attn_v_b = weights_->attn_v_b[layer_idx] ? ((LlaisysTensor*)weights_->attn_v_b[layer_idx])->tensor : nullptr;
    auto attn_o_w = weights_->attn_o_w[layer_idx] ? ((LlaisysTensor*)weights_->attn_o_w[layer_idx])->tensor : nullptr;

    auto q_slice = _q->slice(0, start_pos, start_pos + seq_len);  // [seq_len, nh, dh]
    if (attn_q_w) ops::linear(q_slice, norm_out, attn_q_w, attn_q_b);

    auto k_cache = key_cache_[layer_idx];
    auto k_cache_slice = k_cache->slice(0, start_pos, start_pos + seq_len);  // [seq_len, nkvh, dh]
    if (attn_k_w) ops::linear(k_cache_slice, norm_out, attn_k_w, attn_k_b);

    auto v_cache = value_cache_[layer_idx];
    auto v_cache_slice = v_cache->slice(0, start_pos, start_pos + seq_len);  // [seq_len, nkvh, dh]
    if (attn_v_w) ops::linear(v_cache_slice, norm_out, attn_v_w, attn_v_b);

    // Apply RoPE
    ops::rope(q_slice, q_slice, pos_slice, meta_.theta);
    ops::rope(k_cache_slice, k_cache_slice, pos_slice, meta_.theta);

    // Self-attention
    auto k_total = k_cache->slice(0, 0, start_pos + seq_len);  // [start_pos+seq_len, nkvh, dh]
    auto v_total = v_cache->slice(0, 0, start_pos + seq_len);  // [start_pos+seq_len, nkvh, dh]
    auto attn_out = attn_output->slice(0, start_pos, start_pos + seq_len);  // [seq_len, nh, dh]

    float scale = 1.0f / std::sqrt(static_cast<float>(meta_.dh));
    ops::self_attention(attn_out, q_slice, k_total, v_total, scale);

    // Attention output projection: [seq_len, nh, dh] -> [seq_len, hs]
    auto attn_out_hs = hidden_states->slice(0, start_pos, start_pos + seq_len);  // [seq_len, hs]
    if (attn_o_w) ops::linear(attn_out_hs, attn_out, attn_o_w, nullptr);

    // Residual: hidden = hidden + attn_out_proj
    ops::add(hidden_slice, hidden_slice, attn_out_hs);

    // ===== MLP Block =====
    auto mlp_norm_w = weights_->mlp_norm_w[layer_idx] ? ((LlaisysTensor*)weights_->mlp_norm_w[layer_idx])->tensor : nullptr;
    auto mlp_norm_out = norm_output->slice(0, start_pos, start_pos + seq_len);  // [seq_len, hs]
    if (mlp_norm_w) {
        ops::rms_norm(mlp_norm_out, hidden_slice, mlp_norm_w, meta_.epsilon);
    } else {
        ops::add(mlp_norm_out, hidden_slice, hidden_slice);
    }

    // Gate and Up projections
    auto mlp_gate_w = weights_->mlp_gate_w[layer_idx] ? ((LlaisysTensor*)weights_->mlp_gate_w[layer_idx])->tensor : nullptr;
    auto mlp_up_w = weights_->mlp_up_w[layer_idx] ? ((LlaisysTensor*)weights_->mlp_up_w[layer_idx])->tensor : nullptr;
    auto mlp_down_w = weights_->mlp_down_w[layer_idx] ? ((LlaisysTensor*)weights_->mlp_down_w[layer_idx])->tensor : nullptr;

    auto gate_slice = gate->slice(0, start_pos, start_pos + seq_len);  // [seq_len, di]
    auto up_slice = up->slice(0, start_pos, start_pos + seq_len);      // [seq_len, di]
    if (mlp_gate_w) ops::linear(gate_slice, mlp_norm_out, mlp_gate_w, nullptr);
    if (mlp_up_w) ops::linear(up_slice, mlp_norm_out, mlp_up_w, nullptr);

    // SwiGLU activation
    auto mlp_out = mlp_output->slice(0, start_pos, start_pos + seq_len);  // [seq_len, di]
    ops::swiglu(mlp_out, gate_slice, up_slice);

    // Down projection: [seq_len, di] -> [seq_len, hs]
    auto mlp_out_hs = hidden_states->slice(0, start_pos, start_pos + seq_len);  // [seq_len, hs]
    if (mlp_down_w) ops::linear(mlp_out_hs, mlp_out, mlp_down_w, nullptr);

    // Residual: hidden = hidden + mlp_out_proj
    ops::add(hidden_slice, hidden_slice, mlp_out_hs);
}

int64_t Modelqwen2::_sample_token(tensor_t logits) {
    using namespace llaisys;
    
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    auto max_val = Tensor::create({1}, meta_.dtype, device_type_, device_id_);
    
    ops::argmax(max_idx, max_val, logits);
    
    core::context().setDevice(device_type_, device_id_);
    core::context().runtime().api()->device_synchronize();
    
    int64_t token;
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        token = *reinterpret_cast<int64_t*>(max_idx->data());
    } else {
        auto host_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        core::context().runtime().api()->memcpy_sync(
            host_idx->data(),
            max_idx->data(),
            sizeof(int64_t),
            LLAISYS_MEMCPY_D2H
        );
        token = *reinterpret_cast<int64_t*>(host_idx->data());
    }
    
    return token;
}

}