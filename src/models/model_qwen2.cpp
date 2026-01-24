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

    // 隐藏状态张量
    hidden_states = Tensor::create({meta_.maxseq, meta_.hs}, meta_.dtype, device_type_, device_id_);
    residual = Tensor::create({meta_.maxseq, meta_.hs}, meta_.dtype, device_type_, device_id_);
    norm_output = Tensor::create({meta_.maxseq, meta_.hs}, meta_.dtype, device_type_, device_id_);
    attn_output = Tensor::create({meta_.maxseq, meta_.hs}, meta_.dtype, device_type_, device_id_);
    mlp_output = Tensor::create({meta_.maxseq, meta_.hs}, meta_.dtype, device_type_, device_id_);

    // Attention张量
    _q = Tensor::create({meta_.maxseq, meta_.nh, meta_.dh}, meta_.dtype, device_type_, device_id_);
    _k = Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, meta_.dtype, device_type_, device_id_);
    _v = Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, meta_.dtype, device_type_, device_id_);

    // MLP张量
    gate = Tensor::create({meta_.maxseq, meta_.di}, meta_.dtype, device_type_, device_id_);
    up = Tensor::create({meta_.maxseq, meta_.di}, meta_.dtype, device_type_, device_id_);

    // 输出张量
    logits = Tensor::create({meta_.voc}, meta_.dtype, device_type_, device_id_);

    // KV缓存
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
tensor_t Modelqwen2::_forward_layer(tensor_t hidden_states, size_t layer_idx, size_t seq_len, size_t start_pos) {
    using namespace llaisys;
    // residual connection (in-place updates to hidden_states)
    // 1. Attention block
    // layer norm
    auto hidden_slice = hidden_states; // expected shape [seq_len, hs]
    auto attn_norm_w = weights_->attn_norm_w[layer_idx] ? ((LlaisysTensor*)weights_->attn_norm_w[layer_idx])->tensor : nullptr;
    auto attn_q_w = weights_->attn_q_w[layer_idx] ? ((LlaisysTensor*)weights_->attn_q_w[layer_idx])->tensor : nullptr;
    auto attn_q_b = weights_->attn_q_b[layer_idx] ? ((LlaisysTensor*)weights_->attn_q_b[layer_idx])->tensor : nullptr;
    auto attn_k_w = weights_->attn_k_w[layer_idx] ? ((LlaisysTensor*)weights_->attn_k_w[layer_idx])->tensor : nullptr;
    auto attn_k_b = weights_->attn_k_b[layer_idx] ? ((LlaisysTensor*)weights_->attn_k_b[layer_idx])->tensor : nullptr;
    auto attn_v_w = weights_->attn_v_w[layer_idx] ? ((LlaisysTensor*)weights_->attn_v_w[layer_idx])->tensor : nullptr;
    auto attn_v_b = weights_->attn_v_b[layer_idx] ? ((LlaisysTensor*)weights_->attn_v_b[layer_idx])->tensor : nullptr;
    auto attn_o_w = weights_->attn_o_w[layer_idx] ? ((LlaisysTensor*)weights_->attn_o_w[layer_idx])->tensor : nullptr;

    auto mlp_norm_w = weights_->mlp_norm_w[layer_idx] ? ((LlaisysTensor*)weights_->mlp_norm_w[layer_idx])->tensor : nullptr;
    auto mlp_gate_w = weights_->mlp_gate_w[layer_idx] ? ((LlaisysTensor*)weights_->mlp_gate_w[layer_idx])->tensor : nullptr;
    auto mlp_up_w = weights_->mlp_up_w[layer_idx] ? ((LlaisysTensor*)weights_->mlp_up_w[layer_idx])->tensor : nullptr;
    auto mlp_down_w = weights_->mlp_down_w[layer_idx] ? ((LlaisysTensor*)weights_->mlp_down_w[layer_idx])->tensor : nullptr;

    // slices/views for current sequence range
    auto hs_slice = hidden_slice; // [seq_len, hs]
    auto pos_slice = positions->slice(0, start_pos, start_pos + seq_len);

    // norm output
    auto norm_out = norm_output->slice(0, start_pos, start_pos + seq_len);
    if (attn_norm_w) {
        ops::rms_norm(norm_out, hs_slice, attn_norm_w, meta_.epsilon);
    } else {
        // if no norm weight provided, copy input to norm_out
        ops::add(norm_out, hs_slice, hs_slice);
    }

    // Q, K, V linear projections
    // Q -> write into _q at [start_pos : start_pos+seq_len]
    auto q_slice_3d = _q->slice(0, start_pos, start_pos + seq_len); // [seq_len, nh, dh]
    auto q_slice_2d = q_slice_3d->view({seq_len, meta_.nh * meta_.dh});
    auto norm_view_2d = norm_out->view({seq_len, meta_.hs});
    if (attn_q_w) ops::linear(q_slice_2d, norm_view_2d, attn_q_w, attn_q_b);

    // K -> write into key cache
    auto k_cache = key_cache_[layer_idx];
    auto k_target_3d = k_cache->slice(0, start_pos, start_pos + seq_len); // [seq_len, nkvh, dh]
    auto k_target_2d = k_target_3d->view({seq_len, meta_.nkvh * meta_.dh});
    if (attn_k_w) ops::linear(k_target_2d, norm_view_2d, attn_k_w, attn_k_b);

    // V -> write into value cache
    auto v_cache = value_cache_[layer_idx];
    auto v_target_3d = v_cache->slice(0, start_pos, start_pos + seq_len);
    auto v_target_2d = v_target_3d->view({seq_len, meta_.nkvh * meta_.dh});
    if (attn_v_w) ops::linear(v_target_2d, norm_view_2d, attn_v_w, attn_v_b);

    // Apply RoPE to Q and the portion of K we just wrote
    ops::rope(q_slice_3d, q_slice_3d, pos_slice, meta_.theta);
    ops::rope(k_target_3d, k_target_3d, pos_slice, meta_.theta);

    // Prepare attention inputs: full key/value up to start_pos+seq_len
    auto k_total_3d = k_cache->slice(0, 0, start_pos + seq_len);
    auto v_total_3d = v_cache->slice(0, 0, start_pos + seq_len);

    // attention output -> write into attn_output slice
    auto attn_out_3d = attn_output->slice(0, start_pos, start_pos + seq_len)->view({seq_len, meta_.nh, meta_.dh});

    float scale = 1.0f / std::sqrt(static_cast<float>(meta_.dh));
    ops::self_attention(attn_out_3d, q_slice_3d, k_total_3d, v_total_3d, scale);

    // Project attention output back to hidden size
    auto attn_out_2d = attn_output->slice(0, start_pos, start_pos + seq_len)->view({seq_len, meta_.hs});
    auto attn_val_2d = attn_out_3d->view({seq_len, meta_.nh * meta_.dh});
    if (attn_o_w) ops::linear(attn_out_2d, attn_val_2d, attn_o_w, nullptr);

    // Residual add: hidden = hidden + attn_out
    ops::add(hs_slice, hs_slice, attn_out_2d);

    // 2. MLP block
    auto mlp_norm_out = norm_output->slice(0, start_pos, start_pos + seq_len);
    if (mlp_norm_w) {
        ops::rms_norm(mlp_norm_out, hs_slice, mlp_norm_w, meta_.epsilon);
    } else {
        ops::add(mlp_norm_out, hs_slice, hs_slice);
    }

    // Gate and Up projections
    auto gate_slice_2d = gate->slice(0, start_pos, start_pos + seq_len)->view({seq_len, meta_.di});
    auto up_slice_2d = up->slice(0, start_pos, start_pos + seq_len)->view({seq_len, meta_.di});
    auto mlp_norm_view_2d = mlp_norm_out->view({seq_len, meta_.hs});
    if (mlp_gate_w) ops::linear(gate_slice_2d, mlp_norm_view_2d, mlp_gate_w, nullptr);
    if (mlp_up_w) ops::linear(up_slice_2d, mlp_norm_view_2d, mlp_up_w, nullptr);

    // Swiglu activation -> mlp_output (shape [seq_len, hs])
    auto mlp_out_2d = mlp_output->slice(0, start_pos, start_pos + seq_len)->view({seq_len, meta_.hs});
    auto gate_3d = gate->slice(0, start_pos, start_pos + seq_len)->view({seq_len, meta_.di});
    auto up_3d = up->slice(0, start_pos, start_pos + seq_len)->view({seq_len, meta_.di});
    ops::swiglu(mlp_out_2d, gate_3d, up_3d);

    // Down projection
    if (mlp_down_w) ops::linear(mlp_out_2d, mlp_out_2d, mlp_down_w, nullptr);

    // Residual add: hidden = hidden + mlp_out
    ops::add(hs_slice, hs_slice, mlp_out_2d);

    return hidden_states;
}
int64_t Modelqwen2::_sample_token(tensor_t logits) {
    using namespace llaisys;
    
    // Create temporary tensors for argmax output
    // argmax returns indices and values
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    auto max_val = Tensor::create({1}, meta_.dtype, device_type_, device_id_);
    
    // logits shape: [voc]
    // Reshape to [1, voc] for argmax compatibility (if needed)
    auto logits_2d = logits->view({1, meta_.voc});
    
    // Call argmax to find the token with highest logit
    // argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals)
    ops::argmax(max_idx, max_val, logits_2d);
    
    // Extract the index value from device memory
    core::context().setDevice(device_type_, device_id_);
    core::context().runtime().api()->device_synchronize();
    
    // Copy result to host if needed
    int64_t token;
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        token = *reinterpret_cast<int64_t*>(max_idx->data());
    } else {
        // Allocate host buffer and copy
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
int64_t Modelqwen2::infer(int64_t* token_ids, size_t ntoken) {
    using namespace llaisys;
    core::context().setDevice(device_type_, device_id_);

    for (size_t i = 0; i < ntoken; i++) {
        // Load current token id into input_ids tensor at position seq_len_
        auto input_id_slice = input_ids->slice(0, seq_len_, seq_len_ + 1);
        core::context().runtime().api()->memcpy_sync(
            input_id_slice->data(),
            &token_ids[i],
            sizeof(int64_t),
            LLAISYS_MEMCPY_H2D
        );

        // Set position id
        int64_t pos_id = static_cast<int64_t>(seq_len_);
        auto pos_slice = positions->slice(0, seq_len_, seq_len_ + 1);
        core::context().runtime().api()->memcpy_sync(
            pos_slice->data(),
            &pos_id,
            sizeof(int64_t),
            LLAISYS_MEMCPY_H2D
        );

        // Embedding lookup
        auto embed_weights = weights_->in_embed ? ((LlaisysTensor*)weights_->in_embed)->tensor : nullptr;
        if (embed_weights) {
            auto hidden_slice = hidden_states->slice(0, seq_len_, seq_len_ + 1);
            ops::embedding(hidden_slice, input_id_slice, embed_weights);
        }

        // Forward through each layer
        hidden_states = _forward_layer(hidden_states, 0, 1, seq_len_); // process one token at a time

        // Increment sequence length
        seq_len_ += 1;
    }

    // Final layer norm
    auto out_norm_w = weights_->out_norm_w ? ((LlaisysTensor*)weights_->out_norm_w)->tensor : nullptr;
    auto final_hidden = hidden_states->slice(0, seq_len_ - 1, seq_len_); // last token hidden state
    auto normed_output = norm_output->slice(0, seq_len_ - 1, seq_len_);
    if (out_norm_w) {
        ops::rms_norm(normed_output, final_hidden, out_norm_w, meta_.epsilon);
    } else {
        ops::add(normed_output, final_hidden, final_hidden);
    }

    // Output projection to logits
    auto out_embed = weights_->out_embed ? ((LlaisysTensor*)weights_->out_embed)->tensor : nullptr;
    if (out_embed) {
        auto logits_slice = logits; // shape [voc]
        auto normed_view_2d = normed_output->view({1, meta_.hs});
        ops::linear(logits_slice, normed_view_2d, out_embed, nullptr);
    }
    // Sample token from logits
    int64_t sampled_token = _sample_token(logits);
    return sampled_token;
}
}