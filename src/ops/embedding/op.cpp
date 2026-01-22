#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);  
      
    ASSERT(index->shape().size() == 1, "Embedding: index must be 1D tensor");  
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be Int64 type");  
      
 
    ASSERT(weight->shape().size() == 2, "Embedding: weight must be 2D tensor");  
      

    ASSERT(out->shape().size() == 2, "Embedding: out must be 2D tensor");  
      

    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: out and index first dimension must match");  
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: out and weight second dimension must match");  

    size_t vocab_size = weight->shape()[0];  
    auto index_data = reinterpret_cast<const int64_t*>(index->data());  
    for(size_t i = 0; i < index->numel(); i++) {  
        ASSERT(index_data[i] >= 0 && static_cast<size_t>(index_data[i]) < vocab_size,   
               "Embedding: index out of bounds");  
    }  
  
    // always support cpu calculation  
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {  
        return cpu::embedding(out->data(), index->data(), weight->data(),   
                             out->dtype(), index->numel(), weight->shape()[1]);  
    }  
  
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());  
  
    switch (out->deviceType()) {  
    case LLAISYS_DEVICE_CPU:  
        return cpu::embedding(out->data(), index->data(), weight->data(),   
                             out->dtype(), index->numel(), weight->shape()[1]);  
#ifdef ENABLE_NVIDIA_API  
    case LLAISYS_DEVICE_NVIDIA:  
        TO_BE_IMPLEMENTED();  
        return;  
#endif  
    default:  
        EXCEPTION_UNSUPPORTED_DEVICE;  
    }  
}  

} // namespace llaisys::ops
