#include "llaisys/models/qwen2.h"
#include "../models/model_qwen2.hpp"
#include "../core/context/context.hpp"

__C {
    struct LlaisysQwen2Model {
        llaisys::models::Modelqwen2 *model;
    };

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        // For now, use the first device
        int device_id = (ndevice > 0) ? device_ids[0] : 0;
        
        // Create the model
        auto model = new llaisys::models::Modelqwen2(meta, device, device_id);
        
        // Wrap in opaque struct
        auto wrapper = new LlaisysQwen2Model{model};
        return wrapper;
    }

    void llaisysQwen2ModelDestroy(
        struct LlaisysQwen2Model *model) {
        if (model) {
            delete model->model;
            delete model;
        }
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(
        struct LlaisysQwen2Model *model) {
        if (!model || !model->model) {
            return nullptr;
        }
        return model->model->weights();
    }

    int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model *model,
        int64_t *token_ids,
        size_t ntoken) {
        if (!model || !model->model) {
            return -1;
        }
        return model->model->infer(token_ids, ntoken);
    }
}
