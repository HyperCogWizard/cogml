// Test suite for Hardware-Optimized Data-Free Quantization-Aware Training (QAT) Framework

#include "ggml-qat.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define MAX_TEST_ERROR 0.001f
#define MAX_COMPRESSION_RATIO 0.5f
#define MIN_ACCURACY_RETENTION 0.95f

static bool test_qat_initialization() {
    printf("Testing QAT initialization...\n");
    
    ggml_qat_config_t config = {
        .embedding_qtype = GGML_TYPE_Q8_0,
        .attention_qtype = GGML_TYPE_Q4_K,
        .ffn_qtype = GGML_TYPE_Q6_K,
        .layernorm_qtype = GGML_TYPE_Q8_0,
        .kl_divergence_weight = 0.5f,
        .temperature = 3.0f,
        .progressive_layers = 10,
        .calibration_samples = 1000,
        .memory_reduction_target = 0.75f,
        .accuracy_threshold = 0.98f,
        .enable_atomspace = true,
        .enable_moses = true,
        .enable_ecan = true
    };
    
    ggml_qat_context_t * ctx = ggml_qat_init(&config);
    if (!ctx) {
        printf("FAILED: Could not initialize QAT context\n");
        return false;
    }
    
    ggml_qat_free(ctx);
    printf("PASSED: QAT initialization successful\n");
    return true;
}

static bool test_synthetic_calibration_data() {
    printf("Testing synthetic calibration data generation...\n");
    
    ggml_qat_config_t config = {
        .embedding_qtype = GGML_TYPE_Q8_0,
        .attention_qtype = GGML_TYPE_Q4_K,
        .ffn_qtype = GGML_TYPE_Q6_K,
        .layernorm_qtype = GGML_TYPE_Q8_0,
        .kl_divergence_weight = 0.5f,
        .temperature = 3.0f,
        .progressive_layers = 10,
        .calibration_samples = 1000,
        .memory_reduction_target = 0.75f,
        .accuracy_threshold = 0.98f,
        .enable_atomspace = true,
        .enable_moses = true,
        .enable_ecan = true
    };
    
    ggml_qat_context_t * qat_ctx = ggml_qat_init(&config);
    if (!qat_ctx) {
        printf("FAILED: Could not initialize QAT context\n");
        return false;
    }
    
    // Create ggml context for tensor operations
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,  // 16MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAILED: Could not initialize ggml context\n");
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Test calibration data generation
    int64_t shape[2] = {512, 1024};  // Example embedding dimensions
    struct ggml_tensor * calibration_data = ggml_qat_generate_calibration_data(
        qat_ctx, ctx, shape, 2, 0.0f, 1.0f);
    
    if (!calibration_data) {
        printf("FAILED: Could not generate calibration data\n");
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Verify data properties
    if (calibration_data->ne[0] != shape[0] || calibration_data->ne[1] != shape[1]) {
        printf("FAILED: Calibration data has incorrect shape\n");
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Check that data is within reasonable range (approximately normal distribution)
    const float * data = (const float *)calibration_data->data;
    int64_t n_elements = ggml_nelements(calibration_data);
    
    float mean = 0.0f, variance = 0.0f;
    for (int64_t i = 0; i < n_elements; i++) {
        mean += data[i];
    }
    mean /= n_elements;
    
    for (int64_t i = 0; i < n_elements; i++) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= n_elements;
    float std_dev = sqrtf(variance);
    
    if (fabsf(mean) > 0.1f || fabsf(std_dev - 1.0f) > 0.2f) {
        printf("FAILED: Calibration data statistics incorrect (mean=%.3f, std=%.3f)\n", mean, std_dev);
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    ggml_free(ctx);
    ggml_qat_free(qat_ctx);
    printf("PASSED: Synthetic calibration data generation successful\n");
    return true;
}

static bool test_layer_quantization() {
    printf("Testing architecture-aware layer quantization...\n");
    
    ggml_qat_config_t config = {
        .embedding_qtype = GGML_TYPE_Q8_0,
        .attention_qtype = GGML_TYPE_Q4_K,
        .ffn_qtype = GGML_TYPE_Q6_K,
        .layernorm_qtype = GGML_TYPE_Q8_0,
        .kl_divergence_weight = 0.5f,
        .temperature = 3.0f,
        .progressive_layers = 10,
        .calibration_samples = 1000,
        .memory_reduction_target = 0.75f,
        .accuracy_threshold = 0.98f,
        .enable_atomspace = true,
        .enable_moses = true,
        .enable_ecan = true
    };
    
    ggml_qat_context_t * qat_ctx = ggml_qat_init(&config);
    if (!qat_ctx) {
        printf("FAILED: Could not initialize QAT context\n");
        return false;
    }
    
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAILED: Could not initialize ggml context\n");
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Test different layer types
    ggml_qat_layer_type_t layer_types[] = {
        GGML_QAT_LAYER_EMBEDDING,
        GGML_QAT_LAYER_ATTENTION,
        GGML_QAT_LAYER_FFN,
        GGML_QAT_LAYER_LAYERNORM
    };
    
    int64_t shape[2] = {256, 512};
    
    for (int i = 0; i < 4; i++) {
        // Create test tensor
        struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape[0], shape[1]);
        if (!src) {
            printf("FAILED: Could not create source tensor for layer type %d\n", i);
            ggml_free(ctx);
            ggml_qat_free(qat_ctx);
            return false;
        }
        
        // Fill with test data
        float * data = (float *)src->data;
        for (int64_t j = 0; j < ggml_nelements(src); j++) {
            data[j] = sinf(j * 0.01f);  // Simple test pattern
        }
        
        // Quantize layer
        struct ggml_tensor * quantized = ggml_qat_quantize_layer(qat_ctx, ctx, src, layer_types[i]);
        if (!quantized) {
            printf("FAILED: Could not quantize layer type %d\n", i);
            ggml_free(ctx);
            ggml_qat_free(qat_ctx);
            return false;
        }
        
        // Verify quantized tensor has expected shape
        if (quantized->ne[0] != src->ne[0] || quantized->ne[1] != src->ne[1]) {
            printf("FAILED: Quantized tensor has incorrect shape for layer type %d\n", i);
            ggml_free(ctx);
            ggml_qat_free(qat_ctx);
            return false;
        }
    }
    
    ggml_free(ctx);
    ggml_qat_free(qat_ctx);
    printf("PASSED: Architecture-aware layer quantization successful\n");
    return true;
}

static bool test_opencog_components() {
    printf("Testing OpenCog-aligned components...\n");
    
    ggml_qat_config_t config = {
        .embedding_qtype = GGML_TYPE_Q8_0,
        .attention_qtype = GGML_TYPE_Q4_K,
        .ffn_qtype = GGML_TYPE_Q6_K,
        .layernorm_qtype = GGML_TYPE_Q8_0,
        .kl_divergence_weight = 0.5f,
        .temperature = 3.0f,
        .progressive_layers = 10,
        .calibration_samples = 1000,
        .memory_reduction_target = 0.75f,
        .accuracy_threshold = 0.98f,
        .enable_atomspace = true,
        .enable_moses = true,
        .enable_ecan = true
    };
    
    ggml_qat_context_t * qat_ctx = ggml_qat_init(&config);
    if (!qat_ctx) {
        printf("FAILED: Could not initialize QAT context\n");
        return false;
    }
    
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAILED: Could not initialize ggml context\n");
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Test AtomSpace creation
    ggml_qat_atomspace_t * atomspace = ggml_qat_create_atomspace(qat_ctx, ctx, 100, 50);
    if (!atomspace) {
        printf("FAILED: Could not create AtomSpace\n");
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    if (atomspace->n_atoms != 100 || atomspace->n_connections != 50) {
        printf("FAILED: AtomSpace has incorrect size\n");
        ggml_qat_free_atomspace(atomspace);
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Test MOSES tree creation
    ggml_qat_moses_tree_t * moses_tree = ggml_qat_create_moses_tree(qat_ctx, ctx, 50, 5);
    if (!moses_tree) {
        printf("FAILED: Could not create MOSES tree\n");
        ggml_qat_free_atomspace(atomspace);
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    if (moses_tree->n_nodes != 50) {
        printf("FAILED: MOSES tree has incorrect size\n");
        ggml_qat_free_moses_tree(moses_tree);
        ggml_qat_free_atomspace(atomspace);
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Test ECAN creation
    ggml_qat_ecan_t * ecan = ggml_qat_create_ecan(qat_ctx, ctx, 200);
    if (!ecan) {
        printf("FAILED: Could not create ECAN\n");
        ggml_qat_free_moses_tree(moses_tree);
        ggml_qat_free_atomspace(atomspace);
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    if (ecan->n_elements != 200) {
        printf("FAILED: ECAN has incorrect size\n");
        ggml_qat_free_ecan(ecan);
        ggml_qat_free_moses_tree(moses_tree);
        ggml_qat_free_atomspace(atomspace);
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Verify attention scores sum to approximately 1.0
    float attention_sum = 0.0f;
    for (int i = 0; i < ecan->n_elements; i++) {
        attention_sum += ecan->attention_scores[i];
    }
    
    if (fabsf(attention_sum - 1.0f) > 0.01f) {
        printf("FAILED: ECAN attention scores don't sum to 1.0 (sum=%.3f)\n", attention_sum);
        ggml_qat_free_ecan(ecan);
        ggml_qat_free_moses_tree(moses_tree);
        ggml_qat_free_atomspace(atomspace);
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Clean up
    ggml_qat_free_ecan(ecan);
    ggml_qat_free_moses_tree(moses_tree);
    ggml_qat_free_atomspace(atomspace);
    ggml_free(ctx);
    ggml_qat_free(qat_ctx);
    
    printf("PASSED: OpenCog-aligned components successful\n");
    return true;
}

static bool test_kl_divergence_loss() {
    printf("Testing KL divergence loss calculation...\n");
    
    ggml_qat_config_t config = {
        .embedding_qtype = GGML_TYPE_Q8_0,
        .attention_qtype = GGML_TYPE_Q4_K,
        .ffn_qtype = GGML_TYPE_Q6_K,
        .layernorm_qtype = GGML_TYPE_Q8_0,
        .kl_divergence_weight = 0.5f,
        .temperature = 3.0f,
        .progressive_layers = 10,
        .calibration_samples = 1000,
        .memory_reduction_target = 0.75f,
        .accuracy_threshold = 0.98f,
        .enable_atomspace = true,
        .enable_moses = true,
        .enable_ecan = true
    };
    
    ggml_qat_context_t * qat_ctx = ggml_qat_init(&config);
    if (!qat_ctx) {
        printf("FAILED: Could not initialize QAT context\n");
        return false;
    }
    
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAILED: Could not initialize ggml context\n");
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Create test tensors
    int64_t shape[1] = {1000};
    struct ggml_tensor * original = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape[0]);
    struct ggml_tensor * quantized = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape[0]);
    
    if (!original || !quantized) {
        printf("FAILED: Could not create test tensors\n");
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Fill with test data
    float * orig_data = (float *)original->data;
    float * quant_data = (float *)quantized->data;
    
    for (int i = 0; i < shape[0]; i++) {
        orig_data[i] = sinf(i * 0.01f);
        quant_data[i] = orig_data[i] + 0.01f * cosf(i * 0.02f);  // Slightly different
    }
    
    // Calculate KL divergence
    float kl_loss = ggml_qat_kl_divergence_loss(qat_ctx, ctx, original, quantized, config.temperature);
    
    if (kl_loss < 0.0f || kl_loss > 10.0f) {
        printf("FAILED: KL divergence loss out of reasonable range (%.6f)\n", kl_loss);
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    // Test identical tensors (should give near-zero KL divergence)
    for (int i = 0; i < shape[0]; i++) {
        quant_data[i] = orig_data[i];
    }
    
    float kl_loss_identical = ggml_qat_kl_divergence_loss(qat_ctx, ctx, original, quantized, config.temperature);
    
    if (kl_loss_identical > 0.01f) {
        printf("FAILED: KL divergence for identical tensors too high (%.6f)\n", kl_loss_identical);
        ggml_free(ctx);
        ggml_qat_free(qat_ctx);
        return false;
    }
    
    ggml_free(ctx);
    ggml_qat_free(qat_ctx);
    printf("PASSED: KL divergence loss calculation successful\n");
    return true;
}

int main() {
    printf("Running Hardware-Optimized Data-Free QAT Framework Tests\n");
    printf("========================================================\n\n");
    
    bool all_passed = true;
    
    all_passed &= test_qat_initialization();
    all_passed &= test_synthetic_calibration_data();
    all_passed &= test_layer_quantization();
    all_passed &= test_opencog_components();
    all_passed &= test_kl_divergence_loss();
    
    printf("\n========================================================\n");
    if (all_passed) {
        printf("All QAT framework tests PASSED!\n");
        printf("✓ QAT initialization and configuration\n");
        printf("✓ Synthetic calibration data generation\n");
        printf("✓ Architecture-aware layer quantization\n");
        printf("✓ OpenCog components (AtomSpace, MOSES, ECAN)\n");
        printf("✓ KL divergence loss calculation\n");
        return 0;
    } else {
        printf("Some QAT framework tests FAILED!\n");
        return 1;
    }
}