#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Hardware-Optimized Data-Free Quantization-Aware Training (QAT) Framework
// ============================================================================

// QAT configuration structure
typedef struct {
    // Architecture-aware quantization settings
    enum ggml_type embedding_qtype;     // 8-bit uniform for embeddings
    enum ggml_type attention_qtype;     // 4-bit row-wise for attention
    enum ggml_type ffn_qtype;          // 6-bit group-wise for FFN
    enum ggml_type layernorm_qtype;    // 8-bit uniform for layer norms
    
    // Training protocol parameters
    float kl_divergence_weight;        // Weight for KL divergence loss
    float temperature;                 // Temperature for softmax in KL loss
    int progressive_layers;            // Number of layers to quantize progressively
    int calibration_samples;           // Number of synthetic calibration samples
    
    // Hardware optimization targets
    float memory_reduction_target;     // Target memory reduction (e.g., 0.75 for 75%)
    float accuracy_threshold;          // Minimum accuracy retention (e.g., 0.98 for 98%)
    
    // OpenCog-specific settings
    bool enable_atomspace;             // Enable AtomSpace quantization
    bool enable_moses;                 // Enable MOSES program tree quantization
    bool enable_ecan;                  // Enable ECAN attention quantization
} ggml_qat_config_t;

// QAT model context
typedef struct ggml_qat_context ggml_qat_context_t;

// Layer quantization strategy
typedef enum {
    GGML_QAT_LAYER_EMBEDDING,
    GGML_QAT_LAYER_ATTENTION,
    GGML_QAT_LAYER_FFN,
    GGML_QAT_LAYER_LAYERNORM,
    GGML_QAT_LAYER_ATOMSPACE,
    GGML_QAT_LAYER_MOSES,
    GGML_QAT_LAYER_ECAN,
} ggml_qat_layer_type_t;

// Quantization statistics
typedef struct {
    float original_size_mb;
    float quantized_size_mb;
    float compression_ratio;
    float accuracy_retention;
    float kl_divergence_loss;
    float inference_speedup;
} ggml_qat_stats_t;

// ============================================================================
// Core QAT Functions
// ============================================================================

// Initialize QAT context with configuration
GGML_API ggml_qat_context_t * ggml_qat_init(const ggml_qat_config_t * config);

// Free QAT context
GGML_API void ggml_qat_free(ggml_qat_context_t * ctx);

// Generate synthetic calibration data for data-free training
GGML_API struct ggml_tensor * ggml_qat_generate_calibration_data(
    ggml_qat_context_t * ctx,
    struct ggml_context * ggml_ctx,
    int64_t * shape,
    int n_dims,
    float mean,
    float std);

// Apply architecture-aware quantization to a tensor based on layer type
GGML_API struct ggml_tensor * ggml_qat_quantize_layer(
    ggml_qat_context_t * ctx,
    struct ggml_context * ggml_ctx,
    struct ggml_tensor * src,
    ggml_qat_layer_type_t layer_type);

// Progressive layer-wise quantization
GGML_API bool ggml_qat_progressive_quantize(
    ggml_qat_context_t * ctx,
    struct ggml_context * ggml_ctx,
    struct ggml_tensor ** layers,
    ggml_qat_layer_type_t * layer_types,
    int n_layers,
    int current_layer);

// Calculate KL divergence loss between original and quantized tensors
GGML_API float ggml_qat_kl_divergence_loss(
    ggml_qat_context_t * ctx,
    struct ggml_context * ggml_ctx,
    struct ggml_tensor * original,
    struct ggml_tensor * quantized,
    float temperature);

// Validate quantization performance
GGML_API ggml_qat_stats_t ggml_qat_validate_performance(
    ggml_qat_context_t * ctx,
    struct ggml_context * ggml_ctx,
    struct ggml_tensor * original_model,
    struct ggml_tensor * quantized_model,
    struct ggml_tensor * test_data);

// ============================================================================
// OpenCog-Aligned Components
// ============================================================================

// AtomSpace structure for hypergraph representation
typedef struct {
    float * truth_values;     // Quantized truth values
    int64_t * atom_types;     // Atom type indices
    int64_t * connections;    // Hypergraph connections
    int n_atoms;             // Number of atoms
    int n_connections;       // Number of connections
} ggml_qat_atomspace_t;

// MOSES program tree node
typedef struct {
    float * weights;         // Quantized node weights
    int * operations;        // Operation types
    int * connections;       // Tree structure connections
    int n_nodes;            // Number of nodes
} ggml_qat_moses_tree_t;

// ECAN attention scores
typedef struct {
    float * sti_values;      // Short-term importance (quantized)
    float * lti_values;      // Long-term importance (quantized)
    float * attention_scores; // Attention allocation scores
    int n_elements;         // Number of elements
} ggml_qat_ecan_t;

// Create and quantize AtomSpace structure
GGML_API ggml_qat_atomspace_t * ggml_qat_create_atomspace(
    ggml_qat_context_t * ctx,
    struct ggml_context * ggml_ctx,
    int n_atoms,
    int n_connections);

// Create and quantize MOSES program tree
GGML_API ggml_qat_moses_tree_t * ggml_qat_create_moses_tree(
    ggml_qat_context_t * ctx,
    struct ggml_context * ggml_ctx,
    int n_nodes,
    int tree_depth);

// Create and quantize ECAN attention mechanism
GGML_API ggml_qat_ecan_t * ggml_qat_create_ecan(
    ggml_qat_context_t * ctx,
    struct ggml_context * ggml_ctx,
    int n_elements);

// Free OpenCog structures
GGML_API void ggml_qat_free_atomspace(ggml_qat_atomspace_t * atomspace);
GGML_API void ggml_qat_free_moses_tree(ggml_qat_moses_tree_t * tree);
GGML_API void ggml_qat_free_ecan(ggml_qat_ecan_t * ecan);

// ============================================================================
// Hardware Optimization Functions
// ============================================================================

// Get optimal quantization type for target hardware
GGML_API enum ggml_type ggml_qat_get_optimal_qtype(
    ggml_qat_layer_type_t layer_type,
    ggml_backend_t backend);

// Estimate memory usage for quantized model
GGML_API size_t ggml_qat_estimate_memory_usage(
    ggml_qat_context_t * ctx,
    struct ggml_tensor ** tensors,
    ggml_qat_layer_type_t * layer_types,
    int n_tensors);

// Benchmark quantized model performance
GGML_API float ggml_qat_benchmark_inference_speed(
    ggml_qat_context_t * ctx,
    ggml_backend_t backend,
    struct ggml_tensor * model,
    struct ggml_tensor * test_input,
    int n_iterations);

#ifdef __cplusplus
}
#endif