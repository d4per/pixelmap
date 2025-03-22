# CUDA Acceleration for PixelMap

This module provides CUDA-accelerated implementations for computationally intensive operations in the PixelMap library.

## Module Structure

- `bindings.rs` - Low-level bindings to the CUDA driver API
- `memory.rs` - Memory management (allocations, transfers, etc.)
- `kernels/` - CUDA kernel implementations organized by functionality
  - `core.rs` - Core functionality for launching kernels
  - `feature_extraction.rs` - Feature extraction kernel launchers
  - `feature_matching.rs` - Feature matching kernel launchers
  - `photo_processing.rs` - Photo processing kernel launchers
  - `dense_map.rs` - Dense map operations kernel launchers
  - `correspondence.rs` - Correspondence mapping kernel launchers
  - `transform.rs` - Transform operations kernel launchers
  - `model_3d.rs` - 3D model generation kernel launchers
- `feature_grid.rs` - CUDA-accelerated feature grid
- `dense_photo_map.rs` - CUDA-accelerated dense photo map
- `feature_matcher.rs` - CUDA-accelerated feature matcher
- `photo_processor.rs` - CUDA-accelerated photo processor
- `transform.rs` - CUDA-accelerated transform processor
- `model_3d.rs` - CUDA-accelerated 3D model generator

## Requirements

- CUDA Toolkit 11.0 or higher
- Compatible NVIDIA GPU with compute capability 5.0 or higher
- Appropriate NVIDIA drivers installed

## Building

To build with CUDA support, enable the "cuda" feature:

```sh
cargo build --features cuda
```

The `build.rs` script will automatically compile the CUDA kernels to PTX files during the build process.

## Environment Variables

- `CUDA_COMPUTE_CAPABILITY` - Set the CUDA compute capability for kernel compilation (default: 50)
- `CUDA_PTX_DIR` - (Set automatically) Directory where PTX files are stored
- `CUDA_PATH` - Custom path to CUDA installation (optional)

## Architecture

### API Design Pattern

The CUDA implementations follow a fallback pattern. When CUDA acceleration is not available (due to hardware limitations or feature flags), operations seamlessly fall back to CPU implementations.

### Memory Management

- Device memory is managed through RAII patterns with the `CudaBuffer` type
- Explicit synchronization is required between host and device memory
- Pinned memory is used for frequent host-device transfers

### Performance Configuration

Key performance parameters can be tuned through the `CudaConfig` struct:
- Block dimensions: Default 16x16 threads per block
- Grid dimensions: Automatically calculated based on input size
- Shared memory allocation: Configurable per kernel

## Usage Examples

### Basic CUDA-accelerated Processing

```rust
use pixelmap::{PixelMapProcessor, Photo};

// Load your photos
let photo1 = Photo::load("path/to/image1.jpg")?;
let photo2 = Photo::load("path/to/image2.jpg")?;

// Create a CUDA-accelerated processor
let mut processor = PixelMapProcessor::new_with_cuda(photo1, photo2, 1024)?;

// Process the photos with CUDA acceleration
let (forward_map, backward_map) = processor.generate_mapping_cuda(3, 2.0)?;

// Use the resulting maps
let warped_image = forward_map.warp_photo(&photo1);
```

### CPU Fallback

The implementation will automatically fall back to CPU processing if:
- CUDA is not available on the system
- The `cuda` feature is not enabled
- A CUDA operation fails

## Testing

Run the CUDA-specific tests with:

```sh
cargo test --features cuda cuda_
```

For benchmarking CUDA performance:

```sh
cargo bench --features cuda cuda_benchmarks
```

## Adding New Kernels

1. Create a new CUDA kernel file in `src/cuda/kernels/`
2. Add Rust wrapper functions in the appropriate module
3. Update the build script to include the new kernel file
4. Add corresponding tests in the test module

## Error Handling

CUDA errors are propagated through Rust's Result type. Common error categories:
- Device initialization errors
- Memory allocation failures
- Kernel launch failures
- Synchronization errors

Example error handling:
```rust
let result = cuda_operation()?; // Propagates CUDA errors
```

## Troubleshooting

### Common Issues

- **CUDA not found during build**: Ensure CUDA toolkit is installed and in PATH
- **Kernel launch failures**: Check compute capability compatibility
- **Memory errors**: Verify memory allocations and transfers

### Debug Mode

Enable debug logging for CUDA operations:

```sh
RUST_LOG=pixelmap::cuda=debug cargo run --features cuda
```

## Driver vs Runtime API

This implementation uses the CUDA driver API rather than the runtime API, providing:
- More explicit control over device resources
- Support for loading PTX at runtime
- Better integration with Rust's memory management model

## Version Compatibility

| PixelMap Version | CUDA Version | Compute Capability |
|------------------|--------------|-------------------|
| 1.0.x            | 11.0+        | 5.0+             |
| 0.9.x            | 10.1+        | 3.5+             |
