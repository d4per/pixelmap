use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only try to configure CUDA if the "cuda" feature is enabled
    if cfg!(feature = "cuda") {
        println!("cargo:rustc-cfg=cuda");
        
        // Try to find CUDA installation
        let cuda_path = find_cuda();
        if let Some(cuda_path) = cuda_path {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-lib=dylib=cuda");
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-lib=dylib=cublas");
            
            // Pass CUDA include path to bindgen if generating bindings
            println!("cargo:include={}/include", cuda_path);
            
            // Add custom NVCC flags if needed
            if let Ok(compute_capability) = env::var("CUDA_COMPUTE_CAPABILITY") {
                println!("cargo:rustc-env=CUDA_COMPUTE_CAPABILITY={}", compute_capability);
            }
        } else {
            eprintln!("CUDA installation not found");
        }
    }

    // Only compile CUDA if the feature is enabled
    #[cfg(feature = "cuda")]
    compile_cuda();
    
    println!("cargo:rerun-if-changed=src/cuda/kernels");
}

#[cfg(feature = "cuda")]
fn compile_cuda() {
    use std::fs;
    use std::process::Command;
    use std::path::Path;

    println!("cargo:rerun-if-changed=src/cuda/kernels");

    // Create output directory for PTX files
    let out_dir = env::var("OUT_DIR").unwrap();
    let ptx_dir = Path::new(&out_dir).join("cuda-ptx");
    fs::create_dir_all(&ptx_dir).unwrap();

    // Set environment variable for runtime lookup
    println!("cargo:rustc-env=CUDA_PTX_DIR={}", ptx_dir.display());

    // Check for NVCC
    let nvcc_check = Command::new("nvcc").arg("--version").output();
    if nvcc_check.is_err() {
        println!("cargo:warning=NVCC not found, skipping CUDA kernel compilation");
        return;
    }

    // Determine CUDA compute capability
    let compute_capability = env::var("CUDA_COMPUTE_CAPABILITY").unwrap_or("50".to_string());
    println!("cargo:warning=Using CUDA compute capability {}", compute_capability);

    // Make sure the kernel directory exists
    let kernel_dir = Path::new("src/cuda/kernels");
    if !kernel_dir.exists() {
        fs::create_dir_all(kernel_dir).expect("Failed to create kernel directory");
        println!("cargo:warning=Created kernel directory: {:?}", kernel_dir);
    }

    // List of kernel files to compile
    let kernel_files = [
        "feature_extraction.cu",
        "correspondence_mapping.cu",
        "optimized_operations.cu",
        "feature_matching.cu",
        "photo_processing.cu",
        "interpolate_grid.cu",
        "transform.cu",
        "model_3d.cu",
    ];

    // Compile each kernel file
    for kernel_name in &kernel_files {
        let kernel_path = kernel_dir.join(kernel_name);
        
        // Create the file if it doesn't exist (for placeholder)
        if !kernel_path.exists() {
            println!("cargo:warning=Kernel file not found, creating placeholder: {}", kernel_path.display());
            let content = format!("// CUDA kernel file: {}\n// Placeholder implementation\n", kernel_name);
            fs::write(&kernel_path, content).expect("Failed to create kernel file");
        }
        
        println!("cargo:warning=Compiling CUDA kernel {}", kernel_path.display());
        
        let output_path = ptx_dir.join(format!("{}.ptx", kernel_name.strip_suffix(".cu").unwrap()));
        
        let status = Command::new("nvcc")
            .args(&[
                "--ptx",
                &format!("--gpu-architecture=sm_{}", compute_capability),
                "-O3", // Optimize for performance
                "-o", output_path.to_str().unwrap(),
                kernel_path.to_str().unwrap()
            ])
            .status();
        
        match status {
            Ok(exit_status) => {
                if !exit_status.success() {
                    println!("cargo:warning=Failed to compile CUDA kernel: {}", kernel_path.display());
                } else {
                    println!("cargo:warning=Successfully compiled {}", kernel_path.display());
                }
            },
            Err(e) => {
                println!("cargo:warning=Failed to run NVCC: {}", e);
            }
        }
    }
}

fn find_cuda() -> Option<String> {
    // Check environment variable first
    if let Ok(path) = env::var("CUDA_PATH") {
        return Some(path);
    }
    
    // Common installation locations
    let common_paths = vec![
        "/usr/local/cuda",
        "/opt/cuda",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
    ];
    
    for path in common_paths {
        let path_buf = PathBuf::from(path);
        if path_buf.exists() {
            return Some(path.to_string());
        }
    }
    
    // Try to query nvcc location
    let nvcc_output = Command::new("which").arg("nvcc").output().ok()?;
    if nvcc_output.status.success() {
        let nvcc_path = String::from_utf8_lossy(&nvcc_output.stdout);
        let path = PathBuf::from(nvcc_path.trim());
        return path.parent()?.parent()?.to_str().map(|s| s.to_string());
    }
    
    None
}
