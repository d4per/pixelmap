use std::ptr::slice_from_raw_parts;
use crate::circular_feature_descriptor::CircularFeatureDescriptor;

mod circular_feature_descriptor;

#[link(name = "pixelmap_cuda")]
extern "C" {
    pub fn alloc_cuda_memory(bytes: usize) -> *mut u8;

    pub fn free_cuda_memory(x: *mut u8) -> std::os::raw::c_int;

    pub fn sync_cuda_device();
    pub fn computeDescriptorsOnGPU(
        image_rgba: *mut u8,  // device pointer
        width: u32,
        height: u32,
        radius: u32,
        circular_descriptors_buffer: *mut u8 // device pointer
    );
}

fn main() {

    let mut gpu_mem1 = unsafe { alloc_cuda_memory(100000) };

    let mut gpu_mem2 = unsafe { alloc_cuda_memory(100000)};

    let slice: *const [CircularFeatureDescriptor] = slice_from_raw_parts(gpu_mem2 as *mut CircularFeatureDescriptor, 10);

    // Convert raw pointers to Rust slices
    unsafe { computeDescriptorsOnGPU(gpu_mem1, 10, 10, 10, gpu_mem2) };
    unsafe { sync_cuda_device() }

    println!("aa");

    unsafe { free_cuda_memory(gpu_mem1); }
    unsafe { free_cuda_memory(gpu_mem2 as *mut u8); }
}
