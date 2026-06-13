//#[cfg(feature = "pretrained")]
//#[allow(unused_imports)]
//use crate::pretrained::{self, ModelMeta};


#[cfg(feature = "ndarray")]
pub mod selected {
    pub type Backend = burn::backend::NdArray;
    pub type Device = burn::backend::ndarray::NdArrayDevice;

    pub fn device() -> Device {
        Device::Cpu
    }
}

#[cfg(feature = "vulkan")]
pub mod selected {
    pub type Backend = burn::backend::wgpu::Vulkan;
    pub type Device = burn::backend::wgpu::WgpuDevice;

    pub fn device() -> Device {
        Device::default()
    }
}

#[cfg(feature = "cuda")]
pub mod selected {
    pub type Backend = burn::backend::cuda::Cuda;
    pub type Device = burn::backend::cuda::CudaDevice;

    pub fn device() -> Device {
        Device::default()
    }
}

#[cfg(feature = "tch-cpu")]
pub mod selected {
    pub type Backend = burn::backend::libtorch::LibTorch;
    pub type Device = burn::backend::libtorch::LibTorchDevice;

    pub fn device() -> Device {
        Device::default()
    }
}

#[cfg(feature = "tch-gpu")]
mod selected {
    pub type Backend = burn::backend::libtorch::LibTorch;
    pub type Device = burn::backend::libtorch::LibTorchDevice;
    pub fn device() -> Device {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        device
    }
}
