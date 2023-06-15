#pragma once

#include "optix.h"

#include <stdexcept>
#include <cassert>

struct CUDABuffer
{
	void Free()
	{
		cudaFree(Data);
		Data = nullptr;
		Size_bytes = 0;
	}

	void Alloc(const size_t& size_bytes)
	{
		assert(!Data);
		Size_bytes = size_bytes;
		cudaError result = cudaMalloc(&Data, Size_bytes);
		if (result != CUDA_SUCCESS)
		{
			throw std::runtime_error("Could not allocate cuda memory!");
		}
	}

	void Resize(const size_t& size)
	{
		if(Data)
		{
			Free();
		}
		Alloc(size);
	}

	template<typename T>
	void Upload(const T* t, const size_t& count)
	{
		assert(Data);
		assert(Size_bytes == count * sizeof(T));
		cudaError result = cudaMemcpy(Data, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice);
		if (result != CUDA_SUCCESS)
		{
			throw std::runtime_error("Could not copy buffer to CUDA!");
		}

	}

	template<typename T>
	void Download(const T* t, const size_t& count)
	{
		assert(Data);
		assert(Size_bytes == count * sizeof(T));
		cudaError result = cudaMemcpy((void*)t, Data, count * sizeof(T), cudaMemcpyDeviceToHost);
		if (result != CUDA_SUCCESS)
		{
			throw std::runtime_error("Could not copy buffer from CUDA!");
		}
	}

	template<typename T>
	void AllocAndUpload(const std::vector<T>& vt)
	{
		Alloc(vt.size() * sizeof(T));
		Upload((const T*)vt.data(), vt.size());
	}

	CUdeviceptr CudaPtr() const
	{
		return (CUdeviceptr)(Data);
	}

	size_t Size_bytes{ 0 };
	void* Data{ nullptr };
};