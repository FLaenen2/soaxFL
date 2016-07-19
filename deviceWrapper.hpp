#ifndef SOAX_deviceWrapper_hpp
#define SOAX_deviceWrapper_hpp

#include <thrust/device_vector.h>


// adapter Design Pattern
template<typename T>
class deviceWrapper{
	public:

		#ifdef __CUDA_ARCH__
			typedef T return_ref_type;
		#else
			typedef thrust::device_reference<T> return_ref_type;
		#endif

		deviceWrapper(){
			// calls default constructor for thrust::device_vector
			tvec = thrust::device_vector<T>();
			tptr = tvec.data();
			devPtrWrapper.dptr = thrust::raw_pointer_cast(tptr);
		}
		deviceWrapper(deviceWrapper const& other) : dptr(other.ptr){
			// copy constructor. Mandatory to pass pointer to kernel and construct expression template.
			// we don't want to copy the whole data, only the pointers
			// hence overloading
		}

 		__host__ return_ref_type operator[](unsigned int i){ // writes single value from host through thrust vector
			#ifdef __CUDA_ARCH__
				return tvec[i];
            #else
				return dptr[i];
            #endif
		}
		__host__ __device__ T operator[](unsigned int i) const { // writes single value from host through thrust vector
			#ifdef __CUDA_ARCH__
        		return tvec[i];
            #else
				return dptr[i];
            #endif
		}

		size_t size(void){
			return this->m_size;
		}

		void resize(int n){
			tvec.resize(n);
			// rebind device_ptr and pointer in case memory location has changed
			tptr = tvec.data();
			dptr = thrust::raw_pointer_cast(tptr);
		}

		void shrink_to_fit(void){
			tvec.shrink_to_fit();
			// rebind device_ptr and pointer in case it has changed
			tptr = tvec.data();
			dptr = thrust::raw_pointer_cast<T>(tptr);
		}

		T* data(void){ // return underlying device pointer
			return dptr;
		};

		size_t m_size;
		T *dptr;
	private:
		thrust::device_vector<T> tvec;
		thrust::device_ptr<T> tptr;
};
#endif
