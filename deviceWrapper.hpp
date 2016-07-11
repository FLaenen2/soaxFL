#ifndef SOAX_deviceWrapper_hpp
#define SOAX_deviceWrapper_hpp

#include <thrust/device_vector.h>


// adapter Design Pattern
template<typename T>
class deviceWrapper{
	public:
		// delegate design pattern
		class ptrWrapper{	// user second class as member to be able to redefine operator[] with a different return type
			public:
				using return_ref_type = T&;
				__device__ return_ref_type operator[](unsigned int i){ // writes single value from host through thrust vector
					return dptr[i];
				}//*/
				ptrWrapper(void){} // default empty constructor
				ptrWrapper(ptrWrapper const &other) : dptr(other.dptr){}; // kernels manipulate only pointer and instances
			public:
				T *dptr;
		} devPtrWrapper;

		using return_ref_type = thrust::device_reference<T>;

		deviceWrapper(){
			// calls default constructor for thrust device vector
			tvec = thrust::device_vector<T>();
			tptr = tvec.data();
			devPtrWrapper.dptr = thrust::raw_pointer_cast(tptr);
		}

		// cannot be passed by reference in kernel. Solution : pass the instance by copy, but copy only the ptr, no use to copy everything.

 		__host__ return_ref_type operator[](unsigned int i){ // writes single value from host through thrust vector
			return tvec[i];
		}//*/

		/*__host__ T operator[](int i) const{ // reads single value from host through thrust vector
			return tvec[i];
		}//*/

		/*__device__ T operator[](int i) const { // reads single value from device through pointer
			return dptr[i];
		}//*/

		/*__device__ T& operator[](int i) { // reads single value from device through pointer
			return dptr[i];
		}//*/

		size_t size(void){
			return this->m_size;
		}//*/

		void resize(int n){
			tvec.resize(n);
			// rebind device_ptr and pointer in case it has changed
			tptr = tvec.data();
			devPtrWrapper.dptr = thrust::raw_pointer_cast(tptr);
		}

		void shrink_to_fit(void){
			tvec.shrink_to_fit();
			// rebind device_ptr and pointer in case it has changed
			tptr = tvec.data();
			devPtrWrapper.dptr = thrust::raw_pointer_cast<T>(tptr);
		}

		T* data(void){ // return underlying device pointer
			return devPtrWrapper.dptr;
		};

		size_t m_size;

	private:	

		thrust::device_vector<T> tvec;
		thrust::device_ptr<T> tptr;
};
#endif
