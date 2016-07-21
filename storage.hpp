#ifndef storage_hpp
#define storage_hpp

#include <array>
#include <vector>

#ifdef __NVCC__
	#include "./deviceWrapper.hpp"
	//#include <thrust/host_vector.h>
	#include <thrust/device_vector.h>
	#include <thrust/device_reference.h>
#elif defined __BOOST_COMPUTE__
	#include <boost/compute/container/vector.hpp>
#endif
#include "tupleManip.hpp"
#include "arrayWrapper.hpp"

namespace SOAX {



template<class T, int dim>
class Storage
{
 public:
  template<class AttributeTuple, class Attribute, int startIndex, int N> friend struct FillAttributeInVec;
  template<class AttributeTuple, int N> friend struct TupleObjectFromSoax;
  template<class AttributeTuple, int N> friend struct TupleObjectToSoax;
  template<class AttributeTuple, int N, class Writer> friend struct TupleWrite;
  template<class AttributeTuple, int N> friend struct ReadArrayOfProperty;

  typedef T type;

#ifdef __NVCC__
  //template<class TYPE> using ContainerTypeTemplate = ArrayWrapper<thrust::device_vector<TYPE>>;
  template<class TYPE> using ContainerTypeTemplate = ArrayWrapper<deviceWrapper<TYPE>>;
	typedef T return_type;
	typedef thrust::device_reference<T> return_ref_type;
	//typedef float return_ref_type;
#elif defined __BOOST_COMPUTE__
	template<class TYPE> using ContainerTypeTemplate = ArrayWrapper<boost::compute::vector<TYPE>>;
	typedef T return_type;
//	typedef T& return_ref_type;//*/
	typedef boost::compute::detail::buffer_value<T> return_ref_type;
#else
  template<class TYPE> using ContainerTypeTemplate = ArrayWrapper<std::vector<TYPE>>;
	typedef T return_type;
	typedef T& return_ref_type;
#endif

  typedef ContainerTypeTemplate<type> ContainerType;

  typedef typename TupleTypeDim<type,dim>::type Types;

  enum { dim_ = dim };

 protected:

	return_ref_type data(int coord, int particle){
    	assert((coord < dim) && (particle < data_[0].size()));
		return data_[coord][particle];
	}

	return_type data(int coord, int particle) const { 
		assert((coord < dim) && (particle < data_[0].size()));
		return data_[coord][particle];
	 }//*/


  std::array<ContainerType,dim> data_;
};

} // namespace SOAX

#endif
