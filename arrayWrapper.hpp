#ifndef arrayWrapper_hpp
#define arrayWrapper_hpp

#include <stddef.h>
#include <cassert>
#include <algorithm>
#ifdef __BOOST_COMPUTE__
	#include <boost/compute/algorithm/transform.hpp>
#endif
#ifdef __NVCC__
//#include <thrust/copy.h>
template<typename OP>
__global__ void testArrayWrapper(OP);
template<typename T, typename OP>
__global__ void kern(T, OP , const int);

#endif

#include "iterator.hpp"


namespace SOAX {

template<class AW_Type>
Iterator<AW_Type> bbeg(AW_Type&& aw)
{
  DEBUGMSG("Iterator<AW_Type> bbeg(AW_Type&& aw)\n");
  return Iterator<AW_Type>(std::forward<AW_Type>(aw),0);
}  
  
template<class AW_Type>
CIterator<AW_Type> cbeg(AW_Type&& aw)
{
  return CIterator<AW_Type>(std::forward<AW_Type>(aw),0);
}  
  
template<class AW_Type>
Iterator<AW_Type> eend(AW_Type&& aw)
{
  DEBUGMSG("Iterator<AW_Type> eend(AW_Type&& aw)\n");
  return Iterator<AW_Type>(std::forward<AW_Type>(aw),aw.size());
}  
  
template<class AW_Type>
CIterator<AW_Type> ccend(AW_Type&& aw)
{
  return CIterator<AW_Type>(std::forward<AW_Type>(aw),aw.size());
}  

template<typename T>
class AW_Scalar
{
 private:
  T const& s;

 public:
 AW_Scalar(T const& v) : s(v) {}

  T operator[](size_t) const {
    return s;
  }
  
  size_t size() const {
    return 0;
  };
};

template<typename T>
class AW_Traits {
 public:
  typedef T const& ExprRef;
};

template<typename T>
class AW_Traits<AW_Scalar<T>> {
 public:
  typedef AW_Scalar<T> ExprRef;
};

template<typename OP1, typename OP2>
class AW_Add 
{
 private:
  typename AW_Traits<OP1>::ExprRef op1;
  typename AW_Traits<OP2>::ExprRef op2;
  
 public:
  
  typedef decltype(op1[0] + op2[0]) value_type;
  
 AW_Add(OP1 const& a, OP2 const& b) : op1(a), op2(b) {}
  
  value_type operator[] (size_t idx) const  {
    return op1[idx] + op2[idx];
  }

  size_t size() const {
    assert(op1.size() == 0 || op2.size() == 0 || op1.size() == op2.size());
    return op1.size() != 0 ? op1.size() : op2.size();
  }
};

template<typename OP1, typename OP2>
class AW_Minus
{
 private:
  typename AW_Traits<OP1>::ExprRef op1;
  typename AW_Traits<OP2>::ExprRef op2;
  
 public:

  typedef decltype(op1[0] - op2[0]) value_type;
  
 AW_Minus(OP1 const& a, OP2 const& b) : op1(a), op2(b) {}
  
  value_type operator[] (size_t idx) const  {
    return op1[idx] - op2[idx];
  }

  size_t size() const {
    assert(op1.size() == 0 || op2.size() == 0 || op1.size() == op2.size());
    return op1.size() != 0 ? op1.size() : op2.size();
  }
};

template<typename OP1, typename OP2>
class AW_Mult
{
 private:
  typename AW_Traits<OP1>::ExprRef op1;
  typename AW_Traits<OP2>::ExprRef op2;
  
 public:
  
  typedef decltype(op1[0]*op2[0]) value_type;
  
 AW_Mult(OP1 const& a, OP2 const& b) : op1(a), op2(b) {}
  
  value_type operator[] (size_t idx) const {
    return op1[idx]*op2[idx];
  }

  size_t size() const {
    assert(op1.size() == 0 || op2.size() == 0 || op1.size() == op2.size());
    return op1.size() != 0 ? op1.size() : op2.size();
  }
};

template<typename OP1, typename OP2>
class AW_Div
{
 private:
  typename AW_Traits<OP1>::ExprRef op1;
  typename AW_Traits<OP2>::ExprRef op2;
  
 public:
  
  typedef decltype(op1[0]*op2[0]) value_type;
  
 AW_Div(OP1 const& a, OP2 const& b) : op1(a), op2(b) {}
  
  value_type operator[] (size_t idx) const {
    return op1[idx]/op2[idx];
  }

  size_t size() const {
    assert(op1.size() == 0 || op2.size() == 0 || op1.size() == op2.size());
    return op1.size() != 0 ? op1.size() : op2.size();
  }
};


template<class ArrayType, typename Rep = ArrayType>
  class ArrayWrapper : public Rep
{
 private:

public:
  typedef ArrayWrapper<ArrayType,Rep> AW_Type;

 ArrayWrapper() : Rep() {};
 
 using Rep::Rep;

 ArrayWrapper (Rep const& rb) : Rep(rb) {}

 ArrayWrapper& operator=(ArrayWrapper const& b) {
   assert(this->size() == b.size());

   DEBUGMSG("ArrayWrapper& operator=(ArrayWrapper const& b)\n");
#ifdef __NVCC__
	//testArrayWrapper<<<1,1>>>((*this));
	//kern<<<1,1>>>(b.data(), b.size(), *this); // b.data return underlying pointer)
	DEBUGCMD(cudaDeviceSynchronize());
   //thrust::copy(cbeg(b), ccend(b), bbeg(*this));
   //thrust::copy(cbeg(b), cbeg(b), this->begin());
   //   thrust::copy(b.begin(), b.end(), this->begin());
#elif defined __BOOST_COMPUTE__
	boost::compute::copy(cbeg(b), ccend(b), bbeg(*this));
#else
   std::copy(cbeg(b), ccend(b), bbeg(*this));

#endif

// #pragma omp simd
//    for(size_t idx=0; idx<b.size();++idx) {
//      (*this)[idx] = b[idx];
//    }
   return *this;
 }


 template<typename ArrayType2, typename Rep2>
 ArrayWrapper& operator=(ArrayWrapper<ArrayType2,Rep2> const& b) {
   assert(this->size() == b.size());

   DEBUGMSG("ArrayWrapper& operator=(ArrayWrapper<T2,Rep2> const& b)\n");
   size_t size = b.size();

#ifdef __NVCC__
	kern<<<1,1>>>(*this);
	cudaDeviceSynchronize();
   //thrust::copy(cbeg(b), ccend(b), bbeg(*this));
   //thrust::copy(b.begin(), b.end(), this->begin());
   //thrust::device_vector<float> dvec;
   //thrust::copy(dvec.begin(), dvec.end(), dvec.begin());
   //   thrust::copy(cbeg(b), ccend(b), dvec.begin());
#else
   std::copy(cbeg(b), ccend(b), bbeg(*this));
#endif

// #pragma omp simd
//    for(size_t idx=0;idx<size;++idx) {
//      (*this)[idx] = b[idx];
//    }
   return *this;
 }

 template<typename ArrayType2, typename Rep2>
 ArrayWrapper& operator+=(ArrayWrapper<ArrayType2,Rep2> const& b) {
   assert(this->size() == b.size());

   //   DEBUGMSG("ArrayWrapper& operator=(ArrayWrapper<T2,Rep2> const& b)\n");
   size_t size = b.size();
   //#pragma ivdep
   //#pragma simd
#ifdef __NVCC__
   //   thrust::transform(cbeg(*this),ccend(*this), cbeg(b), bbeg(*this), thrust::plus<float>());
   //   thrust::device_vector<float> df(10,3.);
   //   thrust::fill(bbeg(*this),eend(*this),3);
   //   thrust::fill(bbeg(df),eend(df),5);
   //   thrust::fill(this->begin(),this->end(),3);
   //   thrust::transform(bbeg(*this),eend(*this), bbeg(*this), thrust::negate<float>());
#else
   std::transform(this->cbeg(),this->ccend(),b.cbeg(),this->beg(), std::plus<decltype(b[0])>());
#endif
   // for(size_t idx=0;idx<size;++idx) {
   //   (*this)[idx] += b[idx];
   // }
   return *this;
 }

 template<typename ArrayType2, typename Rep2>
 ArrayWrapper& operator-=(ArrayWrapper<ArrayType2,Rep2> const& b) {
   assert(this->size() == b.size());

   //   DEBUGMSG("ArrayWrapper& operator=(ArrayWrapper<T2,Rep2> const& b)\n");
   size_t size = b.size();
// #pragma simd
//#pragma ivdep
#ifdef __NVCC__
   thrust::transform(this->cbeg(),this->ccend(),b.cbeg(),this->beg(), std::minus<decltype(b[0])>());
#else
   std::transform(this->cbeg(),this->ccend(),b.cbeg(),this->beg(), std::minus<decltype(b[0])>());
#endif
// #pragma simd
//    for(size_t idx=0;idx<size;++idx) {
//      (*this)[idx] -= b[idx];
//    }
   return *this;
 }

 Rep const& rep() const {
   return *this;
 }

 Rep& rep() {
   return *this;
 }
};

template<class ArrayType1, class ArrayType2, int sameType1, int sameType2>
class ArrayTypeDeductionHelper {};

template<class ArrayType1, class ArrayType2>
class ArrayTypeDeductionHelper<ArrayType1,ArrayType2,1,0> {
 public:
  typedef ArrayType1 ArrayType;
};

template<class ArrayType1, class ArrayType2>
class ArrayTypeDeductionHelper<ArrayType1,ArrayType2,0,1> {
 public:
  typedef ArrayType2 ArrayType;
};

template<class ArrayType1, class ArrayType2>
class ArrayTypeDeductionHelper<ArrayType1,ArrayType2,1,1> {
 public:
  typedef ArrayType1 ArrayType;
};

template<class ArrayType1, class ArrayType2>
class ArrayTypeDeduction {
 public:
  typedef typename ArrayType1::value_type T1;
  typedef typename ArrayType2::value_type T2;

  typedef decltype(T1()*T2()) DeclType;

  typedef typename ArrayTypeDeductionHelper<ArrayType1,ArrayType2,
    std::is_same<typename ArrayType1::value_type,DeclType>::value,
    std::is_same<typename ArrayType2::value_type,DeclType>::value>::ArrayType ArrayType;
};

template<typename ArrayType1, typename ArrayType2, typename R1, typename R2>
  ArrayWrapper<typename ArrayTypeDeduction<ArrayType1,ArrayType2>::ArrayType,AW_Add<R1,R2>> operator+(ArrayWrapper<ArrayType1,R1> const& a, ArrayWrapper<ArrayType2,R2> const& b) {
  return ArrayWrapper<typename ArrayTypeDeduction<ArrayType1,ArrayType2>::ArrayType,
    AW_Add<R1,R2>>(AW_Add<R1,R2>(a.rep(),b.rep()));
}

template<typename ArrayType, typename T, typename R>
  ArrayWrapper<ArrayType,AW_Add<AW_Scalar<T>,R>> operator+(ArrayWrapper<ArrayType,R> const& a, T const& s) {
  return ArrayWrapper<ArrayType,AW_Add<AW_Scalar<T>,R>>(AW_Add<AW_Scalar<T>,R>(AW_Scalar<T>(s),a.rep()));
}

template<typename ArrayType, typename T, typename R>
  ArrayWrapper<ArrayType,AW_Minus<R,AW_Scalar<T>>> operator-(ArrayWrapper<ArrayType,R> const& a, T const& s) {
  return ArrayWrapper<ArrayType,AW_Minus<R,AW_Scalar<T>>>(AW_Minus<R,AW_Scalar<T>>(a.rep(),AW_Scalar<T>(s)));
}

template<typename ArrayType1, typename ArrayType2, typename R1, typename R2>
  ArrayWrapper<typename ArrayTypeDeduction<ArrayType1,ArrayType2>::ArrayType,AW_Minus<R1,R2>> operator-(ArrayWrapper<ArrayType1,R1> const& a, ArrayWrapper<ArrayType2,R2> const& b) {
  return ArrayWrapper<typename ArrayTypeDeduction<ArrayType1,ArrayType2>::ArrayType,
    AW_Minus<R1,R2>>(AW_Minus<R1,R2>(a.rep(),b.rep()));
}

template<typename ArrayType, typename T2, typename R2>
  ArrayWrapper<ArrayType,AW_Mult<AW_Scalar<T2>,R2>> operator*(T2 const& s, ArrayWrapper<ArrayType,R2> const& b) {
  return ArrayWrapper<ArrayType,AW_Mult<AW_Scalar<T2>,R2>>(AW_Mult<AW_Scalar<T2>,R2>(AW_Scalar<T2>(s),b.rep()));
}

template<typename ArrayType, typename T2, typename R2>
  ArrayWrapper<ArrayType,AW_Mult<AW_Scalar<T2>,R2>> operator*(ArrayWrapper<ArrayType,R2> const& b, T2 const& s) {
  return ArrayWrapper<ArrayType,AW_Mult<AW_Scalar<T2>,R2>>(AW_Mult<AW_Scalar<T2>,R2>(AW_Scalar<T2>(s),b.rep()));
}

template<typename ArrayType, typename T2, typename R2>
  ArrayWrapper<ArrayType,AW_Div<R2,AW_Scalar<T2>>> operator/(ArrayWrapper<ArrayType,R2> const& b, T2 const& s) {
  return ArrayWrapper<ArrayType,AW_Div<R2,AW_Scalar<T2>>>(AW_Div<R2,AW_Scalar<T2>>(b.rep(),AW_Scalar<T2>(s)));
}
  

} // namespace SOAX
#endif
