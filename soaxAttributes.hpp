#ifndef soaxAttributes_hpp
#define soaxAttributes_hpp

#include <string>
#include <type_traits>

#include "soax.hpp"
#include "tupleManip.hpp"

namespace SOAX {

#define SOAX_ATTRIBUTE(Name, Letter)                                          \
  template<typename T, int Dim>						                                     \
  class Name 				                                     \
  {                                                                                                          \
    public:                                                                                                  \
    enum { dim = Dim};							\
    typedef T type;	                                                \
    typedef typename SOAX::TupleTypeDim<T,dim>::type Types;		\
    static std::string name() { return Letter; }			\
    template<class TYPE> using ContainerTypeTemplate = typename SOAX::Storage<T,1>::template ContainerTypeTemplate<TYPE>; \
									\
    template<int attIndex, typename Attributes>			      	\
      class Access : public SOAX::Storage<T,Dim> {			\
    public:								\
      enum { index = attIndex };					\
									\
      template<int DIM = dim>						\
      typename std::enable_if<DIM != 1,typename SOAX::Storage<T,DIM>::return_type>::type Name(int particle, int coord) const { static_assert(DIM == dim, "DIM != dim"); return this->data(coord,particle);} \
      template<int DIM = dim>			\
      typename std::enable_if<DIM != 1,typename SOAX::Storage<T,DIM>::return_ref_type>::type Name(int particle, int coord) { static_assert(DIM == dim, "DIM != dim"); return this->data(coord,particle);} \
      template<int DIM = dim>						\
      typename std::enable_if<DIM == 1,typename SOAX::Storage<T,DIM>::return_type>::type Name(int particle, int coord) const { static_assert(DIM == dim, "DIM != dim"); static_assert(Dim != 1, "Dim == 1"); } \
      template<int DIM = dim>			\
      typename std::enable_if<DIM == 1,typename SOAX::Storage<T,DIM>::return_ref_type>::type Name(int particle, int coord) { static_assert(DIM == dim, "DIM != dim"); static_assert(Dim != 1, "Dim == 1");} \
      template<int DIM = dim>						\
      typename std::enable_if<DIM != 1,typename SOAX::Storage<T,DIM>::return_type>::type Name(int particle) const { static_assert(DIM == dim, "DIM != dim"); static_assert(Dim == 1, "Dim != 1"); } \
      template<int DIM = dim>			\
      typename std::enable_if<DIM != 1,typename SOAX::Storage<T,DIM>::return_ref_type>::type Name(int particle) { static_assert(DIM == dim, "DIM != dim"); static_assert(Dim == 1, "Dim != 1"); } \
      template<int DIM = dim>						\
      typename std::enable_if<DIM == 1,typename SOAX::Storage<T,DIM>::return_type>::type Name(int particle) const { static_assert(DIM == dim, "DIM != dim"); return this->data(0,particle); } \
      template<int DIM = dim>			\
      typename std::enable_if<DIM == 1,typename SOAX::Storage<T,DIM>::return_ref_type>::type Name(int particle) { static_assert(DIM == dim, "DIM != dim"); return this->data(0,particle); } \
      template<int DIM = dim>			\
      typename std::enable_if<DIM == 1,typename SOAX::Storage<T,DIM>::return_type>::type Name##test(int particle) const{return this->data(0, particle); } \
	  template<int DIM = dim>			\
      typename std::enable_if<DIM == 1,typename SOAX::Storage<T,DIM>::return_ref_type>::type Name##test(int particle){return this->data(0, particle); } \
		template<int DIM = dim>			\
      typename std::enable_if<DIM != 1,typename SOAX::Storage<T,dim>::ContainerType&>::type Name##Arr(int coord) { static_assert(DIM == dim, "DIM != dim"); return this->data_[coord];} \
      template<int DIM = dim>			\
      typename std::enable_if<DIM == 1,typename SOAX::Storage<T,dim>::ContainerType&>::type Name##Arr(int coord) { static_assert(DIM == dim, "DIM != dim"); static_assert(Dim != 1, "Dim == 1"); } \
      template<int DIM = dim>			\
      typename std::enable_if<DIM != 1,typename SOAX::Storage<T,dim>::ContainerType&>::type Name##Arr() { static_assert(DIM == dim, "DIM != dim"); static_assert(Dim == 1, "Dim != 1"); } \
      template<int DIM = dim>			\
      typename std::enable_if<DIM == 1,typename SOAX::Storage<T,dim>::ContainerType&>::type Name##Arr() { static_assert(DIM == dim, "DIM != dim"); return this->data_[0];} \
									\
      constexpr static int Name##Index() {return index;}	       	\
									\
      T Name(typename SOAX::TupleCatAttributes<Attributes,std::tuple_size<Attributes>::value>::type tuple) const { static_assert( dim == 1, "dim == 1" ); return std::get<index>(tuple);} \
      template <int coord> T Name(typename SOAX::TupleCatAttributes<Attributes,std::tuple_size<Attributes>::value>::type tuple) const { static_assert( coord < dim, "coord < dim" ); return std::get<index+coord>(tuple);} \
									\
      };								\
    class Object							\
    {								        \
    public:								\
      type value[dim];							\
      template<int DIM = dim>						\
      typename std::enable_if<DIM != 1,type&>::type Name(int i) { return value[i]; }       \
      template<int DIM = dim>						\
      typename std::enable_if<DIM != 1,type>::type Name(int i) const { return value[i]; }  \
      template<int DIM = dim>						\
      typename std::enable_if<DIM == 1,type&>::type Name(int i) { static_assert(Dim != 1, "Dim == 1"); }	\
      template<int DIM = dim>						\
      typename std::enable_if<DIM == 1,type>::type Name(int i) const { static_assert(Dim != 1, "Dim == 1"); }  \
      template<int DIM = dim>						\
      typename std::enable_if<DIM == 1,type&>::type Name() { return value[0]; }       \
      template<int DIM = dim>						\
      typename std::enable_if<DIM == 1,type>::type Name() const { return value[0]; }  \
      template<int DIM = dim>						\
      typename std::enable_if<DIM != 1,type&>::type Name() { static_assert(Dim == 1, "Dim != 1"); }	\
      template<int DIM = dim>						\
      typename std::enable_if<DIM != 1,type>::type Name() const { static_assert(Dim == 1, "Dim != 1"); }  \
    };									\
  };

  // examples of macro class creation:
  SOAX_ATTRIBUTE(id, "N");
  SOAX_ATTRIBUTE(pos, "P");
  SOAX_ATTRIBUTE(vel, "V");
  SOAX_ATTRIBUTE(acc, "A");

} // namespace SOAX

#endif
