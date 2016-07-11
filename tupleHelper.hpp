#ifndef tupleHelper_hpp
#define tupleHelper_hpp

#include <algorithm>

#include "tupleManip.hpp"

namespace SOAX {

// AttributeInheritor

template<class Tuple>
struct SoaxAttributeInheritor
{
  template<class T>
  class Inheritor : public T::template Access<IndexOfAtt<T,Tuple>::value,Tuple> {};
};

// ElementInheritor

template<class Tuple>
struct SoaxElementInheritor
{
  template<class T>
  class Inheritor : public T::Object {};
};

// SumDim

template<class Tuple, int N>
struct TupleSumDim
{
  enum { dim = TupleSumDim<Tuple,N-1>::dim + std::tuple_element<N-1,Tuple>::type::dim };
};

template<class Tuple>
  struct TupleSumDim<Tuple,1>
{
  enum { dim = std::tuple_element<0,Tuple>::type::dim };
};

// TupleObjectFromSoax

template<class AttributeTuple, int N>
struct TupleObjectFromSoax
{
  template<class Obj, class Soax>
  static void doIt(Obj& obj, const Soax& soax, int i) {
    TupleObjectFromSoax<AttributeTuple,N-1>::doIt(obj,soax,i);
    for (int j = 0; j < std::tuple_element<N-1,AttributeTuple>::type::dim; j++) {
      obj.std::tuple_element<N - 1, AttributeTuple>::type::Object::value[j] = soax.std::tuple_element<
              N - 1, AttributeTuple>::type::template Access<IndexOfAtt<typename std::tuple_element<
              N - 1, AttributeTuple>::type, AttributeTuple>::value, AttributeTuple>::data_[j][i];
    }
  }
};
  
template<class AttributeTuple>
struct TupleObjectFromSoax<AttributeTuple,1>
{
  template<class Obj, class Soax>
  static void doIt(Obj& obj, const Soax& soax, int i) {
    for(int j=0; j<std::tuple_element<0,AttributeTuple>::type::dim; j++) {
      obj.std::tuple_element<0, AttributeTuple>::type::Object::value[j] = soax.std::tuple_element<0, AttributeTuple>::type::template Access<0, AttributeTuple>::data_[j][i];
    }
  }
};

// TupleObjectToSoax

template<class AttributeTuple, int N>
struct TupleObjectToSoax
{
  template<class Obj, class Soax>
  static void doIt(const Obj& obj, Soax& soax, int i) {
    TupleObjectToSoax<AttributeTuple,N-1>::doIt(obj,soax,i);
    for(int j=0; j<std::tuple_element<N-1,AttributeTuple>::type::dim; j++)

      soax.std::tuple_element<N-1,AttributeTuple>::type::template Access<IndexOfAtt<typename std::tuple_element<N-1,AttributeTuple>::type,AttributeTuple>::value,AttributeTuple>::data_[j][i] =  
	obj.std::tuple_element<N-1,AttributeTuple>::type::Object::value[j];
  }
};
  
template<class AttributeTuple>
struct TupleObjectToSoax<AttributeTuple,1>
{
  template<class Obj, class Soax>
  static void doIt(const Obj& obj, Soax& soax, int i) {
    for(int j=0; j<std::tuple_element<0,AttributeTuple>::type::dim; j++)
      soax.std::tuple_element<0,AttributeTuple>::type::template Access<0,AttributeTuple>::data_[j][i] = obj.std::tuple_element<0,AttributeTuple>::type::Object::value[j]; 
  }
};

// SumTypeSize

template<class Tuple, int N>
struct TupleSumTypeSize
{
  enum { size = TupleSumTypeSize<Tuple,N-1>::size + std::tuple_element<N-1,Tuple>::type::dim*sizeof(typename std::tuple_element<N-1,Tuple>::type::type) };
};

template<class Tuple>
  struct TupleSumTypeSize<Tuple,1>
{
  enum { size = std::tuple_element<0,Tuple>::type::dim*sizeof(typename std::tuple_element<0,Tuple>::type::type) };
};

// SumTypeSizeOfAttributeTuple

template<class AttributeTuple, int N>
struct TupleTypeSizeOfAttributeTuple
{
  template<class Vector>
  static void doIt(std::vector<std::string> attributeVec, Vector& vec)
  {
    TupleTypeSizeOfAttributeTuple<AttributeTuple, N-1>::doIt(attributeVec, vec);
    if(std::find(attributeVec.begin(),attributeVec.end(), std::tuple_element<N-1,AttributeTuple>::type::name()) != attributeVec.end())
      vec.push_back(std::tuple_element<N-1,AttributeTuple>::type::dim*sizeof(typename std::tuple_element<N-1,AttributeTuple>::type::type));

  }
};

template<class AttributeTuple>
struct TupleTypeSizeOfAttributeTuple<AttributeTuple,1>
{
  template<class Vector>
  static void doIt(std::vector<std::string> attributeVec, Vector& vec)
  {
    if(std::find(attributeVec.begin(),attributeVec.end(), std::tuple_element<0,AttributeTuple>::type::name()) != attributeVec.end())
      vec.push_back(std::tuple_element<0,AttributeTuple>::type::dim*sizeof(typename std::tuple_element<0,AttributeTuple>::type::type));

  }
};

// sizeOfAttributes
template<class Attributes, class Vector>
void sizeOfAttributes(std::vector<std::string> attributeVec, Vector& vec) {
    TupleTypeSizeOfAttributeTuple<Attributes,std::tuple_size<Attributes>::value>::doIt(attributeVec,vec);
}

// SumTypeSizeOfAttributeTuple

template<class AttributeTuple, int N>
struct TupleSumTypeSizeOfAttributeTuple
{
  static void doIt(std::vector<std::string> attributeVec, size_t& size)
  {
    TupleSumTypeSizeOfAttributeTuple<AttributeTuple, N-1>::doIt(attributeVec, size);
    if(std::find(attributeVec.begin(),attributeVec.end(), std::tuple_element<N-1,AttributeTuple>::type::name()) != attributeVec.end())
      size += std::tuple_element<N-1,AttributeTuple>::type::dim*sizeof(typename std::tuple_element<N-1,AttributeTuple>::type::type);

  }
};

template<class AttributeTuple>
struct TupleSumTypeSizeOfAttributeTuple<AttributeTuple,1>
{
  static void doIt(std::vector<std::string> attributeVec, size_t& size)
  {
    size = 0;
    if(std::find(attributeVec.begin(),attributeVec.end(), std::tuple_element<0,AttributeTuple>::type::name()) != attributeVec.end())
      size += std::tuple_element<0,AttributeTuple>::type::dim*sizeof(typename std::tuple_element<0,AttributeTuple>::type::type);

  }
};

// sizeOfAttributes
template<class Attributes>
size_t sizeOfAttributes(std::vector<std::string> attributeVec) {
    size_t size;
    TupleSumTypeSizeOfAttributeTuple<Attributes,std::tuple_size<Attributes>::value>::doIt(attributeVec,size);
    return size;
}


// CatAttributes

template<class Tuple, int N>
struct TupleCatAttributes
{
  typedef typename TupleTypeCat<typename TupleCatAttributes<Tuple,N-1>::type,typename std::tuple_element<N-1,Tuple>::type::Types>::type type; 
};

template<class Tuple>
  struct TupleCatAttributes<Tuple,1>
{
  typedef typename std::tuple_element<0,Tuple>::type::Types type;
};

// PushBackAttributes

template<class Tuple, int N>
struct TuplePushBackAttributes
{
  typedef typename TupleTypePushBack<typename TupleCatAttributes<Tuple,N-1>::type,typename std::tuple_element<N-1,Tuple>::type::Types>::type type; 
};

template<class Tuple>
  struct TuplePushBackAttributes<Tuple,1>
{
  typedef typename std::tuple_element<0,Tuple>::type::Types type;
};

// Fill Attributes in Vec --------------------------------------------------------------------------------------

template<class AttributeTuple, class Attribute, int startIndex, int N>
struct FillAttributeInVec
{
  template<class DataTuple, class PArray>
    static void fillIn(DataTuple& dataVec, PArray pArray) {
    FillAttributeInVec<AttributeTuple, Attribute, startIndex, N-1>::fillIn(dataVec, pArray);
    std::get<N-1 + startIndex>(dataVec) = &pArray->Attribute::template Access<startIndex,AttributeTuple>::data_[N-1];
  }
};

template<class AttributeTuple, class Attribute, int startIndex>
struct FillAttributeInVec<AttributeTuple, Attribute, startIndex, 1>
{
  template<class DataTuple, class PArray>
  static void fillIn(DataTuple& dataVec, PArray pArray) {
    std::get<startIndex>(dataVec) = &pArray->Attribute::template Access<startIndex,AttributeTuple>::data_[0];
  }
};

template<class AttributeTuple, int N>
struct FillAttributeTupleInVec
{
  template<class DataTuple, class PArray>
  static void fillIn(DataTuple& dataVec, PArray pArray) {
    FillAttributeTupleInVec<AttributeTuple, N-1>::fillIn(dataVec, pArray);
    FillAttributeInVec<AttributeTuple, typename std::tuple_element<N-1,AttributeTuple>::type,
      TupleSumDim<AttributeTuple,N-1>::dim,
      std::tuple_element<N-1,AttributeTuple>::type::dim>::fillIn(dataVec,pArray);
  }
};

template<class AttributeTuple>
struct FillAttributeTupleInVec<AttributeTuple,1>
{
  template<class DataTuple, class PArray>
  static void fillIn(DataTuple& dataVec, PArray pArray) {
    FillAttributeInVec<AttributeTuple,typename std::tuple_element<0,AttributeTuple>::type,
      0,
      std::tuple_element<0,AttributeTuple>::type::dim>::fillIn(dataVec,pArray);
  }
};

// TupleFillFromVector

template<class Tuple, std::size_t N, class TupleVec>
struct TupleFillFromVector {
  static void fill(Tuple& t, const TupleVec& v, int i) 
  {
    TupleFillFromVector<Tuple, N-1, TupleVec>::fill(t,v,i);
    std::get<N-1>(t) = std::get<N-1>(v)->operator[](i);
  }
};
  
template<class Tuple, class TupleVec>
struct TupleFillFromVector<Tuple, 1, TupleVec> {
  static void fill(Tuple& t, const TupleVec& v, int i) 
  {
    std::get<0>(t) = std::get<0>(v)->operator[](i);
  }
};

// Tuple2Vector

template<class Tuple, std::size_t N, class TupleVec>
struct Tuple2Vector {
  static void fill(TupleVec& v, const Tuple& t, int i) 
  {
    Tuple2Vector<Tuple, N-1, TupleVec>::fill(v,t,i);
    std::get<N-1>(v)->operator[](i) = std::get<N-1>(t);
  }
};
  
template<class Tuple, class TupleVec>
struct Tuple2Vector<Tuple, 1, TupleVec> {
  static void fill(TupleVec& v, const Tuple& t, int i) 
  {
    std::get<0>(v)->operator[](i) = std::get<0>(t);
  }
};

// FillInName --------------------------------------------------------------------------------------

template<class Attributes, std::size_t N>
struct FillInName {
  static void fillInName(std::string& name) 
  {
    FillInName<Attributes, N-1>::fillInName(name);
    name += toString(std::tuple_element<N-1,Attributes>::type::name(),
		     std::tuple_element<N-1,Attributes>::type::dim,
		     Type2String<typename std::tuple_element<N-1,Attributes>::type::type>::get(),
		     sizeof(typename std::tuple_element<N-1,Attributes>::type::type),"_");
  }
};
  
template<class Attributes>
struct FillInName<Attributes, 1> {
  static void fillInName(std::string& name) 
  {
    name += toString(std::tuple_element<0,Attributes>::type::name(),
		     std::tuple_element<0,Attributes>::type::dim,
		     Type2String<typename std::tuple_element<0,Attributes>::type::type>::get(),
		     sizeof(typename std::tuple_element<0,Attributes>::type::type),"_");
  }
};

// Writer

template<class Array, class Writer>
  void writeProperty(Array array, std::string filename, Type2Type<Writer> temp) {
  
  typedef typename std::remove_pointer<Array>::type StdArray;
  
  Writer::write(*array,filename);
}

template<class AttributeTuple, int N, class Writer>
struct TupleWrite
{
  template<class PArray>
  static void write(PArray pArray, std::string letter, std::string filename, bool& written) {
    TupleWrite<AttributeTuple, N-1, Writer>::write(pArray, letter, filename, written);
    if(letter == std::tuple_element<N-1,AttributeTuple>::type::name()) {
      typedef typename std::tuple_element<N-1,AttributeTuple>::type ATTRIBUTE;
      writeProperty(&pArray->ATTRIBUTE::template Access<IndexOfAtt<ATTRIBUTE,AttributeTuple>::value,AttributeTuple>::data_, filename,Type2Type<Writer>());
      written = true;
    }
  }
};

template<class AttributeTuple,class Writer>
struct TupleWrite<AttributeTuple,1,Writer>
{
  template<class PArray>
    static void write(PArray pArray, std::string letter, std::string filename, bool& written) {
    if(letter == std::tuple_element<0, AttributeTuple>::type::name()) {
      typedef typename std::tuple_element<0,AttributeTuple>::type ATTRIBUTE;
      writeProperty(&pArray->ATTRIBUTE::template Access<0,AttributeTuple>::data_, filename, Type2Type<Writer>());
      written = true;
    }
  }
};

// ReadArrayOfProperty

template<class Array, class Reader>
void readProperty(Array array, Reader& reader, std::string property, size_t startElement, size_t stopElement) {
  
  typedef typename std::remove_pointer<Array>::type StdArray;
  
  reader.read(*array, property, startElement, stopElement);
}


template<class AttributeTuple, int N>
struct ReadArrayOfProperty
{
  template<class PArray, class Reader>
  static void read(PArray pArray, std::string property, Reader& reader, size_t startElement, size_t stopElement, bool& read) {
    ReadArrayOfProperty<AttributeTuple, N-1>::read(pArray, property, reader, startElement, stopElement, read);
    if(property == std::tuple_element<N-1,AttributeTuple>::type::name()) {
      typedef typename std::tuple_element<N-1,AttributeTuple>::type ATTRIBUTE;
      readProperty(&pArray->ATTRIBUTE::template Access<IndexOfAtt<ATTRIBUTE,AttributeTuple>::value,AttributeTuple>::data_, reader, property, startElement, stopElement);
      read = true;
    }
  }
};

template<class AttributeTuple>
struct ReadArrayOfProperty<AttributeTuple,1>
{
  template<class PArray, class Reader>
  static void read(PArray pArray, std::string property, Reader& reader, size_t startElement, size_t stopElement, bool& read) {
    if(property == std::tuple_element<0,AttributeTuple>::type::name()) {
      typedef typename std::tuple_element<0,AttributeTuple>::type ATTRIBUTE;
      readProperty(&pArray->ATTRIBUTE::template Access<IndexOfAtt<ATTRIBUTE,AttributeTuple>::value,AttributeTuple>::data_, reader, property, startElement, stopElement);
      read = true;
    }
  }
};



} // namespace SOAX

#endif
