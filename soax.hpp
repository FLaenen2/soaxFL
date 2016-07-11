#ifndef soax_hpp
#define soax_hpp

#include <cassert>
#include <fstream>
#include <type_traits>
#include <vector>

#include "tupleManip.hpp"
#include "type2String.hpp"
#include "storage.hpp"
#include "toString.hpp"
#include "tupleHelper.hpp"
#include "tupleFunctions.hpp"
#include "soaxReader.hpp"

namespace SOAX {

template <class AttributeTuple>
  class Soax : public GenScatterHierarchyTuple<SoaxAttributeInheritor<AttributeTuple>::template Inheritor, AttributeTuple>
{
 public:

  class Element :  public GenScatterHierarchyTuple<SoaxElementInheritor<AttributeTuple>::template Inheritor, AttributeTuple>
  {
  public:
    
    template<class SOAX>
    Element(SOAX& soax, int i) {
      TupleObjectFromSoax<AttributeTuple,std::tuple_size<AttributeTuple>::value>::doIt(*this,soax,i);
      // for(int j=0; j<std::tuple_element<1,AttributeTuple>::type::dim; j++)
      // 	std::tuple_element<1,AttributeTuple>::type::Object::value[j] =
      // 	  soax.std::tuple_element<1,AttributeTuple>::type::template Access<IndexOfAtt<typename std::tuple_element<1,AttributeTuple>::type,AttributeTuple>::value,AttributeTuple>::data_[j][i];1
    }
  };
  
  typedef AttributeTuple Attributes;

  enum { totalAttributeNumber = TupleSumDim<Attributes,std::tuple_size<Attributes>::value>::dim };
  
  typedef Element ElementType;
  typedef typename TupleCatAttributes<Attributes,std::tuple_size<Attributes>::value>::type TupleType;
  typedef typename transformTupleT2VecP<std::tuple_element<0,AttributeTuple>::type::template ContainerTypeTemplate,TupleType>::type AttributeTypeVec; /// ????
  
  enum { attributeNumber = std::tuple_size<AttributeTypeVec>::value };
  enum { elementSize = TupleSumTypeSize<Attributes,std::tuple_size<Attributes>::value>::size };

  Soax() {
    FillAttributeTupleInVec<Attributes,std::tuple_size<Attributes>::value>::fillIn(dataTuple_, this);
    OperationOnTuple(dataTuple_,Type2Type<Resize>(),0);
    FillInName<Attributes,std::tuple_size<Attributes>::value>::fillInName(name_);
  }

  Soax(int particleNumber) {

    FillAttributeTupleInVec<Attributes,std::tuple_size<Attributes>::value>::fillIn(dataTuple_, this);
    OperationOnTuple(dataTuple_,Type2Type<Resize>(),particleNumber);
    FillInName<Attributes,std::tuple_size<Attributes>::value>::fillInName(name_);
  }
  
  AttributeTypeVec dataTuple_;

  ElementType getElement(int i) const {
    assert(i < size());
    return Element(*this,i);
  }

  TupleType getTuple(int i) const {
    TupleType particle;
    TupleFillFromVector<TupleType,std::tuple_size<TupleType>::value,AttributeTypeVec>::fill(particle,dataTuple_,i);
    return particle;
  }

  template<class Type>
  long long int push_back(Type element) {
    OperationOnTuple(dataTuple_,Type2Type<IncreaseNumber>(),1);
    pushBackImpl(element);
    
    return size()-1;
  }

  long long int push_back() {
    OperationOnTuple(dataTuple_,Type2Type<IncreaseNumber>(),1);
    return size()-1;
  }

  void read(std::vector<std::string> propertyLetterVec, std::string filename, size_t startElement = 42, size_t stopElement = 23)   {

    Reader<Attributes> reader(filename);
    
    size_t firstElement = startElement;
    size_t lastElement = stopElement;
    
    if((startElement == 42) && (stopElement == 23)) { 
      // reading all elements
    
      firstElement = 0;
      lastElement = reader.get_elementNumber() - 1;
    }  
    else if( (stopElement - startElement) > reader.get_elementNumber() || 
	     startElement > stopElement || startElement > reader.get_elementNumber() || stopElement > reader.get_elementNumber()) {

      std::cerr << "ERROR in Soax::read: startElement < 0 || stopElement < 0 || (stopElement - startElement) > elementNumber_\n" <<
	"|| startElement > stopElement || startElement > elementNumber || stopElement > elementNumber\n";

      std::cerr << "startElement = " << startElement << std::endl;
      std::cerr << "stopElement = " << stopElement << std::endl;
      std::cerr << "file elements = " << reader.get_elementNumber() << std::endl;
      
      exit(1);
    }
    
    clear();

    resize(lastElement - firstElement + 1);
    
    for(std::string property : propertyLetterVec) {
      bool read = false;
      ReadArrayOfProperty<Attributes,std::tuple_size<Attributes>::value>::read(this, property, reader, firstElement, lastElement, read);
      if(read == false) {
	std::cerr << "ERROR in Soax::read: reading of property " << property << " failed\n";
	exit(1);
      }
    }
  }
  
  template<class Writer>
  void write(std::vector<std::string> propertyLetterVec, std::string dir = ".", std::string filenameExtension = "") 
  {
    std::string filename = dir + "/" + "p_";
    for(std::string letter : propertyLetterVec) 
      filename += letter;

    filename += "_";

    if(filenameExtension != "")
      filename += filenameExtension;
    filename += ".bin";
    
    std::remove(filename.c_str());
    
    for(std::string letter : propertyLetterVec)
      write<Writer>(letter, filename);
    
    Writer::close();
    
    filename = dir + "/" + "soax_name.txt";
    std::ofstream out(filename.c_str());
    out << name() << std::endl;
    out.close();
  }

  template<class Writer>
  void write(std::string letter, std::string filename) 
  {
    bool written = false;
    TupleWrite<Attributes,std::tuple_size<Attributes>::value,Writer>::write(this, letter, filename, written);
    if(!written) {
      std::cerr << "ERROR in write for letter = " << letter << " : written = false\n";
      exit(1);
    }
  }

  int size() const {
    return std::tuple_element<0,Attributes>::type::template Access<0,Attributes>::data_[0].size();
  }

  int capacity() const {
    return std::tuple_element<0,Attributes>::type::template Access<0,Attributes>::data_[0].capacity();
  }

  void erase(int i) {
    OperationOnTuple(dataTuple_,Type2Type<Erase>(),i);
  }

  void reserve(int n) {
    OperationOnTuple(dataTuple_,Type2Type<Reserve>(),n);
  }

  void resize(int n) {
    OperationOnTuple(dataTuple_,Type2Type<Resize>(),n);
  }

  void shrink_to_fit() {
    OperationOnTuple(dataTuple_,Type2Type<Shrink_to_fit>());
  }

  void clear() {
    OperationOnTuple(dataTuple_,Type2Type<Clear>());
  }

  template<class Operation, class... Args>
  void apply(Args... args) {
    OperationOnTuple(dataTuple_, Type2Type<Operation>(), args...);
  }

  std::string name() const { return name_; }
  
 private:
  int attributeNumber_;
  int particleNumber_;
  std::string name_;

  void  pushBackImpl(ElementType element) {
    TupleObjectToSoax<AttributeTuple,std::tuple_size<AttributeTuple>::value>::doIt(element,*this,size()-1);
  }

  void pushBackImpl(TupleType tuple) {
    Tuple2Vector<TupleType,std::tuple_size<AttributeTypeVec>::value,AttributeTypeVec>::fill(dataTuple_,tuple,size()-1);    
  }
};

// ---------------------------------------------------------
// Object Printer
// ---------------------------------------------------------

/*template<class Object, std::size_t N>
struct ObjectPrinter {
  static void print(std::ostream& of, const Object& obj) 
  {
    size_t dim = std::tuple_element<N-1,typename Object::AttributeTuple>::type::dim;

    ObjectPrinter<Object, N-1>::print(of,obj);
    of << "\t";
    for(int j=0; j < dim - 1; j++)
      obj.std::tuple_element<N-1,typename Object::AttributeTuple>::type::Object::value[j];
    
    of << obj.std::tuple_element<N-1,typename Object::AttributeTuple>::type::Object::value[dim-1];
  }
};//*/
 
/*template<class Object>
struct ObjectPrinter<Object, 1> {
  static void print(std::ostream& of, const Object& obj) 
  {
    size_t dim = std::tuple_element<0,typename Object::AttributeTuple>::type::dim;
    
    for(int j=0; j < dim - 1; j++)
      of << obj.std::tuple_element<0,typename Object::AttributeTuple>::type::Object::value[j] << "\t";
    
    of << obj.std::tuple_element<0,typename Object::AttributeTuple>::type::Object::value[dim-1];
  }
};//*/

} // namespace SOAX

// template<class... Args>
// std::ostream& operator<<(std::ostream& of, const typename SOAX::Soax<std::tuple<Args...>>::ElementType& obj) 
// {
//   typedef std::tuple<Args...> AttributeTuple;

//   SOAX::ObjectPrinter<decltype(obj), std::tuple_size<AttributeTuple>::value>::print(of,obj);
  
//   return of;
// }

#endif
