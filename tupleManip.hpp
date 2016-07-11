#ifndef tupleManip_hpp
#define tupleManip_hpp

#include <sstream>
#include <limits>
#include <iostream>
#include <tuple>

namespace SOAX {

template <int v>
struct Int2Type
{
  enum { value = v };
};

template <typename T>
struct Type2Type
{
  typedef T OriginalType;
};

struct Printer
{
  template<class T>
  static void doIt(T& t) {
    std::cout << "printing " << t << std::endl;;
  }
};

struct AddToStringStream
{
  template<class T>
  static void doIt(T& t, std::stringstream* s, std::string delimiter = "") {
    *s << t << delimiter;
  }
};

template<class Tuple, std::size_t N, class DoItClass>
struct TupleDo {
    
  template<class... Args>
  static void doIt(Tuple& t, Args... args)
  {
    TupleDo<Tuple, N-1, DoItClass>::doIt(t,args...);
    DoItClass::doIt(std::get<N-1>(t),args...);
  }
  
};

template<class Tuple, class DoItClass>
  struct TupleDo<Tuple, 1, DoItClass> {
  template<class... Args>
  static void doIt(Tuple& t, Args... args)
  {
     DoItClass::doIt(std::get<0>(t),args...);
  }
};

template<class Tuple, class Operation, class... Args>
void OperationOnTuple(Tuple& t, Type2Type<Operation>, Args... args)
{
  TupleDo<Tuple,std::tuple_size<Tuple>::value,Operation>::doIt(t,args...);
}

// -----------------------------------------------------------------------

template <template <typename> class VecType, typename... T>
struct transformTupleT2VecP {};

template <template <typename> class VecType, typename... T>
struct transformTupleT2VecP<VecType, std::tuple<T...>>
{
  using type = std::tuple<VecType<T>*...>;
};

// -------------------------------------------------------------------------
// Concatenate types of std::tuple
// -------------------------------------------------------------------------

template<class T, class... Types>
struct TupleTypeCat;

template<class T, class... Types>
struct TupleTypeCat<T, std::tuple<Types...>> {
  typedef std::tuple<T,Types...> type;
};

template<class T, class... Types>
  struct TupleTypeCat<std::tuple<Types...>, T> {
  typedef std::tuple<Types...,T> type;
};

template<class... Types1, class... Types2>
  struct TupleTypeCat<std::tuple<Types1...>, std::tuple<Types2...>> {
  typedef std::tuple<Types1...,Types2...> type;
};

template<class T, class T2>
  struct TupleTypeCat<T, T2> {
  typedef std::tuple<T,T2> type;
};


// -------------------------------------------------------------------------
// Push-back for type of std::tuple
// -------------------------------------------------------------------------

template < typename Tuple, typename T >
struct TupleTypePushBack;

template < typename T, typename ... Args >
struct TupleTypePushBack<std::tuple<Args...>, T>
{
    typedef std::tuple<Args...,T> type;
};

template < typename ... Args2, typename ... Args1 >
struct TupleTypePushBack<std::tuple<Args1...>, std::tuple<Args2...>>
{
    typedef std::tuple<Args1...,Args2...> type;
};

template < typename T, typename T2 >
struct TupleTypePushBack<T2, T>
{
    typedef std::tuple<T2,T> type;
};

// -------------------------------------------------------------------------
// Create tuple of type dim x T
// -------------------------------------------------------------------------

template <class T, int dim>
struct TupleTypeDim {
  typedef typename TupleTypeCat<T,typename TupleTypeDim<T,dim-1>::type>::type type;
};

template <class T>
struct TupleTypeDim<T,1> {
  typedef std::tuple<T> type;
};

// -------------------------------------------------------------------------
// Get index of type T
// -------------------------------------------------------------------------

template <class T, class Tuple>
struct Index;

template <class T, class... Types>
struct Index<T, std::tuple<T, Types...>> {
    static const std::size_t value = 0;
};

template <class T, class U, class... Types>
struct Index<T, std::tuple<U, Types...>> {
    static const std::size_t value = 1 + Index<T, std::tuple<Types...>>::value;
};

// -------------------------------------------------------------------------
// get index of type T having enum dim
// -------------------------------------------------------------------------

template <class T, class Tuple>
struct IndexOfAtt;

 template <class T>
struct IndexOfAtt<T, std::tuple<>> {
    static const int value = std::numeric_limits<int>::min();
};

template <class T, class... Types>
struct IndexOfAtt<T, std::tuple<T, Types...>> {
    static const int value = 0;
};

template <class T, class U, class... Types>
struct IndexOfAtt<T, std::tuple<U, Types...>> {
  static const int value = U::dim + IndexOfAtt<T, std::tuple<Types...>>::value;
};

// ---------------------------------------------------------
// Class Generator
// ---------------------------------------------------------

template<template<class T> class Unit, class Type>
class GenScatterHierarchyTuple;

template<template<class T> class Unit, class AtomicType, class... TypeList>
class GenScatterHierarchyTuple<Unit, std::tuple<AtomicType, TypeList...>>
  : public GenScatterHierarchyTuple<Unit, AtomicType>
  , public GenScatterHierarchyTuple<Unit, std::tuple<TypeList...>>
{};

template<template<class T> class Unit>
class GenScatterHierarchyTuple<Unit, std::tuple<>>
{};

template<template<class T> class Unit, class AtomicType>
class GenScatterHierarchyTuple
  : public Unit<AtomicType>
{};


// ---------------------------------------------------------
// Tuple Printer
// ---------------------------------------------------------

template<class Tuple, std::size_t N>
struct TuplePrinter {
  static void print(const Tuple& t) 
  {
    TuplePrinter<Tuple, N-1>::print(t);
    std::cout << "\t" << std::get<N-1>(t);
  }
  static void print(std::ostream& of, const Tuple& t) 
  {
    TuplePrinter<Tuple, N-1>::print(of,t);
    of << "\t" << std::get<N-1>(t);
  }
};
 
template<class Tuple>
struct TuplePrinter<Tuple, 1> {
    static void print(const Tuple& t) 
    {
        std::cout << std::get<0>(t);
    }
    static void print(std::ostream& of, const Tuple& t) 
    {
      of << std::get<0>(t);
    }
};
 
template<class... Args>
void print(const std::tuple<Args...>& t) 
{
  //    std::cout << "(";
    TuplePrinter<decltype(t), sizeof...(Args)>::print(t);
    std::cout << std::endl;
    //    std::cout << ")\n";
}
 
} // namespace SOAX

template<class... Args>
std::ostream& operator<<(std::ostream& of, const std::tuple<Args...>& t) 
{
  SOAX::TuplePrinter<decltype(t), sizeof...(Args)>::print(of,t);
  
  return of;
}

#endif
