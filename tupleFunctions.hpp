#ifndef tupleFunctions_hpp
#define tupleFunctions_hpp

namespace SOAX {

// Operation on Tuple functions -------------------------------------------------------------------------

struct Resize
{
  template<class T>
  static void doIt(T& t, int particleNumber) {
    t->resize(particleNumber);
  }
};

struct Reserve
{
  template<class T>
  static void doIt(T& t, int particleNumber) {
    t->reserve(particleNumber);
  }
};

struct Shrink_to_fit
{
  template<class T>
  static void doIt(T& t) {
    t->shrink_to_fit();
  }
};

struct Clear
{
  template<class T>
  static void doIt(T& t) {
    t->clear();
  }
};

struct Erase
{
  template<class T>
  static void doIt(T& t, int i) {
    t->operator[](i) = t->operator[](t->size()-1);
    t->pop_back();
  }
};

struct IncreaseNumber
{
  template<class T>
  static void doIt(T& t, int i) {
    t->resize(t->size()+i);
    //    t->increaseNumber(i);
  }
};

struct SetToValue
{
  template<class T, class Type>
  static void doIt(T& t, Type value) {
    for(int i=0;i<t->size();i++)
      t->operator[](i) = value;
  }
};

} // namespace SOAX

#endif
