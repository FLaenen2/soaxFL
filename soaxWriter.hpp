#ifndef soaxWriter_hpp
#define soaxWriter_hpp

#include <fstream>
#include <thread>
#include <mutex>

namespace SOAX {

// template to implement:

struct WriteTemplate
{
  template<class ArrayOfListVector>
  static void write(const ArrayOfListVector& array, std::string filename) {
    
    // type of data (int, float, double, ...)
    // typedef typename ArrayOfListVector::value_type::value_type DataType;

    // your code is here!

  }

  static void close() {

    // your code is here!

  }
};

} // namespace SOAX

#endif
