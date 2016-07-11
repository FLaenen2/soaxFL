#ifndef array_hpp
#define array_hpp

#include <array>
#include "type2String.hpp"

// define the stream operator<<
template<std::size_t N>
std::ostream& operator<<(std::ostream& of, const std::array<float,N>& t) 
{
  for(int i=0;i<N;i++)
    of << t[i];
  
  return of;
}

// define identification letter
namespace SOAX {

template<typename T, std::size_t N>
struct Type2String<std::array<T,N>>
{
  static std::string get() { return "A"; }
};
}

#endif
