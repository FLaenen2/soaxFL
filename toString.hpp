#ifndef toString_hpp
#define toString_hpp

#include <string>
#include <sstream>

namespace SOAX {

// add an argument of type T to string
template<typename T>
void stringToStream(std::ostringstream& oss, T arg)
{
  oss << arg;
}

// termination point of recursion
template<typename T>
void convertToString(std::ostringstream& oss, T arg)
{
  stringToStream(oss,arg);
}

// recursion until only one template argument is left
template<typename First, typename ... Rest>
void convertToString(std::ostringstream& oss, First first, Rest ... rest) 
{
  stringToStream(oss,first);
  convertToString(oss,rest...);
}

// function adding a list (of variable length) of arguments (of
// different types) to a stream
template<typename First, typename ... Rest>
std::string toString(First first, Rest ... rest)
{
  std::ostringstream oss;
  convertToString(oss,first,rest...);
  return oss.str();
}

} // namespace SOAX

#endif
