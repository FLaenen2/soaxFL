#ifndef type2String_hpp
#define type2String_hpp

namespace SOAX {

template<bool intType, bool floatType, class... T>
struct Type2StringPod {};

template<class... T>
struct Type2String {
  static std::string get() {
    return Type2StringPod<std::is_integral<T...>::value,std::is_floating_point<T...>::value,T...>::get();
  }
};

template<class... T>
struct Type2StringPod<true,false,T...> 
{
  static std::string get() { return "i"; }
};

template<class... T>
struct Type2StringPod<false,true,T...> 
{
  static std::string get() { return "f"; }
};

template<class... T>
struct Type2StringPod<false,false,T...> 
{
};

} // namespace SOAX

#endif
