#ifndef testClass_hpp
#define testClass_hpp

// define your custom flatt class
struct Test
{
public:
  float test1 = 1.1;
  double test2 = 2.2;
};

// define the stream operator<<
std::ostream& operator<<(std::ostream& of, const Test& t) 
{
  of << t.test1 << "\t" << t.test2;
  
  return of;
}


// define identification letter
namespace SOAX {

template<>
struct Type2String<Test>
{
  static std::string get() { return "t"; }
};
}


#endif
