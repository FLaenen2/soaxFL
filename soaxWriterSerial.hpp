#ifndef soaxWriterSerial_hpp
#define soaxWriterSerial_hpp

#include <fstream>

namespace SOAX {

// simple serial writer ---------------------------------------------------------------------------------------

struct WriteSerial
{
  template<class ArrayOfVector>
  static void write(const ArrayOfVector& array, std::string filename) {
    
    typedef typename ArrayOfVector::value_type::value_type DataType;
    
    std::ofstream myFile;
    myFile.open(filename.c_str(), std::ios::out | std::ios::binary | std::ios::app);

    for(int i=0;i<array.size();i++) {
      typename ArrayOfVector::value_type temp_array = array[i];
      myFile.write(reinterpret_cast<char const *>(temp_array.data()),sizeof(DataType)*array[i].size());
    }
    myFile.close();
  }

  static void close() { // close of not appending !!!
  }
};

} // namespace SOAX

#endif
