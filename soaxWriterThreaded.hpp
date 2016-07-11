#ifndef soaxWriterThreaded_hpp
#define soaxWriterThreaded_hpp

#include <fstream>
#include <thread>
#include <mutex>

namespace SOAX {

// simple threaded writer ---------------------------------------------------------------------------------------

extern std::mutex writeMutex;

template<class ArrayOfListVector>
void writeThreaded(const ArrayOfListVector& array, std::string filename) {
  typedef typename ArrayOfListVector::value_type::value_type DataType;

  writeMutex.lock();
  
  std::ofstream myFile;
  myFile.open(filename.c_str(), std::ios::out | std::ios::binary | std::ios::app);
  for(int i=0;i<array.size();i++)
    myFile.write(reinterpret_cast<const char *>(array[i].get_data()),sizeof(DataType)*array[i].number());
  myFile.close();

  writeMutex.unlock();
}

struct WriteThreaded
{
  template<class ArrayOfListVector>
  static void write(const ArrayOfListVector& array, std::string filename) {
    std::thread writeThread(writeThreaded<ArrayOfListVector>,array,filename);
    writeThread.detach();
  }
  
  static void close () { // close of not appending !!!
  }
};

} // namespace SOAX

#endif
