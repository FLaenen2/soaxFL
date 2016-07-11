#ifndef soaxReader_hpp
#define soaxReader_hpp

#include <string>
#include <algorithm>

#include "tupleHelper.hpp"

namespace SOAX {

template<class AttributeTuple>
class Reader
{
public:
  typedef AttributeTuple Attributes;

  Reader(std::string filename);
  ~Reader();
  
  template<class Storage>
  void read(Storage& array, std::string property, size_t startElement, size_t stopElement);

  std::vector<std::string> get_fileProperties() const { return fileProperties_; }
  std::vector<int> get_typeSizes() const { return typeSizes_;}
  size_t get_length() const { return length_; }
  size_t get_elementNumber() const { return elementNumber_; }
  std::string get_filename() const { return filename_; }
  
  
private:
  std::ifstream inputFileStream;

  std::vector<std::string> fileProperties_;
  std::vector<int> typeSizes_;
  size_t elementNumber_;
  size_t length_;
  std::string filename_;
  
  std::vector<std::string> getPropertyVec(std::string filename);
};

// implementations

template<class AttributeTuple>
Reader<AttributeTuple>::Reader(std::string filename) {
  filename_ = filename;
  
  inputFileStream.open(filename.c_str(), std::ifstream::binary);
  if(!inputFileStream) {
    std::cerr << "ERROR in Reader<AttributeTuple>::Reader(std::string filename):\n";
    std::cerr << "unable to open file " << filename << std::endl;
    exit(1);
  }
  inputFileStream.seekg(0, inputFileStream.end);
  length_ = inputFileStream.tellg();
  
  fileProperties_ = getPropertyVec(filename_);
  elementNumber_ = length_/sizeOfAttributes<Attributes>(fileProperties_);    
  sizeOfAttributes<Attributes>(fileProperties_, typeSizes_);

  if(length_%sizeOfAttributes<Attributes>(fileProperties_) != 0) {
    std::cerr << "ERROR in Reader<AttributeTuple>::Reader(std::string filename): length%sizeOfAttributes(fileProperties) != 0\n";
    for(int i=0;i<fileProperties_.size();i++)
      std::cerr << fileProperties_[i] << std::endl;
    exit(1);
  }
  
  // std::cerr << "length of file " << filename << " = " << length_ << std::endl;
  // std::cerr << "number of elements = " << elementNumber_ << std::endl;
}

template<class AttributeTuple>
Reader<AttributeTuple>::~Reader() 
{
  inputFileStream.close();
}

template<class AttributeTuple>
template<class Storage>  
void Reader<AttributeTuple>::read(Storage& array, std::string property, size_t startElement, size_t stopElement) {
  
  auto it = std::find(fileProperties_.begin(), fileProperties_.end(), property);
  
  if(it == fileProperties_.end()) {
    std::cerr << "ERROR in read: it == fileProperties_.end()" << std::endl;
    exit(1);
  }
  
  int index = it - fileProperties_.begin();
  
  // std::cout << "array.size() = " << array.size() << std::endl;
  // std::cout << "index = " << index << std::endl;
  // std::cout << "sizeType of index = " << typeSizes_[index] << std::endl;
  
  size_t offset = 0;
  for(int i=0;i<index;i++)
    offset += typeSizes_[i];
  
  offset *= elementNumber_;
  
  //  std::cerr << "offset = " << offset << std::endl;

  int typeSize = typeSizes_[index]/array.size();
  size_t elementOffset = typeSize*startElement;

  for(int i=0;i<array.size();i++) {  
    inputFileStream.seekg(offset + elementOffset + typeSize*elementNumber_*i, inputFileStream.beg);
    inputFileStream.read(reinterpret_cast<char*>(&array[i][0]),typeSize*(stopElement-startElement+1));
  }
}

template<class Storage>
std::vector<std::string> Reader<Storage>::getPropertyVec(std::string filename) {
  std::vector<std::string> fileProperties;
  
  std::string capitals = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  int foundCapital = filename.find_first_of(capitals);
  int foundCapital2 = filename.find_first_of(capitals,foundCapital+1);
  
  while(foundCapital2 != std::string::npos) {
    std::string property = filename.substr(foundCapital,foundCapital2-foundCapital);
    fileProperties.push_back(property);
    
    //  std::cout << filename.substr(foundCapital,foundCapital2-foundCapital) << std::endl;
    
    foundCapital = filename.find_first_of(capitals, foundCapital+1);
    foundCapital2 = filename.find_first_of(capitals, foundCapital+1);
  }
  foundCapital2 = filename.find_first_of("_", foundCapital+1);
  //  std::cout << filename.substr(foundCapital,foundCapital2-foundCapital) << std::endl;
  
  std::string property = filename.substr(foundCapital,foundCapital2-foundCapital);
  fileProperties.push_back(property);
  
  return fileProperties;
}
  
} // namespace SOAX

#endif
