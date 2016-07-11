// oarsub -I -p "gpu='YES'"
// Example file for the use of SoAx in the context of scientific
// particle simulations

// compilation example: g++ -std=c++11 -DNDEBUG -o -O3 -mavx main main.cpp
#ifdef __NVCC__
	#include <thrust/device_vector.h>
	#include <thrust/iterator/iterator_adaptor.h>
#endif
#include <iostream>
#include <tuple>
#include <string>
#include <fstream>
#include <array>
#include <algorithm>

#include "array.hpp"
#include "soax.hpp"
#include "soaxAttributes.hpp"
#include "soaxWriterSerial.hpp"

//#include "testClass.hpp"

//#include <mpi.h>

// derive repeat_iterator from iterator_adaptor
/*template<typename Iterator>
  class repeat_iterator
    : public thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator>
{
  public:
    // shorthand for the name of the iterator_adaptor we're deriving from
    typedef thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator> super_t;
    __host__ __device__
    repeat_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
    // befriend thrust::iterator_core_access to allow it access to the private interface below
    friend class thrust::iterator_core_access;
  private:
    // repeat each element of the adapted range n times
    unsigned int n;
    // used to keep track of where we began
    const Iterator begin;
    // it is private because only thrust::iterator_core_access needs access to it
    __host__ __device__
    typename super_t::reference dereference() const
    {
      return *(begin + (this->base() - begin) / n);
    }
};//*/

// Definitions of attributes for the particle
SOAX_ATTRIBUTE(id, "N");       // identity (number) 
SOAX_ATTRIBUTE(pos, "P");      // position 
SOAX_ATTRIBUTE(vel, "V");      // velocity 
SOAX_ATTRIBUTE(acc, "A");      // acceleration 
SOAX_ATTRIBUTE(radius, "R");   // radius 
SOAX_ATTRIBUTE(test, "T");     // custom test class
SOAX_ATTRIBUTE(hist, "H");     // some history of the particle

void testfunction(std::string a) {}
/*
class Iterator
{
public:
  using difference_type = std::ptrdiff_t;
  using value_type = float;
  using pointer = float*;
  using reference = float&;
  using iterator_category = std::random_access_iterator_tag;
  
  __host__ __device__ Iterator(thrust::device_vector<float> &df, int i) : ptr_(&df), i_(i) { }

  __host__ Iterator(thrust::device_vector<float> *df, int i) : i_(i) {
    printf("Iterator(thrust::device_vector<float> *df, int i)\n");
    thrust::device_vector<float> *tmp; 
    cudaMalloc(&tmp,sizeof(thrust::device_vector<float>*));
    cudaMemcpy(ptr_,df,sizeof(thrust::device_vector<float>*),cudaMemcpyHostToDevice);
  }
  
  // __host__ __device__ Iterator(const Iterator& it) : ptr_(it.ptr_), i_(it.i_) {
  //   printf("  Iterator(const Iterator it)\n");
  //   //    std::cerr << "  Iterator(const Iterator it)\n";
  // }

  thrust::device_reference<float> operator*() {
    printf("thrust::device_reference<float> operator*()\n");
    //    std::cerr << "  thrust::device_reference<float> operator*() : " << i_ << std::endl;
    return (*ptr_)[i_];
  }

  __host__ __device__ bool operator!=(const Iterator& that) const {
    printf("bool operator!=(const Iterator& that)\n");
    //    std::cerr << "  bool operator!=(const Iterator& that) const : " << !(*this == that) << std::endl;
    return !( i_ == that.i_);
  }
  
  __host__ __device__ bool operator==(const Iterator& that) const {
    printf("bool operator==(const Iterator& that)\n");
    //    std::cerr << "  bool operator==(const Iterator& that) const : " << bool(i_ == that.i_) << std::endl;
    return i_ == that.i_;
  }

  __host__ __device__ Iterator& operator++() { 
    printf("  Iterator& operator++()\n");
    //    std::cerr << "  Iterator& operator++()\n";
    ++i_; return *this; 
  }

  thrust::device_vector<float>* ptr_;
  int i_;
};
//*/
int main(int argc, char** argv) {
    std::cerr << "Soax test starts ...\n\n";

    // thrust::device_vector<float> d_vec(3, 3.14); // 3 elemenst in vector
    // auto itbeg = repeat_iterator<thrust::device_vector<float>::iterator>(d_vec.begin(), 2); // repeat each element 2 times
    // for (int i = 0; i < 6; i++){
    //   std::cout << itbeg[i] << std::endl;
    // }

    // exit(0);

//SOAX::Storage<float, 1>::return_type test = 3.1;
//std::cout << typeid(test).name() << std::endl;
//return 0;
    // MPI - Folklore for MPI output

    // MPI_Init(&argc,&argv);
    // int commRank;   // int commSize;
    // MPI_Comm_rank(MPI_COMM_WORLD,&commRank);
    // MPI_Comm_size(MPI_COMM_WORLD,&commSize);

    // ---------------------------------------

    // number of elements (particles)
    int elementNumber = 10;

    // thrust::device_vector<float> df(elementNumber,3.);

    // thrust::fill(df.begin(), df.end(), 1.);

    // for(int i=0;i<elementNumber;i++)
    //   std::cerr << df[i] << std::endl;

    // thrust::device_vector<float>* dfp = &df;

    // thrust::fill(dfp->begin(), dfp->end(),1.3);
    // (*dfp)[0] = 1.7;
    // for(int i=0;i<elementNumber;i++)
    //   std::cerr << df[i] << std::endl;

    // std::cerr << "using Iterator ...\n";
    // Iterator it0(df,0);
    // Iterator itEnd(df,elementNumber);

    // // *it0 = 1.4;

    // for(int i=0;i<elementNumber;i++)
    //   std::cerr << df[i] << std::endl;

    // thrust::copy(it0, itEnd, std::ostream_iterator<float>(std::cout,"  "));

    // //  thrust::fill(it0, itEnd, 1.);


    // exit(0);

    // Definition of the element type by std::tuple and macro-created
    // classes
    typedef std::tuple <id<float, 1>,
    pos<float, 3>,
    vel<float, 3>,
    acc<float, 3>,
    radius<float, 1>,
    hist<float, 5>> ArrayTypes;


    // creation of Structure of Array
    SOAX::Soax<ArrayTypes> p(elementNumber);
    float dt = 1e-3;
    // initialization
    /*for (int i = 0; i < elementNumber; i++){
        for (int coord = 0; coord < 3; coord++) {

            // note: 1D attributes have different access functions than multi-D attributes
            float a = 100 + (i + 1) * 10 + coord;
            p.idtest(i) = i;                               // setting identity
            //p.postest(i) = i;
            std::cout << p.idtest(i) << std::endl;
            p.pos(i, coord) = 100 + (i + 1) * 10 + coord;   // setting position
            p.vel(i, coord) = 200 + (i + 1) * 10 + coord;   // setting velocity
            p.acc(i, coord) = 300 + (i + 1) * 10 + coord;   // setting acceleration
        }
    }//*/

  // operation on entiere arrays (using expression templates)
  //p.velArr(1) = p.velArr(0) + p.velArr(2);
  //p.posArr(0) = p.posArr(0) + p.velArr(0) * dt;
  //p.posArr(0) = p.posArr(0) + p.velArr(0);
  //p.posArr(0) = p.velArr(0);
  // p.posArr(0) += p.velArr(1);

  //for(int i = 0; i < p.size(); i++)
    //std::cerr << p.getTuple(i) << std::endl;
  
  exit(0);
  
  // reserve explicitely space for 100 element
  /*p.reserve(100); 
  
  // remove the third element
  p.erase(3);
  
  // add an element (uninitialized)
  p.push_back();
	//std::cout << p.pos(2, 1) << std::endl;
  
  // usage of stl algorithms
  std::cout << "maximal x-velocity = " << *std::max_element(p.velArr(0).begin(),p.velArr(0).end()) << std::endl;
//*/
 
  // Access and usage of elements as tuples
 /* auto elementTuple = p.getTuple(2);

  // get first coordinate of the particle position
  p.id(elementTuple);
  p.pos<1>(elementTuple);
  // get index of the first coordinate of the particle position within the tuple
  p.posIndex();

  // Access and usage of elements as objects independently from Soax structure p.
  auto element = p.getElement(3);
  element.id() = 111;
  element.pos(0) = 4.1;
  element.vel(2) = 4.4;
  
  // add element
  p.push_back(element);
/*
  // standard output of the elements
  for(int i=0;i<p.size();i++)
    std::cerr << p.getTuple(i) << std::endl;
  
  // write certain attributes (identified by their strings) to disc;
  std::vector<std::string> attributeLetterVec = {"N","P","V","A","R"};
  p.write<SOAX::WriteSerial>(attributeLetterVec);
  
  // read certain attributes from file
  p.read(std::vector<std::string>{"N","P","V"},"p_NPVAR_.bin",1,3);

  // standard output of the elements
  for(int i=0;i<p.size();i++)
    std::cerr << p.getTuple(i) << std::endl;

  std::cout << "size of attributes = " << SOAX::sizeOfAttributes<decltype(p)::Attributes>(std::vector<std::string>{"N","V"}) << std::endl;
  
  // name of the Soax structure
  std::cout << "name = " << p.name() << std::endl;
  
  // shrink capacity to fit size
  p.shrink_to_fit();
  // clear all data -> size = 0
  p.clear();//*/
return 0;

  //  MPI_Finalize();  
}
