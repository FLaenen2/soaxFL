#ifndef soaxWriterMPI_hpp
#define soaxWriterMPI_hpp

#include <fstream>
#include <mpi.h>

namespace SOAX {

extern MPI_File mpiFile;
extern unsigned int propertyOffset;

struct WriteMPI
{
  template<class ArrayOfListVector>
  static void write(ArrayOfListVector& array, std::string filename) {
    
    //    std::cerr << "starting MPI Write\n";

    typedef typename ArrayOfListVector::value_type::value_type DataType;

    int mpiCommRank;
    int mpiCommSize;
    MPI_Comm_rank(MPI_COMM_WORLD,&mpiCommRank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpiCommSize);
    
    unsigned int localParticleNumber = array[0].size();
    unsigned int globalParticleNumbers[mpiCommSize];
    
    //    std::cerr << mpiCommRank << " : localParticleNumber = " << localParticleNumber << std::endl;

    MPI_Gather(&localParticleNumber,1,MPI_UNSIGNED,&globalParticleNumbers[0],1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
    MPI_Bcast(&globalParticleNumbers[0],mpiCommSize,MPI_UNSIGNED,0,MPI_COMM_WORLD);
  
    long long globalNumber = 0;
    unsigned int rank[mpiCommSize];
    for(int i=0;i<mpiCommSize;i++) {
      rank[i] = i;
      globalNumber += globalParticleNumbers[i];
    }
    
    //    std::cerr << mpiCommRank << " : globalNumber = " << globalNumber << std::endl;
    
    if(mpiCommRank == 0) {
      std::string particleNumberFilename = filename + "_numbers.txt";

      std::ofstream out(particleNumberFilename.c_str());
      for(int i=0;i<mpiCommSize;i++) {
	out << rank[i] << "  " << globalParticleNumbers[i] << std::endl;
      }
      out << "-1  " << globalNumber << std::endl;
    }

    char file[100];
  
    sprintf(file,"%s",filename.c_str());

    if(mpiFile == MPI_FILE_NULL) {
      //      std::cerr << mpiCommRank << " opening file\n";

      propertyOffset = 0;

      MPI_File_open(MPI_COMM_WORLD, file, MPI_MODE_WRONLY | MPI_MODE_CREATE , MPI_INFO_NULL, &mpiFile );
    }

    //    std::cerr << mpiCommRank << " writing file " << mpiCommSize << "  " <<  localParticleNumber << " \n";
    for(int i=0;i<array.size();i++) {
      MPI_Offset my_offset = propertyOffset + i*globalNumber*sizeof(DataType);
      
      for(int commRank=0;commRank<mpiCommRank;commRank++) 
      	my_offset += globalParticleNumbers[commRank]*sizeof(DataType);
      
      //      std::cerr << mpiCommRank << "\t offset = " << my_offset << std::endl;
      MPI_File_seek(mpiFile, my_offset, MPI_SEEK_SET);
      
      MPI_Status status;
      MPI_File_write_all(mpiFile,reinterpret_cast<char *>(array[i].data()) ,localParticleNumber*sizeof(DataType) , MPI_BYTE, &status);
    }
    propertyOffset += array.size()*globalNumber*sizeof(DataType);
    // std::cerr << mpiCommRank << " : propertyOffset = " << propertyOffset << std::endl;
  }
  
  static void close() {
    if(mpiFile != MPI_FILE_NULL)
      MPI_File_close(&mpiFile);
  }
};



} // namespace SOAX

#endif
