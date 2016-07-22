//
// Created by François Laenen on 8/07/16.
//

template<typename OP>
__global__ void testArrayWrapper(OP expr){
    printf("from device %g\n", expr[1]);
}


template<typename T, typename OP>
__global__ void kern(T *output, OP expr, const int size){
    /// OP2  an expression template
    /// *output a pointer to write to (must be valid, writable from the device)

    if (i < n){
        output[i] = expr[i];
    }
}