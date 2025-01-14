#ifndef CUCH_HPP
#define CUCH_HPP

#include <sstream>

#include <cuda_runtime.h>

/**
 * A stream exception that is base for all runtime errors.
 */
class CudaError : public std::exception {
  protected:
    std::string mMessage; ///< Internal buffer where the message is kept.
    cudaError_t mStatus;

  public:
    CudaError(cudaError_t status = cudaSuccess) : std::exception(), mStatus(status) {
    }
    CudaError(const char* msg, cudaError_t status = cudaSuccess) : std::exception(), mMessage(msg), mStatus(status) {
    }
    CudaError(const std::string& msg, cudaError_t status = cudaSuccess)
        : std::exception(), mMessage(msg), mStatus(status) {
    }

    virtual const char* what() const noexcept override {
        return mMessage.c_str();
    }

    // Overloading << operator that uses stringstream to append data to mMessage.
    template <typename T>
    CudaError& operator<<(const T& data) {
        std::stringstream stream;
        stream << mMessage << data;
        mMessage = stream.str();
        return *this;
    }
};

/**
 * CUDA error code check. This is internal function used by CUCH macro.
 */
inline void _cuda_check(cudaError_t status, int line, const char* srcFile, const char* errMsg = NULL) {
    if (status != cudaSuccess) {
        throw(CudaError(status) << "CUDA Error (" << status << "): " << cudaGetErrorString(status) << "\n"
                                << "at " << srcFile << ":" << line << ":" << errMsg);
    }
}

/**
 * Macro wrapper for CUDA calls checking.
 */
#define CUCH(status) _cuda_check(status, __LINE__, __FILE__, #status)

#endif // CUCH_HPP
