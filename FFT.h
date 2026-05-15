#ifndef FFT_H
#define FFT_H
#include <complex>
#include <vector>
#include <cmath>

class FFT
{
private:
    std::vector<std::complex<double>> smallFFT2(const std::vector<std::complex<double>>& x, bool inverse);
    std::vector<std::complex<double>> smallFFT3(const std::vector<std::complex<double>>& x, bool inverse);
    std::vector<std::complex<double>> smallFFT5(const std::vector<std::complex<double>>& input, bool inverse);
    std::vector<std::complex<double>> dft(const std::vector<std::complex<double>>& x, bool inverse);

public:
    std::vector<std::complex<double>> ctFFT(const std::vector<std::complex<double>>& x, bool inverse);

};

#endif // FFT_H
