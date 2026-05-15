#include <random>
#include <iomanip>
#include <iostream>
#include <FFT.h>


std::vector<std::complex<double>> generateRandomComplex(size_t n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::complex<double>> data(n);
    for (size_t i = 0; i < n; ++i)
        data[i] = { dist(gen), dist(gen) };
    return data;
}

double maxAbsoluteError(const std::vector<std::complex<double>>& a,
                        const std::vector<std::complex<double>>& b) {
    double max_err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double err = std::abs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

double rmsError(const std::vector<std::complex<double>>& a,
                const std::vector<std::complex<double>>& b) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double err = std::abs(a[i] - b[i]);
        sum_sq += err * err;
    }
    return std::sqrt(sum_sq / a.size());
}

int main()
{
    FFT F;
    const size_t N = 518400;
    std::random_device rd;
    auto original = generateRandomComplex(N, rd());
    std::vector<std::complex<double>> spectrum = F.ctFFT(original, false);
    std::vector<std::complex<double>> reconstructed = F.ctFFT(spectrum, true);
    double max_err = maxAbsoluteError(original, reconstructed);
    double rms_err = rmsError(original, reconstructed);

    std::cout << std::setprecision(15) << std::scientific;
    std::cout << " N          : " << N << "\n";
    std::cout << "Abs ERROR : " << max_err << "\n";
    std::cout << "RMS ERROR  : " << rms_err << "\n";



    return 0;
}
