#include "FFT.h"
std::vector<std::complex<double>> FFT::smallFFT2(const std::vector<std::complex<double>>& x, bool inverse)
{
    std::vector<std::complex<double>> y(2);
    y[0] = x[0] + x[1];
    y[1] = x[0] - x[1];
    if(inverse)
    {
        y[0]/=2.0;
        y[1]/=2.0;
    }

    return y;
}

std::vector<std::complex<double>> FFT::smallFFT3(const std::vector<std::complex<double>>& x, bool inverse)
{

    // Константы алгоритма Винограда для N = 3
    // c1 = cos(2*pi/3) - 1 = -0.5 - 1 = -1.5
    // c2 = sin(2*pi/3) = sqrt(3)/2
    constexpr double c1 = -1.5;
    const double c2 = std::sqrt(3.0) / 2.0;

    // Шаг 1: Предварительное сложение (Pre-weave additions)
    std::complex<double> t1 = x[1] + x[2];
    std::complex<double> t2 = x[1] - x[2];
    std::complex<double> t3 = x[0] + t1;

    // Шаг 2: Умножение на константы ядра (Core multiplications)
    // Вещественные константы умножаются на комплексные промежуточные значения
    std::complex<double> m1 = t1 * c1;
    std::complex<double> m2 = t2 * c2;

    // Меняем знак мнимой части для обратного преобразования (IFFT)
    if (inverse)
    {
        m2 = std::complex<double>(-m2.imag(), m2.real()); // Эквивалентно m2 * i
    }
    else
    {
        m2 = std::complex<double>(m2.imag(), -m2.real());  // Эквивалентно m2 * -i
    }

    // Шаг 3: Пост-сложение (Post-weave additions)
    std::complex<double> s1 = t3 + m1;

    // Шаг 4: Формирование выходного вектора
    std::vector<std::complex<double>> y(3);
    y[0] = t3;
    y[1] = s1 + m2;
    y[2] = s1 - m2;

    // Нормировка для обратного преобразования Фурье
    if (inverse)
    {
        y[0] /= 3.0;
        y[1] /= 3.0;
        y[2] /= 3.0;
    }

    return y;
}

std::vector<std::complex<double>> FFT::smallFFT5(const std::vector<std::complex<double>>& input, bool inverse)
{
    if (input.size() != 5) return {};

    // Шаг 1: Комплексное сопряжение для ОДПФ
    std::vector<std::complex<double>> x(5);
    for (int i = 0; i < 5; ++i)
    {
        x[i] = inverse ? std::conj(input[i]) : input[i];
    }

    // Тригонометрические базисы
    const double cos1 = std::cos(2.0 * M_PI / 5.0);
    const double cos2 = std::cos(4.0 * M_PI / 5.0);
    const double sin1 = std::sin(2.0 * M_PI / 5.0);
    const double sin2 = std::sin(4.0 * M_PI / 5.0);

    // Масштабирующие константы вещественной части Винограда
    const double c1 = (cos1 + cos2) / 2.0 - 1.0;
    const double c2 = (cos1 - cos2) / 2.0;

    // Шаг 2: Входной граф пре-сложений (Строгие индексы элементов)
    std::complex<double> t1 = x[1] + x[4];
    std::complex<double> t2 = x[2] + x[3];
    std::complex<double> t3 = x[1] - x[4];
    std::complex<double> t4 = x[2] - x[3];
    std::complex<double> t5 = t1 + t2;

    // Шаг 3: Вещественные умножения (Масштабирование)
    std::complex<double> m0 = x[0] + t5;
    std::complex<double> m1 = t5 * c1;
    std::complex<double> m2 = (t1 - t2) * c2;

    // Вычисление вещественных проекций выходного графа
    std::complex<double> s1 = m0 + m1;
    std::complex<double> r1 = s1 + m2;
    std::complex<double> r2 = s1 - m2;

    // Шаг 4: Мнимые проекции (Умножение на мнимую единицу -j без вызова тригонометрии)
    // Математически эквивалентно: i = -1j * (t * sin)
    std::complex<double> i1 = std::complex<double>(t3.imag() * sin1 + t4.imag() * sin2,
                              -(t3.real() * sin1 + t4.real() * sin2));
    std::complex<double> i2 = std::complex<double>(t3.imag() * sin2 - t4.imag() * sin1,
                              -(t3.real() * sin2 - t4.real() * sin1));

    // Шаг 5: Пост-сложения спектра
    std::vector<std::complex<double>> Y(5);
    Y[0] = m0;
    Y[1] = r1 + i1;
    Y[2] = r2 + i2;
    Y[3] = r2 - i2;
    Y[4] = r1 - i1;

    // Финальная нормировка и сопряжение для ОДПФ
    for (int i = 0; i < 5; ++i)
    {
        if (inverse)
        {
            Y[i] = std::conj(Y[i]) / 5.0;
        }
    }
    return Y;
}

std::vector<std::complex<double>> FFT::dft(const std::vector<std::complex<double>>& x, bool inverse)
{
    const std::size_t N = x.size();
    std::vector<std::complex<double>> result(N);

    if (N == 0)
        return result;

    const double pi = std::acos(-1.0);
    const double sign = inverse ? 1.0 : -1.0;   // знак в экспоненте
    const double scale = inverse ? 1.0 / N : 1.0; // масштаб для обратного ДПФ

    for (std::size_t k = 0; k < N; ++k)
    {
        std::complex<double> sum(0.0, 0.0);
        for (std::size_t n = 0; n < N; ++n)
        {
            // угол поворота: ±2π·k·n / N
            const double theta = 2.0 * pi * sign * static_cast<double>(k * n) / N;
            sum += x[n] * std::exp(std::complex<double>(0.0, theta));
        }
        result[k] = sum * scale;
    }

    return result;
}

std::vector<std::complex<double>> FFT::ctFFT(const std::vector<std::complex<double>>& x, bool inverse)
{
    size_t N = x.size();

    // Базовые случаи
    if (N <= 1) return x;
    if (N == 2) return smallFFT2(x, inverse);
    if (N == 3) return smallFFT3(x, inverse);
    if (N == 4) { /* Можно разложить на 2x2, обработается ниже автоматически */ }
    if (N == 5) return smallFFT5(x, inverse);

    // Выбор оптимального множителя (радикса)
    size_t N1 = 0;
    if (N % 2 == 0) N1 = 2;
    else if (N % 3 == 0) N1 = 3;
    else if (N % 5 == 0) N1 = 5;

    // Если число не делится на 2, 3 и 5 — используем стандартный DFT
    if (N1 == 0)
    {
        return dft(x, inverse);
    }

    size_t N2 = N / N1;

    // Шаг 1: Разделение входного вектора на N1 подпоследовательностей длины N2
    // Формируем матрицы размера N1 x N2 (строки — подпоследовательности)
    std::vector<std::vector<std::complex<double>>> subVectors(N1, std::vector<std::complex<double>>(N2));
    for (size_t i = 0; i < N1; ++i)
    {
        for (size_t j = 0; j < N2; ++j)
        {
            subVectors[i][j] = x[i + j * N1];
        }
    }

    // Шаг 2: Рекурсивный вызов FFT для каждой подпоследовательности длины N2
    std::vector<std::vector<std::complex<double>>> subFFTs(N1);
    for (size_t i = 0; i < N1; ++i)
    {
        subFFTs[i] = ctFFT(subVectors[i], inverse);
    }

    // Шаг 3: Умножение на поворотные коэффициенты (Twiddle factors) и финальная сборка
    std::vector<std::complex<double>> result(N, 0.0);
    const double angle_sign = inverse ? 1.0 : -1.0;
    const double pi = std::acos(-1.0);

    for (size_t k2 = 0; k2 < N2; ++k2)
    {
        // Подготовка временного вектора размера N1 для финального преобразования
        std::vector<std::complex<double>> temp(N1);

        for (size_t k1 = 0; k1 < N1; ++k1)
        {
            // Применяем поворотный коэффициент к результатам под-FFT
            double theta = angle_sign * 2.0 * pi * k1 * k2 / N;
            std::complex<double> twiddle = std::polar(1.0, theta);
            temp[k1] = subFFTs[k1][k2] * twiddle;
        }

        // Шаг 4: Вызов малого FFT размера N1 для объединения результатов
        std::vector<std::complex<double>> combined;
        if (N1 == 2) combined = smallFFT2(temp, inverse);
        else if (N1 == 3) combined = smallFFT3(temp, inverse);
        else if (N1 == 5) combined = smallFFT5(temp, inverse);

        // Запись в итоговый вектор
        for (size_t k1 = 0; k1 < N1; ++k1)
        {
            result[k2 + k1 * N2] = combined[k1];
        }
    }

    return result;
}

