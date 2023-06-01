#include <cmath>

float pow_3(float base)
{
    return base * base * base;
}

inline int ngp_weights(float x, float *W)
{
    int i = std::floor(x);
    W[0] = 1.0;
    return i;
}

inline int cic_weights(float x, float *W)
{
    int order = 2;
    int start = std::floor(x - 0.5);
    float s[order];
    for (int i = start; i < start + order; i++)
    {
        s[i - start] = std::abs(i + 0.5 - x);
    }
    W[0] = 1.0 - s[0];
    W[1] = 1.0 - s[1];
    return start;
}

inline int tsc_weights(float x, float *W)
{
    int order = 3;
    int start = std::floor(x - 1.0);
    float s[order];
    for (int i = start; i < start + order; i++)
    {
        s[i - start] = std::abs(i + 0.5 - x);
    }

    W[0] = 0.5 * (1.5 - s[0]) * (1.5 - s[0]);
    W[1] = 0.75 - s[1] * s[1];
    W[2] = 0.5 * (1.5 - s[2]) * (1.5 - s[2]);
    return start;
}

inline int pcs_weights(float x, float *W)
{
    int order = 4;
    int start = std::floor(x - 1.5);
    float s[order];
    for (int i = start; i < start + order; i++)
    {
        s[i - start] = std::abs(i + 0.5 - x);
    }
    W[0] = 1.0 / 6.0 * pow_3(2.0 - s[0]);
    W[1] = 1.0 / 6.0 * (4.0 - 6.0 * s[1] * s[1] + 3 * pow_3(s[1]));
    W[2] = 1.0 / 6.0 * (4.0 - 6.0 * s[2] * s[2] + 3 * pow_3(s[2]));
    W[3] = 1.0 / 6.0 * pow_3(2.0 - s[3]);
    return start;
}
