#pragma once

#include <array>
#include <cmath>

using Vec2 = std::array<double, 2>;
using Matrix = std::array<std::array<double, 2>, 2>;

inline std::array<double, 2>& operator+=(std::array<double, 2>& a, const std::array<double, 2>& b) noexcept {
  a[0] += b[0];
  a[1] += b[1];
  return a;
}

inline std::array<double, 2>& operator-=(std::array<double, 2>& a, const std::array<double, 2>& b) noexcept {
  a[0] -= b[0];
  a[1] -= b[1];
  return a;
}

inline std::array<double, 2>& operator*=(std::array<double, 2>& a, double s) noexcept {
  a[0] *= s;
  a[1] *= s;
  return a;
}

inline double dot(const Vec2& a, const Vec2& b) { return a[0] * b[0] + a[1] * b[1]; }
inline double norm2(const Vec2& a) { return dot(a, a); }
inline double norm(const Vec2& a) { return std::sqrt(dot(a, a)); }

inline Vec2 operator+(const Vec2& a, const Vec2& b) { return {a[0] + b[0], a[1] + b[1]}; }
inline Vec2 operator-(const Vec2& a, const Vec2& b) { return {a[0] - b[0], a[1] - b[1]}; }
inline Vec2 operator*(double s, const Vec2& v) { return {s * v[0], s * v[1]}; }
inline Vec2 operator*(const Vec2& v, double s) { return s * v; }
inline Vec2 operator/(const Vec2& v, double s) { return {v[0] / s, v[1] / s}; }

// Periodic boundary
inline double wrap_coordinate(double x, double box) {
    double y = std::fmod(x, box);
    return (y < 0) ? (y + box) : y;
}


inline Vec2 pbc(const Vec2& a, const double box) {
  Vec2 r;
  for (int d = 0; d < 2; ++d) r[d] = wrap_coordinate(a[d], box);
  return r;
}

inline double pbc(double x, double box) { return x - box * std::round(x / box); }
inline Vec2 pbc_vec(const Vec2& a, const Vec2& b, const double box) {
  Vec2 r;
  for (int d = 0; d < 2; ++d) r[d] = pbc(a[d] - b[d], box);
  return r;
}

inline Vec2 pbc_vec(const Vec2& a, const Vec2& b, const Vec2& box) {
  Vec2 r;
  for (int d = 0; d < 2; ++d) r[d] = pbc(a[d] - b[d], box[d]);
  return r;
}

inline std::array<double,2> displacement(const Vec2& v1, const Vec2& v2) {
  return {v1[0] - v2[0], v1[1] - v2[1]};
}

inline std::array<double,3> separation(const Vec2& v1, const Vec2& v2) {
  Vec2 dr = v1 - v2;
  return {dr[0], dr[1], norm(dr)};
}
