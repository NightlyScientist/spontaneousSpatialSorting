#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "SimulationContext.h"

void cutoffMatrix(SimulationParameters& config, NeighborList& nlist);

void read_parameters(int argc, char* argv[], SimulationParameters& config);
void save_data_csv(int l_itime, double time, const SimulationContext& ctx);

template <typename T>
T convert(const std::unordered_map<std::string, std::string>& argMap, const std::string& str) {
  if (!argMap.count(str)) {
    std::cerr << "Conversion error: unable to convert argument '" << str << "'" << std::endl;
    std::exit(1);
  }
  const auto& input = argMap.at(str);
  std::istringstream iss(input);
  T value;
  if constexpr (std::is_same_v<T, bool>) {
    iss >> std::boolalpha;
  }
  iss >> value;
  return value;
}

template <typename T>
std::vector<T> splitAndConvert(const std::unordered_map<std::string, std::string>& argMap, const std::string& str,
                               char delimiter = ',') {
  if (!argMap.count(str)) {
    std::cerr << "Conversion error: unable to convert argument '" << str << "'" << std::endl;
    std::exit(1);
  }
  const auto& input = argMap.at(str);
  std::vector<T> result;
  std::istringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    std::istringstream converter(token);
    T value;
    converter >> value;
    result.push_back(value);
  }
  return result;
}

void interactionMatrix(SimulationParameters& ctx);

// CSV Writer helper (header-only)
class CsvWriter {
  std::ofstream ofs;
  char delimiter = ',';

 public:
  explicit CsvWriter(std::string_view filename, char delim = ',') : ofs(filename.data()), delimiter(delim) {
    if (!ofs) throw std::runtime_error("Failed to open file: " + std::string(filename));
  }

  void write_string(std::string row) { ofs << row << std::endl; }

  // Write CSV header row from string vector
  void write_header(const std::vector<std::string>& header) { write_row(header); }

  // Write a row (vector of strings)
  void write_row(const std::vector<std::string>& fields) {
    for (size_t i = 0; i < fields.size(); ++i) {
      if (i > 0) ofs << delimiter;
      ofs << escape(fields[i]);
    }
    ofs << '\n';
  }

  // Write a row (variadic template for convenience)
  template <typename... Args>
  void write_row(Args&&... args) {
    std::vector<std::string> fields{to_string(std::forward<Args>(args))...};
    write_row(fields);
  }

 private:
  // Escape fields if needed for CSV (quotes, commas, etc.)
  static std::string escape(std::string_view sv) {
    if (sv.find_first_of(",\"\n\r") == std::string_view::npos) {
      return std::string(sv);
    }
    std::string esc = "\"";
    for (char c : sv) {
      if (c == '"')
        esc += "\"\"";
      else
        esc += c;
    }
    esc += "\"";
    return esc;
  }

  // Helper: number to string with controlled formatting (6 significant digits for floats)
  template <typename T>
  static typename std::enable_if_t<std::is_integral_v<T>, std::string>
  to_string(const T& val) {
    return std::to_string(val);
  }

  template <typename T>
  static typename std::enable_if_t<std::is_floating_point_v<T>, std::string>
  to_string(const T& val) {
    std::ostringstream oss;
    oss.precision(6);
    oss << std::defaultfloat << val;
    return oss.str();
  }

  // std::string overload
  static std::string to_string(const std::string& s) { return s; }

  // const char* overload
  static std::string to_string(const char* s) { return std::string(s); }

  // Generic vector overload (delimited by ":")
  template <typename T>
  static std::string to_string(const std::vector<T>& v) {
    if (v.empty()) return "";
    std::ostringstream os;
    os << to_string(v[0]);
    for (size_t i = 1; i < v.size(); ++i) {
      os << ':' << to_string(v[i]);
    }
    return os.str();
  }
};
