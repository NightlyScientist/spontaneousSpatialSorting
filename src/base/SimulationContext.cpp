#include "SimulationContext.h"
#include <cmath>
#include <cstddef>

SimulationContext::SimulationContext(const SimulationParameters& params) : SimulationParameters(params), ptcl_mutexes(params.maxPtcls) {
  const size_t max_ptcls = static_cast<size_t>(params.maxPtcls);
  ptcl.resize(max_ptcls);
}
