functions {
/* Survival Model is described by
 1 log hazard        log h(t) -the instantaneous risk at time t
 2 cumulative hazard H(t)     -the total risk accumulated up to time t
*/

// Step 1 Write down the log Hazard functions + function overloading to generalize to vectors

// Gompertz log Hazard functions ==============================================================
// t     - Time 
// gamma - Change in log hazard per unit of t 
// eta   - log hazard at t = 0 

// @laplace
// @brief Gompertz log hazard at a single time point. The hazard grows (gamma > 0) or decays (gamma < 0) exponentially in time.
// @param t real, Time since origin (t >= 0).
// @param gamma real, Change in log hazard per unit of time.
// @param eta real, Log hazard at t = 0 (typically a linear predictor).
// @return real, Log hazard log h(t).
// @example real lh = survival::srv_gompertz_log_hazard(2.0, gamma, eta0);
// @math
// \log h(t) = \eta + \gamma t
real survival__srv_gompertz_log_hazard(real t, real gamma, real eta){
    return eta + gamma * t;
}

// @laplace
// @brief Gompertz log hazard, vectorised over observations with a shared gamma.
// @param t vector, Times since origin (t >= 0).
// @param gamma real, Change in log hazard per unit of time, shared by all observations.
// @param eta vector, Per-observation log hazard at t = 0.
// @return vector, Log hazard for each observation.
vector survival__srv_gompertz_log_hazard(vector t, real gamma, vector eta){
    return eta + gamma * t;
}

// @laplace
// @brief Gompertz log hazard, vectorised over observations with per-observation gamma.
// @param t vector, Times since origin (t >= 0).
// @param gamma vector, Per-observation change in log hazard per unit of time.
// @param eta vector, Per-observation log hazard at t = 0.
// @return vector, Log hazard for each observation.
vector survival__srv_gompertz_log_hazard(vector t, vector gamma, vector eta){
    return eta + gamma .* t;
}

// Log-Logistics log Hazard functions ==========================================================
// t     - Time 
// alpha - Scale parameter characteristic time
// beta  - Shape parameter controlling hazard shape 

// @laplace
// @brief Log-logistic log hazard at a single time point. The hazard is monotone decreasing for beta <= 1 and rises then falls (unimodal) for beta > 1.
// @param t real, Time since origin (t > 0).
// @param alpha real, Scale parameter (alpha > 0); the median survival time.
// @param beta real, Shape parameter (beta > 0) controlling the hazard shape.
// @return real, Log hazard log h(t).
// @example real lh = survival::srv_log_logistics_log_hazard(2.0, alpha, beta);
// @math
// \log h(t) = \log\beta - \log\alpha + (\beta - 1)\log\left(\frac{t}{\alpha}\right)
//   - \log\left(1 + \left(\frac{t}{\alpha}\right)^{\beta}\right)
real survival__srv_log_logistics_log_hazard(real t, real alpha, real beta){
    return log(beta) - log(alpha) + (beta - 1) * log(t / alpha) - log(1 + pow(t / alpha, beta));
}

// @laplace
// @brief Log-logistic log hazard, vectorised over observations with shared alpha and beta.
// @param t vector, Times since origin (t > 0).
// @param alpha real, Scale parameter (alpha > 0), shared by all observations.
// @param beta real, Shape parameter (beta > 0), shared by all observations.
// @return vector, Log hazard for each observation.
vector survival__srv_log_logistics_log_hazard(vector t, real alpha, real beta){
    return log(beta) - log(alpha) + (beta - 1) * log(t / alpha) - log(1 + pow(t / alpha, beta));
}

// @laplace
// @brief Log-logistic log hazard, vectorised over observations with per-observation alpha and beta.
// @param t vector, Times since origin (t > 0).
// @param alpha vector, Per-observation scale parameter (alpha > 0).
// @param beta vector, Per-observation shape parameter (beta > 0).
// @return vector, Log hazard for each observation.
vector survival__srv_log_logistics_log_hazard(vector t, vector alpha, vector beta){
    return log(beta) - log(alpha) + (beta - 1) .* log(t ./ alpha) - log(1 + pow(t ./ alpha, beta));
}

// Makeham log Hazard functions =================================================================
// t     - Time 
// a     - Constant/background hazard
// gamma - Exponential hazard growth rate
// eta   - Log of exponential hazard component b

// @laplace
// @brief Gompertz-Makeham log hazard at a single time point: a constant background hazard plus an exponentially changing Gompertz component. Reduces to Gompertz when a = 0.
// @param t real, Time since origin (t >= 0).
// @param a real, Constant background hazard on the natural scale (a >= 0).
// @param gamma real, Growth rate of the exponential component.
// @param eta real, Log of the exponential component's scale b (so b = exp(eta)).
// @return real, Log hazard log h(t).
// @example real lh = survival::srv_makeham_log_hazard(2.0, a, gamma, eta0);
// @math
// \log h(t) = \log\left(a + e^{\eta + \gamma t}\right)
real survival__srv_makeham_log_hazard(real t, real a, real gamma, real eta){
    return log(a + exp(eta + gamma * t));
}

// @laplace
// @brief Gompertz-Makeham log hazard, vectorised with per-observation background hazard and shared eta.
// @param t vector, Times since origin (t >= 0).
// @param a vector, Per-observation constant background hazard (a >= 0).
// @param gamma real, Growth rate of the exponential component.
// @param eta real, Log of the exponential component's scale b, shared by all observations.
// @return vector, Log hazard for each observation.
vector survival__srv_makeham_log_hazard(vector t, vector a, real gamma, real eta){
    return log(a + exp(eta + gamma * t));
}

// @laplace
// @brief Gompertz-Makeham log hazard, vectorised with shared background hazard and per-observation eta.
// @param t vector, Times since origin (t >= 0).
// @param a real, Constant background hazard (a >= 0), shared by all observations.
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Per-observation log of the exponential component's scale b.
// @return vector, Log hazard for each observation.
vector survival__srv_makeham_log_hazard(vector t, real a, real gamma, vector eta){
    return log(a + exp(eta + gamma * t));
}

// @laplace
// @brief Gompertz-Makeham log hazard, vectorised with per-observation background hazard and per-observation eta.
// @param t vector, Times since origin (t >= 0).
// @param a vector, Per-observation constant background hazard (a >= 0).
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Per-observation log of the exponential component's scale b.
// @return vector, Log hazard for each observation.
vector survival__srv_makeham_log_hazard(vector t, vector a, real gamma,  vector eta){
    return log(a + exp(eta + gamma * t));
}
// Piecewise exponential log Hazard Function ======================================================
// t   - Time 
// tau - Breaks Interval boundaries
// eta - Log-hazard for each interval
// Which interval each time falls in: the largest j with starts[j] < t

// @laplace
// @brief Piecewise-exponential (PWE) interval lookup: for each time, the index of the interval it falls in. Call once in transformed data. A time exactly on a boundary belongs to the earlier interval.
// @param t data vector, Observed times (N).
// @param starts data vector, Increasing interval start points (J); starts[1] should be 0 (or <= min(t)).
// @return array[] int, Interval index (1..J) for each observation.
// @example
// transformed data {
//   array[N] int interval = survival::srv_pwe_interval(t, starts);
// }
// @math
// j(t_n) = \max\{\, j \in 1..J : s_j < t_n \,\}, \quad j(t_n) = 1 \text{ if no such } j > 1
array[] int survival__srv_pwe_interval(data vector t, data vector starts) {
  int N = rows(t);
  int J = rows(starts);
  array[N] int idx;
  for (n in 1:N) {
    idx[n] = 1;
    for (j in 2:J) {
      if (t[n] > starts[j]) idx[n] = j;
    }
  }
  return idx;
}

// Time each person spends in each interval (N x J matrix)

// @laplace
// @brief Piecewise-exponential (PWE) exposure matrix: time each observation spends at risk in each interval. The last interval is open-ended. Call once in transformed data.
// @param t data vector, Observed times (N).
// @param starts data vector, Increasing interval start points (J); starts[1] should be 0 (or <= min(t)).
// @return matrix, N x J matrix of time at risk per observation and interval.
// @example
// transformed data {
//   matrix[N, J] E = survival::srv_pwe_exposure(t, starts);
// }
// @math
// E_{nj} = \max\left(0,\ \min(t_n, s_{j+1}) - s_j\right), \quad s_{J+1} = \infty
matrix survival__srv_pwe_exposure(data vector t, data vector starts) {
  int N = rows(t);
  int J = rows(starts);
  matrix[N, J] E;
  for (n in 1:N) {
    for (j in 1:J) {
      real end_j = (j < J) ? starts[j + 1] : positive_infinity();
      E[n, j] = fmax(0, fmin(t[n], end_j) - starts[j]);
    }
  }
  return E;
}

// @laplace
// @brief Piecewise-exponential (PWE) log hazard at each observation's time: the baseline log hazard of its interval plus its linear predictor (proportional hazards).
// @param interval data array[] int, Interval index for each observation, from srv_pwe_interval.
// @param log_lambda vector, Baseline log hazard for each of the J intervals.
// @param eta vector, Per-observation linear predictor (log hazard ratio).
// @return vector, Log hazard for each observation.
// @example vector[N] log_h = survival::srv_pwe_log_hazard(interval, log_lambda, X * beta);
// @math
// \log h_n = \log\lambda_{j(t_n)} + \eta_n
vector survival__srv_pwe_log_hazard(data array[] int interval, vector log_lambda, vector eta) {
  return log_lambda[interval] + eta;
}

// Step 2 Write down the Cumulative Hazard functions + function overloading to generalize to vectors

// Gompertz Cumulative Hazard function =============================================================

// Helper functions for save div
real srv_expm1_div(real x) {                             // computes (exp(x) - 1) / x safely
  if (abs(x) < 1e-5) return 1 + x / 2 + square(x) / 6;   // Taylor series near 0
  return expm1(x) / x;
}

vector srv_expm1_div(vector x) {                         
    int n = size(x);
    vector[n] x_new;
    for(i in 1:n){
        if (abs(x[i]) < 1e-5){
            x_new[i] =  1 + x[i] / 2 + square(x[i]) / 6; 
        } 
        else{
            x_new[i] =  expm1(x[i]) / x[i];
        }   
    }
    return x_new;
}

// Cumulative Hazard Functions 

// @laplace
// @brief Gompertz cumulative hazard at a single time point. Numerically stable as gamma -> 0, where it reduces to the exponential model exp(eta) * t.
// @param t real, Time since origin (t >= 0).
// @param gamma real, Change in log hazard per unit of time.
// @param eta real, Log hazard at t = 0.
// @return real, Cumulative hazard H(t).
// @example real H = survival::srv_gompertz_cumulative_hazard(2.0, gamma, eta0);
// @math
// H(t) = \int_0^t h(u)\,du = e^{\eta}\,\frac{e^{\gamma t} - 1}{\gamma}
//      = e^{\eta}\, t\, \frac{\operatorname{expm1}(\gamma t)}{\gamma t}
real survival__srv_gompertz_cumulative_hazard(real t, real gamma, real eta) {
  return exp(eta) * t * srv_expm1_div(gamma * t);
}

// @laplace
// @brief Gompertz cumulative hazard, vectorised over times with shared gamma and eta (e.g. a survival curve on a time grid).
// @param t vector, Times since origin (t >= 0).
// @param gamma real, Change in log hazard per unit of time.
// @param eta real, Log hazard at t = 0, shared by all times.
// @return vector, Cumulative hazard at each time.
vector survival__srv_gompertz_cumulative_hazard(vector t, real gamma, real eta) {
  return exp(eta) .* t .* srv_expm1_div(gamma * t);
}

// @laplace
// @brief Gompertz cumulative hazard, vectorised over observations with shared gamma.
// @param t vector, Times since origin (t >= 0).
// @param gamma real, Change in log hazard per unit of time, shared by all observations.
// @param eta vector, Per-observation log hazard at t = 0.
// @return vector, Cumulative hazard for each observation.
vector survival__srv_gompertz_cumulative_hazard(vector t, real gamma, vector eta) {
  return exp(eta) .* t .* srv_expm1_div(gamma .* t);
}

// @laplace
// @brief Gompertz cumulative hazard, vectorised over observations with per-observation gamma.
// @param t vector, Times since origin (t >= 0).
// @param gamma vector, Per-observation change in log hazard per unit of time.
// @param eta vector, Per-observation log hazard at t = 0.
// @return vector, Cumulative hazard for each observation.
vector survival__srv_gompertz_cumulative_hazard(vector t, vector gamma, vector eta) {
  return exp(eta) .* t .* srv_expm1_div(gamma .* t);
}

// Log-Logistics Cumulative Hazard functions ==========================================================

// @laplace
// @brief Log-logistic cumulative hazard at a single time point.
// @param t real, Time since origin (t >= 0).
// @param alpha real, Scale parameter (alpha > 0).
// @param beta real, Shape parameter (beta > 0).
// @return real, Cumulative hazard H(t).
// @example real H = survival::srv_log_logistics_cumulative_hazard(2.0, alpha, beta);
// @math
// H(t) = \log\left(1 + \left(\frac{t}{\alpha}\right)^{\beta}\right)
real survival__srv_log_logistics_cumulative_hazard(real t, real alpha, real beta){
    return log(1 + pow(t / alpha, beta));
}

// @laplace
// @brief Log-logistic cumulative hazard, vectorised over observations with shared alpha and beta.
// @param t vector, Times since origin (t >= 0).
// @param alpha real, Scale parameter (alpha > 0), shared by all observations.
// @param beta real, Shape parameter (beta > 0), shared by all observations.
// @return vector, Cumulative hazard for each observation.
vector survival__srv_log_logistics_cumulative_hazard(vector t, real alpha, real beta){
    return log(1 + pow(t / alpha, beta));
}

// @laplace
// @brief Log-logistic cumulative hazard, vectorised over observations with per-observation alpha and beta.
// @param t vector, Times since origin (t >= 0).
// @param alpha vector, Per-observation scale parameter (alpha > 0).
// @param beta vector, Per-observation shape parameter (beta > 0).
// @return vector, Cumulative hazard for each observation.
vector survival__srv_log_logistics_cumulative_hazard(vector t, vector alpha, vector beta){
    return log(1 + pow(t ./ alpha, beta));
}

// Makeham Cumulative Hazard functions ==========================================================

// @laplace
// @brief Gompertz-Makeham cumulative hazard at a single time point. Numerically stable as gamma -> 0.
// @param t real, Time since origin (t >= 0).
// @param a real, Constant background hazard (a >= 0).
// @param gamma real, Growth rate of the exponential component.
// @param eta real, Log of the exponential component's scale b.
// @return real, Cumulative hazard H(t).
// @example real H = survival::srv_makeham_cumulative_hazard(2.0, a, gamma, eta0);
// @math
// H(t) = a t + e^{\eta}\,\frac{e^{\gamma t} - 1}{\gamma}
real survival__srv_makeham_cumulative_hazard(real t, real a, real gamma, real eta){
    return a * t + exp(eta) * t * srv_expm1_div(gamma * t);
}

// @laplace
// @brief Gompertz-Makeham cumulative hazard, vectorised over times with shared parameters (e.g. a survival curve on a time grid).
// @param t vector, Times since origin (t >= 0).
// @param a real, Constant background hazard (a >= 0).
// @param gamma real, Growth rate of the exponential component.
// @param eta real, Log of the exponential component's scale b, shared by all times.
// @return vector, Cumulative hazard at each time.
vector survival__srv_makeham_cumulative_hazard(vector t, real a, real gamma, real eta){
    return a * t + exp(eta) * t .* srv_expm1_div(gamma * t);
}

// @laplace
// @brief Gompertz-Makeham cumulative hazard, vectorised with shared background hazard and per-observation eta.
// @param t vector, Times since origin (t >= 0).
// @param a real, Constant background hazard (a >= 0), shared by all observations.
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Per-observation log of the exponential component's scale b.
// @return vector, Cumulative hazard for each observation.
vector survival__srv_makeham_cumulative_hazard(vector t, real a, real gamma, vector eta){
    return a * t + exp(eta) .* t .* srv_expm1_div(gamma * t);
}

// @laplace
// @brief Gompertz-Makeham cumulative hazard, vectorised with per-observation background hazard and per-observation eta.
// @param t vector, Times since origin (t >= 0).
// @param a vector, Per-observation constant background hazard (a >= 0).
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Per-observation log of the exponential component's scale b.
// @return vector, Cumulative hazard for each observation.
vector survival__srv_makeham_cumulative_hazard(vector t, vector a, real gamma, vector eta){
    return a .* t + exp(eta) .* t .* srv_expm1_div(gamma * t);
}

// Piecewise Cumulative Hazard Function 

// @laplace
// @brief Piecewise-exponential (PWE) cumulative hazard for each observation, using the exposure matrix from srv_pwe_exposure.
// @param E data matrix, N x J exposure matrix from srv_pwe_exposure.
// @param log_lambda vector, Baseline log hazard for each of the J intervals.
// @param eta vector, Per-observation linear predictor (log hazard ratio).
// @return vector, Cumulative hazard for each observation.
// @example vector[N] H = survival::srv_pwe_cumulative_hazard(E, log_lambda, X * beta);
// @math
// H_n = e^{\eta_n} \sum_{j=1}^{J} E_{nj}\, \lambda_j
vector survival__srv_pwe_cumulative_hazard(data matrix E, vector log_lambda, vector eta) {
  return (E * exp(log_lambda)) .* exp(eta);
}

// Add wrappers that follow Stan's naming rules for lpdf or lccdf

// Gompertz Log pointwise density function and log censured cumulative density function ==========

// @laplace
// @brief Gompertz right-censored log likelihood, summed over observations. Handles censoring itself through d, so all observations (events and censored) go in one call. Use this OR srv_gompertz_lccdf for censored rows, never both.
// @param t vector, Observed times: event time if d = 1, censoring time if d = 0.
// @param gamma real, Change in log hazard per unit of time, shared by all observations.
// @param eta vector, Per-observation log hazard at t = 0 (e.g. X * beta).
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @return real, Total log likelihood.
// @example target += survival::srv_gompertz_lpdf(t | gamma, X * beta, d);
// @math
// \log L = \sum_{i=1}^{N} \left[ d_i \log h(t_i) - H(t_i) \right]
real survival__srv_gompertz_lpdf(vector t,  real gamma, vector eta, data vector d){
    vector[num_elements(t)] log_h;
    log_h = survival__srv_gompertz_log_hazard(t, gamma, eta);
    return dot_product(d, log_h) - sum(survival__srv_gompertz_cumulative_hazard(t, gamma, eta));
}

// @laplace
// @brief Gompertz right-censored log likelihood with per-observation gamma, summed over observations.
// @param t vector, Observed times: event time if d = 1, censoring time if d = 0.
// @param gamma vector, Per-observation change in log hazard per unit of time.
// @param eta vector, Per-observation log hazard at t = 0.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @return real, Total log likelihood.
real survival__srv_gompertz_lpdf(vector t, vector gamma, vector eta, data vector d){
    vector[num_elements(t)] log_h;
    log_h = survival__srv_gompertz_log_hazard(t, gamma, eta);
    return dot_product(d, log_h) - sum(survival__srv_gompertz_cumulative_hazard(t, gamma, eta));
}

// @laplace
// @brief Gompertz log survival (log complementary CDF), summed over observations. For the split-data pattern where censored rows are passed separately.
// @param t vector, Censoring times.
// @param gamma real, Change in log hazard per unit of time, shared by all observations.
// @param eta vector, Per-observation log hazard at t = 0.
// @return real, Sum of log S(t_i).
// @example target += survival::srv_gompertz_lccdf(t_cens | gamma, eta_cens);
// @math
// \sum_{i} \log S(t_i) = -\sum_{i} H(t_i)
real survival__srv_gompertz_lccdf(vector t, real gamma, vector eta){
    return -sum(survival__srv_gompertz_cumulative_hazard(t, gamma, eta));
}

// @laplace
// @brief Gompertz log survival (log complementary CDF) with per-observation gamma, summed over observations.
// @param t vector, Censoring times.
// @param gamma vector, Per-observation change in log hazard per unit of time.
// @param eta vector, Per-observation log hazard at t = 0.
// @return real, Sum of log S(t_i).
real survival__srv_gompertz_lccdf(vector t, vector gamma, vector eta){
    return -sum(survival__srv_gompertz_cumulative_hazard(t, gamma, eta));
}

// Add a per-observation version and derived quantities

// @laplace
// @brief Gompertz per-observation right-censored log likelihood, e.g. for log_lik in generated quantities (LOO / WAIC).
// @param t vector, Observed times: event time if d = 1, censoring time if d = 0.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @param gamma real, Change in log hazard per unit of time, shared by all observations.
// @param eta vector, Per-observation log hazard at t = 0.
// @return vector, Log likelihood contribution of each observation.
// @example vector[N] log_lik = survival::srv_gompertz_loglik(t, d, gamma, eta);
// @math
// \ell_i = d_i \log h(t_i) - H(t_i)
vector survival__srv_gompertz_loglik(vector t, data vector d, real gamma, vector eta) {
  return d .* survival__srv_gompertz_log_hazard(t, gamma, eta) - survival__srv_gompertz_cumulative_hazard(t, gamma, eta);
}

// @laplace
// @brief Gompertz per-observation right-censored log likelihood with per-observation gamma.
// @param t vector, Observed times: event time if d = 1, censoring time if d = 0.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @param gamma vector, Per-observation change in log hazard per unit of time.
// @param eta vector, Per-observation log hazard at t = 0.
// @return vector, Log likelihood contribution of each observation.
vector survival__srv_gompertz_loglik(vector t, data vector d, vector gamma, vector eta) {
  return d .* survival__srv_gompertz_log_hazard(t, gamma, eta) - survival__srv_gompertz_cumulative_hazard(t, gamma, eta);
}

// Log_logistics Log pointwise density function and log censured cumulative density function =============

// @laplace
// @brief Log-logistic right-censored log likelihood, summed over observations. Handles censoring itself through d; use this OR srv_log_logistics_lccdf for censored rows, never both.
// @param t vector, Observed times (t > 0): event time if d = 1, censoring time if d = 0.
// @param alpha real, Scale parameter (alpha > 0), shared by all observations.
// @param beta real, Shape parameter (beta > 0), shared by all observations.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @return real, Total log likelihood.
// @example target += survival::srv_log_logistics_lpdf(t | alpha, beta, d);
// @math
// \log L = \sum_{i=1}^{N} \left[ d_i \log h(t_i) - H(t_i) \right]
real survival__srv_log_logistics_lpdf(vector t, real alpha, real beta, data vector d){
    return dot_product(d, survival__srv_log_logistics_log_hazard(t, alpha, beta)) - sum(survival__srv_log_logistics_cumulative_hazard(t, alpha, beta));
}

// @laplace
// @brief Log-logistic right-censored log likelihood with per-observation alpha and beta (e.g. alpha = exp(X * beta_x) for an AFT model), summed over observations.
// @param t vector, Observed times (t > 0): event time if d = 1, censoring time if d = 0.
// @param alpha vector, Per-observation scale parameter (alpha > 0).
// @param beta vector, Per-observation shape parameter (beta > 0).
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @return real, Total log likelihood.
real survival__srv_log_logistics_lpdf(vector t, vector alpha, vector beta, data vector d){
    return dot_product(d, survival__srv_log_logistics_log_hazard(t, alpha, beta)) - sum(survival__srv_log_logistics_cumulative_hazard(t, alpha, beta));
}

// @laplace
// @brief Log-logistic log survival (log complementary CDF), summed over observations. For the split-data pattern where censored rows are passed separately.
// @param t vector, Censoring times.
// @param alpha real, Scale parameter (alpha > 0), shared by all observations.
// @param beta real, Shape parameter (beta > 0), shared by all observations.
// @return real, Sum of log S(t_i).
// @example target += survival::srv_log_logistics_lccdf(t_cens | alpha, beta);
// @math
// \sum_{i} \log S(t_i) = -\sum_{i} \log\left(1 + \left(\frac{t_i}{\alpha}\right)^{\beta}\right)
real survival__srv_log_logistics_lccdf(vector t, real alpha, real beta){
    return -sum(survival__srv_log_logistics_cumulative_hazard(t, alpha, beta));
}

// @laplace
// @brief Log-logistic log survival (log complementary CDF) with per-observation alpha and beta, summed over observations.
// @param t vector, Censoring times.
// @param alpha vector, Per-observation scale parameter (alpha > 0).
// @param beta vector, Per-observation shape parameter (beta > 0).
// @return real, Sum of log S(t_i).
real survival__srv_log_logistics_lccdf(vector t, vector alpha, vector beta){
    return -sum(survival__srv_log_logistics_cumulative_hazard(t, alpha, beta));
}

// Add a per-observation version and derived quantities

// @laplace
// @brief Log-logistic per-observation right-censored log likelihood, e.g. for log_lik in generated quantities (LOO / WAIC).
// @param t vector, Observed times (t > 0): event time if d = 1, censoring time if d = 0.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @param alpha real, Scale parameter (alpha > 0), shared by all observations.
// @param beta real, Shape parameter (beta > 0), shared by all observations.
// @return vector, Log likelihood contribution of each observation.
// @example vector[N] log_lik = survival::srv_log_logistics_loglik(t, d, alpha, beta);
// @math
// \ell_i = d_i \log h(t_i) - H(t_i)
vector survival__srv_log_logistics_loglik(vector t, data vector d, real alpha, real beta){
    return d .* survival__srv_log_logistics_log_hazard(t, alpha, beta) - survival__srv_log_logistics_cumulative_hazard(t, alpha, beta);
}

// @laplace
// @brief Log-logistic per-observation right-censored log likelihood with per-observation alpha and beta.
// @param t vector, Observed times (t > 0): event time if d = 1, censoring time if d = 0.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @param alpha vector, Per-observation scale parameter (alpha > 0).
// @param beta vector, Per-observation shape parameter (beta > 0).
// @return vector, Log likelihood contribution of each observation.
vector survival__srv_log_logistics_loglik(vector t, data vector d, vector alpha, vector beta){
    return d .* survival__srv_log_logistics_log_hazard(t, alpha, beta) - survival__srv_log_logistics_cumulative_hazard(t, alpha, beta);
}

// Makeham Log pointwise density function and log censured cumulative density function ========================

// @laplace
// @brief Gompertz-Makeham right-censored log likelihood, summed over observations. Handles censoring itself through d; use this OR srv_makeham_lccdf for censored rows, never both.
// @param t vector, Observed times: event time if d = 1, censoring time if d = 0.
// @param a real, Constant background hazard (a >= 0), shared by all observations.
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Per-observation log of the exponential component's scale b.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @return real, Total log likelihood.
// @example target += survival::srv_makeham_lpdf(t | a, gamma, X * beta, d);
// @math
// \log L = \sum_{i=1}^{N} \left[ d_i \log\left(a + e^{\eta_i + \gamma t_i}\right)
//   - a t_i - e^{\eta_i}\,\frac{e^{\gamma t_i} - 1}{\gamma} \right]
real survival__srv_makeham_lpdf(vector t, real a, real gamma, vector eta, data vector d){
    return dot_product(d, survival__srv_makeham_log_hazard(t, a, gamma, eta)) - sum(survival__srv_makeham_cumulative_hazard(t, a, gamma, eta));
}

// @laplace
// @brief Gompertz-Makeham right-censored log likelihood with per-observation background hazard, summed over observations.
// @param t vector, Observed times: event time if d = 1, censoring time if d = 0.
// @param a vector, Per-observation constant background hazard (a >= 0).
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Per-observation log of the exponential component's scale b.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @return real, Total log likelihood.
real survival__srv_makeham_lpdf(vector t, vector a, real gamma, vector eta, data vector d){
    return dot_product(d, survival__srv_makeham_log_hazard(t, a, gamma, eta)) - sum(survival__srv_makeham_cumulative_hazard(t, a, gamma, eta));
}

// @laplace
// @brief Gompertz-Makeham log survival (log complementary CDF), summed over observations. For the split-data pattern where censored rows are passed separately.
// @param t vector, Censoring times.
// @param a real, Constant background hazard (a >= 0), shared by all observations.
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Per-observation log of the exponential component's scale b.
// @return real, Sum of log S(t_i).
// @example target += survival::srv_makeham_lccdf(t_cens | a, gamma, eta_cens);
// @math
// \sum_{i} \log S(t_i) = -\sum_{i} H(t_i)
real survival__srv_makeham_lccdf(vector t, real a, real gamma, vector eta){
    return -sum(survival__srv_makeham_cumulative_hazard(t, a, gamma, eta));
}

// @laplace
// @brief Gompertz-Makeham log survival (log complementary CDF) with per-observation background hazard, summed over observations.
// @param t vector, Censoring times.
// @param a vector, Per-observation constant background hazard (a >= 0).
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Per-observation log of the exponential component's scale b.
// @return real, Sum of log S(t_i).
real survival__srv_makeham_lccdf(vector t, vector a, real gamma, vector eta){
    return -sum(survival__srv_makeham_cumulative_hazard(t, a, gamma, eta));
}

// Add a per-observation version and derived quantities

// @laplace
// @brief Gompertz-Makeham per-observation right-censored log likelihood, e.g. for log_lik in generated quantities (LOO / WAIC). Note: d is the LAST argument here.
// @param t vector, Observed times: event time if d = 1, censoring time if d = 0.
// @param a real, Constant background hazard (a >= 0), shared by all observations.
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Per-observation log of the exponential component's scale b.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @return vector, Log likelihood contribution of each observation.
// @example vector[N] log_lik = survival::srv_makeham_loglik(t, a, gamma, eta, d);
// @math
// \ell_i = d_i \log h(t_i) - H(t_i)
vector survival__srv_makeham_loglik(vector t, real a, real gamma, vector eta, data vector d){
    return d .* survival__srv_makeham_log_hazard(t, a, gamma, eta) - survival__srv_makeham_cumulative_hazard(t, a, gamma, eta);
}

// @laplace
// @brief Gompertz-Makeham per-observation right-censored log likelihood with per-observation background hazard.
// @param t vector, Observed times: event time if d = 1, censoring time if d = 0.
// @param a vector, Per-observation constant background hazard (a >= 0).
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Per-observation log of the exponential component's scale b.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @return vector, Log likelihood contribution of each observation.
vector survival__srv_makeham_loglik(vector t, vector a, real gamma, vector eta, data vector d){
    return d .* survival__srv_makeham_log_hazard(t, a, gamma, eta) - survival__srv_makeham_cumulative_hazard(t, a, gamma, eta);
}

// PWE Log pointwise density function and log censured cumulative density function =============

// @laplace
// @brief Piecewise-exponential (PWE) right-censored log likelihood, summed over observations. Handles censoring itself through d.
// @param interval data array[] int, Interval index for each observation, from srv_pwe_interval.
// @param E data matrix, N x J exposure matrix from srv_pwe_exposure.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @param log_lambda vector, Baseline log hazard for each of the J intervals.
// @param eta vector, Per-observation linear predictor (log hazard ratio).
// @return real, Total log likelihood.
// @example target += survival::srv_pwe_loglik_sum(interval, E, d, log_lambda, X * beta);
// @math
// \log L = \sum_{n=1}^{N} \left[ d_n \left(\log\lambda_{j(t_n)} + \eta_n\right)
//   - e^{\eta_n} \sum_{j=1}^{J} E_{nj}\,\lambda_j \right]
real survival__srv_pwe_loglik_sum(data array[] int interval, data matrix E, data vector d,vector log_lambda, vector eta) {
  return dot_product(d, survival__srv_pwe_log_hazard(interval, log_lambda, eta)) - sum(survival__srv_pwe_cumulative_hazard(E, log_lambda, eta));
}

// @laplace
// @brief Piecewise-exponential (PWE) per-observation right-censored log likelihood, e.g. for log_lik in generated quantities (LOO / WAIC).
// @param interval data array[] int, Interval index for each observation, from srv_pwe_interval.
// @param E data matrix, N x J exposure matrix from srv_pwe_exposure.
// @param d data vector, Event indicator: 1 = event observed, 0 = right-censored.
// @param log_lambda vector, Baseline log hazard for each of the J intervals.
// @param eta vector, Per-observation linear predictor (log hazard ratio).
// @return vector, Log likelihood contribution of each observation.
// @example vector[N] log_lik = survival::srv_pwe_loglik(interval, E, d, log_lambda, eta);
// @math
// \ell_n = d_n \log h_n - H_n
vector survival__srv_pwe_loglik(data array[] int interval, data matrix E, data vector d, vector log_lambda, vector eta) {
  return d .* survival__srv_pwe_log_hazard(interval, log_lambda, eta) - survival__srv_pwe_cumulative_hazard(E, log_lambda, eta);
}

// Add Survival Curves for all types of models 

// @laplace
// @brief Gompertz survival probability S(t) = P(T > t) for each element.
// @param t vector, Times at which to evaluate survival (t >= 0).
// @param gamma real, Change in log hazard per unit of time.
// @param eta vector, Log hazard at t = 0, one per element of t (use rep_vector for a single profile).
// @return vector, Survival probability at each time.
// @example vector[G] S = survival::srv_gompertz_survival_curve(t_grid, gamma, rep_vector(eta0, G));
// @math
// S(t) = \exp\left(-H(t)\right) = \exp\left(-e^{\eta}\,\frac{e^{\gamma t} - 1}{\gamma}\right)
vector survival__srv_gompertz_survival_curve(vector t, real gamma, vector eta) {
  return exp(-survival__srv_gompertz_cumulative_hazard(t, gamma, eta));
}

// @laplace
// @brief Gompertz survival probability with per-element gamma.
// @param t vector, Times at which to evaluate survival (t >= 0).
// @param gamma vector, Per-element change in log hazard per unit of time.
// @param eta vector, Per-element log hazard at t = 0.
// @return vector, Survival probability at each time.
vector survival__srv_gompertz_survival_curve(vector t, vector gamma, vector eta) {
  return exp(-survival__srv_gompertz_cumulative_hazard(t, gamma, eta));
}

// @laplace
// @brief Log-logistic survival probability S(t) = P(T > t) for each element.
// @param t vector, Times at which to evaluate survival (t >= 0).
// @param alpha real, Scale parameter (alpha > 0).
// @param beta real, Shape parameter (beta > 0).
// @return vector, Survival probability at each time.
// @example vector[G] S = survival::srv_log_logistics_survival_curve(t_grid, alpha, beta);
// @math
// S(t) = \frac{1}{1 + \left(t / \alpha\right)^{\beta}}
vector survival__srv_log_logistics_survival_curve(vector t, real alpha, real beta){
    return exp(-survival__srv_log_logistics_cumulative_hazard(t, alpha, beta));
}

// @laplace
// @brief Log-logistic survival probability with per-element alpha and beta.
// @param t vector, Times at which to evaluate survival (t >= 0).
// @param alpha vector, Per-element scale parameter (alpha > 0).
// @param beta vector, Per-element shape parameter (beta > 0).
// @return vector, Survival probability at each time.
vector survival__srv_log_logistics_survival_curve(vector t, vector alpha, vector beta){
    return exp(-survival__srv_log_logistics_cumulative_hazard(t, alpha, beta));
}

// @laplace
// @brief Gompertz-Makeham survival probability S(t) = P(T > t) for each element.
// @param t vector, Times at which to evaluate survival (t >= 0).
// @param a real, Constant background hazard (a >= 0).
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Log of the exponential component's scale b, one per element of t.
// @return vector, Survival probability at each time.
// @example vector[G] S = survival::srv_makeham_survival_curve(t_grid, a, gamma, rep_vector(eta0, G));
// @math
// S(t) = \exp\left(-a t - e^{\eta}\,\frac{e^{\gamma t} - 1}{\gamma}\right)
vector survival__srv_makeham_survival_curve(vector t, real a, real gamma, vector eta){
    return exp(-survival__srv_makeham_cumulative_hazard(t, a, gamma, eta));
}

// @laplace
// @brief Gompertz-Makeham survival probability with per-element background hazard.
// @param t vector, Times at which to evaluate survival (t >= 0).
// @param a vector, Per-element constant background hazard (a >= 0).
// @param gamma real, Growth rate of the exponential component.
// @param eta vector, Per-element log of the exponential component's scale b.
// @return vector, Survival probability at each time.
vector survival__srv_makeham_survival_curve(vector t, vector a, real gamma, vector eta){
    return exp(-survival__srv_makeham_cumulative_hazard(t, a, gamma, eta));
}


// @laplace
// @brief Piecewise-exponential (PWE) survival probability for each row of the exposure matrix. To draw a curve, build E from a time grid with srv_pwe_exposure.
// @param E data matrix, Exposure matrix from srv_pwe_exposure (one row per time point).
// @param log_lambda vector, Baseline log hazard for each of the J intervals.
// @param eta vector, Linear predictor, one per row of E.
// @return vector, Survival probability for each row.
// @example vector[G] S = survival::srv_pwe_survival_curve(E_grid, log_lambda, rep_vector(eta0, G));
// @math
// S_n = \exp\left(-e^{\eta_n} \sum_{j=1}^{J} E_{nj}\,\lambda_j\right)
vector survival__srv_pwe_survival_curve(data matrix E, vector log_lambda, vector eta){
    return exp(-survival__srv_pwe_cumulative_hazard(E, log_lambda, eta));
}

// Random Number Generation from main functions (inverse CDF)
real srv_log1p_div(real x) {          // log1p(x) / x, safe near 0
  if (abs(x) < 1e-5) return 1 - x / 2 + square(x) / 3;
  return log1p(x) / x;
}

// @laplace
// @brief Draw one Gompertz event time by inverting the cumulative hazard. With gamma < 0 the model has a cure fraction, and draws that never experience the event return positive_infinity().
// @param gamma real, Change in log hazard per unit of time.
// @param eta real, Log hazard at t = 0.
// @return real, Simulated event time (may be +infinity when gamma < 0).
// @example real t_sim = survival::srv_gompertz_rng(gamma, eta_new);
// @math
// T = \frac{1}{\gamma}\log\left(1 + \gamma\, E\, e^{-\eta}\right), \quad E \sim \mathrm{Exponential}(1)
// T = \infty \ \text{if}\ 1 + \gamma E e^{-\eta} \le 0
real survival__srv_gompertz_rng(real gamma, real eta) {
  real z = exponential_rng(1) * exp(-eta);    // E / b
  real x = gamma * z;
  if (x <= -1) return positive_infinity();    // cured: never has the event
  return z * srv_log1p_div(x);
}

// @laplace
// @brief Draw one Gompertz-Makeham event time as the minimum of an exponential (background) time and a Gompertz time. Note: unlike the other Makeham functions, the background hazard is on the LOG scale here.
// @param log_c real, Log of the constant background hazard (log a).
// @param gamma real, Growth rate of the exponential component.
// @param eta real, Log of the exponential component's scale b.
// @return real, Simulated event time.
// @example real t_sim = survival::srv_makeham_rng(log(a), gamma, eta_new);
// @math
// T = \min(T_c, T_G), \quad T_c \sim \mathrm{Exponential}\left(e^{\log c}\right), \quad T_G \sim \mathrm{Gompertz}(\gamma, \eta)
real survival__srv_makeham_rng(real log_c, real gamma, real eta) {
  return fmin(exponential_rng(exp(log_c)), survival__srv_gompertz_rng(gamma, eta));
}

// @laplace
// @brief Draw one log-logistic event time by inverse-CDF sampling.
// @param alpha real, Scale parameter (alpha > 0); the median of the distribution.
// @param beta real, Shape parameter (beta > 0).
// @return real, Simulated event time.
// @example real t_sim = survival::srv_log_logistics_rng(alpha, beta);
// @math
// T = \alpha \left(\frac{U}{1 - U}\right)^{1/\beta}, \quad U \sim \mathrm{Uniform}(0, 1)
real survival__srv_log_logistics_rng(real alpha, real beta) {
    real u = uniform_rng(0, 1);
    return alpha * exp((log(u) - log1m(u)) / beta);
}


}
// Drawn functions (seed 20260918): srv_makeham_cumulative_hazard, srv_makeham_lccdf.
// Gompertz-Makeham proportional-hazards model with two covariates, split-data
// (events / right-censored) likelihood, exactly as the library's examples show.


data {
  int<lower=1> N_ev;
  int<lower=1> N_cen;
  int<lower=1> K;
  vector<lower=0>[N_ev] t_ev;
  vector<lower=0>[N_cen] t_cen;
  matrix[N_ev, K] X_ev;
  matrix[N_cen, K] X_cen;
}

parameters {
  real<lower=0> a;
  real<lower=0> gamma;
  real eta0;
  vector[K] beta;
}

model {
  vector[N_ev]  eta_ev  = eta0 + X_ev * beta;
  vector[N_cen] eta_cen = eta0 + X_cen * beta;

  a ~ normal(0, 0.5);
  gamma ~ normal(0, 1);
  eta0 ~ normal(0, 3);
  beta ~ normal(0, 1);

  target += sum(survival__srv_makeham_log_hazard(t_ev, a, gamma, eta_ev));
  target += -sum(survival__srv_makeham_cumulative_hazard(t_ev, a, gamma, eta_ev));
  target += survival__srv_makeham_lccdf(t_cen | a, gamma, eta_cen);
}
