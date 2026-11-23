functions {
// =============================================================================================================
// ts — time series module for Laplace
//
// Conventions used throughout this file:
//   * Conditional likelihood only. The first observations are conditioned on, never modelled.
//     Stationarity / invertibility are NOT enforced; put priors on phi and theta instead.
//   * `_lpdf` / `_lpmf`  score an observed series.
//   * `_rng`             simulate a fresh series from explicit seed values (prior predictive, SBC).
//   * `_forecast_rng`    continue an observed series H steps ahead.
//   * Covariate overloads always append `matrix X, vector beta` after `sigma`.
// =============================================================================================================

// Helpers =====================================================================================================

//@laplace
//@brief Build a Lag Design Matrix
//@param int p Number of previous observation to consider
//@param vector y Observations of interest
//@example matrix X = ts::lag_design(10, y);
//@math X[n, i] = y_{p + n - i}, \quad n = 1,\dots,T-p, \quad i = 1,\dots,p
//@return matrix[T - p, p] X. Row n holds y_{t-1},\dots,y_{t-p} for t = p + n.
matrix ts__lag_design(int p, vector y){
    int T = num_elements(y);
    int N = T - p;
    matrix[N, p] X;

    for(i in 1:p){
        X[, i] = y[(p + 1 - i):(T - i)];
    }
    return X;
}

//@laplace
//@brief Build a Lag Design Matrix starting from a caller-supplied offset
//@param int p Number of previous observation to consider
//@param vector y Observations of interest
//@param int L Row offset. Rows correspond to t = L+1,...,T. Requires L >= p.
//@example matrix X = ts::lag_design(2, y, max(2, 1 * 12));
//@math X[n, i] = y_{L + n - i}, \quad n = 1,\dots,T-L
//@return matrix[T - L, p] X
matrix ts__lag_design(int p, vector y, int L){
    int T = num_elements(y);
    int N = T - L;
    matrix[N, p] X;

    for(i in 1:p){
        X[, i] = y[(L + 1 - i):(T - i)];
    }
    return X;
}

//@laplace
//@brief Build a Seasonal Lag Design Matrix
//@param int P Number of seasonal lags
//@param int s Seasonal period
//@param vector y Observations of interest
//@example matrix X = ts::seasonal_lag_design(1, 12, y);
//@math X[n, k] = y_{Ps + n - ks}, \quad n = 1,\dots,T-Ps, \quad k = 1,\dots,P
//@return matrix[T - P*s, P] X
matrix ts__seasonal_lag_design(int P, int s, vector y) {
    int T = num_elements(y);
    int N = T - P * s;
    matrix[N, P] X;

    for (k in 1:P)
        X[, k] = y[(P * s + 1 - k * s):(T - k * s)];

    return X;
}

//@laplace
//@brief Build a Seasonal Lag Design Matrix starting from a caller-supplied offset
//@param int P Number of seasonal lags
//@param int s Seasonal period
//@param vector y Observations of interest
//@param int L Row offset. Rows correspond to t = L+1,...,T. Requires L >= P*s.
//@example matrix X = ts::seasonal_lag_design(1, 12, y, max(2, 12));
//@math X[n, k] = y_{L + n - ks}, \quad n = 1,\dots,T-L
//@return matrix[T - L, P] X
matrix ts__seasonal_lag_design(int P, int s, vector y, int L) {
    int T = num_elements(y);
    int N = T - L;
    matrix[N, P] X;

    for (k in 1:P)
        X[, k] = y[(L + 1 - k * s):(T - k * s)];

    return X;
}

//@laplace
//@brief Difference a time series d times
//@param vector y
//@param int d
//@math \nabla^d y_t = \nabla^{d-1} y_t - \nabla^{d-1} y_{t-1}, \quad \nabla^0 y = y
//@return vector[T-d]
vector ts__difference(vector y, int d) {
    int T = num_elements(y);
    vector[T] current = y;
    vector[T] next;

    for (k in 1:d) {
        int n = T - k + 1;

        for (t in 1:(n - 1))
            next[t] = current[t + 1] - current[t];

        current[1:(n - 1)] = next[1:(n - 1)];
    }

    return current[1:(T - d)];
}

//@laplace
//@brief Seasonal differencing
//@param vector y Observations
//@param int s Seasonal period
//@example vector y_s = ts::seasonal_difference(y, 12);
//@math \nabla_s y_t = y_t - y_{t-s}
//@return vector Seasonally differenced observations
vector ts__seasonal_difference(vector y, int s) {
    int T = num_elements(y);
    vector[T - s] result;

    for (t in 1:(T - s))
        result[t] = y[t + s] - y[t];

    return result;
}

//@laplace
//@brief Apply seasonal differencing D times
//@param vector y Observations
//@param int s Seasonal period
//@param int D Number of seasonal differences. D = 0 returns y unchanged.
//@example vector y_s = ts::seasonal_difference(y, 12, 1);
//@math \nabla_s^D y_t
//@return vector[T - D*s] Seasonally differenced observations
vector ts__seasonal_difference(vector y, int s, int D) {
    int T = num_elements(y);
    vector[T] current = y;

    for (k in 1:D) {
        int n = T - (k - 1) * s;
        for (t in 1:(n - s))
            current[t] = current[t + s] - current[t];
    }
    return current[1:(T - D * s)];
}

//@laplace
//@brief Integrate a forecast of d-th differences back onto the level scale
//@param vector y_obs The observed series, on the level scale
//@param vector delta_f Forecast d-th differences, length H
//@param int d Order of differencing to undo
//@example vector y_f = ts::undifference(y, delta_f, 1);
//@math y_{T+h} = y_{T+h-1} + \nabla y_{T+h}, applied d times, anchored at \nabla^{k} y_T
//@return vector[H] Forecast on the level scale
vector ts__undifference(vector y_obs, vector delta_f, int d){
    int H = num_elements(delta_f);
    vector[H] current = delta_f;

    for (k in 1:d) {
        int order = d - k;
        vector[num_elements(y_obs) - order] prev = ts__difference(y_obs, order);
        current[1] += prev[num_elements(prev)];
        for (h in 2:H) current[h] += current[h - 1];
    }
    return current;
}

//@laplace
//@brief Integrate a forecast of D-th seasonal differences back onto the level scale
//@param vector y_obs The observed series, on the level scale
//@param vector delta_f Forecast seasonal differences, length H
//@param int s Seasonal period
//@param int D Number of seasonal differences to undo
//@example vector y_f = ts::unseasonal_difference(y, delta_f, 12, 1);
//@math y_{T+h} = y_{T+h-s} + \nabla_s y_{T+h}, applied D times
//@return vector[H] Forecast on the level scale
vector ts__unseasonal_difference(vector y_obs, vector delta_f, int s, int D){
    int T = num_elements(y_obs);
    int H = num_elements(delta_f);
    vector[H] current = delta_f;

    for (k in 1:D) {
        int order = D - k;
        int n = T - order * s;
        vector[n] prev = ts__seasonal_difference(y_obs, s, order);
        vector[n + H] full;

        full[1:n] = prev;
        for (h in 1:H) full[n + h] = current[h] + full[n + h - s];
        current = full[(n + 1):(n + H)];
    }
    return current;
}

// Linear Mean Recursions =======================================================================================


// Autoregression AR(p) Model ==================================================================================
//@laplace
//@brief Autoregression primary for computing mu_t from AR p model
//@param alpha Intercept
//@param vector phi with length of p. Partial Persistence
//@param matrix X_lag coming from a ts::lag_design function
//@example mu_t = ts::ar(alpha, phi, X_lag);
//@math \mu_t = \alpha + \sum_{i=1}^{p} \phi_i y_{t-i}
//@return vector[rows(X_lag)] mu_t
vector ts__ar(real alpha, vector phi, matrix X_lag){
    return alpha + X_lag * phi;
}

//@laplace
//@brief Autoregression primary for computing mu_t from AR p model
//@param alpha Intercept
//@param matrix X Covariates
//@param vector beta
//@param vector phi with length of p. Partial Persistence
//@param matrix X_lag coming from a ts::lag_design function
//@example mu_t = ts::ar(alpha, phi, X_lag, X, beta);
//@math \mu_t = \alpha + \sum_{i=1}^{p} \phi_i y_{t-i} + x_t^\top \beta
//@return vector[rows(X_lag)] mu_t
vector ts__ar(real alpha, vector phi, matrix X_lag, matrix X, vector beta){
    return alpha + X_lag * phi + X * beta;
}

// Seasonal AR(p) Model SAR
//@laplace
//@brief Seasonal Autoregression primary
//@param vector phi with length of p. Partial Persistence
//@param vector Phi
//@param matrix x_lag coming from a ts::lag_design function
//@param matrix X_slag coming from a ts::seasonal_lag_design function
//@example mu_t = ts::sar(alpha, phi, Phi, x_lag, X_slag, X, beta);
//@math \mu_t = \alpha + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{k=1}^{P} \Phi_k y_{t-ks} + x_t^\top \beta
//@return vector[rows(x_lag)] mu_t. Both design matrices MUST be built with the same offset L = max(p, P*s).
vector ts__sar(real alpha, vector phi, vector Phi, matrix x_lag, matrix X_slag, matrix X, vector beta){
    return alpha + x_lag * phi + X_slag * Phi + X * beta;
}

// Seasonal AR(p) Model SAR
//@laplace
//@brief Seasonal Autoregression primary
//@param vector phi with length of p. Partial Persistence
//@param vector Phi
//@param matrix x_lag coming from a ts::lag_design function
//@param matrix X_slag coming from a ts::seasonal_lag_design function
//@example mu_t = ts::sar(alpha, phi, Phi, x_lag, X_slag);
//@math \mu_t = \alpha + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{k=1}^{P} \Phi_k y_{t-ks}
//@return vector[rows(x_lag)] mu_t. Both design matrices MUST be built with the same offset L = max(p, P*s).
vector ts__sar(real alpha, vector phi, vector Phi, matrix x_lag, matrix X_slag){
    return alpha + x_lag * phi + X_slag * Phi;
}

// Autoregression Log Pointwise distribution function
//@laplace
//@brief Autoregression Model AR Log Pointwise distribution function
//@param alpha Intercept
//@param vector phi with length of p. Partial Persistence
//@param matrix X_lag coming from a ts::lag_design function
//@param vector y_obs with length N Computed as T - P
//@param real sigma
//@math \log p(y) = \sum_{t=p+1}^{T} \log \mathcal{N}(y_t \mid \alpha + \sum_i \phi_i y_{t-i}, \sigma)
//@return real Log density of y_obs
real ts__ar_lpdf(vector y_obs, real alpha, vector phi, matrix X_lag, real sigma){
    return normal_lpdf(y_obs | alpha + X_lag * phi, sigma);
}

// Autoregression Log Pointwise distribution function
//@laplace
//@brief Autoregression Model AR Log Pointwise distribution function
//@param alpha Intercept
//@param vector phi with length of p. Partial Persistence
//@param matrix X_lag coming from a ts::lag_design function
//@param vector y_obs with length N Computed as T - P
//@param real sigma
//@math \log p(y) = \sum_{t=p+1}^{T} \log \mathcal{N}(y_t \mid \alpha + \sum_i \phi_i y_{t-i} + x_t^\top\beta, \sigma)
//@return real Log density of y_obs
real ts__ar_lpdf(vector y_obs, real alpha, vector phi, matrix X_lag, real sigma, matrix X, vector beta){
    return normal_lpdf(y_obs | alpha + X_lag * phi + X * beta, sigma);
}

// SAR Log Pointwise Density Function
//@laplace
//@brief SAR Log Pointwise Density Function
//@param vector phi with length of p. Partial Persistence
//@param vector Phi
//@param matrix x_lag coming from a ts::lag_design function
//@param matrix X_slag coming from a ts::seasonal_lag_design function
//@param real sigma
//@param vector y_obs
//@math \log p(y) = \sum_t \log \mathcal{N}(y_t \mid \alpha + \sum_i \phi_i y_{t-i} + \sum_k \Phi_k y_{t-ks} + x_t^\top\beta, \sigma)
//@return real Log density of y_obs
real ts__sar_lpdf(vector y_obs, real alpha, vector phi, vector Phi, matrix x_lag, matrix X_slag, real sigma, matrix X, vector beta){
    return  normal_lpdf(y_obs | alpha + x_lag * phi + X_slag * Phi + X * beta, sigma);
}

// SAR Log Pointwise Density Function
//@laplace
//@brief SAR Log Pointwise Density Function
//@param vector phi with length of p. Partial Persistence
//@param vector Phi
//@param matrix x_lag coming from a ts::lag_design function
//@param matrix X_slag coming from a ts::seasonal_lag_design function
//@param real sigma
//@param vector y_obs
//@math \log p(y) = \sum_t \log \mathcal{N}(y_t \mid \alpha + \sum_i \phi_i y_{t-i} + \sum_k \Phi_k y_{t-ks}, \sigma)
//@return real Log density of y_obs
real ts__sar_lpdf(vector y_obs, real alpha, vector phi, vector Phi, matrix x_lag, matrix X_slag, real sigma){
    return  normal_lpdf(y_obs | alpha + x_lag * phi + X_slag * Phi, sigma);
}

//@laplace
//@brief Simulate an AR(p) series forward from explicit seeds
//@param int H Number of values to simulate
//@param vector y_init Seed values, length p, ordered oldest to newest
//@param real alpha Intercept
//@param vector phi Partial persistence, length p
//@param real sigma
//@example vector y_sim = ts::ar_rng(100, y_init, alpha, phi, sigma);
//@math y_{t} \sim \mathcal{N}(\alpha + \sum_{i=1}^{p}\phi_i y_{t-i},\ \sigma)
//@return vector[H] Simulated observations
vector ts__ar_rng(int H, vector y_init, real alpha, vector phi, real sigma){
    int p = num_elements(phi);
    vector[p + H] full;
    full[1:p] = y_init;

    for (h in 1:H) {
        real mu = alpha;
        for (i in 1:p) mu += phi[i] * full[p + h - i];
        full[p + h] = normal_rng(mu, sigma);
    }
    return full[(p + 1):(p + H)];
}

//@laplace
//@brief Forecast an AR(p) series H steps beyond the observed data
//@param vector y_obs Observed series, length T
//@param int H Forecast horizon
//@param real alpha Intercept
//@param vector phi Partial persistence, length p
//@param real sigma
//@example vector y_f = ts::ar_forecast_rng(y, 12, alpha, phi, sigma);
//@math y_{T+h} \sim \mathcal{N}(\alpha + \sum_{i=1}^{p}\phi_i y_{T+h-i},\ \sigma)
//@return vector[H] Posterior predictive draws for t = T+1,...,T+H
vector ts__ar_forecast_rng(vector y_obs, int H, real alpha, vector phi, real sigma){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    vector[T + H] full;
    full[1:T] = y_obs;

    for (h in 1:H) {
        real mu = alpha;
        for (i in 1:p) mu += phi[i] * full[T + h - i];
        full[T + h] = normal_rng(mu, sigma);
    }
    return full[(T + 1):(T + H)];
}

//@laplace
//@brief Forecast an AR(p) series H steps beyond the observed data with covariates
//@param vector y_obs Observed series, length T
//@param int H Forecast horizon
//@param real alpha Intercept
//@param vector phi Partial persistence, length p
//@param real sigma
//@param matrix X_f Future covariates, H rows
//@param vector beta
//@example vector y_f = ts::ar_forecast_rng(y, 12, alpha, phi, sigma, X_f, beta);
//@math y_{T+h} \sim \mathcal{N}(\alpha + \sum_i \phi_i y_{T+h-i} + x_{T+h}^\top\beta,\ \sigma)
//@return vector[H] Posterior predictive draws for t = T+1,...,T+H
vector ts__ar_forecast_rng(vector y_obs, int H, real alpha, vector phi, real sigma, matrix X_f, vector beta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    vector[T + H] full;
    full[1:T] = y_obs;

    for (h in 1:H) {
        real mu = alpha + X_f[h] * beta;
        for (i in 1:p) mu += phi[i] * full[T + h - i];
        full[T + h] = normal_rng(mu, sigma);
    }
    return full[(T + 1):(T + H)];
}

//@laplace
//@brief Simulate a seasonal AR series forward from explicit seeds
//@param int H Number of values to simulate
//@param vector y_init Seed values, length >= max(p, P*s), ordered oldest to newest
//@param real alpha Intercept
//@param vector phi Ordinary AR coefficients
//@param vector Phi Seasonal AR coefficients
//@param int s Seasonal period
//@param real sigma
//@math y_t \sim \mathcal{N}(\alpha + \sum_i \phi_i y_{t-i} + \sum_k \Phi_k y_{t-ks},\ \sigma)
//@return vector[H] Simulated observations
vector ts__sar_rng(int H, vector y_init, real alpha, vector phi, vector Phi, int s, real sigma){
    int L = num_elements(y_init);
    int p = num_elements(phi);
    int P = num_elements(Phi);
    vector[L + H] full;
    full[1:L] = y_init;

    for (h in 1:H) {
        real mu = alpha;
        for (i in 1:p) mu += phi[i] * full[L + h - i];
        for (k in 1:P) mu += Phi[k] * full[L + h - k * s];
        full[L + h] = normal_rng(mu, sigma);
    }
    return full[(L + 1):(L + H)];
}

//@laplace
//@brief Forecast a seasonal AR series H steps beyond the observed data
//@param vector y_obs Observed series, length T
//@param int H Forecast horizon
//@param real alpha Intercept
//@param vector phi Ordinary AR coefficients
//@param vector Phi Seasonal AR coefficients
//@param int s Seasonal period
//@param real sigma
//@math y_{T+h} \sim \mathcal{N}(\alpha + \sum_i \phi_i y_{T+h-i} + \sum_k \Phi_k y_{T+h-ks},\ \sigma)
//@return vector[H] Posterior predictive draws for t = T+1,...,T+H
vector ts__sar_forecast_rng(vector y_obs, int H, real alpha, vector phi, vector Phi, int s, real sigma){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    int P = num_elements(Phi);
    vector[T + H] full;
    full[1:T] = y_obs;

    for (h in 1:H) {
        real mu = alpha;
        for (i in 1:p) mu += phi[i] * full[T + h - i];
        for (k in 1:P) mu += Phi[k] * full[T + h - k * s];
        full[T + h] = normal_rng(mu, sigma);
    }
    return full[(T + 1):(T + H)];
}

// MA(q) Moving Average =========================================================================================

// Moving Average Primary
//@laplace
//@brief MA q Moving Average
//@param real alpha
//@param vector theta
//@param vector y
//@example mu_eps = ts::ma(y, alpha, theta);
//@math \mu_t = \alpha + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j}, \quad \varepsilon_t = y_t - \mu_t, \quad \varepsilon_{t\le 0} = 0
//@return Array of vectors in position array[1] is the mu_t and in array[2] is eps
array[] vector ts__ma(vector y, real alpha, vector theta){
    int T = num_elements(y);
    int q = num_elements(theta);
    array[2] vector[T] results; // array[1] mu_t and array[2] eps

    for(t in 1:T){
        results[1][t] = alpha;

        for(j in 1:min(q, t - 1)){
            results[1][t] += theta[j] * results[2][t - j];
        }

        results[2][t] = y[t] - results[1][t];
    }
    return results;
}

// Moving Average Primary
//@laplace
//@brief MA q Moving Average
//@param real alpha
//@param vector theta
//@param vector y
//@example mu_eps = ts::ma(y, alpha, theta, X, beta);
//@math \mu_t = \alpha + x_t^\top\beta + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j}, \quad \varepsilon_t = y_t - \mu_t
//@return Array of vectors in position array[1] is the mu_t and in array[2] is eps
array[] vector ts__ma(vector y, real alpha, vector theta, matrix X, vector beta){
    int T = num_elements(y);
    int q = num_elements(theta);
    array[2] vector[T] results; // array[1] mu_t and array[2] eps

    for(t in 1:T){
        results[1][t] = alpha + X[t] * beta;

        for(j in 1:min(q, t - 1)){
            results[1][t] += theta[j] * results[2][t - j];
        }

        results[2][t] = y[t] - results[1][t];
    }
    return results;
}


// Seasonal Moving Average Primary
//@laplace
//@brief Seasonal Moving Average primary
//@param real alpha
//@param vector theta
//@param vector Theta
//@param vector y
//@param int s Seasonal period
//@example results = ts::sma(y, alpha, theta, Theta, 12);
//@math \mu_t = \alpha + \sum_{j=1}^{q}\theta_j\varepsilon_{t-j} + \sum_{k=1}^{Q}\Theta_k\varepsilon_{t-ks}
//@return Array of vectors in position array[1] is the mu_t and in array[2] is eps
array[] vector ts__sma(vector y, real alpha, vector theta, vector Theta, int s){
    int T = num_elements(y);
    int q = num_elements(theta);
    int Q = num_elements(Theta);

    array[2] vector[T] results; // array[1] mu_t and array[2] eps

    for(t in 1:T){
        results[1][t] = alpha;

        for(j in 1:min(q, t - 1)){
            results[1][t] += theta[j] * results[2][t - j];
        }
        for(k in 1:min(Q, (t - 1) %/% s)){
            results[1][t] += Theta[k] * results[2][t - k * s];
        }

        results[2][t] = y[t] - results[1][t];
    }
    return results;
}

// Seasonal Moving Average Primary
//@laplace
//@brief Seasonal Moving Average primary
//@param real alpha
//@param vector theta
//@param vector Theta
//@param vector y
//@param int s Seasonal period
//@example results = ts::sma(y, alpha, theta, Theta, 12, X, beta);
//@math \mu_t = \alpha + x_t^\top\beta + \sum_j \theta_j\varepsilon_{t-j} + \sum_k \Theta_k\varepsilon_{t-ks}
//@return Array of vectors in position array[1] is the mu_t and in array[2] is eps
array[] vector ts__sma(vector y, real alpha, vector theta, vector Theta, int s, matrix X, vector beta){
    int T = num_elements(y);
    int q = num_elements(theta);
    int Q = num_elements(Theta);

    array[2] vector[T] results; // array[1] mu_t and array[2] eps

    for(t in 1:T){
        results[1][t] = alpha + X[t] * beta;

        for(j in 1:min(q, t - 1)){
            results[1][t] += theta[j] * results[2][t - j];
        }
        for(k in 1:min(Q, (t - 1) %/% s)){
            results[1][t] += Theta[k] * results[2][t - k * s];
        }

        results[2][t] = y[t] - results[1][t];
    }
    return results;
}


// Moving Average Log Pointwise Probability Function
//@laplace
//@brief Moving Average Log Pointwise Probability Function
//@param vector y_obs
//@param real alpha
//@param vector theta
//@param real sigma
//@math \log p(y) = \sum_{t=1}^{T}\log\mathcal{N}(y_t \mid \alpha + \sum_j \theta_j\varepsilon_{t-j},\ \sigma)
//@return real Log density of y_obs
real ts__ma_lpdf(vector y_obs, real alpha, vector theta, real sigma){
    int T = num_elements(y_obs);
    int q = num_elements(theta);
    vector[T] mu_t;
    vector[T] eps;

    for(t in 1:T){
        mu_t[t] = alpha;

        for(j in 1:min(q, t - 1)){
            mu_t[t] += theta[j] * eps[t - j];
        }
        eps[t] = y_obs[t] - mu_t[t];
    }
    return normal_lpdf(y_obs | mu_t, sigma);
}

// Moving Average Log Pointwise Probability Function
//@laplace
//@brief Moving Average Log Pointwise Probability Function
//@param vector y_obs
//@param real alpha
//@param vector theta
//@param real sigma
//@math \log p(y) = \sum_{t=1}^{T}\log\mathcal{N}(y_t \mid \alpha + x_t^\top\beta + \sum_j \theta_j\varepsilon_{t-j},\ \sigma)
//@return real Log density of y_obs
real ts__ma_lpdf(vector y_obs, real alpha, vector theta, real sigma, matrix X, vector beta){
    int T = num_elements(y_obs);
    int q = num_elements(theta);
    vector[T] mu_t;
    vector[T] eps;

    for(t in 1:T){
        mu_t[t] = alpha + X[t] * beta;

        for(j in 1:min(q, t - 1)){
            mu_t[t] += theta[j] * eps[t - j];
        }
        eps[t] = y_obs[t] - mu_t[t];
    }
    return normal_lpdf(y_obs | mu_t, sigma);
}

// Seasonal Moving Average Log Pointwise Probability Function
//@laplace
//@brief Seasonal Moving Average Log Pointwise Probability Function
//@param vector y_obs
//@param real alpha
//@param vector theta
//@param vector Theta
//@param real sigma
//@param int s Seasonal period
//@math \log p(y) = \sum_t \log\mathcal{N}(y_t \mid \alpha + \sum_j\theta_j\varepsilon_{t-j} + \sum_k\Theta_k\varepsilon_{t-ks},\ \sigma)
//@return real Log density of y_obs
real ts__sma_lpdf(vector y_obs, real alpha, vector theta, vector Theta, real sigma, int s){
    int T = num_elements(y_obs);
    int q = num_elements(theta);
    int Q = num_elements(Theta);

    vector[T] mu_t;
    vector[T] eps;

    for(t in 1:T){
        mu_t[t] = alpha;

        for(j in 1:min(q, t - 1)){
            mu_t[t] += theta[j] * eps[t - j];
        }

        for(k in 1:min(Q, (t - 1) %/% s)){
            mu_t[t] += Theta[k] * eps[t - k * s];
        }
        eps[t] = y_obs[t] - mu_t[t];
    }
    return normal_lpdf(y_obs | mu_t, sigma);
}

// Seasonal Moving Average Log Pointwise Probability Function
//@laplace
//@brief Seasonal Moving Average Log Pointwise Probability Function
//@param vector y_obs
//@param real alpha
//@param vector theta
//@param vector Theta
//@param real sigma
//@param int s Seasonal period
//@math \log p(y) = \sum_t \log\mathcal{N}(y_t \mid \alpha + x_t^\top\beta + \sum_j\theta_j\varepsilon_{t-j} + \sum_k\Theta_k\varepsilon_{t-ks},\ \sigma)
//@return real Log density of y_obs
real ts__sma_lpdf(vector y_obs, real alpha, vector theta, vector Theta, real sigma, int s, matrix X, vector beta){
    int T = num_elements(y_obs);
    int q = num_elements(theta);
    int Q = num_elements(Theta);

    vector[T] mu_t;
    vector[T] eps;

    for(t in 1:T){
        mu_t[t] = alpha + X[t] * beta;

        for(j in 1:min(q, t - 1)){
            mu_t[t] += theta[j] * eps[t - j];
        }

        for(k in 1:min(Q, (t - 1) %/% s)){
            mu_t[t] += Theta[k] * eps[t - k * s];
        }
        eps[t] = y_obs[t] - mu_t[t];
    }
    return normal_lpdf(y_obs | mu_t, sigma);
}

//@laplace
//@brief Simulate an MA(q) series from a cold start
//@param int H Number of values to simulate
//@param real alpha
//@param vector theta
//@param real sigma
//@example vector y_sim = ts::ma_rng(100, alpha, theta, sigma);
//@math \varepsilon_t \sim \mathcal{N}(0,\sigma), \quad y_t = \alpha + \sum_{j=1}^{q}\theta_j\varepsilon_{t-j} + \varepsilon_t
//@return vector[H] Simulated observations
vector ts__ma_rng(int H, real alpha, vector theta, real sigma){
    int q = num_elements(theta);
    vector[H] y;
    vector[H] eps;

    for (h in 1:H) {
        real mu = alpha;
        for (j in 1:min(q, h - 1)) mu += theta[j] * eps[h - j];
        eps[h] = normal_rng(0, sigma);
        y[h]   = mu + eps[h];
    }
    return y;
}

//@laplace
//@brief Forecast an MA(q) series H steps beyond the observed data
//@param vector y_obs Observed series, length T
//@param int H Forecast horizon
//@param real alpha
//@param vector theta
//@param real sigma
//@example vector y_f = ts::ma_forecast_rng(y, 12, alpha, theta, sigma);
//@math Residuals are replayed from y_{1:T}; future \varepsilon_{T+h} \sim \mathcal{N}(0,\sigma). Forecasts are flat beyond h = q.
//@return vector[H] Posterior predictive draws for t = T+1,...,T+H
vector ts__ma_forecast_rng(vector y_obs, int H, real alpha, vector theta, real sigma){
    int T = num_elements(y_obs);
    int q = num_elements(theta);
    vector[T + H] eps;
    vector[H] y_f;

    for (t in 1:T) {
        real mu = alpha;
        for (j in 1:min(q, t - 1)) mu += theta[j] * eps[t - j];
        eps[t] = y_obs[t] - mu;
    }
    for (h in 1:H) {
        real mu = alpha;
        for (j in 1:q) mu += theta[j] * eps[T + h - j];
        eps[T + h] = normal_rng(0, sigma);
        y_f[h]     = mu + eps[T + h];
    }
    return y_f;
}

//@laplace
//@brief Simulate a seasonal MA series from a cold start
//@param int H Number of values to simulate
//@param real alpha
//@param vector theta
//@param vector Theta
//@param int s Seasonal period
//@param real sigma
//@math y_t = \alpha + \sum_j\theta_j\varepsilon_{t-j} + \sum_k\Theta_k\varepsilon_{t-ks} + \varepsilon_t
//@return vector[H] Simulated observations
vector ts__sma_rng(int H, real alpha, vector theta, vector Theta, int s, real sigma){
    int q = num_elements(theta);
    int Q = num_elements(Theta);
    vector[H] y;
    vector[H] eps;

    for (h in 1:H) {
        real mu = alpha;
        for (j in 1:min(q, h - 1))          mu += theta[j] * eps[h - j];
        for (k in 1:min(Q, (h - 1) %/% s))  mu += Theta[k] * eps[h - k * s];
        eps[h] = normal_rng(0, sigma);
        y[h]   = mu + eps[h];
    }
    return y;
}

//@laplace
//@brief Forecast a seasonal MA series H steps beyond the observed data
//@param vector y_obs Observed series, length T
//@param int H Forecast horizon
//@param real alpha
//@param vector theta
//@param vector Theta
//@param int s Seasonal period
//@param real sigma
//@math Residuals replayed from y_{1:T}; future \varepsilon_{T+h}\sim\mathcal{N}(0,\sigma)
//@return vector[H] Posterior predictive draws for t = T+1,...,T+H
vector ts__sma_forecast_rng(vector y_obs, int H, real alpha, vector theta, vector Theta, int s, real sigma){
    int T = num_elements(y_obs);
    int q = num_elements(theta);
    int Q = num_elements(Theta);
    vector[T + H] eps;
    vector[H] y_f;

    for (t in 1:T) {
        real mu = alpha;
        for (j in 1:min(q, t - 1))          mu += theta[j] * eps[t - j];
        for (k in 1:min(Q, (t - 1) %/% s))  mu += Theta[k] * eps[t - k * s];
        eps[t] = y_obs[t] - mu;
    }
    for (h in 1:H) {
        real mu = alpha;
        for (j in 1:q) mu += theta[j] * eps[T + h - j];
        for (k in 1:Q) mu += Theta[k] * eps[T + h - k * s];
        eps[T + h] = normal_rng(0, sigma);
        y_f[h]     = mu + eps[T + h];
    }
    return y_f;
}

// Autoregression Moving Average ARMA(p, q) =================================================================
//@laplace
//@brief Autoregression Moving Average primary
//@param real alpha
//@param vector phi
//@param vector theta
//@param vector y_obs
//@example results = ts::arma(y_obs, alpha, phi, theta);
//@math \mu_t = \alpha + \sum_{i=1}^{p}\phi_i y_{t-i} + \sum_{j=1}^{q}\theta_j\varepsilon_{t-j}, \quad \varepsilon_t = y_t - \mu_t
//@return array of 2 where array[1] is mu_t and array[2] is eps
array [] vector ts__arma(vector y_obs, real alpha, vector phi, vector theta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    int q = num_elements(theta);
    array[2] vector[T] results; // array[1] mu_t , array[2] eps

    for(t in 1:T){
        results[1][t] = alpha;
        for(i in 1:min(p, t - 1)) results[1][t] += phi[i]   * y_obs[t - i];
        for(j in 1:min(q, t - 1)) results[1][t] += theta[j] * results[2][t -j];
        results[2][t] = y_obs[t] - results[1][t];
    }
    return results;
}

//@laplace
//@brief Autoregression Moving Average primary
//@param real alpha
//@param vector phi
//@param vector theta
//@param vector y_obs
//@example results = ts::arma(y_obs, alpha, phi, theta, X, beta);
//@math \mu_t = \alpha + x_t^\top\beta + \sum_i\phi_i y_{t-i} + \sum_j\theta_j\varepsilon_{t-j}
//@return array of 2 where array[1] is mu_t and array[2] is eps
array [] vector ts__arma(vector y_obs, real alpha, vector phi, vector theta, matrix X, vector beta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    int q = num_elements(theta);
    array[2] vector[T] results; // array[1] mu_t , array[2] eps

    for(t in 1:T){
        results[1][t] = alpha + X[t] * beta;
        for(i in 1:min(p, t - 1)) results[1][t] += phi[i]   * y_obs[t - i];
        for(j in 1:min(q, t - 1)) results[1][t] += theta[j] * results[2][t -j];
        results[2][t] = y_obs[t] - results[1][t];
    }
    return results;
}

//@laplace
//@brief Seasonal Autoregression Moving Average primary
//@param real alpha
//@param vector phi
//@param vector Phi
//@param vector theta
//@param vector Theta
//@param vector y_obs
//@param int s Seasonal period
//@math \mu_t = \alpha + \sum_i\phi_i y_{t-i} + \sum_k\Phi_k y_{t-ks} + \sum_j\theta_j\varepsilon_{t-j} + \sum_n\Theta_n\varepsilon_{t-ns}
//@return array of 2 where array[1] is mu_t and array[2] is eps
array [] vector ts__sarma(vector y_obs, real alpha, vector phi, vector Phi, vector theta, vector Theta, int s){
    int T = num_elements(y_obs);
    int p = num_elements(phi);   int P = num_elements(Phi);
    int q = num_elements(theta); int Q = num_elements(Theta);

    array[2] vector[T] results; // array[1] mu_t , array[2] eps

    for(t in 1:T){
        results[1][t] = alpha;

        // AR + SAR
        for(i in 1:min(p, t - 1))          results[1][t] += phi[i]   * y_obs[t - i];
        for(k in 1:min(P, (t - 1) %/% s))  results[1][t] += Phi[k]   * y_obs[t - k * s];

        // MA + SMA
        for(j in 1:min(q, t - 1))          results[1][t] += theta[j] * results[2][t -j];
        for(n in 1:min(Q, (t - 1) %/% s))  results[1][t] += Theta[n] * results[2][t - n * s];

        // Innovation
        results[2][t] = y_obs[t] - results[1][t];
    }
    return results;
}

//@laplace
//@brief Seasonal Autoregression Moving Average primary
//@param real alpha
//@param vector phi
//@param vector Phi
//@param vector theta
//@param vector Theta
//@param vector y_obs
//@param int s Seasonal period
//@math \mu_t = \alpha + x_t^\top\beta + \sum_i\phi_i y_{t-i} + \sum_k\Phi_k y_{t-ks} + \sum_j\theta_j\varepsilon_{t-j} + \sum_n\Theta_n\varepsilon_{t-ns}
//@return array of 2 where array[1] is mu_t and array[2] is eps
array [] vector ts__sarma(vector y_obs, real alpha, vector phi, vector Phi, vector theta, vector Theta, int s, matrix X, vector beta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);   int P = num_elements(Phi);
    int q = num_elements(theta); int Q = num_elements(Theta);

    array[2] vector[T] results; // array[1] mu_t , array[2] eps

    for(t in 1:T){
        results[1][t] = alpha + X[t] * beta;

        // AR + SAR
        for(i in 1:min(p, t - 1))          results[1][t] += phi[i]       * y_obs[t - i];
        for(k in 1:min(P, (t - 1) %/% s))  results[1][t] += Phi[k]       * y_obs[t - k * s];

        // MA + SMA
        for(j in 1:min(q, t - 1))          results[1][t] += theta[j]     * results[2][t -j];
        for(n in 1:min(Q, (t - 1) %/% s))  results[1][t] += Theta[n]     * results[2][t - n * s];

        // Innovation
        results[2][t] = y_obs[t] - results[1][t];
    }
    return results;
}

// ARMA(p, q) Log Pointwise Density Function
//@laplace
//@brief ARMA Log Pointwise Density Function
//@param real alpha
//@param vector phi
//@param vector theta
//@param vector y_obs
//@param real sigma
//@math \log p(y) = \sum_{t=1}^{T}\log\mathcal{N}(y_t \mid \alpha + \sum_i\phi_i y_{t-i} + \sum_j\theta_j\varepsilon_{t-j},\ \sigma)
//@return real Log density of y_obs
real ts__arma_lpdf(vector y_obs, real alpha, vector phi, vector theta, real sigma){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    int q = num_elements(theta);
    vector[T] mu_t;
    vector[T] eps;

    for(t in 1:T){
        mu_t[t] = alpha;
        for(i in 1:min(p, t - 1)) mu_t[t] += phi[i]   * y_obs[t - i];
        for(j in 1:min(q, t - 1)) mu_t[t] += theta[j] * eps[t -j];
        eps[t] = y_obs[t] - mu_t[t];
    }
    return normal_lpdf(y_obs | mu_t, sigma);
}

// ARMA(p, q) Log Pointwise Density Function
//@laplace
//@brief ARMA Log Pointwise Density Function
//@param real alpha
//@param vector phi
//@param vector theta
//@param vector y_obs
//@param real sigma
//@math \log p(y) = \sum_{t=1}^{T}\log\mathcal{N}(y_t \mid \alpha + x_t^\top\beta + \sum_i\phi_i y_{t-i} + \sum_j\theta_j\varepsilon_{t-j},\ \sigma)
//@return real Log density of y_obs
real ts__arma_lpdf(vector y_obs, real alpha, vector phi, vector theta, real sigma, matrix X, vector beta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    int q = num_elements(theta);

    vector[T] mu_t;
    vector[T] eps;

    for(t in 1:T){
        mu_t[t] = alpha + X[t] * beta;
        for(i in 1:min(p, t - 1)) mu_t[t] += phi[i]   * y_obs[t - i];
        for(j in 1:min(q, t - 1)) mu_t[t] += theta[j] * eps[t -j];
        eps[t] = y_obs[t] - mu_t[t];
    }
    return normal_lpdf(y_obs | mu_t, sigma);
}

//@laplace
//@brief SARMA Log Pointwise Density Function
//@param real alpha
//@param vector phi
//@param vector Phi
//@param vector theta
//@param vector Theta
//@param vector y_obs
//@param real sigma
//@param int s Seasonal period
//@math \log p(y) = \sum_t \log\mathcal{N}(y_t \mid \mu_t, \sigma) with the SARMA recursion for \mu_t
//@return real Log density of y_obs
real ts__sarma_lpdf(vector y_obs, real alpha, vector phi, vector Phi, vector theta, vector Theta, real sigma, int s){
    int T = num_elements(y_obs);
    int p = num_elements(phi);   int P = num_elements(Phi);
    int q = num_elements(theta); int Q = num_elements(Theta);

    vector[T] mu_t;
    vector[T] eps;

    for(t in 1:T){
        mu_t[t] = alpha;
        // AR + SAR
        for(i in 1:min(p, t - 1))          mu_t[t] += phi[i]   * y_obs[t - i];
        for(k in 1:min(P, (t - 1) %/% s))  mu_t[t] += Phi[k]   * y_obs[t - k * s];

        // MA + SMA
        for(j in 1:min(q, t - 1))          mu_t[t] += theta[j] * eps[t -j];
        for(n in 1:min(Q, (t - 1) %/% s))  mu_t[t] += Theta[n] * eps[t - n * s];

        // Innovation
        eps[t] = y_obs[t] - mu_t[t];
    }
    return normal_lpdf(y_obs | mu_t, sigma);
}

//@laplace
//@brief SARMA Log Pointwise Density Function
//@param real alpha
//@param vector phi
//@param vector Phi
//@param vector theta
//@param vector Theta
//@param vector y_obs
//@param real sigma
//@param int s Seasonal period
//@math \log p(y) = \sum_t \log\mathcal{N}(y_t \mid \mu_t, \sigma) with x_t^\top\beta added to \mu_t
//@return real Log density of y_obs
real ts__sarma_lpdf(vector y_obs, real alpha, vector phi, vector Phi, vector theta, vector Theta, real sigma, int s, matrix X, vector beta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);   int P = num_elements(Phi);
    int q = num_elements(theta); int Q = num_elements(Theta);

    vector[T] mu_t;
    vector[T] eps;

    for(t in 1:T){
        mu_t[t] = alpha + X[t] * beta;
        // AR + SAR
        for(i in 1:min(p, t - 1))          mu_t[t] += phi[i]   * y_obs[t - i];
        for(k in 1:min(P, (t - 1) %/% s))  mu_t[t] += Phi[k]   * y_obs[t - k * s];

        // MA + SMA
        for(j in 1:min(q, t - 1))          mu_t[t] += theta[j] * eps[t -j];
        for(n in 1:min(Q, (t - 1) %/% s))  mu_t[t] += Theta[n] * eps[t - n * s];

        // Innovation
        eps[t] = y_obs[t] - mu_t[t];
    }
    return normal_lpdf(y_obs | mu_t, sigma);
}

//@laplace
//@brief Simulate an ARMA(p, q) series from a cold start
//@param int H Number of values to simulate
//@param real alpha
//@param vector phi
//@param vector theta
//@param real sigma
//@example vector y_sim = ts::arma_rng(200, alpha, phi, theta, sigma);
//@math y_t = \alpha + \sum_i\phi_i y_{t-i} + \sum_j\theta_j\varepsilon_{t-j} + \varepsilon_t, \quad \varepsilon_t\sim\mathcal{N}(0,\sigma)
//@return vector[H] Simulated observations
vector ts__arma_rng(int H, real alpha, vector phi, vector theta, real sigma){
    int p = num_elements(phi);
    int q = num_elements(theta);
    vector[H] y;
    vector[H] eps;

    for (h in 1:H) {
        real mu = alpha;
        for (i in 1:min(p, h - 1)) mu += phi[i]   * y[h - i];
        for (j in 1:min(q, h - 1)) mu += theta[j] * eps[h - j];
        eps[h] = normal_rng(0, sigma);
        y[h]   = mu + eps[h];
    }
    return y;
}

//@laplace
//@brief Forecast an ARMA(p, q) series H steps beyond the observed data
//@param vector y_obs Observed series, length T
//@param int H Forecast horizon
//@param real alpha
//@param vector phi
//@param vector theta
//@param real sigma
//@example vector y_f = ts::arma_forecast_rng(y, 12, alpha, phi, theta, sigma);
//@math Residuals replayed from y_{1:T}; then y_{T+h} = \alpha + \sum_i\phi_i y_{T+h-i} + \sum_j\theta_j\varepsilon_{T+h-j} + \varepsilon_{T+h}
//@return vector[H] Posterior predictive draws for t = T+1,...,T+H
vector ts__arma_forecast_rng(vector y_obs, int H, real alpha, vector phi, vector theta, real sigma){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    int q = num_elements(theta);
    vector[T + H] y;
    vector[T + H] eps;
    y[1:T] = y_obs;

    for (t in 1:T) {
        real mu = alpha;
        for (i in 1:min(p, t - 1)) mu += phi[i]   * y[t - i];
        for (j in 1:min(q, t - 1)) mu += theta[j] * eps[t - j];
        eps[t] = y_obs[t] - mu;
    }
    for (h in 1:H) {
        real mu = alpha;
        for (i in 1:p) mu += phi[i]   * y[T + h - i];
        for (j in 1:q) mu += theta[j] * eps[T + h - j];
        eps[T + h] = normal_rng(0, sigma);
        y[T + h]   = mu + eps[T + h];
    }
    return y[(T + 1):(T + H)];
}

//@laplace
//@brief Forecast an ARMA(p, q) series H steps beyond the observed data with covariates
//@param vector y_obs Observed series, length T
//@param int H Forecast horizon
//@param real alpha
//@param vector phi
//@param vector theta
//@param real sigma
//@param matrix X Observed covariates, T rows
//@param matrix X_f Future covariates, H rows
//@param vector beta
//@math As arma_forecast_rng, with x_t^\top\beta in \mu_t for both the replay and the forecast
//@return vector[H] Posterior predictive draws for t = T+1,...,T+H
vector ts__arma_forecast_rng(vector y_obs, int H, real alpha, vector phi, vector theta, real sigma, matrix X, matrix X_f, vector beta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    int q = num_elements(theta);
    vector[T + H] y;
    vector[T + H] eps;
    y[1:T] = y_obs;

    for (t in 1:T) {
        real mu = alpha + X[t] * beta;
        for (i in 1:min(p, t - 1)) mu += phi[i]   * y[t - i];
        for (j in 1:min(q, t - 1)) mu += theta[j] * eps[t - j];
        eps[t] = y_obs[t] - mu;
    }
    for (h in 1:H) {
        real mu = alpha + X_f[h] * beta;
        for (i in 1:p) mu += phi[i]   * y[T + h - i];
        for (j in 1:q) mu += theta[j] * eps[T + h - j];
        eps[T + h] = normal_rng(0, sigma);
        y[T + h]   = mu + eps[T + h];
    }
    return y[(T + 1):(T + H)];
}

//@laplace
//@brief Simulate a SARMA series from a cold start
//@param int H Number of values to simulate
//@param real alpha
//@param vector phi
//@param vector Phi
//@param vector theta
//@param vector Theta
//@param int s Seasonal period
//@param real sigma
//@math y_t = \mu_t + \varepsilon_t with the SARMA recursion for \mu_t
//@return vector[H] Simulated observations
vector ts__sarma_rng(int H, real alpha, vector phi, vector Phi, vector theta, vector Theta, int s, real sigma){
    int p = num_elements(phi);   int P = num_elements(Phi);
    int q = num_elements(theta); int Q = num_elements(Theta);
    vector[H] y;
    vector[H] eps;

    for (h in 1:H) {
        real mu = alpha;
        for (i in 1:min(p, h - 1))         mu += phi[i]   * y[h - i];
        for (k in 1:min(P, (h - 1) %/% s)) mu += Phi[k]   * y[h - k * s];
        for (j in 1:min(q, h - 1))         mu += theta[j] * eps[h - j];
        for (n in 1:min(Q, (h - 1) %/% s)) mu += Theta[n] * eps[h - n * s];
        eps[h] = normal_rng(0, sigma);
        y[h]   = mu + eps[h];
    }
    return y;
}

//@laplace
//@brief Forecast a SARMA series H steps beyond the observed data
//@param vector y_obs Observed series, length T
//@param int H Forecast horizon
//@param real alpha
//@param vector phi
//@param vector Phi
//@param vector theta
//@param vector Theta
//@param int s Seasonal period
//@param real sigma
//@math Residuals replayed from y_{1:T}, then the SARMA recursion is run forward
//@return vector[H] Posterior predictive draws for t = T+1,...,T+H
vector ts__sarma_forecast_rng(vector y_obs, int H, real alpha, vector phi, vector Phi, vector theta, vector Theta, int s, real sigma){
    int T = num_elements(y_obs);
    int p = num_elements(phi);   int P = num_elements(Phi);
    int q = num_elements(theta); int Q = num_elements(Theta);
    vector[T + H] y;
    vector[T + H] eps;
    y[1:T] = y_obs;

    for (t in 1:T) {
        real mu = alpha;
        for (i in 1:min(p, t - 1))         mu += phi[i]   * y[t - i];
        for (k in 1:min(P, (t - 1) %/% s)) mu += Phi[k]   * y[t - k * s];
        for (j in 1:min(q, t - 1))         mu += theta[j] * eps[t - j];
        for (n in 1:min(Q, (t - 1) %/% s)) mu += Theta[n] * eps[t - n * s];
        eps[t] = y_obs[t] - mu;
    }
    for (h in 1:H) {
        real mu = alpha;
        for (i in 1:p) mu += phi[i]   * y[T + h - i];
        for (k in 1:P) mu += Phi[k]   * y[T + h - k * s];
        for (j in 1:q) mu += theta[j] * eps[T + h - j];
        for (n in 1:Q) mu += Theta[n] * eps[T + h - n * s];
        eps[T + h] = normal_rng(0, sigma);
        y[T + h]   = mu + eps[T + h];
    }
    return y[(T + 1):(T + H)];
}

// Autoregression Moving Average with Differencing ARIMA(p, d, q) =====================================================
//@laplace
//@brief Autoregression Moving Average Differencing primary
//@param real alpha
//@param int d
//@param vector phi
//@param vector theta
//@param vector y_obs
//@math z_t = \nabla^d y_t, \quad \mu_t = \alpha + \sum_i\phi_i z_{t-i} + \sum_j\theta_j\varepsilon_{t-j}, \quad \varepsilon_t = z_t - \mu_t
//@return array of 2 over the differenced scale, length T-d: array[1] is mu_t, array[2] is eps
array[] vector ts__arima(vector y_obs, real alpha, int d, vector phi, vector theta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    int q = num_elements(theta);

    int N = T - d;
    array[2] vector[N] results; // array[1] mu_t , array[2] eps
    vector[N] delta_y = ts__difference(y_obs, d);

    for(t in 1:N){
        results[1][t] = alpha;
        for(i in 1:min(p, t - 1)) results[1][t] += phi[i]   * delta_y[t - i];
        for(j in 1:min(q, t - 1)) results[1][t] += theta[j] * results[2][t -j];
        results[2][t] = delta_y[t] - results[1][t];
    }
    return results;
}

//@laplace
//@brief Autoregression Moving Average Differencing primary
//@param real alpha
//@param int d
//@param vector phi
//@param vector theta
//@param vector y_obs
//@param matrix X Covariates aligned to the DIFFERENCED scale: X must have T-d rows
//@param vector beta
//@math \mu_t = \alpha + x_t^\top\beta + \sum_i\phi_i z_{t-i} + \sum_j\theta_j\varepsilon_{t-j}, \quad z = \nabla^d y
//@return array of 2 over the differenced scale, length T-d: array[1] is mu_t, array[2] is eps
array[] vector ts__arima(vector y_obs, real alpha, int d, vector phi, vector theta, matrix X, vector beta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    int q = num_elements(theta);

    int N = T - d;
    array[2] vector[N] results; // array[1] mu_t , array[2] eps
    vector[N] delta_y = ts__difference(y_obs, d);

    for(t in 1:N){
        results[1][t] = alpha + X[t] * beta;
        for(i in 1:min(p, t - 1)) results[1][t] += phi[i]   * delta_y[t - i];
        for(j in 1:min(q, t - 1)) results[1][t] += theta[j] * results[2][t -j];
        results[2][t] = delta_y[t] - results[1][t];
    }
    return results;
}

//@laplace
//@brief Seasonal Autoregression Moving Average Differencing primary
//@param real alpha
//@param vector phi Ordinary AR coefficients
//@param vector Phi Seasonal AR coefficients
//@param vector theta Ordinary MA coefficients
//@param vector Theta Seasonal MA coefficients
//@param vector y_obs Observations
//@param int d Ordinary differencing order
//@param int D Seasonal differencing order
//@param int s Seasonal period
//@math z_t = \nabla^d \nabla_s^D y_t, then the SARMA recursion is applied to z
//@return Array of vectors over the differenced scale, length T - D*s - d: [1] mu_t, [2] eps
array[] vector ts__sarima(vector y_obs, real alpha, vector phi, vector Phi, vector theta, vector Theta, int d, int D, int s){
    int T = num_elements(y_obs);
    int p = num_elements(phi);   int P = num_elements(Phi);
    int q = num_elements(theta); int Q = num_elements(Theta);

    int N = T - D * s - d;
    vector[N] delta_y = ts__difference(ts__seasonal_difference(y_obs, s, D), d);
    array[2] vector[N] results;

    for (t in 1:N) {
        results[1][t] = alpha;

        // AR
        for (i in 1:min(p, t - 1))         results[1][t] += phi[i]   * delta_y[t - i];
        // Seasonal AR
        for (i in 1:min(P, (t - 1) %/% s)) results[1][t] += Phi[i]   * delta_y[t - i * s];
        // MA
        for (i in 1:min(q, t - 1))         results[1][t] += theta[i] * results[2][t - i];
        // Seasonal MA
        for (i in 1:min(Q, (t - 1) %/% s)) results[1][t] += Theta[i] * results[2][t - i * s];

        // Innovation
        results[2][t] = delta_y[t] - results[1][t];
    }
    return results;
}

//@laplace
//@brief Seasonal Autoregression Moving Average Differencing primary
//@param real alpha
//@param vector phi Ordinary AR coefficients
//@param vector Phi Seasonal AR coefficients
//@param vector theta Ordinary MA coefficients
//@param vector Theta Seasonal MA coefficients
//@param vector y_obs Observations
//@param int d Ordinary differencing order
//@param int D Seasonal differencing order
//@param int s Seasonal period
//@param matrix X Covariates aligned to the DIFFERENCED scale: T - D*s - d rows
//@param vector beta
//@math z_t = \nabla^d \nabla_s^D y_t, \quad \mu_t = \alpha + x_t^\top\beta + \text{SARMA}(z)_t
//@return Array of vectors over the differenced scale: [1] mu_t, [2] eps
array[] vector ts__sarima(vector y_obs, real alpha, vector phi, vector Phi, vector theta, vector Theta, int d, int D, int s, matrix X, vector beta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);   int P = num_elements(Phi);
    int q = num_elements(theta); int Q = num_elements(Theta);

    int N = T - D * s - d;
    vector[N] delta_y = ts__difference(ts__seasonal_difference(y_obs, s, D), d);
    array[2] vector[N] results;

    for (t in 1:N) {
        results[1][t] = alpha + X[t] * beta;

        // AR
        for (i in 1:min(p, t - 1))         results[1][t] += phi[i]   * delta_y[t - i];
        // Seasonal AR
        for (i in 1:min(P, (t - 1) %/% s)) results[1][t] += Phi[i]   * delta_y[t - i * s];
        // MA
        for (i in 1:min(q, t - 1))         results[1][t] += theta[i] * results[2][t - i];
        // Seasonal MA
        for (i in 1:min(Q, (t - 1) %/% s)) results[1][t] += Theta[i] * results[2][t - i * s];

        // Innovation
        results[2][t] = delta_y[t] - results[1][t];
    }
    return results;
}

//@laplace
//@brief ARIMA Log Pointwise Density Function
//@param real alpha
//@param int d
//@param vector phi
//@param vector theta
//@param vector y_obs
//@param real sigma
//@math \log p(\nabla^d y) = \sum_t \log\mathcal{N}(z_t \mid \mu_t, \sigma). NOTE the density is on the differenced scale, so models with different d are NOT comparable by lp__ or loo.
//@return real Log density of the differenced series
real ts__arima_lpdf(vector y_obs, real alpha, int d, vector phi, vector theta, real sigma){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    int q = num_elements(theta);

    int N = T - d;
    vector[N] mu_t;
    vector[N] eps;
    vector[N] delta_y = ts__difference(y_obs, d);

    for(t in 1:N){
        mu_t[t] = alpha;
        for(i in 1:min(p, t - 1)) mu_t[t] += phi[i]   * delta_y[t - i];
        for(j in 1:min(q, t - 1)) mu_t[t] += theta[j] * eps[t -j];
        eps[t] = delta_y[t] - mu_t[t];
    }
    return normal_lpdf(delta_y | mu_t, sigma);
}

//@laplace
//@brief ARIMA Log Pointwise Density Function
//@param real alpha
//@param int d
//@param vector phi
//@param vector theta
//@param vector y_obs
//@param real sigma
//@param matrix X Covariates aligned to the DIFFERENCED scale: T-d rows
//@param vector beta
//@math \log p(\nabla^d y) = \sum_t \log\mathcal{N}(z_t \mid \alpha + x_t^\top\beta + \dots,\ \sigma)
//@return real Log density of the differenced series
real ts__arima_lpdf(vector y_obs, real alpha, int d, vector phi, vector theta, real sigma, matrix X, vector beta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);
    int q = num_elements(theta);

    int N = T - d;
    vector[N] mu_t;
    vector[N] eps;
    vector[N] delta_y = ts__difference(y_obs, d);

    for(t in 1:N){
        mu_t[t] = alpha + X[t] * beta;
        for(i in 1:min(p, t - 1)) mu_t[t] += phi[i]   * delta_y[t - i];
        for(j in 1:min(q, t - 1)) mu_t[t] += theta[j] * eps[t -j];
        eps[t] = delta_y[t] - mu_t[t];
    }
    return normal_lpdf(delta_y | mu_t, sigma);
}

//@laplace
//@brief SARIMA Log Pointwise Density Function
//@param real alpha
//@param vector phi Ordinary AR coefficients
//@param vector Phi Seasonal AR coefficients
//@param vector theta Ordinary MA coefficients
//@param vector Theta Seasonal MA coefficients
//@param vector y_obs Observations
//@param int d Ordinary differencing order
//@param int D Seasonal differencing order
//@param int s Seasonal period
//@param real sigma
//@math \log p(z) = \sum_t \log\mathcal{N}(z_t \mid \mu_t, \sigma), \quad z = \nabla^d\nabla_s^D y
//@return real Log density of the differenced series
real ts__sarima_lpdf(vector y_obs, real alpha, vector phi, vector Phi, vector theta, vector Theta, int d, int D, int s, real sigma){
    int T = num_elements(y_obs);
    int p = num_elements(phi);   int P = num_elements(Phi);
    int q = num_elements(theta); int Q = num_elements(Theta);

    int N = T - D * s - d;
    vector[N] delta_y = ts__difference(ts__seasonal_difference(y_obs, s, D), d);
    vector[N] mu_t;
    vector[N] eps;

    for (t in 1:N) {
        mu_t[t] = alpha;
        // AR
        for (i in 1:min(p, t - 1))         mu_t[t] += phi[i]   * delta_y[t - i];
        // Seasonal AR
        for (i in 1:min(P, (t - 1) %/% s)) mu_t[t] += Phi[i]   * delta_y[t - i * s];
        // MA
        for (i in 1:min(q, t - 1))         mu_t[t] += theta[i] * eps[t - i];
        // Seasonal MA
        for (i in 1:min(Q, (t - 1) %/% s)) mu_t[t] += Theta[i] * eps[t - i * s];

        // Innovation
        eps[t] = delta_y[t] - mu_t[t];
    }
    return normal_lpdf(delta_y | mu_t, sigma);
}

//@laplace
//@brief SARIMA Log Pointwise Density Function
//@param real alpha
//@param vector phi Ordinary AR coefficients
//@param vector Phi Seasonal AR coefficients
//@param vector theta Ordinary MA coefficients
//@param vector Theta Seasonal MA coefficients
//@param vector y_obs Observations
//@param int d Ordinary differencing order
//@param int D Seasonal differencing order
//@param int s Seasonal period
//@param real sigma
//@param matrix X Covariates aligned to the DIFFERENCED scale
//@param vector beta
//@math \log p(z) = \sum_t \log\mathcal{N}(z_t \mid \alpha + x_t^\top\beta + \dots,\ \sigma)
//@return real Log density of the differenced series
real ts__sarima_lpdf(vector y_obs, real alpha, vector phi, vector Phi, vector theta, vector Theta, int d, int D, int s, real sigma, matrix X, vector beta){
    int T = num_elements(y_obs);
    int p = num_elements(phi);   int P = num_elements(Phi);
    int q = num_elements(theta); int Q = num_elements(Theta);

    int N = T - D * s - d;
    vector[N] delta_y = ts__difference(ts__seasonal_difference(y_obs, s, D), d);
    vector[N] mu_t;
    vector[N] eps;

    for (t in 1:N) {
        mu_t[t] = alpha + X[t] * beta;

        // AR
        for (i in 1:min(p, t - 1))         mu_t[t] += phi[i]   * delta_y[t - i];
        // Seasonal AR
        for (i in 1:min(P, (t - 1) %/% s)) mu_t[t] += Phi[i]   * delta_y[t - i * s];
        // MA
        for (i in 1:min(q, t - 1))         mu_t[t] += theta[i] * eps[t - i];
        // Seasonal MA
        for (i in 1:min(Q, (t - 1) %/% s)) mu_t[t] += Theta[i] * eps[t - i * s];

        // Innovation
        eps[t] = delta_y[t] - mu_t[t];
    }
    return normal_lpdf(delta_y | mu_t, sigma);
}

//@laplace
//@brief Simulate an ARIMA(p, d, q) series from a cold start
//@param int H Number of values to simulate
//@param vector y_init Seed levels, length d, oldest to newest. Used only to anchor the integration.
//@param real alpha
//@param int d
//@param vector phi
//@param vector theta
//@param real sigma
//@math Simulate z \sim ARMA(p,q), then integrate d times anchored at y_init
//@return vector[H] Simulated observations on the level scale
vector ts__arima_rng(int H, vector y_init, real alpha, int d, vector phi, vector theta, real sigma){
    vector[H] delta_f = ts__arma_rng(H, alpha, phi, theta, sigma);
    return ts__undifference(y_init, delta_f, d);
}

//@laplace
//@brief Forecast an ARIMA(p, d, q) series H steps beyond the observed data
//@param vector y_obs Observed series on the level scale, length T
//@param int H Forecast horizon
//@param real alpha
//@param int d
//@param vector phi
//@param vector theta
//@param real sigma
//@example vector y_f = ts::arima_forecast_rng(y, 12, alpha, 1, phi, theta, sigma);
//@math Forecast \nabla^d y forward with the ARMA recursion, then integrate back: y_{T+h} accumulates the forecast differences onto y_T
//@return vector[H] Posterior predictive draws on the level scale
vector ts__arima_forecast_rng(vector y_obs, int H, real alpha, int d, vector phi, vector theta, real sigma){
    int T = num_elements(y_obs);
    vector[T - d] delta_y = ts__difference(y_obs, d);
    vector[H] delta_f = ts__arma_forecast_rng(delta_y, H, alpha, phi, theta, sigma);
    return ts__undifference(y_obs, delta_f, d);
}

//@laplace
//@brief Forecast an ARIMA(p, d, q) series H steps ahead with covariates
//@param vector y_obs Observed series on the level scale, length T
//@param int H Forecast horizon
//@param real alpha
//@param int d
//@param vector phi
//@param vector theta
//@param real sigma
//@param matrix X Covariates on the DIFFERENCED scale, T-d rows
//@param matrix X_f Future covariates on the DIFFERENCED scale, H rows
//@param vector beta
//@math As arima_forecast_rng, with x_t^\top\beta carried through the ARMA recursion on the differenced scale
//@return vector[H] Posterior predictive draws on the level scale
vector ts__arima_forecast_rng(vector y_obs, int H, real alpha, int d, vector phi, vector theta, real sigma, matrix X, matrix X_f, vector beta){
    int T = num_elements(y_obs);
    vector[T - d] delta_y = ts__difference(y_obs, d);
    vector[H] delta_f = ts__arma_forecast_rng(delta_y, H, alpha, phi, theta, sigma, X, X_f, beta);
    return ts__undifference(y_obs, delta_f, d);
}

//@laplace
//@brief Forecast a SARIMA series H steps beyond the observed data
//@param vector y_obs Observed series on the level scale, length T
//@param int H Forecast horizon
//@param real alpha
//@param vector phi
//@param vector Phi
//@param vector theta
//@param vector Theta
//@param int d Ordinary differencing order
//@param int D Seasonal differencing order
//@param int s Seasonal period
//@param real sigma
//@math Forecast z = \nabla^d\nabla_s^D y with the SARMA recursion, undo \nabla^d, then undo \nabla_s^D
//@return vector[H] Posterior predictive draws on the level scale
vector ts__sarima_forecast_rng(vector y_obs, int H, real alpha, vector phi, vector Phi, vector theta, vector Theta, int d, int D, int s, real sigma){
    int T = num_elements(y_obs);
    vector[T - D * s] y_seasonal = ts__seasonal_difference(y_obs, s, D);
    vector[T - D * s - d] delta_y = ts__difference(y_seasonal, d);

    vector[H] delta_f   = ts__sarma_forecast_rng(delta_y, H, alpha, phi, Phi, theta, Theta, s, sigma);
    vector[H] seasonal_f = ts__undifference(y_seasonal, delta_f, d);

    return ts__unseasonal_difference(y_obs, seasonal_f, s, D);
}

// Beyond the linear Gaussian mean ========================================================================================================

// Exponential smoothing (Holt–Winters) ===================================================================================================
//@laplace
//@brief Additive Holt–Winters one-step-ahead predictions
//@param real alpha   level smoothing, (0,1)
//@param real beta    trend smoothing, (0,1)
//@param real gamma   seasonal smoothing, (0,1)
//@param real l0      level seed
//@param real b0      trend seed
//@param vector s0    seasonal seeds s_{-m+1..0}, length m, MUST sum to zero
//@param int m        seasonal period, m >= 1
//@math \hat{y}_t = \ell_{t-1} + b_{t-1} + s_{t-m} \\ \ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\ b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1} \\ s_t = \gamma(y_t - \ell_t) + (1-\gamma)s_{t-m}
//@return vector[N] One-step-ahead predictions \hat{y}_t
vector ts__exponential_smoothing(vector y_obs, real alpha, real beta, real gamma, real l0, real b0, vector s0, int m){
    int N = num_elements(y_obs);
    vector[N] y_hat;  vector[N + 1] lt;
    vector[N + 1] bt; vector[N + m] st;

    // Initialize the process
    lt[1]   = l0;
    bt[1]   = b0;
    st[1:m] = s0;

    for (t in 1:N) {
        // Resolve y_hat
        y_hat[t]  = lt[t] + bt[t] + st[t];

        // Level
        lt[t + 1] = alpha * (y_obs[t] - st[t])    + (1 - alpha) * (lt[t] + bt[t]);

        // Trend
        bt[t + 1] = beta  * (lt[t + 1] - lt[t])   + (1 - beta)  * bt[t];

        // Seasonal
        st[t + m] = gamma * (y_obs[t] - lt[t + 1])+ (1 - gamma) * st[t];
    }
    return y_hat;
}

//@laplace
//@brief Additive Holt–Winters one-step-ahead predictions
//@param real alpha   level smoothing, (0,1)
//@param real beta    trend smoothing, (0,1)
//@param real gamma   seasonal smoothing, (0,1)
//@param real l0      level seed
//@param real b0      trend seed
//@param vector s0    seasonal seeds s_{-m+1..0}, length m, MUST sum to zero
//@param int m        seasonal period, m >= 1
//@param matrix X     Covariates, N rows
//@param vector beta_c Covariate coefficients
//@math \hat{y}_t = \ell_{t-1} + b_{t-1} + s_{t-m} + x_t^\top\beta_c. The regression part is removed from y_t before the level and seasonal updates, so it is not absorbed twice.
//@return vector[N] One-step-ahead predictions \hat{y}_t
vector ts__exponential_smoothing(vector y_obs, real alpha, real beta, real gamma, real l0, real b0, vector s0, int m, matrix X, vector beta_c){
    int N = num_elements(y_obs);
    vector[N] y_hat;  vector[N + 1] lt;
    vector[N + 1] bt; vector[N + m] st;

    // Initialize the process
    lt[1]   = l0;
    bt[1]   = b0;
    st[1:m] = s0;

    for (t in 1:N) {
        real xb = X[t] * beta_c;

        // Resolve y_hat
        y_hat[t]  = lt[t] + bt[t] + st[t] + xb;

        // Level, on the covariate-adjusted series
        lt[t + 1] = alpha * (y_obs[t] - xb - st[t])     + (1 - alpha) * (lt[t] + bt[t]);

        // Trend
        bt[t + 1] = beta  * (lt[t + 1] - lt[t])         + (1 - beta)  * bt[t];

        // Seasonal
        st[t + m] = gamma * (y_obs[t] - xb - lt[t + 1]) + (1 - gamma) * st[t];
    }
    return y_hat;
}

//@laplace
//@brief Additive Holt–Winters Log Pointwise Density Function
//@param real alpha   level smoothing, (0,1)
//@param real beta    trend smoothing, (0,1)
//@param real gamma   seasonal smoothing, (0,1)
//@param real l0      level seed
//@param real b0      trend seed
//@param vector s0    seasonal seeds s_{-m+1..0}, length m, MUST sum to zero
//@param int m        seasonal period, m >= 1
//@param real sigma
//@math \log p(y) = \sum_{t=1}^{N}\log\mathcal{N}(y_t \mid \hat{y}_t, \sigma)
//@return real Log density of y_obs
real ts__exponential_smoothing_lpdf(vector y_obs, real alpha, real beta, real gamma, real l0, real b0, vector s0, int m, real sigma){
    return normal_lpdf(y_obs | ts__exponential_smoothing(y_obs, alpha, beta, gamma, l0, b0, s0, m), sigma);
}

//@laplace
//@brief Additive Holt–Winters Log Pointwise Density Function
//@param real alpha   level smoothing, (0,1)
//@param real beta    trend smoothing, (0,1)
//@param real gamma   seasonal smoothing, (0,1)
//@param real l0      level seed
//@param real b0      trend seed
//@param vector s0    seasonal seeds s_{-m+1..0}, length m, MUST sum to zero
//@param int m        seasonal period, m >= 1
//@param real sigma
//@param matrix X     Covariates, N rows
//@param vector beta_c Covariate coefficients
//@math \log p(y) = \sum_{t=1}^{N}\log\mathcal{N}(y_t \mid \hat{y}_t, \sigma)
//@return real Log density of y_obs
real ts__exponential_smoothing_lpdf(vector y_obs, real alpha, real beta, real gamma, real l0, real b0, vector s0, int m, real sigma, matrix X, vector beta_c){
    return normal_lpdf(y_obs | ts__exponential_smoothing(y_obs, alpha, beta, gamma, l0, b0, s0, m, X, beta_c), sigma);
}

//@laplace
//@brief Simulate an additive Holt–Winters series from explicit seeds
//@param int H Number of values to simulate
//@param real alpha   level smoothing, (0,1)
//@param real beta    trend smoothing, (0,1)
//@param real gamma   seasonal smoothing, (0,1)
//@param real l0      level seed
//@param real b0      trend seed
//@param vector s0    seasonal seeds, length m, MUST sum to zero
//@param int m        seasonal period
//@param real sigma
//@math y_t \sim \mathcal{N}(\ell_{t-1} + b_{t-1} + s_{t-m},\ \sigma), with the filter then updated using the drawn y_t
//@return vector[H] Simulated observations
vector ts__exponential_smoothing_rng(int H, real alpha, real beta, real gamma, real l0, real b0, vector s0, int m, real sigma){
    vector[H] y;      vector[H + 1] lt;
    vector[H + 1] bt; vector[H + m] st;

    lt[1]   = l0;
    bt[1]   = b0;
    st[1:m] = s0;

    for (t in 1:H) {
        y[t]      = normal_rng(lt[t] + bt[t] + st[t], sigma);
        lt[t + 1] = alpha * (y[t] - st[t])      + (1 - alpha) * (lt[t] + bt[t]);
        bt[t + 1] = beta  * (lt[t + 1] - lt[t]) + (1 - beta)  * bt[t];
        st[t + m] = gamma * (y[t] - lt[t + 1])  + (1 - gamma) * st[t];
    }
    return y;
}

//@laplace
//@brief Forecast an additive Holt–Winters series H steps beyond the observed data
//@param vector y_obs Observed series, length N
//@param int H Forecast horizon
//@param real alpha, beta, gamma Smoothing parameters
//@param real l0, b0 Seeds
//@param vector s0 Seasonal seeds, length m
//@param int m Seasonal period
//@param real sigma
//@example vector y_f = ts::exponential_smoothing_forecast_rng(y, 12, alpha, beta, gamma, l0, b0, s0, 12, sigma);
//@math The filter is replayed over y_{1:N} to recover \ell_N, b_N, s; then \hat{y}_{N+h} = \ell_N + h\,b_N + s_{N+h-m} and y_{N+h}\sim\mathcal{N}(\hat{y}_{N+h},\sigma). The seasonal states are frozen at their final values over the horizon.
//@return vector[H] Posterior predictive draws for t = N+1,...,N+H
vector ts__exponential_smoothing_forecast_rng(vector y_obs, int H, real alpha, real beta, real gamma, real l0, real b0, vector s0, int m, real sigma){
    int N = num_elements(y_obs);
    vector[N + 1] lt; vector[N + 1] bt; vector[N + m] st;
    vector[H] y_f;

    lt[1]   = l0;
    bt[1]   = b0;
    st[1:m] = s0;

    for (t in 1:N) {
        lt[t + 1] = alpha * (y_obs[t] - st[t])     + (1 - alpha) * (lt[t] + bt[t]);
        bt[t + 1] = beta  * (lt[t + 1] - lt[t])    + (1 - beta)  * bt[t];
        st[t + m] = gamma * (y_obs[t] - lt[t + 1]) + (1 - gamma) * st[t];
    }
    for (h in 1:H) {
        // st[j] holds s_{j-m}, so the effect needed at time N+h sits at index N+h.
        // Beyond one full period the seasonal states are frozen and recycled.
        int idx = N + h;
        while (idx > N + m) idx -= m;
        y_f[h] = normal_rng(lt[N + 1] + h * bt[N + 1] + st[idx], sigma);
    }
    return y_f;
}

//@laplace
//@brief Forecast an additive Holt–Winters series H steps ahead with covariates
//@param vector y_obs Observed series, length N
//@param int H Forecast horizon
//@param real alpha, beta, gamma Smoothing parameters
//@param real l0, b0 Seeds
//@param vector s0 Seasonal seeds, length m
//@param int m Seasonal period
//@param real sigma
//@param matrix X Observed covariates, N rows
//@param matrix X_f Future covariates, H rows
//@param vector beta_c
//@math As exponential_smoothing_forecast_rng, with x_{N+h}^\top\beta_c added to the forecast mean
//@return vector[H] Posterior predictive draws for t = N+1,...,N+H
vector ts__exponential_smoothing_forecast_rng(vector y_obs, int H, real alpha, real beta, real gamma, real l0, real b0, vector s0, int m, real sigma, matrix X, matrix X_f, vector beta_c){
    int N = num_elements(y_obs);
    vector[N + 1] lt; vector[N + 1] bt; vector[N + m] st;
    vector[H] y_f;

    lt[1]   = l0;
    bt[1]   = b0;
    st[1:m] = s0;

    for (t in 1:N) {
        real xb = X[t] * beta_c;
        lt[t + 1] = alpha * (y_obs[t] - xb - st[t])     + (1 - alpha) * (lt[t] + bt[t]);
        bt[t + 1] = beta  * (lt[t + 1] - lt[t])         + (1 - beta)  * bt[t];
        st[t + m] = gamma * (y_obs[t] - xb - lt[t + 1]) + (1 - gamma) * st[t];
    }
    for (h in 1:H) {
        int idx = N + h;
        while (idx > N + m) idx -= m;
        y_f[h] = normal_rng(lt[N + 1] + h * bt[N + 1] + st[idx] + X_f[h] * beta_c, sigma);
    }
    return y_f;
}

// GARCH ==================================================================================================================================
//@laplace
//@brief GARCH conditional standard deviations
//@param real omega  variance floor, omega > 0. NOT the long-run variance.
//@param real alpha  reactivity, alpha >= 0
//@param real beta   persistence, beta >= 0, with alpha + beta < 1
//@param vector y_obs
//@param vector mu   Conditional mean, supplied by any mean model
//@math \varepsilon_t = y_t - \mu_t, \quad \sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2, \quad \sigma_1^2 = \frac{\omega}{1 - \alpha - \beta}
//@return vector[N] Conditional standard deviations \sigma_t
vector ts__garch(vector y_obs, real omega, real alpha, real beta, vector mu){
    int N = num_elements(y_obs);
    vector[N] sd2;
    vector[N] eps;

    // stationary initialisation
    sd2[1] = omega / (1 - alpha - beta);
    eps   = y_obs - mu;

    for(t in 2:N){
        sd2[t]  = omega + alpha * square(eps[t - 1]) + beta * sd2[t - 1];
    }
    return sqrt(sd2);
}

//@laplace
//@brief GARCH Log Pointwise Density Function
//@param real omega  variance floor, omega > 0. NOT the long-run variance.
//@param real alpha  reactivity, alpha >= 0
//@param real beta   persistence, beta >= 0, with alpha + beta < 1
//@param vector y_obs
//@param vector mu   Conditional mean, supplied by any mean model
//@math \log p(y) = \sum_{t=1}^{N}\log\mathcal{N}(y_t \mid \mu_t, \sigma_t)
//@return real Log density of y_obs
real ts__garch_lpdf(vector y_obs, real omega, real alpha, real beta, vector mu){
    return normal_lpdf(y_obs | mu, ts__garch(y_obs, omega, alpha, beta, mu));
}

//@laplace
//@brief Simulate a GARCH series from the stationary initialisation
//@param int H Number of values to simulate
//@param real omega Variance floor
//@param real alpha Reactivity
//@param real beta  Persistence
//@param vector mu  Conditional mean over the H simulated points
//@math \sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2, \quad y_t = \mu_t + \varepsilon_t, \quad \varepsilon_t\sim\mathcal{N}(0,\sigma_t)
//@return vector[H] Simulated observations
vector ts__garch_rng(int H, real omega, real alpha, real beta, vector mu){
    vector[H] y;
    vector[H] sd2;
    vector[H] eps;

    sd2[1] = omega / (1 - alpha - beta);
    eps[1] = normal_rng(0, sqrt(sd2[1]));
    y[1]   = mu[1] + eps[1];

    for (t in 2:H) {
        sd2[t] = omega + alpha * square(eps[t - 1]) + beta * sd2[t - 1];
        eps[t] = normal_rng(0, sqrt(sd2[t]));
        y[t]   = mu[t] + eps[t];
    }
    return y;
}

//@laplace
//@brief Forecast a GARCH series H steps beyond the observed data
//@param vector y_obs Observed series, length N
//@param int H Forecast horizon
//@param real omega Variance floor
//@param real alpha Reactivity
//@param real beta  Persistence
//@param vector mu  Conditional mean over the observed points, length N
//@param vector mu_f Conditional mean over the forecast horizon, length H
//@math The variance recursion is replayed over y_{1:N}, then continued: \sigma_{N+h}^2 = \omega + \alpha\varepsilon_{N+h-1}^2 + \beta\sigma_{N+h-1}^2
//@return vector[H] Posterior predictive draws for t = N+1,...,N+H
vector ts__garch_forecast_rng(vector y_obs, int H, real omega, real alpha, real beta, vector mu, vector mu_f){
    int N = num_elements(y_obs);
    vector[N + H] sd2;
    vector[N + H] eps;
    vector[H] y_f;

    sd2[1] = omega / (1 - alpha - beta);
    eps[1:N] = y_obs - mu;

    for (t in 2:N) sd2[t] = omega + alpha * square(eps[t - 1]) + beta * sd2[t - 1];

    for (h in 1:H) {
        sd2[N + h] = omega + alpha * square(eps[N + h - 1]) + beta * sd2[N + h - 1];
        eps[N + h] = normal_rng(0, sqrt(sd2[N + h]));
        y_f[h]     = mu_f[h] + eps[N + h];
    }
    return y_f;
}

// STAR ===================================================================================================================================
//@laplace
//@brief Smooth transition AR (STAR). Returns mu for t = L+1..N, L = max(p, d).
//@param real alpha1, alpha2  regime intercepts
//@param vector phi1, phi2    regime AR coefficients, both length p
//@param real c               threshold location, on the scale of y
//@param real kappa           transition sharpness, > 0, dimensionless
//@param int d                delay, >= 1. DATA, never a parameter.
//@param real scale           spread of y; makes kappa dimensionless
//@math G_t = \mathrm{logit}^{-1}\!\left(\kappa\frac{y_{t-d} - c}{\text{scale}}\right) \\ \mu_t = (1 - G_t)\left(\alpha_1 + \sum_j \phi^{(1)}_j y_{t-j}\right) + G_t\left(\alpha_2 + \sum_j \phi^{(2)}_j y_{t-j}\right)
//@return vector[N - L] mu, aligned to t = L+1,...,N
vector ts__star(vector y_obs, real alpha1, real alpha2, vector phi1, vector phi2, real c, real kappa, int d, real scale){
    int N = num_elements(y_obs);
    int p = num_elements(phi1);
    int L = max(p, d);
    vector[N - L] mu;

    for (t in (L + 1):N) {
        real G  = inv_logit(kappa * (y_obs[t - d] - c) / scale);
        real m1 = alpha1;
        real m2 = alpha2;
        for (j in 1:p) {
            m1 += phi1[j] * y_obs[t - j];
            m2 += phi2[j] * y_obs[t - j];
        }
        mu[t - L] = (1 - G) * m1 + G * m2;
    }
    return mu;
}

//@laplace
//@brief Smooth transition AR (STAR)
//@param real alpha1, alpha2  regime intercepts
//@param vector phi1, phi2    regime AR coefficients, both length p
//@param real c               threshold location, on the scale of y
//@param real kappa           transition sharpness, > 0, dimensionless
//@param int d                delay, >= 1. DATA, never a parameter.
//@param real scale           spread of y; makes kappa dimensionless
//@param matrix X             Covariates indexed on the FULL series, N rows
//@param vector beta
//@math \mu_t = (1 - G_t)m^{(1)}_t + G_t m^{(2)}_t + x_t^\top\beta
//@return vector[N - L] mu, aligned to t = L+1,...,N
vector ts__star(vector y_obs, real alpha1, real alpha2, vector phi1, vector phi2, real c, real kappa, int d, real scale, matrix X, vector beta){
    int N = num_elements(y_obs);
    int p = num_elements(phi1);
    int L = max(p, d);
    vector[N - L] mu;

    for (t in (L + 1):N) {
        real G  = inv_logit(kappa * (y_obs[t - d] - c) / scale);
        real m1 = alpha1;
        real m2 = alpha2;
        for (j in 1:p) {
            m1 += phi1[j] * y_obs[t - j];
            m2 += phi2[j] * y_obs[t - j];
        }
        mu[t - L] = (1 - G) * m1 + G * m2 + X[t] * beta;
    }
    return mu;
}

//@laplace
//@brief Smooth transition AR (STAR) Log Pointwise Density Function
//@param real alpha1, alpha2  regime intercepts
//@param vector phi1, phi2    regime AR coefficients, both length p
//@param real c               threshold location, on the scale of y
//@param real kappa           transition sharpness, > 0, dimensionless
//@param int d                delay, >= 1. DATA, never a parameter.
//@param real scale           spread of y; makes kappa dimensionless
//@param real sigma
//@math \log p(y) = \sum_{t=L+1}^{N}\log\mathcal{N}(y_t \mid \mu_t, \sigma), \quad L = \max(p, d)
//@return real Log density of y_obs[(L+1):N]
real ts__star_lpdf(vector y_obs, real alpha1, real alpha2, vector phi1, vector phi2, real c, real kappa, int d, real scale, real sigma){
    int N = num_elements(y_obs);
    int L = max(num_elements(phi1), d);
    return normal_lpdf(y_obs[(L + 1):N] | ts__star(y_obs, alpha1, alpha2, phi1, phi2, c, kappa, d, scale), sigma);
}

//@laplace
//@brief Smooth transition AR (STAR) Log Pointwise Density Function
//@param real alpha1, alpha2  regime intercepts
//@param vector phi1, phi2    regime AR coefficients, both length p
//@param real c               threshold location, on the scale of y
//@param real kappa           transition sharpness, > 0, dimensionless
//@param int d                delay, >= 1. DATA, never a parameter.
//@param real scale           spread of y; makes kappa dimensionless
//@param real sigma
//@param matrix X             Covariates indexed on the FULL series, N rows
//@param vector beta
//@math \log p(y) = \sum_{t=L+1}^{N}\log\mathcal{N}(y_t \mid \mu_t, \sigma), \quad L = \max(p, d)
//@return real Log density of y_obs[(L+1):N]
real ts__star_lpdf(vector y_obs, real alpha1, real alpha2, vector phi1, vector phi2, real c, real kappa, int d, real scale, real sigma, matrix X, vector beta){
    int N = num_elements(y_obs);
    int L = max(num_elements(phi1), d);
    return normal_lpdf(y_obs[(L + 1):N] | ts__star(y_obs, alpha1, alpha2, phi1, phi2, c, kappa, d, scale, X, beta), sigma);
}

//@laplace
//@brief Simulate a STAR series forward from explicit seeds
//@param int H Number of values to simulate
//@param vector y_init Seed values, length >= max(p, d), oldest to newest
//@param real alpha1, alpha2 Regime intercepts
//@param vector phi1, phi2 Regime AR coefficients
//@param real c Threshold location
//@param real kappa Transition sharpness
//@param int d Delay
//@param real scale Spread of y
//@param real sigma
//@math y_t \sim \mathcal{N}((1-G_t)m^{(1)}_t + G_t m^{(2)}_t,\ \sigma)
//@return vector[H] Simulated observations
vector ts__star_rng(int H, vector y_init, real alpha1, real alpha2, vector phi1, vector phi2, real c, real kappa, int d, real scale, real sigma){
    int L = num_elements(y_init);
    int p = num_elements(phi1);
    vector[L + H] full;
    full[1:L] = y_init;

    for (h in 1:H) {
        int t = L + h;
        real G  = inv_logit(kappa * (full[t - d] - c) / scale);
        real m1 = alpha1;
        real m2 = alpha2;
        for (j in 1:p) {
            m1 += phi1[j] * full[t - j];
            m2 += phi2[j] * full[t - j];
        }
        full[t] = normal_rng((1 - G) * m1 + G * m2, sigma);
    }
    return full[(L + 1):(L + H)];
}

//@laplace
//@brief Forecast a STAR series H steps beyond the observed data
//@param vector y_obs Observed series, length N
//@param int H Forecast horizon
//@param real alpha1, alpha2 Regime intercepts
//@param vector phi1, phi2 Regime AR coefficients
//@param real c Threshold location
//@param real kappa Transition sharpness
//@param int d Delay
//@param real scale Spread of y
//@param real sigma
//@math The regime weight G_{N+h} is computed from the drawn y_{N+h-d}, so the forecast can switch regimes along the horizon
//@return vector[H] Posterior predictive draws for t = N+1,...,N+H
vector ts__star_forecast_rng(vector y_obs, int H, real alpha1, real alpha2, vector phi1, vector phi2, real c, real kappa, int d, real scale, real sigma){
    int N = num_elements(y_obs);
    int p = num_elements(phi1);
    vector[N + H] full;
    full[1:N] = y_obs;

    for (h in 1:H) {
        int t = N + h;
        real G  = inv_logit(kappa * (full[t - d] - c) / scale);
        real m1 = alpha1;
        real m2 = alpha2;
        for (j in 1:p) {
            m1 += phi1[j] * full[t - j];
            m2 += phi2[j] * full[t - j];
        }
        full[t] = normal_rng((1 - G) * m1 + G * m2, sigma);
    }
    return full[(N + 1):(N + H)];
}

// INGARCH ================================================================================================================================
//@laplace
//@brief Log-linear INGARCH / count autoregression
//@param real omega
//@param vector alpha Observation-lag coefficients
//@param vector beta Intensity-lag coefficients
//@param array[] int y_obs
//@math \eta_t = \omega + \sum_{p}\alpha_p\log(1 + y_{t-p}) + \sum_{q}\beta_q\log\lambda_{t-q}, \quad \lambda_t = e^{\eta_t}, \quad y_t \sim \text{Poisson}(\lambda_t)
//@return vector[N] Conditional intensities \lambda_t
vector ts__ingarch(array[] int y_obs, real omega, vector alpha, vector beta){
    int N = size(y_obs);
    int P = num_elements(alpha);
    int Q = num_elements(beta);
    vector[N] lambda_t;

    for(t in 1:N){
        // Initialize
        real eta_t = omega;
        // Observation lags
        for (p in 1:min(P, t - 1)) eta_t += alpha[p] * log1p(y_obs[t - p]);
        // Intensity lags
        for(q in 1:min(Q, t - 1))  eta_t += beta[q] * log(lambda_t[t - q]);
        // Accumulate
        lambda_t[t] = exp(eta_t);
    }
    return lambda_t;
}

//@laplace
//@brief Log-linear INGARCH / count autoregression
//@param real omega
//@param vector alpha Observation-lag coefficients
//@param vector beta Intensity-lag coefficients
//@param array[] int y_obs
//@param matrix X Covariates, N rows
//@param vector beta_c Covariate coefficients
//@math \eta_t = \omega + x_t^\top\beta_c + \sum_p\alpha_p\log(1 + y_{t-p}) + \sum_q\beta_q\log\lambda_{t-q}. The regression sits INSIDE the log link, so \lambda_t stays positive.
//@return vector[N] Conditional intensities \lambda_t
vector ts__ingarch(array[] int y_obs, real omega, vector alpha, vector beta, matrix X, vector beta_c){
    int N = size(y_obs);
    int P = num_elements(alpha);
    int Q = num_elements(beta);
    vector[N] lambda_t;

    for(t in 1:N){
        // Initialize
        real eta_t = omega + X[t] * beta_c;
        // Observation lags
        for (p in 1:min(P, t - 1)) eta_t += alpha[p] * log1p(y_obs[t - p]);
        // Intensity lags
        for(q in 1:min(Q, t - 1))  eta_t += beta[q] * log(lambda_t[t - q]);
        // Accumulate
        lambda_t[t] = exp(eta_t);
    }
    return lambda_t;
}

//@laplace
//@brief Log-linear INGARCH / count autoregression Log Pointwise Mass Function
//@param real omega
//@param vector alpha Observation-lag coefficients
//@param vector beta Intensity-lag coefficients
//@param array[] int y_obs
//@math \log p(y) = \sum_{t=1}^{N}\log\text{Poisson}(y_t \mid \lambda_t)
//@return real Log mass of y_obs
real ts__ingarch_lpmf(array[] int y_obs, real omega, vector alpha, vector beta){
    return poisson_lpmf(y_obs | ts__ingarch(y_obs, omega, alpha, beta));
}

//@laplace
//@brief Log-linear INGARCH / count autoregression Log Pointwise Mass Function
//@param real omega
//@param vector alpha Observation-lag coefficients
//@param vector beta Intensity-lag coefficients
//@param array[] int y_obs
//@param matrix X Covariates, N rows
//@param vector beta_c Covariate coefficients
//@math \log p(y) = \sum_{t=1}^{N}\log\text{Poisson}(y_t \mid \lambda_t)
//@return real Log mass of y_obs
real ts__ingarch_lpmf(array[] int y_obs, real omega, vector alpha, vector beta, matrix X, vector beta_c){
    return poisson_lpmf(y_obs | ts__ingarch(y_obs, omega, alpha, beta, X, beta_c));
}

//@laplace
//@brief Simulate a log-linear INGARCH series from a cold start
//@param int H Number of values to simulate
//@param real omega
//@param vector alpha Observation-lag coefficients
//@param vector beta Intensity-lag coefficients
//@math \lambda_t = \exp(\eta_t), \quad y_t \sim \text{Poisson}(\lambda_t), with \eta_t built from the drawn counts
//@return array[H] int Simulated counts. poisson_rng errors if \lambda exceeds 2^{30}, so keep \omega modest.
array[] int ts__ingarch_rng(int H, real omega, vector alpha, vector beta){
    int P = num_elements(alpha);
    int Q = num_elements(beta);
    vector[H] lambda_t;
    array[H] int y;

    for (h in 1:H) {
        real eta = omega;
        for (p in 1:min(P, h - 1)) eta += alpha[p] * log1p(y[h - p]);
        for (q in 1:min(Q, h - 1)) eta += beta[q]  * log(lambda_t[h - q]);
        lambda_t[h] = exp(eta);
        y[h] = poisson_rng(lambda_t[h]);
    }
    return y;
}

//@laplace
//@brief Forecast a log-linear INGARCH series H steps beyond the observed data
//@param array[] int y_obs Observed counts, length N
//@param int H Forecast horizon
//@param real omega
//@param vector alpha Observation-lag coefficients
//@param vector beta Intensity-lag coefficients
//@math The intensity recursion is replayed over y_{1:N}, then continued using the drawn counts
//@return array[H] int Posterior predictive counts for t = N+1,...,N+H
array[] int ts__ingarch_forecast_rng(array[] int y_obs, int H, real omega, vector alpha, vector beta){
    int N = size(y_obs);
    int P = num_elements(alpha);
    int Q = num_elements(beta);
    vector[N + H] lambda_t;
    array[N + H] int y;
    array[H] int y_f;

    y[1:N] = y_obs;
    lambda_t[1:N] = ts__ingarch(y_obs, omega, alpha, beta);

    for (h in 1:H) {
        int t = N + h;
        real eta = omega;
        for (p in 1:P) eta += alpha[p] * log1p(y[t - p]);
        for (q in 1:Q) eta += beta[q]  * log(lambda_t[t - q]);
        lambda_t[t] = exp(eta);
        y[t]   = poisson_rng(lambda_t[t]);
        y_f[h] = y[t];
    }
    return y_f;
}


}
// Drawn functions (seed 20260918): difference, ar_lpdf, arma_rng.
// ARI(p, 1): the series is differenced once, then an AR(p) is fitted to the
// differences through the library's lag-design + ar_lpdf pair.  arma_rng draws
// a cold-start replicate of the differenced series in generated quantities.


data {
  int<lower=2> T;
  vector[T] y;
  int<lower=1> p;
}

transformed data {
  vector[T - 1] dy = ts__difference(y, 1);
  matrix[T - 1 - p, p] X_lag = ts__lag_design(p, dy);
  vector[T - 1 - p] dy_obs = dy[(p + 1):(T - 1)];
  vector[0] theta_none;
}

parameters {
  real alpha;
  vector[p] phi;
  real<lower=0> sigma;
}

model {
  alpha ~ normal(0, 2);
  phi ~ normal(0, 1);
  sigma ~ normal(0, 2);
  target += ts__ar_lpdf(dy_obs | alpha, phi, X_lag, sigma);
}

generated quantities {
  vector[T - 1] dy_rep = ts__arma_rng(T - 1, alpha, phi, theta_none, sigma);
}
