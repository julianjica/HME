#include <iostream>
#include <cmath>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <nlopt.hpp>  // NLopt for optimization
#include <boost/math/distributions/normal.hpp>
#include <boost/math/quadrature/gauss.hpp>

using namespace std;
using namespace Eigen;
using namespace boost::math;
using namespace boost::math::quadrature;

// Struct to hold optimization data
struct OptimizationData {
    const vector<double>& params;
    const vector<double>& a;
    const Matrix3d& C;
    const vector<double>& sigma;
};

// Manager problem function
tuple<double, double, double> manager_problem(double x, double y, const Vector3d& t, 
                                              const vector<double>& params, const vector<double>& a, 
                                              const Matrix3d& C, const vector<double>& sigma) {
    double Ac = params[0], Ad = params[1], lc = params[2], ld = params[3], sc = params[4], sk = params[5], sd = params[6], 
           beta = params[7], r = params[8];
    double a1 = a[0], a2 = a[1];
    double sigmac = sigma[0], sigmad = sigma[1];
    
    double tc = t[0], td = t[1], tk = t[2];
    
    // Net production
    double x1 = (1 + Ac * exp(-lc * x * x)) * (sc * tc + sk * tk) + x;
    double x2 = (1 + Ad * exp(-ld * y * y)) * (sd * td + sk * tk) + y;
    double w = a1 * x1 + a2 * x2 + beta;

    return {x1, x2, w};
}

// Pre-integration function
double pre_integration(double x, double y, const Vector3d& t, const vector<double>& params, 
                       const vector<double>& a, const Matrix3d& C, const vector<double>& sigma) {
    double r = params.back();
    double sigmac = sigma[0], sigmad = sigma[1];

    auto [x1, x2, w] = manager_problem(x, y, t, params, a, C, sigma);
    Vector3d C_t = C * t;
    double exponent = -r * (w - t.dot(C_t));
    
    normal_distribution<double> norm_x(0, sigmac);
    normal_distribution<double> norm_y(0, sigmad);

    return exp(exponent) * pdf(norm_x, x) * pdf(norm_y, y);
}

// Payoff function
double payoff(const Vector3d& t, const vector<double>& params, const vector<double>& a, 
              const Matrix3d& C, const vector<double>& sigma) {
    double sigmac = sigma[0], sigmad = sigma[1];
    
    auto integrand = [&](double x, double y) {
        return pre_integration(x, y, t, params, a, C, sigma);
    };

    double integral = gauss<double, 64>::integrate(
        [&](double x) {
            return gauss<double, 64>::integrate(
                [&](double y) { return integrand(x, y); }, -3 * sigmad, 3 * sigmad);
        }, -3 * sigmac, 3 * sigmac);

    double r = params.back();
    return -log(integral) / r;
}

// Wrapper for the objective function to use with NLopt
double objective_wrapper(const std::vector<double>& t, std::vector<double>& grad, void* data) {
    // Cast the void* data back to our OptimizationData struct
    OptimizationData* opt_data = reinterpret_cast<OptimizationData*>(data);
    
    // Reconstruct parameters
    const std::vector<double>& params = opt_data->params;
    const std::vector<double>& a = opt_data->a;
    const Eigen::Matrix3d& C = opt_data->C;
    const std::vector<double>& sigma = opt_data->sigma;

    // Convert t vector to Eigen Vector3d for use in the payoff function
    Eigen::Vector3d t_vec(t.data());

    // Call the payoff function (note: minimize the negative payoff)
    double result = -payoff(t_vec, params, a, C, sigma);

    // Debugging: Print the current value of the objective
    //std::cout << "Evaluating payoff at t = [" << t[0] << ", " << t[1] << ", " << t[2] << "] -> " << result << std::endl;

    return result;
}

// Optimal t function using NLopt
Eigen::Vector3d optimal_t(const std::vector<double>& params, const std::vector<double>& a, 
                          const Eigen::Matrix3d& C, const std::vector<double>& sigma) {
    // Prepare the optimization data to pass to the wrapper function
    OptimizationData opt_data = {params, a, C, sigma};

    // Create NLopt optimizer with Nelder-Mead algorithm
    nlopt::opt opt(nlopt::LN_NELDERMEAD, 3);

    // Set the objective function
    opt.set_min_objective(objective_wrapper, &opt_data);

    // Set optimization settings
    opt.set_ftol_rel(1e-6);  // Tolerance for stopping

    // Initial guess for t
    std::vector<double> t_opt = {1.0, 1.0, 1.0};
    double minf;  // Variable to store the minimum function value

    // Perform the optimization
    nlopt::result result = opt.optimize(t_opt, minf);

    // Return the optimal t values as an Eigen::Vector3d
    return Eigen::Vector3d(t_opt.data());
}

int main(int argc, char *argv[]) {
    vector<double> params = {1, 1, 1, 1, 1, 1, 2, 1, 1};
    vector<double> sigma = {atof(argv[1]), atof(argv[2])};
    vector<double> a = {1, 1};
    Matrix3d C;
    C << 2, 0, -1, 0, 2, -1, -1, -1, 2;
    
    // Find optimal t
    Vector3d t = optimal_t(params, a, C, sigma);
    cout << t[0] << " " << t[1] << " " << t[2] << endl;

    return 0;
}

