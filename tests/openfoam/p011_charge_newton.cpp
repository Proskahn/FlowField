// Exercise the production Newton helper without an OpenFOAM installation.
#include "ChargeNewton.H"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = Eigen::VectorXd;
using Matrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

static void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class Action>
static void mustReject(Action action)
{
    bool rejected = false;
    try { action(); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "An invalid Newton state was accepted");
}

static Vec linearSolve(const Matrix& matrix, const Vec& rhs)
{
    return Eigen::MatrixXd(matrix).fullPivLu().solve(rhs);
}

static void cancellation(bool acceptable)
{
    Vec x = Vec::Zero(1);
    const double offset = 1.0029965198152e-5;
    auto assemble = [offset](const Vec& value, Vec& residual,
                             std::vector<Triplet>* jacobian)
    {
        // At 1e12 the ULP is about 1.22e-4. Every Newton/backtracking
        // correction here is smaller than half an ULP, so this residual
        // cannot improve in double precision, despite a nonzero correction.
        volatile double shifted = 1e12 + value[0];
        residual = Vec::Constant(1, (shifted - 1e12) - offset);
        if (jacobian) jacobian->emplace_back(0, 0, 1.);
    };
    if (!acceptable)
    {
        mustReject([&] { p011Charge::solve(x, 1., assemble, linearSolve); });
        return;
    }
    auto result = p011Charge::solve(x, 1e5, assemble, linearSolve);
    require(result.stagnated, "Cancellation plateau was not identified");
    require(x[0] == 0., "An unaccepted line-search trial replaced the iterate");
    require(result.residual > 1e-10 && result.residual < 1e-7,
            "Manufactured residual does not exercise the stopping gap");
    require(std::abs(result.residual - offset/1e5) < 1e-24,
            "Returned residual is not the residual at the retained iterate");
}

static void exactAcceptanceBoundary()
{
    Vec x = Vec::Zero(1);
    auto assemble = [](const Vec&, Vec& residual, std::vector<Triplet>* jacobian)
    {
        residual = Vec::Constant(1, 1e-7);
        if (jacobian) jacobian->emplace_back(0, 0, 1.);
    };
    mustReject([&] { p011Charge::solve(x, 1., assemble, linearSolve); });
}

static void nonlinearConvergence()
{
    Vec x(2); x << .5, .5;
    int assembled = 0;
    auto assemble = [&assembled](const Vec& value, Vec& residual,
                                 std::vector<Triplet>* jacobian)
    {
        ++assembled;
        residual.resize(2);
        residual << value[0]*value[0]-2., std::exp(value[1])-3.;
        if (jacobian)
        {
            jacobian->emplace_back(0, 0, 2.*value[0]);
            jacobian->emplace_back(1, 1, std::exp(value[1]));
        }
    };
    auto result = p011Charge::solve(x, 1., assemble, linearSolve);
    require(!result.stagnated, "A well-conditioned solve was labelled stagnant");
    require(result.residual < 1e-10, "Preferred accuracy was not reached");
    require(std::abs(x[0]-std::sqrt(2.)) < 1e-10, "Wrong quadratic root");
    require(std::abs(x[1]-std::log(3.)) < 1e-10, "Wrong exponential root");
    require(assembled > 2, "Test did not exercise iterative convergence");
    Vec residual;
    assemble(x, residual, nullptr);
    require(std::abs(result.residual-residual.lpNorm<1>()) < 1e-24,
            "Final residual was not recomputed at the returned state");
}

static void nonfiniteState(const std::string& where)
{
    const double invalid[] = {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()};
    for (double bad : invalid)
    {
        Vec x = Vec::Zero(1);
        if (where == "iterate") x[0] = bad;
        auto assemble = [where, bad](const Vec& value, Vec& residual,
                                     std::vector<Triplet>* jacobian)
        {
            residual = Vec::Constant(1, value[0]-.01);
            if (where == "residual" || (where == "trial" && value[0] != 0.))
                residual[0] = bad;
            if (jacobian) jacobian->emplace_back(0, 0, 1.);
        };
        auto linear = [where, bad](const Matrix& matrix, const Vec& rhs) -> Vec
        {
            if (where == "correction") return Vec::Constant(rhs.size(), bad);
            return linearSolve(matrix, rhs);
        };
        mustReject([&] { p011Charge::solve(x, 1., assemble, linear); });
    }
}

static void invalidScale()
{
    for (double scale : {0., -1., std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()})
    {
        Vec x = Vec::Zero(1);
        auto assemble = [](const Vec&, Vec& residual, std::vector<Triplet>* jacobian)
        {
            residual = Vec::Zero(1);
            if (jacobian) jacobian->emplace_back(0, 0, 1.);
        };
        mustReject([&] { p011Charge::solve(x, scale, assemble, linearSolve); });
    }
}

static void butlerVolmer()
{
    const double pref = 1.05e10, beta = 16.9;
    const double tiny = p011Charge::rate(pref, 1e-20);
    require(tiny > 0. && std::abs(tiny/2.1e-10-1.) < 1e-14,
            "Small Butler-Volmer current was lost to cancellation");
    require(p011Charge::rate(pref, 0.) == 0., "Equilibrium rate is nonzero");
    for (double arg : {-40., -1., -1e-6, -1e-20, 0., 1e-20, 1e-6, 1., 40.})
    {
        const long double a = arg, p = pref;
        const double reference = static_cast<double>(p*(std::expm1(a)-std::expm1(-a)));
        const double rate = p011Charge::rate(pref, arg);
        const double slope = p011Charge::slope(pref, beta, arg);
        const double referenceSlope = static_cast<double>(p*beta*(std::exp(a)+std::exp(-a)));
        require(std::abs(rate-reference) <= 2e-14*std::max(std::abs(reference), 1e-300),
                "Butler-Volmer current disagrees with independent expm1 evaluation");
        require(rate == -p011Charge::rate(pref, -arg), "Rate lost odd symmetry");
        require(slope > 0. && std::abs(slope/referenceSlope-1.) < 2e-14,
                "Butler-Volmer derivative is inconsistent");
    }
}

int main(int argc, char** argv)
{
    try
    {
        require(argc == 2, "Expected a scenario name");
        const std::string scenario = argv[1];
        if (scenario == "cancellation") cancellation(true);
        else if (scenario == "unacceptable") cancellation(false);
        else if (scenario == "boundary") exactAcceptanceBoundary();
        else if (scenario == "nonlinear") nonlinearConvergence();
        else if (scenario == "scale") invalidScale();
        else if (scenario == "bv") butlerVolmer();
        else if (scenario == "iterate" || scenario == "residual" ||
                 scenario == "correction" || scenario == "trial") nonfiniteState(scenario);
        else throw std::runtime_error("Unknown scenario: " + scenario);
        std::cout << scenario << ": passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
