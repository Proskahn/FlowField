#pragma once
// The inner Newton target is an accuracy preference. The existing charge
// acceptance limit remains mandatory, including after numerical stagnation.
// Kept independent of OpenFOAM for manufactured floating-point regressions.
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace p011Charge
{
constexpr double target = 1e-10;
constexpr double acceptance = 1e-7;

inline double rate(double pref, double arg)
{
    return 2*pref*std::sinh(arg);
}

inline double slope(double pref, double beta, double arg)
{
    return 2*pref*beta*std::cosh(arg);
}

struct NewtonResult
{
    double residual;
    bool stagnated;
    int iterations;
};

template<class Assemble, class LinearSolve>
NewtonResult solve(Eigen::VectorXd& x, double currentScale,
                   Assemble assemble, LinearSolve linearSolve)
{
    if (x.size() == 0 || !x.allFinite() || !std::isfinite(currentScale) || currentScale <= 0)
        throw std::runtime_error("Nonfinite charge iterate or invalid current scale");
    Eigen::VectorXd residual(x.size());
    auto measure = [&]()
    {
        if (!x.allFinite() || !residual.allFinite())
            throw std::runtime_error("Nonfinite charge iterate or residual");
        const double norm = residual.lpNorm<1>()/currentScale;
        if (!std::isfinite(norm))
            throw std::runtime_error("Nonfinite normalized charge residual");
        return norm;
    };

    bool stagnated = false;
    int iteration = 0;
    for (; iteration < 80; ++iteration)
    {
        std::vector<Eigen::Triplet<double>> triplets;
        assemble(x, residual, &triplets);
        const double norm = measure();
        if (norm < target) break;

        Eigen::SparseMatrix<double> jacobian(x.size(), x.size());
        jacobian.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::VectorXd dx = linearSolve(jacobian, -residual);
        if (dx.size() != x.size() || !dx.allFinite())
            throw std::runtime_error("Nonfinite charge Newton step");
        double step = std::min(1., .1/std::max(dx.cwiseAbs().maxCoeff(), 1e-15));
        Eigen::VectorXd trial(x.size()), candidateResidual(x.size());
        bool accepted = false;
        const double merit = residual.stableNorm();
        for (int search = 0; search < 24; ++search)
        {
            trial = x + step*dx;
            if (!trial.allFinite())
                throw std::runtime_error("Nonfinite charge trial iterate");
            assemble(trial, candidateResidual, nullptr);
            if (!candidateResidual.allFinite())
                throw std::runtime_error("Nonfinite charge trial residual");
            if (candidateResidual.stableNorm() < merit)
            {
                accepted = true;
                break;
            }
            step *= .5;
        }
        if (!accepted)
        {
            // Never install an unaccepted trial. Reassemble the current
            // iterate and allow stagnation only under the unchanged gate.
            assemble(x, residual, nullptr);
            if (measure() >= acceptance)
                throw std::runtime_error("Charge Newton line search failed above acceptance limit");
            stagnated = true;
            break;
        }
        x.swap(trial);
    }
    assemble(x, residual, nullptr);
    const double norm = measure();
    if (norm >= acceptance)
        throw std::runtime_error("Charge equation failed acceptance limit");
    return {norm, stagnated, iteration};
}
}
