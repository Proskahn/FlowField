// Manufactured solutions exercise the SAME stress helper used by P006.
// These are operator checks, not qualification of a full electrolyzer case.
#include "Tools.H"

static vector analyticVelocity(const vector& x, label mode)
{
    if (mode==0) return vector(sqr(x.y()),0,0);
    if (mode==1) return vector(x.x()*x.y(),0,0);
    return vector(sqr(x.x()),0,0);
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    boolList fluid(mesh.nCells(),true), neighbourFluid;
    syncTools::swapBoundaryCellList(mesh,fluid,neighbourFluid);
    wordList types(mesh.boundary().size(),"fixedValue");
    forAll(mesh.boundary(),p)
        if (mesh.boundary()[p].coupled()) types[p]=mesh.boundary()[p].type();
    surfaceScalarField mu(IOobject("mu",runTime.timeName(),mesh),mesh,
        dimensionedScalar("mu",dimDynamicViscosity,1));
    scalar maximumError=0, oldShearError=0;
    label totalChecked=0;
    for(label mode=0;mode<3;++mode)
    {
        volVectorField q(IOobject("q",runTime.timeName(),mesh),mesh,
            dimensionedVector("q",dimVelocity,vector::zero),types);
        forAll(q,c)q[c]=analyticVelocity(mesh.C()[c],mode);
        forAll(mesh.boundary(),p)
            if(!mesh.boundary()[p].coupled())
                forAll(mesh.boundary()[p],f)
                    q.boundaryFieldRef()[p][f]=analyticVelocity(mesh.boundary()[p].Cf()[f],mode);
        q.correctBoundaryConditions();
        volVectorField cross("cross",fvc::surfaceIntegrate(
            phaseTransposeStressFlux(mu,q,fluid,neighbourFluid)));
        volVectorField full("full",fvc::laplacian(mu,q)+cross);
        volVectorField reversed("reversed",fvc::surfaceIntegrate(mu*(
            fvc::interpolate(dev2(Foam::T(phaseGradient(q,fluid,neighbourFluid))))&mesh.Sf())));
        // Analytic div[grad(q)^T - 2/3 div(q) I] = 1/3 grad(div(q)).
        vector expectedCross=mode==0?vector::zero:
            mode==1?vector(0,1.0/3,0):vector(2.0/3,0,0);
        vector expectedFull=expectedCross+(mode==1?vector::zero:vector(2,0,0));
        scalar error=0, mutationError=0;label checked=0;
        forAll(q,c)
        {
            const vector& x=mesh.C()[c];
            // Two stencils away from physical boundaries; all MPI interfaces
            // are included and must give the same manufactured solution.
            if(cmptMin(x)<.25 || cmptMax(x)>.75)continue;
            error=max(error,max(mag(cross[c]-expectedCross),mag(full[c]-expectedFull)));
            mutationError=max(mutationError,mag(reversed[c]-expectedCross));++checked;
        }
        reduce(error,maxOp<scalar>());reduce(mutationError,maxOp<scalar>());
        reduce(checked,sumOp<label>());totalChecked+=checked;
        maximumError=max(maximumError,error);
        if(mode==0)oldShearError=mutationError;
        Info<<"STRESS_CHECK mode="<<mode<<" cells="<<checked
            <<" max_error="<<error<<" reversed_error="<<mutationError<<nl;
    }
    // A known diffusion/reaction solution with 1e18 coefficient contrast tests
    // false convergence under the old global residual normalization. Verify
    // actual velocity error, independently of either solver's reported error.
    volVectorField qCheck(IOobject("qCheck",runTime.timeName(),mesh),mesh,
        dimensionedVector("q",dimVelocity,vector(1,0,0)),types);
    auto diagonal=auxiliary("diagonal",mesh,dimDensity/dimTime,1);
    forAll(qCheck,c)
    {
        if(mesh.C()[c].x()<.5)qCheck[c]=vector::zero;
        else diagonal[c]=1e18;
    }
    diagonal.correctBoundaryConditions();
    volVectorField prescribed(IOobject("prescribed",runTime.timeName(),mesh),mesh,
        dimensionedVector("q",dimVelocity,vector(1,0,0)));
    volVectorField rhs("rhs",diagonal*prescribed);
    fvVectorMatrix equation(fvm::Sp(diagonal,qCheck)-fvm::laplacian(
        dimensionedScalar("mu",dimDynamicViscosity,1e-6),qCheck));
    solve(equation==rhs);
    scalar unscaledError=gMax(mag(qCheck.primitiveField()-vector(1,0,0)));
    solveRowScaled(equation,rhs);
    scalar scaledError=gMax(mag(qCheck.primitiveField()-vector(1,0,0)));
    Info<<"ROW_SCALE_CHECK unscaled_error="<<unscaledError
        <<" scaled_error="<<scaledError<<nl;
    bool passed=totalChecked==192 && maximumError<1e-10 && oldShearError>1
        && unscaledError>.9 && scaledError<1e-10;
    Info<<(passed?"STRESS_CHECK PASSED":"STRESS_CHECK FAILED")<<nl;
    return passed?0:1;
}
