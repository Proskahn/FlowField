/* Jin et al., Fuel 404 (2026) 136144, Eqs. 1--16.
 * Nondimensional Navier--Stokes--Brinkman + gamma-weighted thermal advection.
 * Unit reference L,U,rho; reported integrals are per unit depth. No 3D PEMEC.
 */
#include "fvCFD.H"
#include "constrainHbyA.H"
#include <fstream>
#include <iomanip>
#include <cmath>
using namespace Foam;

int main(int argc,char* argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    if(Pstream::parRun()) FatalErrorInFunction << "P007 is serial; parallelize independent designs" << exit(FatalError);
    IOdictionary props(IOobject("topologyProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
    volVectorField U(IOobject("U",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField p(IOobject("p",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField T(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField gamma(IOobject("gamma",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField alpha(IOobject("brinkman",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    surfaceScalarField phi(IOobject("phi",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),fvc::flux(U));
    const scalar Re=props.get<scalar>("Re"),Pr=props.get<scalar>("Pr"),H=props.get<scalar>("heatCoefficient");
    const scalar depth=props.get<scalar>("depth"),power=props.get<scalar>("pumpingPower");
    const scalar tol=props.get<scalar>("outerTolerance"),btol=props.get<scalar>("balanceTolerance");
    const dimensionedScalar nu("nu",dimViscosity,1/Re),kappa("kappa",dimViscosity,1/(Re*Pr));
    volScalarField source(dimensionedScalar("h",dimless/dimTime,H/(Re*Pr))*(1-gamma));
    const label inlet=mesh.boundaryMesh().findPatchID("inlet"),outlet=mesh.boundaryMesh().findPatchID("outlet");
    fileName folder=runTime.path()/"postProcessing"/"p007";mkDir(folder);
    std::ofstream history((folder/"history.csv").c_str());
    history<<"iteration,heat_transfer,dissipation,inlet_flow,outlet_flow,inlet_pressure,pumping_power,volume_error,local_continuity,power_error,momentum_residual,energy_balance,field_change,min_T,max_T,volume_fraction\n"<<std::setprecision(16);
    label stable=0;bool converged=false;scalar lastPower=0;
    const scalarField& V=mesh.V();
    while(runTime.loop())
    {
        const label iteration=runTime.timeIndex();
        const vectorField oldU(U.primitiveField());const scalarField oldT(T.primitiveField()),oldP(p.primitiveField());
        p.storePrevIter();
        if(lastPower>VSMALL)
        {
            const scalar factor=std::exp(props.get<scalar>("powerRelaxation")*std::log(power/lastPower));
            const scalar pin=p.boundaryField()[inlet][0]*min(1.1,max(.9,factor));
            p.boundaryFieldRef()[inlet] == pin;
        }
        fvVectorMatrix UEqn(fvm::div(phi,U)-fvm::laplacian(nu,U)+fvm::Sp(alpha,U));
        UEqn.relax();solve(UEqn == -fvc::grad(p));
        volScalarField rAU(1/UEqn.A());
        volVectorField HbyA(constrainHbyA(rAU*UEqn.H(),U,p));
        surfaceScalarField phiHbyA(fvc::flux(HbyA));
        fvScalarMatrix pEqn(fvm::laplacian(rAU,p)==fvc::div(phiHbyA));
        pEqn.solve();phi=phiHbyA-pEqn.flux();
        p.relax();U=HbyA-rAU*fvc::grad(p);U.correctBoundaryConditions();
        // gamma is OUTSIDE the advective derivative (paper Eq. 7), not div(gamma*u*T).
        fvScalarMatrix TEqn(gamma*fvm::div(phi,T)-fvm::laplacian(kappa,T)+fvm::Sp(source,T)==source);
        TEqn.solve();T.correctBoundaryConditions();
        const scalar qin=-gSum(phi.boundaryField()[inlet])/depth,qout=gSum(phi.boundaryField()[outlet])/depth;
        const scalar pin=p.boundaryField()[inlet][0];lastPower=pin*qin;
        const scalar volumeErr=mag(qin-qout)/max(mag(qin),SMALL);
        const scalar localErr=gSum(mag(fvc::div(phi)().primitiveField())*V)/(depth*max(mag(qin),SMALL));
        const scalar powerErr=mag(lastPower/power-1);
        const scalar heat=gSum(H*(1-gamma.primitiveField())*(1-T.primitiveField())*V)/depth;
        volTensorField gradU(fvc::grad(U));
        volScalarField diss("dissipationDensity",nu*(gradU && (gradU+gradU.T()))+alpha*magSqr(U));
        const scalar dissip=gSum(diss.primitiveField()*V)/depth;
        fvVectorMatrix checkU(fvm::div(phi,U)-fvm::laplacian(nu,U)+fvm::Sp(alpha,U)==-fvc::grad(p));
        const scalar momScale=gSum((mag(fvc::grad(p)().primitiveField())+alpha.primitiveField()*mag(U.primitiveField()))*V);
        const scalar momErr=gSum(mag(checkU.residual()()))/max(momScale,SMALL);
        const scalar energyErr=gSum(mag(TEqn.residual()()))/max(mag(heat)*depth/(Re*Pr),SMALL);
        const scalar change=max(max(gMax(mag(U.primitiveField()-oldU))/max(gMax(mag(U.primitiveField())),SMALL),
            gMax(mag(p.primitiveField()-oldP))/max(gMax(mag(p.primitiveField())),SMALL)),gMax(mag(T.primitiveField()-oldT)));
        const scalar low=gMin(T.primitiveField()),high=gMax(T.primitiveField());
        const scalar fraction=gSum(gamma.primitiveField()*V)/gSum(V);
        if(!std::isfinite(heat+dissip+change+momErr+energyErr+localErr+powerErr) || low < -1e-7 || high > 1+1e-7)
            FatalErrorInFunction << "Nonfinite or unbounded P007 state" << exit(FatalError);
        history<<iteration<<','<<heat<<','<<dissip<<','<<qin<<','<<qout<<','<<pin<<','<<lastPower<<','
            <<volumeErr<<','<<localErr<<','<<powerErr<<','<<momErr<<','<<energyErr<<','<<change<<','<<low<<','<<high<<','<<fraction<<'\n';
        history.flush();
        if(iteration==1 || iteration%props.get<label>("diagnosticInterval")==0)
            Info<<"Iteration = "<<iteration<<nl<<"P007: heat="<<heat<<"; dissipation="<<dissip<<"; change="<<change
                <<"; continuity="<<localErr<<"; power="<<powerErr<<"; momentum="<<momErr<<"; energy="<<energyErr<<endl;
        const bool ok=iteration>=30 && change<tol && qin>0 && qout>0 && heat>0 && dissip>0 &&
            max(max(localErr,volumeErr),max(powerErr,max(momErr,energyErr)))<btol;
        stable=ok ? stable+1 : 0;
        runTime.write();
        if(stable>=5){converged=true;runTime.writeNow();break;}
    }
    runTime.writeNow();
    std::ofstream status((folder/"status.json").c_str());
    status<<"{\"converged\":"<<(converged?"true":"false")<<",\"iteration\":"<<runTime.timeIndex()
        <<",\"cells\":"<<mesh.nCells()<<",\"processes\":1}\n";
    if(runTime.timeIndex()!=1 && runTime.timeIndex()%props.get<label>("diagnosticInterval")!=0)
        Info<<"Iteration = "<<runTime.timeIndex()<<nl;
    Info<<(converged ? "CONVERGED: P007 flow, heat and pumping power qualified." : "ITERATION LIMIT: P007 is not qualified.")<<nl<<"End"<<endl;
    return converged ? 0 : 2;
}
