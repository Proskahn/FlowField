/* P003: Behrou et al. 2019, DOI 10.1016/j.ijheatmasstransfer.2019.01.050.
 * Serial 2D finite-volume reconstruction. See P003-model.md for departures.
 * U is superficial velocity; p is dimensional gauge Pa. No liquid water.
 */
#include "fvCFD.H"
#include "constrainHbyA.H"
#include <fstream>
#include <iomanip>
#include <cmath>
using namespace Foam;

static volScalarField auxiliary(const word& name,const dimensionSet& dims,
    const scalar initial,const volScalarField& prototype)
{
    const fvMesh& mesh=prototype.mesh();
    return volScalarField(IOobject(name,mesh.time().timeName(),mesh,
        IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,
        dimensionedScalar(name,dims,initial),prototype.boundaryField().types());
}

static scalar reaction(const scalar oxygen,const dictionary& props)
{
    const scalar delta=props.get<scalar>("openCircuit")-props.get<scalar>("voltage");
    const scalar resistance=props.get<scalar>("ohmic");
    const scalar exponent=96485.3/(8.3144*props.get<scalar>("temperature"));
    const scalar alpha=props.get<scalar>("alpha");
    const scalar prefactor=props.get<scalar>("exchangeCurrent")*props.get<scalar>("rho")
        *max(oxygen,0.)/.032/props.get<scalar>("referenceOxygen");
    scalar lo=0.,hi=delta/resistance;
    for (label i=0;i<70;++i)
    {
        const scalar mid=(lo+hi)/2,eta=delta-resistance*mid;
        const scalar rhs=prefactor*(std::exp(alpha*exponent*eta)-std::exp(-(1-alpha)*exponent*eta));
        if(mid>rhs) hi=mid; else lo=mid;
    }
    return (lo+hi)/2;
}

static void chemistry(volScalarField& O,volScalarField& W,volScalarField& N,
    const volScalarField& factor,volScalarField& M,volScalarField& DO,
    volScalarField& DW,volScalarField& current,volScalarField& sink,
    volScalarField& waterSource,const dictionary& props)
{
    const scalar correction=std::pow(props.get<scalar>("temperature")/298.15,1.75);
    const scalar dow=2.64e-5*correction,don=2.07e-5*correction,dwn=2.64e-5*correction;
    const scalar rho=props.get<scalar>("rho"),H=props.get<scalar>("depth");
    auto local=[&](scalar o,scalar w,scalar f,scalar& n,scalar& mm,scalar& dO,scalar& dW,
                   scalar& j,scalar& ko,scalar& sw)
    {
        n=1-o-w;
        // Coefficient safeguards only; the converged mass fractions are checked unbounded.
        scalar oo=max(o,1e-14),ww=max(w,1e-14),nn=max(n,1e-14);
        mm=1/(oo/.032+ww/.018+nn/.028);
        dO=f*(ww+nn)/(ww/dow+nn/don);
        dW=f*(oo+nn)/(oo/dow+nn/dwn);
        j=reaction(o,props);
        ko=.032*j/(4*96485.3*rho*H*oo);
        sw=2*.018*j/(4*96485.3*rho*H);
    };
    forAll(O,c) local(O[c],W[c],factor[c],N[c],M[c],DO[c],DW[c],current[c],sink[c],waterSource[c]);
    forAll(O.boundaryField(),p) forAll(O.boundaryField()[p],f)
        local(O.boundaryField()[p][f],W.boundaryField()[p][f],factor.boundaryField()[p][f],
            N.boundaryFieldRef()[p][f],M.boundaryFieldRef()[p][f],DO.boundaryFieldRef()[p][f],
            DW.boundaryFieldRef()[p][f],current.boundaryFieldRef()[p][f],
            sink.boundaryFieldRef()[p][f],waterSource.boundaryFieldRef()[p][f]);
}

int main(int argc,char* argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    if(Pstream::parRun()) FatalErrorInFunction << "P003 adapter supports serial only" << exit(FatalError);
    IOdictionary props(IOobject("topologyProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
    volVectorField U(IOobject("U",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField p(IOobject("p",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField O(IOobject("oxygen",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField W(IOobject("water",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField eps(IOobject("porosity",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField factor(IOobject("diffusionFactor",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField alphaB(IOobject("brinkman",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    surfaceScalarField phi(IOobject("phi",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),fvc::flux(U));
    volScalarField N=auxiliary("nitrogen",dimless,0.749,O);
    volScalarField M=auxiliary("molarMass",dimMass/dimMoles,.03,O);
    volScalarField DO=auxiliary("oxygenDiffusivity",dimArea/dimTime,1e-5,O);
    volScalarField DW=auxiliary("waterDiffusivity",dimArea/dimTime,1e-5,O);
    volScalarField j=auxiliary("currentDensity",dimCurrent/dimArea,0.,O);
    volScalarField sink=auxiliary("oxygenSink",dimless/dimTime,0.,O);
    volScalarField source=auxiliary("waterSource",dimless/dimTime,0.,O);
    const scalar H=props.get<scalar>("depth"),rhoValue=props.get<scalar>("rho");
    const dimensionedScalar rho("rho",dimDensity,rhoValue);
    const dimensionedScalar mu("mu",dimDynamicViscosity,props.get<scalar>("mu"));
    const dimensionedScalar poiseuille("poiseuille",dimDensity/dimTime,12*mu.value()/(H*H));
    surfaceScalarField oxygenFlux(IOobject("oxygenFlux",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),rho*phi);
    surfaceScalarField waterFlux(IOobject("waterFlux",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),rho*phi);
    fileName folder=runTime.path()/"postProcessing"/"p003";mkDir(folder);
    std::ofstream history((folder/"history_0.csv").c_str());
    history<<"iteration,mean_current_A_m2,std_current_A_m2,power_domain_W,inlet_flow_m3_s,outlet_flow_m3_s,volume_error,oxygen_error,water_error,field_change,min_oxygen,max_water,min_nitrogen,closure_net_mass_kg_s\n";
    history<<std::setprecision(14);
    const label inlet=mesh.boundaryMesh().findPatchID("inlet"),outlet=mesh.boundaryMesh().findPatchID("outlet");
    const scalar area=gSum(mesh.V())/H;
    label stable=0;bool converged=false;
    while(runTime.loop())
    {
        p.storePrevIter();
        vectorField oldU=U.primitiveField();scalarField oldP=p.primitiveField(),oldO=O.primitiveField(),oldW=W.primitiveField();
        chemistry(O,W,N,factor,M,DO,DW,j,sink,source,props);
        // Eq. 4 multiplied by porosity: retain its non-conservative 1/eps convection factor.
        fvVectorMatrix UEqn((rho/eps)*fvm::div(phi,U)-fvm::laplacian(mu,U)
            +fvm::Sp(eps*(alphaB+poiseuille),U)
            -fvc::div(mu*T(fvc::grad(U)),"div((mu*T(grad(U))))"));
        UEqn.relax();solve(UEqn == -eps*fvc::grad(p));
        volScalarField rAU(1/UEqn.A());
        volScalarField rAUeps(rAU*eps);
        volVectorField HbyA(constrainHbyA(rAU*UEqn.H(),U,p));
        surfaceScalarField phiH("phiH",fvc::flux(HbyA));
        fvScalarMatrix pEqn(fvm::laplacian(rAUeps,p)==fvc::div(phiH));
        pEqn.solve();phi=phiH-pEqn.flux();p.relax();
        U=HbyA-rAUeps*fvc::grad(p);U.correctBoundaryConditions();
        // Eq. 8 molecular-weight gradient term: positive on the PDE right-hand side.
        surfaceScalarField crossO("crossO",fvc::interpolate(DO*O/M)*fvc::snGrad(M)*mesh.magSf());
        surfaceScalarField crossW("crossW",fvc::interpolate(DW*W/M)*fvc::snGrad(M)*mesh.magSf());
        fvScalarMatrix OEqn(fvm::div(phi,O)-fvm::laplacian(DO,O)+fvm::Sp(sink,O)==fvc::div(crossO));
        OEqn.relax(.7);OEqn.solve();
        fvScalarMatrix WEqn(fvm::div(phi,W)-fvm::laplacian(DW,W)==fvc::div(crossW)+source);
        WEqn.relax(.7);WEqn.solve();
        oxygenFlux=rho*(OEqn.flux()-crossO);waterFlux=rho*(WEqn.flux()-crossW);
        chemistry(O,W,N,factor,M,DO,DW,j,sink,source,props);
        const scalar minO=gMin(O.primitiveField()),minW=gMin(W.primitiveField()),minN=gMin(N.primitiveField());
        if(minO < -1e-8 || minW < -1e-8 || minN < -1e-8 || gMax(O.primitiveField())>1+1e-8 || gMax(W.primitiveField())>1+1e-8)
            FatalErrorInFunction << "Unphysical species mass fraction; no clipping accepted" << exit(FatalError);
        scalar current=gSum(j.primitiveField()*mesh.V())/H;
        scalar mean=current/area;
        scalar variance=gSum(sqr(j.primitiveField()-mean)*mesh.V())/(area*H);
        scalar qin=-sum(phi.boundaryField()[inlet]),qout=sum(phi.boundaryField()[outlet]);
        scalar oFlux=0,wFlux=0;
        forAll(mesh.boundary(),patchi)
        {
            if(mesh.boundary()[patchi].type()=="empty") continue;
            oFlux+=sum(oxygenFlux.boundaryField()[patchi]);wFlux+=sum(waterFlux.boundaryField()[patchi]);
        }
        const scalar expectedO=.032*current/(4*96485.3),expectedW=2*.018*current/(4*96485.3);
        const scalar vError=mag(qout-qin)/max(qin,SMALL);
        const scalar oError=mag(oFlux+expectedO)/max(expectedO,SMALL);
        const scalar wError=mag(wFlux-expectedW)/max(expectedW,SMALL);
        const scalar change=max(gMax(mag(U.primitiveField()-oldU))/max(gMax(mag(U.primitiveField())),1e-6),
            max(gMax(mag(p.primitiveField()-oldP))/max(gMax(mag(p.primitiveField())),1.),
                max(gMax(mag(O.primitiveField()-oldO)),gMax(mag(W.primitiveField()-oldW)))));
        history<<runTime.value()<<','<<mean<<','<<std::sqrt(max(variance,0.))<<','<<current*props.get<scalar>("voltage")
            <<','<<qin<<','<<qout<<','<<vError<<','<<oError<<','<<wError<<','<<change
            <<','<<minO<<','<<gMax(W.primitiveField())<<','<<minN<<','<<(expectedW-expectedO)<<'\n';history.flush();
        Info<<"P003: iteration="<<runTime.value()<<" meanCurrent="<<mean<<" power="<<current*props.get<scalar>("voltage")
            <<" change="<<change<<" balances="<<vError<<','<<oError<<','<<wError<<endl;
        if(qin>0 && minO>=0 && minW>=0 && minN>=0 && change<props.get<scalar>("outerTolerance")
            && max(vError,max(oError,wError))<props.get<scalar>("balanceTolerance") && runTime.value()>=30) ++stable;
        else stable=0;
        runTime.write();
        if(stable>=5){converged=true;runTime.writeNow();Info<<"CONVERGED: P003 fields, volume and transported-species balances passed."<<endl;break;}
    }
    runTime.writeNow();
    std::ofstream status((folder/"status.json").c_str());
    status<<"{\"converged\":"<<(converged?"true":"false")<<",\"iteration\":"<<runTime.value()<<",\"cells\":"<<mesh.nCells()<<",\"processes\":1}\n";
    Info<<"End"<<endl;return converged?0:2;
}
