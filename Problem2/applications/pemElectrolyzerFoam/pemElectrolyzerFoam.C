/*
 * Steady homogeneous-mixture PEM electrolyzer, OpenCFD OpenFOAM v2312.
 * Based on Zhuang et al., IJHE 49 (2024) 337-352, Sections 2, 3.1-3.3.1.
 * See docs/model.md for closure assumptions and corrections to printed units.
 * This implementation is serial; it rejects decomposed/parallel execution.
 */
#include "fvCFD.H"
#include "ChargeNetwork.H"
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace Foam;

static volScalarField auxiliary(const word& name, const fvMesh& mesh,
                               const dimensionSet& dims, scalar value=0,
                               IOobject::writeOption write=IOobject::NO_WRITE)
{
    return volScalarField(IOobject(name,mesh.time().timeName(),mesh,
        IOobject::NO_READ,write),mesh,dimensionedScalar(name,dims,value),"zeroGradient");
}

// Distance-weighted harmonic coefficient. Exactly zero across an inactive
// material interface; there is no small artificial conductivity or leakage.
static void harmonic(surfaceScalarField& face, const volScalarField& cell,
                     const boolList* active=nullptr)
{
    const fvMesh& mesh=cell.mesh();
    forAll(mesh.neighbour(),f)
    {
        label o=mesh.owner()[f], n=mesh.neighbour()[f];
        if ((active && (!(*active)[o] || !(*active)[n])) || cell[o]<=0 || cell[n]<=0)
        { face[f]=0; continue; }
        scalar dO=mag(mesh.Cf()[f]-mesh.C()[o]), dN=mag(mesh.Cf()[f]-mesh.C()[n]);
        face[f]=(dO+dN)/(dO/cell[o]+dN/cell[n]);
    }
    forAll(mesh.boundary(),p)
        forAll(mesh.boundary()[p],f)
        {
            label c=mesh.boundary()[p].faceCells()[f];
            face.boundaryFieldRef()[p][f]=(active && !(*active)[c]) ? 0 : cell[c];
        }
}

static void pressureGradient(volVectorField& grad, const volScalarField& p,
                             const boolList& fluid)
{
    const fvMesh& mesh=p.mesh();
    surfaceScalarField pf(fvc::interpolate(p));
    forAll(mesh.neighbour(),f)
    {
        label o=mesh.owner()[f], n=mesh.neighbour()[f];
        if (fluid[o] && !fluid[n]) pf[f]=p[o];
        if (!fluid[o] && fluid[n]) pf[f]=p[n];
    }
    grad=fvc::surfaceIntegrate(pf*mesh.Sf());
    forAll(fluid,c) if (!fluid[c]) grad[c]=vector::zero;
}

int main(int argc, char *argv[])
{
    argList::noParallel();
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    IOdictionary props(IOobject("pemProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
    const scalar F=96485.33212, R=8.314462618, MO=0.0319988, MH=0.00201588, MW=0.01801528;
    const scalar pAbs=props.get<scalar>("operatingPressure"), Tref=props.get<scalar>("referenceTemperature");
    const scalar rhoL=props.get<scalar>("liquidDensity"), muL=props.get<scalar>("liquidViscosity");
    const scalar cpL=props.get<scalar>("liquidCp"), kL=props.get<scalar>("liquidConductivity");
    const scalar j=props.get<scalar>("currentDensity"), targetI=j*props.get<scalar>("reactionArea");
    if (j<=0 || props.get<scalar>("flowPseudoDt")<=0)
        FatalErrorInFunction << "Positive currentDensity and flowPseudoDt are required." << exit(FatalError);
    volScalarField material(IOobject("material",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField p(IOobject("p",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField T(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField C(IOobject("Cgas",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField phiS(IOobject("phiS",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField phiE(IOobject("phiE",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volVectorField U(IOobject("U",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    auto eps=auxiliary("porosity",mesh,dimless);
    auto rho=auxiliary("rho",mesh,dimDensity,rhoL,IOobject::AUTO_WRITE);
    auto mu=auxiliary("mu",mesh,dimDynamicViscosity,muL);
    auto drag=auxiliary("drag",mesh,dimDensity/dimTime);
    auto D=auxiliary("Dgas",mesh,dimArea/dimTime);
    auto cp=auxiliary("CpFluid",mesh,dimEnergy/dimMass/dimTemperature,cpL);
    auto k=auxiliary("kEffective",mesh,dimPower/dimLength/dimTemperature);
    auto sigmaS=auxiliary("sigmaS",mesh,dimensionSet(-1,-3,3,0,0,2,0));
    auto sigmaE=auxiliary("sigmaE",mesh,sigmaS.dimensions());
    auto iv=auxiliary("iVol",mesh,dimCurrent/dimVolume,0,IOobject::AUTO_WRITE);
    auto heat=auxiliary("heatSource",mesh,dimPower/dimVolume,0,IOobject::AUTO_WRITE);
    auto coverage=auxiliary("coverage",mesh,dimless,0,IOobject::AUTO_WRITE);
    auto fgas=auxiliary("gasVolumeFraction",mesh,dimless,0,IOobject::AUTO_WRITE);
    auto xgas=auxiliary("gasMoleFraction",mesh,dimless,0,IOobject::AUTO_WRITE);
    auto Sm=auxiliary("massSource",mesh,dimDensity/dimTime,0,IOobject::AUTO_WRITE);
    auto Sgas=auxiliary("gasMolarSource",mesh,C.dimensions()/dimTime,0,IOobject::AUTO_WRITE);
    auto Cwater=auxiliary("Cwater",mesh,C.dimensions(),0,IOobject::AUTO_WRITE);
    boolList fluid(mesh.nCells(),false), anode(mesh.nCells(),false);
    labelList mat(mesh.nCells());
    DynamicList<label> solidList;
    forAll(mat,c)
    {
        mat[c]=label(material[c]+0.5);
        if (mat[c]<0 || mat[c]>8) FatalErrorInFunction << "Invalid material ID" << exit(FatalError);
        fluid[c]=mat[c]==1 || mat[c]==2 || mat[c]==3 || mat[c]==5 || mat[c]==6 || mat[c]==7;
        anode[c]=mat[c]>=5;
        if (!fluid[c]) solidList.append(c);
        eps[c]=(mat[c]==1 || mat[c]==7) ? 1 :
               (mat[c]==2 || mat[c]==6) ? props.get<scalar>("porosityPTL") :
               (mat[c]==3 || mat[c]==5) ? props.get<scalar>("porosityCL") : 0;
        sigmaS[c]=(mat[c]==1 || mat[c]==4 || mat[c]==7) ? 0 : props.get<scalar>("electronConductivity");
    }
    const labelList solids(solidList);
    scalarField solidZero(solids.size(),0);
    vectorField solidU(solids.size(),vector::zero);
    eps.correctBoundaryConditions(); sigmaS.correctBoundaryConditions();
    surfaceScalarField epsF(IOobject("epsF",runTime.timeName(),mesh),fvc::interpolate(eps));
    harmonic(epsF,eps,&fluid);
    surfaceScalarField phi(IOobject("phi",runTime.timeName(),mesh,IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE),epsF*fvc::flux(U));
    surfaceScalarField massPhi(IOobject("massPhi",runTime.timeName(),mesh,IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE),fvc::interpolate(rho)*phi);
    volVectorField gradP(IOobject("gradP",runTime.timeName(),mesh),fvc::grad(p));
    ChargeNetwork charge(mesh,mat);
    fileName outDir=runTime.path()/"postProcessing"/"pem";
    mkDir(outDir);
    std::ofstream history((outDir/("history_"+runTime.timeName()+".csv")).c_str());
    history << "iteration,voltage_V,current_anode_A,current_cathode_A,charge_residual,dp_anode_Pa,dp_cathode_Pa,Tmax_K,gas_volume_max,O2_source_mol_s,H2_source_mol_s,O2_out_mol_s,H2_out_mol_s,mass_error,gas_error,field_change,heat_W,bounded_cells\n";
    history << std::setprecision(12);
    bool converged=false;
    label stable=0;
    Info << "Steady coupled PEM model: " << mesh.nCells() << " cells, " << targetI
         << " A. Time labels are outer iterations, not physical seconds." << endl;
    while (runTime.loop())
    {
        Info << "\nIteration = " << runTime.timeName() << endl;
        scalarField oldT(T.primitiveField()), oldC(C.primitiveField()), oldP(p.primitiveField());
        vectorField oldU(U.primitiveField());
        forAll(mat,c)
        {
            scalar rg=pAbs*(anode[c] ? MO : MH)/(R*T[c]);
            scalar fg=fluid[c] ? min(max(C[c]*R*T[c]/pAbs,0.0),0.995) : 0;
            fgas[c]=fg;
            rho[c]=rhoL*(1-fg)+rg*fg;
            mu[c]=muL*(1-fg)+props.get<scalar>(anode[c] ? "oxygenViscosity" : "hydrogenViscosity")*fg;
            scalar gasCp=props.get<scalar>(anode[c] ? "oxygenCp" : "hydrogenCp");
            cp[c]=(rhoL*(1-fg)*cpL+rg*fg*gasCp)/rho[c];
            scalar kGas=props.get<scalar>(anode[c] ? "oxygenConductivity" : "hydrogenConductivity");
            scalar kFluid=kL*(1-fg)+kGas*fg;
            scalar kSolid=props.get<scalar>((mat[c]==0 || mat[c]==6 || mat[c]==8) ? "kTitanium" :
                mat[c]==2 ? "kCarbon" : mat[c]==3 ? "kCathodeCL" : mat[c]==5 ? "kAnodeCL" : "kMembrane");
            k[c]=(1-eps[c])*kSolid+eps[c]*kFluid;
            D[c]=fluid[c] ? std::pow(eps[c],1.5)*props.get<scalar>(anode[c] ? "diffusivityO2" : "diffusivityH2")*(T[c]/Tref)*(101325/pAbs) : 0;
            scalar perm=(mat[c]==2 || mat[c]==6) ? props.get<scalar>("permeabilityPTL") : props.get<scalar>("permeabilityCL");
            drag[c]=(fluid[c] && eps[c]<1) ? mu[c]*sqr(eps[c])/perm : 0;
            sigmaE[c]=(mat[c]>=3 && mat[c]<=5) ? 6.94*std::exp(1268*(1/303.0-1/T[c])) : 0;
            Cwater[c]=fluid[c] ? rhoL*(1-fg)/MW : 0;
            xgas[c]=fluid[c] ? C[c]/max(C[c]+Cwater[c],SMALL) : 0;
        }
        rho.correctBoundaryConditions(); mu.correctBoundaryConditions(); cp.correctBoundaryConditions();
        k.correctBoundaryConditions(); D.correctBoundaryConditions(); sigmaE.correctBoundaryConditions();
        // Inlets supply pure liquid; do not interpolate an adjacent gas-filled cell density onto them.
        forAll(mesh.boundary(),patchi) if (mesh.boundary()[patchi].name().ends_with("Inlet"))
        {
            rho.boundaryFieldRef()[patchi]=rhoL;
            cp.boundaryFieldRef()[patchi]=cpL;
        }
        scalar chargeError=0;
        scalar voltage=charge.solve(props,T,C,sigmaS,sigmaE,phiS,phiE,iv,heat,coverage,chargeError);
        scalar Ia=0,Ic=0;
        forAll(mat,c)
        {
            Sm[c]=0; Sgas[c]=0;
            if (mat[c]==5)
            {
                Ia+=iv[c]*mesh.V()[c];
                Sgas[c]=iv[c]/(4*F);
                Sm[c]=iv[c]*(MO/4-MW/2)/F;
            }
            if (mat[c]==3)
            {
                Ic-=iv[c]*mesh.V()[c];
                Sgas[c]=-iv[c]/(2*F);
                Sm[c]=-iv[c]*MH/(2*F);
            }
        }
        // SIMPLE momentum/pressure coupling, intrinsic pore velocity U.
        volScalarField epsMu("epsMu",eps*mu);
        surfaceScalarField muF(IOobject("muF",runTime.timeName(),mesh),fvc::interpolate(epsMu));
        harmonic(muF,epsMu,&fluid);
        volScalarField pseudo("pseudo",eps*rho/dimensionedScalar("dt",dimTime,props.get<scalar>("flowPseudoDt")));
        fvVectorMatrix UEqn(fvm::div(massPhi,U)-fvm::laplacian(muF,U)+fvm::Sp(drag+pseudo,U)==pseudo*U);
        // Internal fluid/solid faces are exact no-slip walls at the face location.
        forAll(mesh.neighbour(),f)
        {
            label o=mesh.owner()[f],n=mesh.neighbour()[f];
            if (fluid[o]==fluid[n]) continue;
            label c=fluid[o] ? o : n;
            UEqn.diag()[c]+=epsMu[c]*mesh.magSf()[f]/mag(mesh.Cf()[f]-mesh.C()[c]);
        }
        UEqn.relax();
        const scalar uConstraint=gMax(mag(UEqn.diag()));
        forAll(solids,s) UEqn.diag()[solids[s]]=uConstraint;
        UEqn.setValues(solids,solidU);
        pressureGradient(gradP,p,fluid);
        solve(UEqn==-eps*gradP);
        volScalarField rAU("rAU",1.0/UEqn.A());
        volVectorField HbyA("HbyA",rAU*UEqn.H());
        HbyA.correctBoundaryConditions();
        forAll(mesh.boundary(),patchi) if (U.boundaryField()[patchi].fixesValue())
            HbyA.boundaryFieldRef()[patchi]==U.boundaryField()[patchi];
        surfaceScalarField rhoF(fvc::interpolate(rho));
        surfaceScalarField phiH("phiH",epsF*fvc::flux(HbyA));
        surfaceScalarField massPhiH("massPhiH",rhoF*phiH);
        surfaceScalarField Dp("Dp",rhoF*epsF*fvc::interpolate(rAU*eps));
        fvScalarMatrix pEqn(fvm::laplacian(Dp,p)==fvc::div(massPhiH)-Sm);
        const scalar pConstraint=-gMax(mag(pEqn.diag()));
        forAll(solids,s) pEqn.diag()[solids[s]]=pConstraint;
        pEqn.setValues(solids,solidZero);
        pEqn.solve();
        massPhi=massPhiH-pEqn.flux();
        phi=massPhi/rhoF;
        p.primitiveFieldRef()=0.3*p.primitiveField()+0.7*oldP;
        p.correctBoundaryConditions();
        pressureGradient(gradP,p,fluid);
        U=HbyA-rAU*eps*gradP;
        U.correctBoundaryConditions();
        forAll(solids,s) U[solids[s]]=vector::zero;
        // Molar gas transport. Water is recovered from total mixture mass.
        surfaceScalarField DF(IOobject("DF",runTime.timeName(),mesh),fvc::interpolate(D));
        harmonic(DF,D,&fluid);
        forAll(mesh.boundary(),patchi) if (mesh.boundary()[patchi].name().ends_with("Inlet"))
            DF.boundaryFieldRef()[patchi]=0; // Pure-water advective feed, no diffusive gas sink.
        fvScalarMatrix CEqn(fvm::div(phi,C)-fvm::laplacian(DF,C)==Sgas);
        const scalar cConstraint=gMax(mag(CEqn.diag()));
        forAll(solids,s) CEqn.diag()[solids[s]]=cConstraint;
        CEqn.setValues(solids,solidZero); CEqn.solve();
        // Relax the solved field, not the matrix: matrix relaxation would
        // propagate concentration changes extremely slowly through thin CLs.
        C.primitiveFieldRef()=0.5*C.primitiveField()+0.5*oldC;
        label bounded=0;
        forAll(mat,c)
        {
            scalar limit=0.995*pAbs/(R*T[c]);
            if (fluid[c] && (C[c]<-1e-9 || C[c]>limit)) ++bounded;
            C[c]=min(max(C[c],0.0),limit);
        }
        C.correctBoundaryConditions();
        // Stationary solids conduct heat; only fluid carries sensible enthalpy.
        surfaceScalarField kF(IOobject("kF",runTime.timeName(),mesh),fvc::interpolate(k));
        harmonic(kF,k);
        surfaceScalarField heatPhi("heatPhi",massPhi*fvc::interpolate(cp));
        fvScalarMatrix TEqn(fvm::div(heatPhi,T)-fvm::Sp(fvc::div(heatPhi),T)-fvm::laplacian(kF,T)==heat);
        TEqn.solve();
        T.primitiveFieldRef()=0.5*T.primitiveField()+0.5*oldT;
        T.correctBoundaryConditions();
        scalar tmin=gMin(T.primitiveField()),tmax=gMax(T.primitiveField());
        if (!std::isfinite(tmin) || !std::isfinite(tmax) || tmin<273.15 || tmax>373.15)
            FatalErrorInFunction << "Temperature left the liquid-water validity range: " << tmin << ' ' << tmax << exit(FatalError);
        scalar inflow[2]={0,0},outflow[2]={0,0},gasOut[2]={0,0},dp[2]={0,0};
        forAll(mesh.boundary(),patchi)
        {
            const auto& patch=mesh.boundary()[patchi];
            bool inlet=patch.name().ends_with("Inlet"),outlet=patch.name().ends_with("Outlet");
            if (!inlet && !outlet) continue;
            label side=patch.name().starts_with("anode") ? 0 : 1;
            if (inlet)
            {
                inflow[side]-=sum(massPhi.boundaryField()[patchi]);
                dp[side]=sum(p.boundaryField()[patchi]*mesh.magSf().boundaryField()[patchi])/sum(mesh.magSf().boundaryField()[patchi]);
            }
            else
            {
                outflow[side]+=sum(massPhi.boundaryField()[patchi]);
                gasOut[side]+=sum(phi.boundaryField()[patchi]*C.boundaryField()[patchi]);
            }
        }
        scalar sourceA=Ia*(MO/4-MW/2)/F,sourceC=Ic*MH/(2*F);
        scalar massError=max(mag(outflow[0]-inflow[0]-sourceA)/max(inflow[0],SMALL),
                             mag(outflow[1]-inflow[1]-sourceC)/max(inflow[1],SMALL));
        scalar gasError=max(mag(gasOut[0]-Ia/(4*F))/max(Ia/(4*F),SMALL),
                            mag(gasOut[1]-Ic/(2*F))/max(Ic/(2*F),SMALL));
        scalar change=max(gMax(mag(T.primitiveField()-oldT))/max(tmax-props.get<scalar>("inletTemperature"),1.0),
            max(gMax(mag(C.primitiveField()-oldC))/max(gMax(C.primitiveField()),1.0),
            max(gMax(mag(U.primitiveField()-oldU))/max(gMax(mag(U.primitiveField())),0.01),
                gMax(mag(p.primitiveField()-oldP))/max(gMax(mag(p.primitiveField())),1.0))));
        scalar heatW=sum(heat.primitiveField()*mesh.V());
        Info << "PEM: V=" << voltage << " V; Ia=" << Ia << "; Ic=" << Ic
             << " A; dP=" << dp[0] << ", " << dp[1] << " Pa; Tmax=" << tmax
             << " K; change=" << change << "; charge=" << chargeError
             << "; gas balance=" << gasError << "; bounded=" << bounded << endl;
        history << runTime.value() << ',' << voltage << ',' << Ia << ',' << Ic << ',' << chargeError
                << ',' << dp[0] << ',' << dp[1] << ',' << tmax << ',' << gMax(fgas.primitiveField())
                << ',' << Ia/(4*F) << ',' << Ic/(2*F) << ',' << gasOut[0] << ',' << gasOut[1]
                << ',' << massError << ',' << gasError << ',' << change << ',' << heatW << ',' << bounded << '\n';
        history.flush();
        if (change<props.get<scalar>("outerTolerance") && chargeError<1e-6
            && massError<props.get<scalar>("balanceTolerance") && gasError<props.get<scalar>("balanceTolerance")
            && !bounded && runTime.value()>=props.get<label>("minimumIterations")) ++stable;
        else stable=0;
        runTime.write();
        if (stable>=5)
        {
            converged=true;
            runTime.writeNow();
            Info << "CONVERGED: coupled fields and charge/mass/species balances passed." << endl;
            break;
        }
        runTime.printExecutionTime(Info);
    }
    runTime.writeNow();
    std::ofstream status((outDir/"status.json").c_str());
    status << "{\"converged\":" << (converged ? "true" : "false")
           << ",\"iteration\":" << runTime.value() << ",\"cells\":" << mesh.nCells() << "}\n";
    if (!converged) WarningInFunction << "Iteration limit reached; this is not a converged result." << endl;
    Info << "End" << endl;
    return converged ? 0 : 2;
}
