/* Zhou et al. (2026), steady two-fluid porous electrolyzer reconstruction.
 * Superficial phase momentum, common liquid pressure + Leverett gas pressure.
 * Pseudo-time is solely an iterative device; time labels are outer iterations.
 * All paper ambiguities and closure choices are recorded in benchmarks/P006.
 */
#include "Tools.H"
#include "ChargeNetwork.H"
#include "FlowNetwork.H"
#include "Anderson.H"
#include <fstream>
#include <iomanip>
#include <cmath>

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    IOdictionary props(IOobject("modelProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
    auto par=[&](const word& key) { return props.get<scalar>(key); };
    const scalar targetVoltage=par("cell_voltage");
    scalar appliedVoltage=min(par("initial_voltage"),targetVoltage);
    const scalar F=96485.33212, rhoL=par("liquid_density"), muL=par("liquid_viscosity");
    const scalar MW=.018, MO=.032, MH=.002, cpL=par("liquid_cp");
    const scalar waterC=par("ionomer_density")/par("ionomer_equivalent_weight");
    #define READ_SCALAR(name) volScalarField name(IOobject(#name,runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh)
    READ_SCALAR(material); READ_SCALAR(p); READ_SCALAR(T); READ_SCALAR(sg);
    READ_SCALAR(phiS); READ_SCALAR(phiE);
    volScalarField water(IOobject("lambda",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volVectorField Ul(IOobject("Ul",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volVectorField Ug(IOobject("Ug",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volVectorField qL(IOobject("qL",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),Ul);
    volVectorField qG(IOobject("qG",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),Ug);
    auto eps=auxiliary("porosity",mesh,dimless);
    auto al=auxiliary("liquidFraction",mesh,dimless);
    auto ag=auxiliary("gasFraction",mesh,dimless);
    auto rhoG=auxiliary("rhoG",mesh,dimDensity);
    auto muG=auxiliary("muG",mesh,dimDynamicViscosity);
    auto k=auxiliary("kEffective",mesh,dimPower/dimLength/dimTemperature);
    auto sigmaS=auxiliary("sigmaS",mesh,dimensionSet(-1,-3,3,0,0,2,0));
    auto sigmaE=auxiliary("sigmaE",mesh,sigmaS.dimensions());
    auto iv=auxiliary("iVol",mesh,dimCurrent/dimVolume,0,IOobject::AUTO_WRITE);
    auto heat=auxiliary("heatSource",mesh,dimPower/dimVolume,0,IOobject::AUTO_WRITE);
    auto coverage=auxiliary("coverage",mesh,dimless);
    auto pc=auxiliary("pc",mesh,p.dimensions(),0,IOobject::AUTO_WRITE);
    auto dragL=auxiliary("dragL",mesh,dimDensity/dimTime);
    auto dragG=auxiliary("dragG",mesh,dimDensity/dimTime);
    auto Sl=auxiliary("liquidSource",mesh,dimless/dimTime,0,IOobject::AUTO_WRITE);
    auto Sg=auxiliary("gasSource",mesh,dimless/dimTime,0,IOobject::AUTO_WRITE);
    auto hydration=auxiliary("hydrationSource",mesh,dimless/dimTime,0,IOobject::AUTO_WRITE);
    auto waterD=auxiliary("waterD",mesh,dimArea/dimTime);
    boolList fluid(mesh.nCells(),false),anode(mesh.nCells(),false),ionic(mesh.nCells(),false);
    labelList mat(mesh.nCells()); DynamicList<label> fixed,waterFixed;
    forAll(mat,c)
    {
        mat[c]=label(material[c]+.5);
        if (mat[c]<0 || mat[c]>8) FatalErrorInFunction << "Invalid material" << exit(FatalError);
        fluid[c]=mat[c]==1 || mat[c]==2 || mat[c]==3 || mat[c]==5 || mat[c]==6 || mat[c]==7;
        anode[c]=mat[c]>=5; ionic[c]=mat[c]>=3 && mat[c]<=5;
        if (!fluid[c]) fixed.append(c);
        if (mat[c]!=4) waterFixed.append(c);
        eps[c]=(mat[c]==1 || mat[c]==7) ? 1 : (mat[c]==2 || mat[c]==6) ? par("porosity_ptl") :
            (mat[c]==3 || mat[c]==5) ? par("porosity_cl") : 0;
        sigmaS[c]=mat[c]==1 || mat[c]==4 || mat[c]==7 ? 0 : par("electron_conductivity");
        rhoG[c]=par(anode[c] ? "oxygen_density" : "hydrogen_density");
        muG[c]=par(anode[c] ? "oxygen_viscosity" : "hydrogen_viscosity");
    }
    labelList solids(fixed), lambdaFixed(waterFixed);
    scalarField solidZero(solids.size(),0),fixedWater(lambdaFixed.size(),par("membrane_water_boundary"));
    vectorField solidU(solids.size(),vector::zero);
    boolList nbrFluid; labelList nbrMat; vectorField nbrCentres;
    syncTools::swapBoundaryCellList(mesh,fluid,nbrFluid);
    syncTools::swapBoundaryCellList(mesh,mat,nbrMat);
    syncTools::swapBoundaryCellPositions(mesh,mesh.C().primitiveField(),nbrCentres);
    eps.correctBoundaryConditions();sigmaS.correctBoundaryConditions();rhoG.correctBoundaryConditions();muG.correctBoundaryConditions();
    qL==eps*(scalar(1)-sg)*Ul;qG==eps*sg*Ug;
    surfaceScalarField epsF(IOobject("epsF",runTime.timeName(),mesh),fvc::interpolate(eps));
    harmonic(epsF,eps,nbrCentres,&fluid);
    surfaceScalarField phiL(IOobject("phiL",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    surfaceScalarField phiG(IOobject("phiG",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volVectorField forceL(IOobject("pressureForceL",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),
        mesh,dimensionedVector("zero",p.dimensions()/dimLength,vector::zero));
    volVectorField forceG(IOobject("pressureForceG",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),
        mesh,dimensionedVector("zero",p.dimensions()/dimLength,vector::zero));
    ChargeNetwork charge(mesh,mat);
    FlowNetwork flowNetwork(mesh,mat);
    OuterAcceleration acceleration(mat,eps);
    const label totalCells=returnReduce(mesh.nCells(),sumOp<label>());
    fileName out=runTime.globalPath()/"postProcessing"/"p006";
    std::ofstream history;
    if (Pstream::master())
    {
        mkDir(out);history.open((out/"history.csv").c_str());history<<std::setprecision(12);
        history<<"iteration,current_anode_A,current_cathode_A,charge_residual,T_mean_K,T_std_K,T_min_K,T_max_K,flow_cv,dp_anode_Pa,dp_cathode_Pa,liquid_balance,gas_balance,local_liquid_balance,local_gas_balance,field_change,heat_W,energy_balance,bounded_cells,lambda_min,lambda_max,applied_voltage_V,liquid_momentum_residual,gas_momentum_residual,momentum_fraction,thermal_active";
        for (label i=0;i<9;++i) history<<",branch_"<<i+1<<"_kg_s";
        history<<",field_change_T,field_change_sg,field_change_p,field_change_Ul,field_change_Ug,field_change_lambda";
        history<<'\n';
    }
    bool converged=false,thermalActive=props.lookupOrDefault<Switch>("initial_thermal_active",false);label stable=0;
    if(thermalActive)Info<<"Enabling coupled energy solve at the requested voltage."<<endl;
    scalar pseudoCo=100.0;scalar momentumFraction=props.lookupOrDefault<scalar>("initial_momentum_fraction",0.);
    Info<<"P006 two-fluid/charge/water/heat: "<<totalCells<<" cells, "<<Pstream::nProcs()<<" processes. Iteration labels are not seconds."<<endl;
    while (runTime.loop())
    {
        Info<<"\nIteration = "<<runTime.timeName()<<endl;
        props.set("cell_voltage",appliedVoltage);
        props.set("momentum_fraction",momentumFraction);
        props.set("pressure_relaxation",1.0);
        const bool darcyInitialization=momentumFraction<=.000256;
        pseudoCo=darcyInitialization?1000.:props.lookupOrDefault<scalar>("phase_numerical_courant",.001);
        const scalar dt=darcyInitialization?1.:par("pseudo_dt");
        acceleration.begin(qL,qG,p,sg,T,water,phiL,phiG);
        Ul=qL/max(eps*(scalar(1)-sg),dimensionedScalar("floor",dimless,1e-9));
        Ug=qG/max(eps*sg,dimensionedScalar("floor",dimless,1e-9));
        Ul.correctBoundaryConditions();Ug.correctBoundaryConditions();
        scalarField oldT(T.primitiveField()),oldS(sg.primitiveField()),oldP(p.primitiveField()),oldW(water.primitiveField());
        vectorField oldUl(Ul.primitiveField()),oldUg(Ug.primitiveField());
        forAll(mat,c)
        {
            scalar g=fluid[c] ? min(max(sg[c],1e-6),1-1e-6) : 0, l=1-g;
            al[c]=eps[c]*l;ag[c]=eps[c]*g;
            scalar kFluid=l*par("liquid_conductivity")+g*par(anode[c] ? "oxygen_conductivity" : "hydrogen_conductivity");
            scalar ks=mat[c]==4 ? par("membrane_conductivity") : (mat[c]==3 || mat[c]==5) ? par("catalyst_conductivity") : par("solid_conductivity");
            k[c]=(1-eps[c])*ks+eps[c]*kFluid;
            scalar factor=mat[c]==4 ? 1 : std::pow(par("ionomer_fraction_cl"),1.5);
            sigmaE[c]=ionic[c] ? factor*max(.5139*water[c]-.326,0.01)*std::exp(1268*(1/303.-1/T[c])) : 0;
            // f(lambda) from Eq.18. eta_lambda absorbs Eq.17's unspecified dimensional prefactor.
            waterD[c]=ionic[c] ? par("water_diffusivity_factor")*4.17e-8*water[c]*(1+161*std::exp(-water[c]))*std::exp(-2346/T[c]) : 0;
            pc[c]=0;dragL[c]=0;dragG[c]=0;
            if (fluid[c])
            {
                scalar permeability=par(mat[c]==1 || mat[c]==7 ? "permeability_channel" :
                    mat[c]==2 || mat[c]==6 ? "permeability_ptl" : "permeability_cl");
                scalar cosine=std::cos(par("contact_angle_deg")*constant::mathematical::pi/180);
                scalar z=cosine>0 ? g : l;
                scalar pref=eps[c]<1 ? par("surface_tension")*cosine*std::sqrt(eps[c]/permeability) : 0;
                pc[c]=pref*(1.417*z-2.120*sqr(z)+1.263*pow3(z));
                // Eq.13 acts on superficial velocity u = eps*s*U.
                dragL[c]=muL/(permeability*pow3(l));
                dragG[c]=muG[c]/(permeability*pow3(g));
            }
        }
        for (auto* f : {&al,&ag,&k,&sigmaE,&waterD,&pc,&dragL,&dragG}) f->correctBoundaryConditions();
        forAll(mesh.boundary(),pi) if (mesh.boundary()[pi].name().ends_with("Inlet"))
        {
            al.boundaryFieldRef()[pi]=par("inlet_liquid_saturation");
            ag.boundaryFieldRef()[pi]=1-par("inlet_liquid_saturation");
            pc.boundaryFieldRef()[pi]=0;
        }
        refreshPhasePressureForces(p,pc,dragL,dragG,mat,fluid,nbrMat,nbrFluid,nbrCentres,forceL,forceG);
        scalar chargeError;
        charge.solve(props,T,sg,sigmaS,sigmaE,phiS,phiE,iv,heat,coverage,chargeError);
        // Dissolved water: diffusion and electro-osmotic drag in PEM; CL reservoirs.
        surfaceScalarField sEF(IOobject("sEF",runTime.timeName(),mesh),fvc::interpolate(sigmaE));harmonic(sEF,sigmaE,nbrCentres,&ionic);
        surfaceScalarField ionicCurrent("ionicCurrent",-sEF*fvc::snGrad(phiE)*mesh.magSf());
        surfaceScalarField waterPhi("waterPhi",ionicCurrent/dimensionedScalar("F",dimCurrent*dimTime/dimVolume, F*waterC)*(2.5/22.));
        surfaceScalarField wDF(IOobject("wDF",runTime.timeName(),mesh),fvc::interpolate(waterD));harmonic(wDF,waterD,nbrCentres,&ionic);
        fvScalarMatrix WEqn(fvm::div(waterPhi,water)-fvm::laplacian(wDF,water));
        scalar wdiag=max(gMax(mag(WEqn.diag())),SMALL);
        forAll(lambdaFixed,i) WEqn.diag()[lambdaFixed[i]]=wdiag;
        WEqn.setValues(lambdaFixed,fixedWater);WEqn.solve();water.correctBoundaryConditions();
        surfaceScalarField waterFlux("waterFlux",waterPhi*fvc::interpolate(water,waterPhi,"upwind")-wDF*fvc::snGrad(water)*mesh.magSf());
        volScalarField divWater("divWater",fvc::div(waterFlux));
        scalar Ia=0,Ic=0;
        forAll(mat,c)
        {
            Sl[c]=0;Sg[c]=0;hydration[c]=0;
            if (mat[c]==5) {Ia+=iv[c]*mesh.V()[c];Sl[c]=-MW*iv[c]/(2*F*rhoL);Sg[c]=MO*iv[c]/(4*F*rhoG[c]);}
            if (mat[c]==3) {Ic-=iv[c]*mesh.V()[c];Sg[c]=-MH*iv[c]/(2*F*rhoG[c]);}
            if (mat[c]==3 || mat[c]==5) {hydration[c]=-divWater[c]*waterC*MW/rhoL;Sl[c]+=hydration[c];}
        }
        reduce(Ia,sumOp<scalar>());reduce(Ic,sumOp<scalar>());
        // Discretize Eq.13 directly in superficial velocity. This retains the
        // fraction-gradient stresses lost by an approximate intrinsic conversion.
        qL==al*Ul;qG==ag*Ug;
        volScalarField safeL("safeL",max(al,dimensionedScalar("floor",dimless,1e-9)));
        volScalarField safeG("safeG",max(ag,dimensionedScalar("floor",dimless,1e-9)));
        volScalarField viscL("viscL",dimensionedScalar("muL",dimDynamicViscosity,muL)/safeL);
        volScalarField viscG("viscG",muG/safeG);
        surfaceScalarField muLF(IOobject("muLF",runTime.timeName(),mesh),fvc::interpolate(viscL));harmonic(muLF,viscL,nbrCentres,&fluid);
        surfaceScalarField muGF(IOobject("muGF",runTime.timeName(),mesh),fvc::interpolate(viscG));harmonic(muGF,viscG,nbrCentres,&fluid);
        volScalarField pseudoL("pseudoL",dimensionedScalar("rhoDt",dimDensity/dimTime,rhoL/dt)/safeL);
        volScalarField pseudoG("pseudoG",rhoG/(safeG*dimensionedScalar("dt",dimTime,dt)));
        // Match momentum pseudo-time to local phase convection. A global dt
        // lets fast gas momentum jump while saturation remains nearly frozen.
        volScalarField fluxSumL("fluxSumL",fvc::surfaceSum(mag(phiL)));
        volScalarField fluxSumG("fluxSumG",fvc::surfaceSum(mag(phiG)));
        forAll(mat,c)if(fluid[c])
        {pseudoL[c]+=rhoL*fluxSumL[c]/(pseudoCo*mesh.V()[c]*sqr(safeL[c]));
         pseudoG[c]+=rhoG[c]*fluxSumG[c]/(pseudoCo*mesh.V()[c]*sqr(safeG[c]));}
        pseudoL*=0.;pseudoG*=0.;
        surfaceScalarField massL("massL",phiL*dimensionedScalar("rhoL",dimDensity,rhoL));
        surfaceScalarField massG("massG",phiG*fvc::interpolate(rhoG));
        // The transpose/deviatoric contribution completes the Newtonian tensor
        // accompanying the implicit vector Laplacian in the printed equation.
        surfaceVectorField tauL("tauL",phaseTransposeStressFlux(muLF,qL,fluid,nbrFluid));
        surfaceVectorField tauG("tauG",phaseTransposeStressFlux(muGF,qG,fluid,nbrFluid));
        auto wallStress=[&](label c,const vector& sf,scalar distance,scalar orientation)
        {
            vector normal=orientation*sf/mag(sf);
            return std::array<vector,2>{{-viscL[c]*(qL[c]&normal)*sf/(3*distance),
                                        -viscG[c]*(qG[c]&normal)*sf/(3*distance)}};
        };
        forAll(mesh.neighbour(),f)
        {
            label o=mesh.owner()[f],n=mesh.neighbour()[f];
            if(fluid[o]==fluid[n])continue;
            label c=fluid[o]?o:n;
            auto stress=wallStress(c,mesh.Sf()[f],mag(mesh.Cf()[f]-mesh.C()[c]),fluid[o]?1:-1);
            tauL[f]=stress[0];tauG[f]=stress[1];
        }
        forAll(mesh.boundary(),pi)
        {
            const auto& patch=mesh.boundary()[pi];
            forAll(patch,f)
            {
                label c=patch.faceCells()[f],b=patch.start()-mesh.nInternalFaces()+f;
                bool wall=patch.coupled()?fluid[c]&&!nbrFluid[b]:fluid[c]&&patch.name()=="externalWalls";
                if(!wall)continue;
                auto stress=wallStress(c,patch.Sf()[f],mag(patch.Cf()[f]-mesh.C()[c]),1);
                tauL.boundaryFieldRef()[pi][f]=stress[0];tauG.boundaryFieldRef()[pi][f]=stress[1];
            }
        }
        volVectorField stressL("stressL",fvc::surfaceIntegrate(tauL));
        volVectorField stressG("stressG",fvc::surfaceIntegrate(tauG));
        // Combining Eq.13 with rho*div(q)=Sm (Eqs.11-12) gives the
        // conservative inertia div(rho*q*q)/eps_eff^2. This includes Sm*q
        // exactly at steady conservation, without treating transient phase
        // imbalance as an artificial momentum production term.
        fvVectorMatrix LEqn(momentumFraction*(1/sqr(safeL))*fvm::div(massL,qL)
            -momentumFraction*fvm::laplacian(muLF,qL)+fvm::Sp(dragL+pseudoL,qL)==pseudoL*qL+momentumFraction*stressL);
        fvVectorMatrix GEqn(momentumFraction*(1/sqr(safeG))*fvm::div(massG,qG)
            -momentumFraction*fvm::laplacian(muGF,qG)+fvm::Sp(dragG+pseudoG,qG)==pseudoG*qG+momentumFraction*stressG);
        auto wall=[&](label c,scalar area,scalar distance)
        {LEqn.diag()[c]+=momentumFraction*viscL[c]*area/distance;GEqn.diag()[c]+=momentumFraction*viscG[c]*area/distance;};
        forAll(mesh.neighbour(),f)
        {
            label o=mesh.owner()[f],n=mesh.neighbour()[f];
            if (fluid[o]==fluid[n])continue;
            label c=fluid[o]?o:n;wall(c,mesh.magSf()[f],mag(mesh.Cf()[f]-mesh.C()[c]));
        }
        forAll(mesh.boundary(),pi) if (mesh.boundary()[pi].coupled())
        {
            const auto& patch=mesh.boundary()[pi];
            forAll(patch,f)
            {label c=patch.faceCells()[f],b=patch.start()-mesh.nInternalFaces()+f;
             if (fluid[c]&&!nbrFluid[b])wall(c,patch.magSf()[f],mag(patch.Cf()[f]-mesh.C()[c]));}
        }
        // The outer Anderson method controls the update; use unrelaxed momentum matrices.
        scalar ld=gMax(mag(LEqn.diag())),gd=gMax(mag(GEqn.diag()));
        forAll(solids,s) {LEqn.diag()[solids[s]]=ld;GEqn.diag()[solids[s]]=gd;}
        LEqn.setValues(solids,solidU);GEqn.setValues(solids,solidU);
        // Solve the momentum predictors at the current pressure before the
        // coupled pressure/saturation update. Fresh momentum solves below use
        // the projected saturation; the outer Anderson method closes the coupling.
        solveColumnScaled(LEqn,forceL,dragL,muL/par("permeability_channel"));solveColumnScaled(GEqn,forceG,dragG,par("oxygen_viscosity")/par("permeability_channel"));
        // Pressure-free H predictors; capillary force is applied on faces below.
        volScalarField rqL("rqL",1.0/LEqn.A()),rqG("rqG",1.0/GEqn.A());
        volVectorField HqL("HqL",rqL*LEqn.H()),HqG("HqG",rqG*GEqn.H());
        // The projection parameterizes phase mobility by eps^2*s^2, so expose
        // its coefficient and predictor in the matching intrinsic coordinates.
        volScalarField rL("rL",rqL/sqr(safeL)),rG("rG",rqG/sqr(safeG));
        volVectorField HL("HL",HqL/safeL),HG("HG",HqG/safeG);
        HL.correctBoundaryConditions();HG.correctBoundaryConditions();
        forAll(mesh.boundary(),pi) if (Ul.boundaryField()[pi].fixesValue())
        {HL.boundaryFieldRef()[pi]==Ul.boundaryField()[pi];HG.boundaryFieldRef()[pi]==Ug.boundaryField()[pi];}
        surfaceScalarField pressureFaceL("pressureFaceL",fvc::interpolate(p));
        surfaceScalarField pressureFaceG("pressureFaceG",pressureFaceL);
        flowNetwork.solve(props,eps,rL,rG,HL,HG,Sl,Sg,dt,pseudoCo,p,sg,pc,phiL,phiG,pressureFaceL,pressureFaceG);
        forAll(mat,c){al[c]=eps[c]*(1-sg[c]);ag[c]=eps[c]*sg[c];}
        al.correctBoundaryConditions();ag.correctBoundaryConditions();
        forAll(mesh.boundary(),pi)if(mesh.boundary()[pi].name().ends_with("Inlet"))
        {al.boundaryFieldRef()[pi]=par("inlet_liquid_saturation");
         ag.boundaryFieldRef()[pi]=1-par("inlet_liquid_saturation");}
        // Supply a diagonal velocity guess for the fresh momentum block.
        // Reconstruct pressure force before applying the cell momentum inverse.
        // Reconstructing its coefficient-weighted flux directly would make
        // the pressure force depend on pseudo-time and equation relaxation.
        forceL=-fvc::surfaceIntegrate(pressureFaceL*mesh.Sf());
        forceG=-fvc::surfaceIntegrate(pressureFaceG*mesh.Sf());
        forAll(solids,i){forceL[solids[i]]=vector::zero;forceG[solids[i]]=vector::zero;}
        volVectorField projectedL("projectedL",(al/safeL)*HqL+pow3(al/safeL)*rqL*forceL);
        volVectorField projectedG("projectedG",(ag/safeG)*HqG+pow3(ag/safeG)*rqG*forceG);
        Ul=projectedL/max(al,dimensionedScalar("floor",dimless,1e-9));
        Ug=projectedG/max(ag,dimensionedScalar("floor",dimless,1e-9));
        // Pseudo-time and matrix relaxation already damp the update. Apply
        // the projected superficial velocity without a second field relaxation
        // that would advance phase inventory faster than its momentum.
        Ul.correctBoundaryConditions();Ug.correctBoundaryConditions();
        forAll(solids,s){Ul[solids[s]]=vector::zero;Ug[solids[s]]=vector::zero;}
        qL==al*Ul;qG==ag*Ug;
        // A nonlinear block sweep: refresh momentum at the projected saturation.
        // Face conservation is re-coupled by the next outer block evaluation.
        auto refreshMomentum=[&](bool apply)
        {
            forAll(mat,c)if(fluid[c])
            {
                scalar perm=par(mat[c]==1||mat[c]==7 ? "permeability_channel" : mat[c]==2||mat[c]==6 ? "permeability_ptl" : "permeability_cl");
                dragL[c]=muL/(perm*pow3(max(1-sg[c],1e-6)));
                dragG[c]=muG[c]/(perm*pow3(max(sg[c],1e-6)));
            }
            dragL.correctBoundaryConditions();dragG.correctBoundaryConditions();
        volScalarField safeL("safeL",max(al,dimensionedScalar("floor",dimless,1e-9)));
        volScalarField safeG("safeG",max(ag,dimensionedScalar("floor",dimless,1e-9)));
        volScalarField viscL("viscL",dimensionedScalar("muL",dimDynamicViscosity,muL)/safeL);
        volScalarField viscG("viscG",muG/safeG);
        surfaceScalarField muLF(IOobject("muLF",runTime.timeName(),mesh),fvc::interpolate(viscL));harmonic(muLF,viscL,nbrCentres,&fluid);
        surfaceScalarField muGF(IOobject("muGF",runTime.timeName(),mesh),fvc::interpolate(viscG));harmonic(muGF,viscG,nbrCentres,&fluid);
        volScalarField pseudoL("pseudoL",dimensionedScalar("rhoDt",dimDensity/dimTime,rhoL/dt)/safeL);
        volScalarField pseudoG("pseudoG",rhoG/(safeG*dimensionedScalar("dt",dimTime,dt)));
        // Match momentum pseudo-time to local phase convection. A global dt
        // lets fast gas momentum jump while saturation remains nearly frozen.
        volScalarField fluxSumL("fluxSumL",fvc::surfaceSum(mag(phiL)));
        volScalarField fluxSumG("fluxSumG",fvc::surfaceSum(mag(phiG)));
        forAll(mat,c)if(fluid[c])
        {pseudoL[c]+=rhoL*fluxSumL[c]/(pseudoCo*mesh.V()[c]*sqr(safeL[c]));
         pseudoG[c]+=rhoG[c]*fluxSumG[c]/(pseudoCo*mesh.V()[c]*sqr(safeG[c]));}
        pseudoL*=0.;pseudoG*=0.;
        surfaceScalarField massL("massL",phiL*dimensionedScalar("rhoL",dimDensity,rhoL));
        surfaceScalarField massG("massG",phiG*fvc::interpolate(rhoG));
        // The transpose/deviatoric contribution completes the Newtonian tensor
        // accompanying the implicit vector Laplacian in the printed equation.
        surfaceVectorField tauL("tauL",phaseTransposeStressFlux(muLF,qL,fluid,nbrFluid));
        surfaceVectorField tauG("tauG",phaseTransposeStressFlux(muGF,qG,fluid,nbrFluid));
        auto wallStress=[&](label c,const vector& sf,scalar distance,scalar orientation)
        {
            vector normal=orientation*sf/mag(sf);
            return std::array<vector,2>{{-viscL[c]*(qL[c]&normal)*sf/(3*distance),
                                        -viscG[c]*(qG[c]&normal)*sf/(3*distance)}};
        };
        forAll(mesh.neighbour(),f)
        {
            label o=mesh.owner()[f],n=mesh.neighbour()[f];
            if(fluid[o]==fluid[n])continue;
            label c=fluid[o]?o:n;
            auto stress=wallStress(c,mesh.Sf()[f],mag(mesh.Cf()[f]-mesh.C()[c]),fluid[o]?1:-1);
            tauL[f]=stress[0];tauG[f]=stress[1];
        }
        forAll(mesh.boundary(),pi)
        {
            const auto& patch=mesh.boundary()[pi];
            forAll(patch,f)
            {
                label c=patch.faceCells()[f],b=patch.start()-mesh.nInternalFaces()+f;
                bool wall=patch.coupled()?fluid[c]&&!nbrFluid[b]:fluid[c]&&patch.name()=="externalWalls";
                if(!wall)continue;
                auto stress=wallStress(c,patch.Sf()[f],mag(patch.Cf()[f]-mesh.C()[c]),1);
                tauL.boundaryFieldRef()[pi][f]=stress[0];tauG.boundaryFieldRef()[pi][f]=stress[1];
            }
        }
        volVectorField stressL("stressL",fvc::surfaceIntegrate(tauL));
        volVectorField stressG("stressG",fvc::surfaceIntegrate(tauG));
        // Combining Eq.13 with rho*div(q)=Sm (Eqs.11-12) gives the
        // conservative inertia div(rho*q*q)/eps_eff^2. This includes Sm*q
        // exactly at steady conservation, without treating transient phase
        // imbalance as an artificial momentum production term.
        fvVectorMatrix LEqn(momentumFraction*(1/sqr(safeL))*fvm::div(massL,qL)
            -momentumFraction*fvm::laplacian(muLF,qL)+fvm::Sp(dragL+pseudoL,qL)==pseudoL*qL+momentumFraction*stressL);
        fvVectorMatrix GEqn(momentumFraction*(1/sqr(safeG))*fvm::div(massG,qG)
            -momentumFraction*fvm::laplacian(muGF,qG)+fvm::Sp(dragG+pseudoG,qG)==pseudoG*qG+momentumFraction*stressG);
        auto wall=[&](label c,scalar area,scalar distance)
        {LEqn.diag()[c]+=momentumFraction*viscL[c]*area/distance;GEqn.diag()[c]+=momentumFraction*viscG[c]*area/distance;};
        forAll(mesh.neighbour(),f)
        {
            label o=mesh.owner()[f],n=mesh.neighbour()[f];
            if (fluid[o]==fluid[n])continue;
            label c=fluid[o]?o:n;wall(c,mesh.magSf()[f],mag(mesh.Cf()[f]-mesh.C()[c]));
        }
        forAll(mesh.boundary(),pi) if (mesh.boundary()[pi].coupled())
        {
            const auto& patch=mesh.boundary()[pi];
            forAll(patch,f)
            {label c=patch.faceCells()[f],b=patch.start()-mesh.nInternalFaces()+f;
             if (fluid[c]&&!nbrFluid[b])wall(c,patch.magSf()[f],mag(patch.Cf()[f]-mesh.C()[c]));}
        }

            if(apply)
            {
                scalar ld=gMax(mag(LEqn.diag())),gd=gMax(mag(GEqn.diag()));
                forAll(solids,i){LEqn.diag()[solids[i]]=ld;GEqn.diag()[solids[i]]=gd;}
                LEqn.setValues(solids,solidU);GEqn.setValues(solids,solidU);
                solveColumnScaled(LEqn,forceL,dragL,muL/par("permeability_channel"));
                solveColumnScaled(GEqn,forceG,dragG,par("oxygen_viscosity")/par("permeability_channel"));
                return std::array<scalar,2>{{0,0}};
            }
        // Matrix application includes processor terms once. v2312's vector
        // residual() path also adds coupled boundary sources before its ldu
        // interface update; avoid that decomposition-dependent diagnostic.
        volVectorField residualL("momentumResidualL",(LEqn&qL)-forceL);
        volVectorField residualG("momentumResidualG",(GEqn&qG)-forceG);
        scalar momentumL=0,momentumG=0;
        // Check each fluid material separately so the large porous resistance
        // cannot hide a channel momentum defect in a global normalization.
        for(label zone : {1,2,3,5,6,7})
        {
            scalar numL=0,numG=0,denL=0,denG=0;
            forAll(mat,c)if(mat[c]==zone)
            {
                scalar v=mesh.V()[c];
                numL+=v*mag(residualL[c]);numG+=v*mag(residualG[c]);
                denL+=v*(mag(forceL[c])+mag(stressL[c])+dragL[c]*mag(qL[c]));
                denG+=v*(mag(forceG[c])+mag(stressG[c])+dragG[c]*mag(qG[c]));
            }
            reduce(numL,sumOp<scalar>());reduce(numG,sumOp<scalar>());
            reduce(denL,sumOp<scalar>());reduce(denG,sumOp<scalar>());
            momentumL=max(momentumL,numL/max(denL,1e-12));
            momentumG=max(momentumG,numG/max(denG,1e-12));
        }

            return std::array<scalar,2>{{momentumL,momentumG}};
        };
        refreshMomentum(true);refreshMomentum(true);
        const auto momentumError=refreshMomentum(false);
        const scalar momentumL=momentumError[0],momentumG=momentumError[1];
        Ul=qL/max(al,dimensionedScalar("floor",dimless,1e-9));
        Ug=qG/max(ag,dimensionedScalar("floor",dimless,1e-9));
        Ul.correctBoundaryConditions();Ug.correctBoundaryConditions();
        label bounded=0;
        // LTE conjugate heat transfer; convective cooling only on exposed BP faces.
        surfaceScalarField kF(IOobject("kF",runTime.timeName(),mesh),fvc::interpolate(k));harmonic(kF,k,nbrCentres);
        auto cpG=auxiliary("cpG",mesh,dimEnergy/dimMass/dimTemperature);
        forAll(mat,c) {cpG[c]=par(anode[c]?"oxygen_cp":"hydrogen_cp");}
        cpG.correctBoundaryConditions();
        surfaceScalarField heatPhi("heatPhi",phiL*dimensionedScalar("rhoCpL",dimDensity*dimEnergy/dimMass/dimTemperature,rhoL*cpL)+phiG*fvc::interpolate(rhoG*cpG));
        fvScalarMatrix TEqn(fvm::div(heatPhi,T)-fvm::Sp(fvc::div(heatPhi),T)-fvm::laplacian(kF,T)==heat);
        forAll(mesh.boundary(),pi)
        {
            const auto& patch=mesh.boundary()[pi];if(patch.coupled())continue;
            forAll(patch,f)
            {
                label c=patch.faceCells()[f];if(mat[c]!=0 && mat[c]!=8)continue;
                scalar h=1/(1/par("external_heat_transfer")+1/(patch.deltaCoeffs()[f]*k[c]));
                TEqn.diag()[c]+=h*patch.magSf()[f];TEqn.source()[c]+=h*patch.magSf()[f]*par("ambient_temperature");
            }
        }
        if(thermalActive) TEqn.solve();
        // Solve the thermal block fully. A globally capped inner update can
        // make F(T)-T insensitive to a uniform temperature shift.
        // The outer Anderson update bounds every proposed temperature.
        T.correctBoundaryConditions();
        scalar tmin=gMin(T.primitiveField()),tmax=gMax(T.primitiveField());
        if(!std::isfinite(tmin)||!std::isfinite(tmax)||tmin<273.15||tmax>373.15)
            FatalErrorInFunction<<"Temperature outside no-phase-change validity: "<<tmin<<' '<<tmax<<exit(FatalError);
        if(gMin(water.primitiveField())<=.64 || gMax(water.primitiveField())>30)
            FatalErrorInFunction<<"Membrane water outside closure range"<<exit(FatalError);
        scalar area=0,ta=0,tt=0,lo=GREAT,hi=-GREAT,branches[9]={0};
        const scalar referenceT=par("inlet_temperature");
        auto membraneSample=[&](scalar a,scalar temp) {area+=a;ta+=a*(temp-referenceT);tt+=a*sqr(temp-referenceT);lo=min(lo,temp);hi=max(hi,temp);};
        forAll(mesh.neighbour(),f)
        {
            label o=mesh.owner()[f],n=mesh.neighbour()[f];
            if ((mat[o]==4&&mat[n]==5)||(mat[o]==5&&mat[n]==4))
            {
                scalar go=k[o]/mag(mesh.Cf()[f]-mesh.C()[o]),gn=k[n]/mag(mesh.Cf()[f]-mesh.C()[n]);
                membraneSample(mesh.magSf()[f],(go*T[o]+gn*T[n])/(go+gn));
            }
            if(mat[o]==7&&mat[n]==7&&mag(mesh.Cf()[f].y()-.0125)<1e-10&&mag(mesh.Sf()[f].y())>0)
            {label branch=min(8,max(0,label(mesh.Cf()[f].x()/.002)));branches[branch]+=rhoL*phiL[f]*(mesh.Sf()[f].y()>0?1:-1);}
        }
        scalarField nbrT,nbrK;syncTools::swapBoundaryCellList(mesh,T.primitiveField(),nbrT);syncTools::swapBoundaryCellList(mesh,k.primitiveField(),nbrK);
        scalar netL[2]={0,0},netG[2]={0,0},srcL[2]={0,0},srcG[2]={0,0},dp[2]={0,0},portA[2]={0,0};
        forAll(mesh.boundary(),pi)
        {
            const auto& patch=mesh.boundary()[pi];
            forAll(patch,f)
            {
                label c=patch.faceCells()[f],b=patch.start()-mesh.nInternalFaces()+f;
                if(patch.coupled())
                {
                    if(mat[c]==4&&nbrMat[b]==5)
                    {scalar go=k[c]/mag(patch.Cf()[f]-mesh.C()[c]),gn=nbrK[b]/mag(patch.Cf()[f]-nbrCentres[b]);membraneSample(patch.magSf()[f],(go*T[c]+gn*nbrT[b])/(go+gn));}
                    if(mat[c]==7&&nbrMat[b]==7&&patch.Sf()[f].y()>0&&mag(patch.Cf()[f].y()-.0125)<1e-10)
                    {label branch=min(8,max(0,label(patch.Cf()[f].x()/.002)));branches[branch]+=rhoL*phiL.boundaryField()[pi][f];}
                }
                else if(patch.name().ends_with("Inlet")||patch.name().ends_with("Outlet"))
                {label side=patch.name().starts_with("anode")?0:1;netL[side]+=phiL.boundaryField()[pi][f];netG[side]+=phiG.boundaryField()[pi][f];
                 if(patch.name().ends_with("Inlet")){dp[side]+=p.boundaryField()[pi][f]*patch.magSf()[f];portA[side]+=patch.magSf()[f];}}
            }
        }
        volScalarField rl("rl",fvc::div(phiL)-Sl),rg("rg",fvc::div(phiG)-Sg);
        scalar localL=0,localG=0;
        forAll(mat,c) if(fluid[c])
        {label side=anode[c]?0:1;srcL[side]+=Sl[c]*mesh.V()[c];srcG[side]+=Sg[c]*mesh.V()[c];localL+=mag(rl[c])*mesh.V()[c];localG+=mag(rg[c])*mesh.V()[c];}
        for(label side=0;side<2;++side)
        {reduce(netL[side],sumOp<scalar>());reduce(netG[side],sumOp<scalar>());reduce(srcL[side],sumOp<scalar>());reduce(srcG[side],sumOp<scalar>());reduce(dp[side],sumOp<scalar>());reduce(portA[side],sumOp<scalar>());dp[side]/=max(portA[side],SMALL);}
        reduce(area,sumOp<scalar>());reduce(ta,sumOp<scalar>());reduce(tt,sumOp<scalar>());reduce(lo,minOp<scalar>());reduce(hi,maxOp<scalar>());
        scalar mean=referenceT+ta/area,sdT=std::sqrt(max(tt/area-sqr(ta/area),0.)),sumB=0,sumBB=0;
        for(label i=0;i<9;++i){reduce(branches[i],sumOp<scalar>());sumB+=branches[i];sumBB+=sqr(branches[i]);}
        scalar meanB=sumB/9,cv=std::sqrt(max(sumBB/9-sqr(meanB),0.))/max(mag(meanB),SMALL);
        scalar flow=par("inlet_flow_ml_min")*1e-6/60,errL=0,errG=0;
        for(label i=0;i<2;++i){errL=max(errL,mag(netL[i]-srcL[i])/flow);errG=max(errG,mag(netG[i]-srcG[i])/max(mag(srcG[i]),flow));}
        reduce(localL,sumOp<scalar>());reduce(localG,sumOp<scalar>());localL/=2*flow;localG/=max(srcG[0]+srcG[1],2*flow);
        // Expose the existing normalized block updates without changing their
        // scales or acceptance. old* is the input after Anderson acceleration.
        const char* changeNames[]={"T","sg","p","Ul","Ug","lambda"};
        const scalar changes[]={
            gMax(mag(T.primitiveField()-oldT))/max(tmax-par("inlet_temperature"),1.),
            gMax(mag(sg.primitiveField()-oldS)),
            gMax(mag(p.primitiveField()-oldP))/max(gMax(mag(p.primitiveField())),1.),
            gMax(mag(Ul.primitiveField()-oldUl))/max(gMax(mag(Ul.primitiveField())),.01),
            gMax(mag(Ug.primitiveField()-oldUg))/max(gMax(mag(Ug.primitiveField())),.01),
            gMax(mag(water.primitiveField()-oldW))/par("membrane_water_boundary")};
        scalar change=changes[0];label dominantChange=0;
        for(label i=1;i<6;++i){if(changes[i]>change)dominantChange=i;change=max(change,changes[i]);}
        scalar heatW=gSum(heat.primitiveField()*mesh.V());
        // Discrete nonconservative LTE balance, including all faces and source divergence.
        volScalarField energyResidual("energyResidual",fvc::div(heatPhi,T)-fvc::div(heatPhi)*T-fvc::laplacian(kF,T)-heat);
        scalar energy=gSum(energyResidual.primitiveField()*mesh.V()),cooling=0;
        forAll(mesh.boundary(),pi)
        {const auto& patch=mesh.boundary()[pi];if(patch.coupled())continue;forAll(patch,f)
         {label c=patch.faceCells()[f];if(mat[c]==0||mat[c]==8){scalar h=1/(1/par("external_heat_transfer")+1/(patch.deltaCoeffs()[f]*k[c]));cooling+=h*patch.magSf()[f]*(T[c]-par("ambient_temperature"));}}}
        reduce(cooling,sumOp<scalar>());scalar energyError=mag(energy+cooling)/max(mag(heatW),.01);
        scalar lambdaMin=gMin(water.primitiveField()),lambdaMax=gMax(water.primitiveField());
        Info<<"P006: V="<<appliedVoltage<<"; I="<<Ia<<" A; sigmaT="<<sdT<<" K; CV="<<cv<<"; change="<<change<<"; phase balances="<<localL<<','<<localG<<"; momentum="<<momentumL<<','<<momentumG<<"; momentum fraction="<<momentumFraction<<"; bounded="<<bounded<<endl;
        Info<<"P006 changes:";
        for(label i=0;i<6;++i)Info<<' '<<changeNames[i]<<'='<<changes[i]<<';';
        Info<<" dominant="<<changeNames[dominantChange]<<"; charge="<<chargeError
            <<"; energy="<<energyError<<"; heat="<<(thermalActive?1:0)<<endl;
        if(Pstream::master())
        {history<<runTime.value()<<','<<Ia<<','<<Ic<<','<<chargeError<<','<<mean<<','<<sdT<<','<<lo<<','<<hi<<','<<cv<<','<<dp[0]<<','<<dp[1]<<','<<errL<<','<<errG<<','<<localL<<','<<localG<<','<<change<<','<<heatW<<','<<energyError<<','<<bounded<<','<<lambdaMin<<','<<lambdaMax<<','<<appliedVoltage<<','<<momentumL<<','<<momentumG<<','<<momentumFraction<<','<<(thermalActive?1:0);for(label i=0;i<9;++i)history<<','<<branches[i];for(label i=0;i<6;++i)history<<','<<changes[i];history<<'\n';history.flush();}
        bool modelChanged=false;
        if(momentumFraction<1 && max(localL,localG)<.001 && max(momentumL,momentumG)<.001 && change<.001 && !bounded)
        {momentumFraction=momentumFraction==0 ? 1e-6 : min(1.,2*momentumFraction);modelChanged=true;acceleration.reset();Info<<"Momentum continuation: next fraction="<<momentumFraction<<endl;}
        if(!modelChanged && momentumFraction>=1 && !thermalActive && max(localL,localG)<.01 && max(momentumL,momentumG)<.01 && change<.001 && !bounded)
        {
            modelChanged=true;acceleration.reset();
            if(appliedVoltage<targetVoltage)
            {appliedVoltage=min(targetVoltage,appliedVoltage+.005);Info<<"Voltage continuation: next voltage "<<appliedVoltage<<endl;}
            else {thermalActive=true;Info<<"Enabling coupled energy solve at the requested voltage."<<endl;}
        }
        if(!modelChanged&&momentumFraction>=1&&thermalActive&&change<par("field_tolerance")&&chargeError<1e-6&&max(max(errL,errG),max(localL,localG))<par("balance_tolerance")&&max(momentumL,momentumG)<par("balance_tolerance")&&energyError<par("balance_tolerance")&&!bounded&&runTime.value()>=par("minimum_iterations"))++stable;else stable=0;
        if(modelChanged)runTime.writeNow();
        runTime.write();
        if(stable>=5){converged=true;runTime.writeNow();Info<<"CONVERGED: coupled fields and local/global balances passed."<<endl;break;}
        // Once momentum has settled, accelerate the remaining coupled phase
        // transport even when its steady mass balance is still converging.
        // Waiting for mass conservation first can prevent acceleration of the
        // very transport modes that have stalled. Acceptance is unchanged.
        if(!modelChanged && momentumFraction>.000256)
        {
            // Scale the nonlinear saturation residual by actual conservation.
            // Numerical storage must not hide a remaining phase imbalance.
            const scalarField& phaseResidual=flowNetwork.massCorrection();
            acceleration.observe(qL,qG,p,sg,T,water,phiL,phiG,phaseResidual,momentumFraction>=1 && thermalActive);
        }
        else acceleration.reset();
        runTime.printExecutionTime(Info);
    }
    runTime.writeNow();
    if(Pstream::master())
    {std::ofstream status((out/"status.json").c_str());status<<"{\"converged\":"<<(converged?"true":"false")<<",\"iteration\":"<<runTime.value()<<",\"cells\":"<<totalCells<<",\"processes\":"<<Pstream::nProcs()<<",\"momentum_fraction\":"<<momentumFraction<<",\"thermal_active\":"<<(thermalActive?"true":"false")<<"}\n";}
    Info<<"End"<<endl;return converged?0:2;
}
