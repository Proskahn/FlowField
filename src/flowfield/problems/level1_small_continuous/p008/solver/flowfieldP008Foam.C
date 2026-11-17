/* Li et al. JECE 2026: three-dimensional AEM, channel mixture + porous Darcy,
 * two charge potentials, Butler-Volmer, Faraday sources and conjugate heat.
 * Iteration labels are steady solver iterations, never physical seconds.
 */
#include "Transport.H"
#include "ChargeNetwork.H"
#include <fstream>
#include <iomanip>
int main(int argc,char* argv[])
{
 #include "setRootCase.H"
 #include "createTime.H"
 #include "createMesh.H"
 if(Pstream::parRun())FatalErrorInFunction<<"P008 is serial; no MPI approximation is implemented."<<exit(FatalError);
 IOdictionary props(IOobject("modelProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
 auto par=[&](const word& key){return props.get<scalar>(key);};
 const scalar target=par("cell_voltage"),F=96485.33212,R=8.314462618;
 #define READ_SCALAR(n) volScalarField n(IOobject(#n,runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh)
 #define READ_VECTOR(n) volVectorField n(IOobject(#n,runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh)
 READ_SCALAR(material);READ_SCALAR(p);READ_SCALAR(sg);READ_SCALAR(T);READ_SCALAR(phiS);READ_SCALAR(phiE);
 READ_VECTOR(U);READ_VECTOR(Ul);READ_VECTOR(Ug);
 auto sigmaS=auxiliary("sigmaS",mesh,dimensionSet(-1,-3,3,0,0,2,0));
 auto sigmaE=auxiliary("sigmaE",mesh,sigmaS.dimensions());
 auto iv=auxiliary("iVol",mesh,dimCurrent/dimVolume,0,IOobject::AUTO_WRITE);
 auto heat=auxiliary("heatSource",mesh,dimPower/dimVolume,0,IOobject::AUTO_WRITE);
 auto coverage=auxiliary("coverage",mesh,dimless);
 auto Sl=auxiliary("liquidSource",mesh,dimless/dimTime,0,IOobject::AUTO_WRITE);
 auto Sg=auxiliary("gasSource",mesh,dimless/dimTime,0,IOobject::AUTO_WRITE);
 auto capillary=auxiliary("pc",mesh,p.dimensions(),0,IOobject::AUTO_WRITE);
 surfaceScalarField phiL(IOobject("phiL",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,dimensionedScalar("zero",dimVolume/dimTime,0));
 surfaceScalarField phiG(IOobject("phiG",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,dimensionedScalar("zero",dimVolume/dimTime,0));
 labelList mat(mesh.nCells());forAll(mat,c)mat[c]=label(material[c]+.5);
 Transport transport(mesh,mat,props);ChargeNetwork charge(mesh,mat);
 const scalar ss[9]={200000,0,1250,750,0,750,1250,0,200000};
 forAll(mat,c)sigmaS[c]=ss[mat[c]];sigmaS.correctBoundaryConditions();
 fileName out=runTime.path()/"postProcessing"/"p008";mkDir(out);
 std::ofstream history((out/"history.csv").c_str());history<<std::setprecision(14);
 history<<"iteration,applied_voltage_V,current_anode_A,current_cathode_A,charge_residual,liquid_balance,gas_balance,momentum_residual,energy_balance,field_change,min_T,max_T,min_sg,max_sg,current_density_A_cm2,membrane_temperature_uniformity,temperature_gradient_K_m,cathode_mean_velocity_m_s,anode_mean_velocity_m_s,cathode_cl_gas_fraction,anode_cl_gas_fraction,cathode_pressure_drop_Pa,anode_pressure_drop_Pa,heat_W\n";
 bool converged=false;label stable=0;scalar chargeResidual=GREAT;
 Info<<"P008 3D coupled AEM model: "<<mesh.nCells()<<" cells. Labels are outer iterations."<<endl;
 while(runTime.loop()){
  label iteration=runTime.timeIndex();bool report=iteration==1||iteration%10==0;
  scalar applied=min(target,1.4+(target-1.4)*iteration/80.);props.set("cell_voltage",applied);
  scalarField oldS(sg.primitiveField()),oldT(T.primitiveField()),oldIV(iv.primitiveField());vectorField oldU(U.primitiveField());
  forAll(mat,c){scalar temp=T[c],m=par("koh_molarity");
   scalar koh=par("koh_conductivity_scale")*(-2.04*m-.0027*m*m+.005332*m*temp+207.2*m/temp+.00105*m*m*m-.0000004*m*m*temp*temp);
   if(mat[c]==4)sigmaE[c]=(.524*par("membrane_water_content")-.318)*std::exp(1270*(1/303.-1/temp));
   else if(mat[c]==3||mat[c]==5)sigmaE[c]=std::pow(transport.porosity[c]*(1-sg[c]),1.5)*koh;
   else sigmaE[c]=0;
   if((mat[c]>=3&&mat[c]<=5)&&sigmaE[c]<=0)FatalErrorInFunction<<"Nonpositive ionic conductivity"<<exit(FatalError);
  }
  sigmaE.correctBoundaryConditions();charge.solve(props,T,sg,sigmaS,sigmaE,phiS,phiE,iv,heat,coverage,chargeResidual);
  scalar Ia=0,Ic=0;
  forAll(mat,c){Sl[c]=0;Sg[c]=0;
   if(mat[c]==5){Ia+=iv[c]*mesh.V()[c];Sl[c]=iv[c]*.01801528/(2*F*par("liquid_density"));Sg[c]=iv[c]*.0319988/(4*F*transport.rhoGas[c]);}
   if(mat[c]==3){Ic-=iv[c]*mesh.V()[c];Sl[c]=iv[c]*.01801528/(F*par("liquid_density"));Sg[c]=-iv[c]*.00201588/(2*F*transport.rhoGas[c]);}}
  transport.momentum(sg,U);transport.project(p,sg,Sl,Sg);transport.velocities(p,sg,U,Ul,Ug);
  transport.thermal(sg,heat,T);transport.writeFlux(phiL,phiG,p);
  capillary.primitiveFieldRef()=transport.pc;capillary.correctBoundaryConditions();
  scalar change=0;
  forAll(mat,c){change=max(change,mag(sg[c]-oldS[c]));change=max(change,mag(T[c]-oldT[c])/max(mag(T[c]),1.));
   if(transport.channel(c))change=max(change,mag(U[c]-oldU[c])/max(mag(U[c]),.1));
   if(mat[c]==3||mat[c]==5)change=max(change,mag(iv[c]-oldIV[c])/max(mag(iv[c]),1e5));}
  scalarField volumes(9,0),sumsS(9,0),sumsU(9,0),sumsT(9,0);
  forAll(mat,c){scalar v=mesh.V()[c];volumes[mat[c]]+=v;sumsS[mat[c]]+=sg[c]*v;sumsU[mat[c]]+=mag(U[c])*v;sumsT[mat[c]]+=T[c]*v;}
  scalar meanT=sumsT[4]/volumes[4],uniformity=0,meanGrad=0,totalV=sum(mesh.V().field());
  auto gradT=fvc::grad(T);forAll(mat,c){meanGrad+=mag(gradT()[c])*mesh.V()[c];if(mat[c]==4)uniformity+=mag(T[c]-meanT)*mesh.V()[c];}
  uniformity/=meanT*volumes[4];meanGrad/=totalV;
  scalar dpA=0,dpC=0;forAll(mesh.boundary(),pi){const auto& patch=mesh.boundary()[pi];
   if(patch.name().ends_with("Inlet")){scalar dp=sum(p.boundaryField()[pi]*patch.magSf())/sum(patch.magSf());
    if(patch.name()=="anodeInlet")dpA=dp;else dpC=dp;}}
  scalar minT=min(T.primitiveField()),maxT=max(T.primitiveField()),minS=1,maxS=0;
  forAll(mat,c)if(transport.fluid(c)){minS=min(minS,sg[c]);maxS=max(maxS,sg[c]);}
  scalar heatW=sum(heat.primitiveField()*mesh.V());
  if(!std::isfinite(change+Ia+heatW+maxT)||minT<273.15||maxT>373.15||minS<0||maxS>=par("maximum_packing"))
   {runTime.writeNow(); FatalErrorInFunction<<"P008 fields outside numerical admissibility bounds: T="<<minT<<","<<maxT<<"; s="<<minS<<","<<maxS<<"; I="<<Ia<<"; change="<<change<<exit(FatalError);}
  history<<iteration<<','<<applied<<','<<Ia<<','<<Ic<<','<<chargeResidual<<','<<transport.liquidResidual<<','<<transport.gasResidual<<','<<transport.momentumResidual<<','<<transport.energyResidual<<','<<change<<','<<minT<<','<<maxT<<','<<minS<<','<<maxS<<','<<Ia/(par("reactionArea")*1e4)<<','<<uniformity<<','<<meanGrad<<','<<sumsU[1]/volumes[1]<<','<<sumsU[7]/volumes[7]<<','<<sumsS[3]/volumes[3]<<','<<sumsS[5]/volumes[5]<<','<<dpC<<','<<dpA<<','<<heatW<<'\n';history.flush();
  bool good=iteration>=par("minimum_iterations")&&applied==target&&change<par("field_tolerance")&&chargeResidual<1e-7
   &&mag(Ia-Ic)/max(mag(Ia),1e-10)<par("balance_tolerance")&&Ia>0&&dpA>0&&dpC>0
   &&transport.liquidResidual<par("balance_tolerance")&&transport.gasResidual<par("balance_tolerance")
   &&transport.momentumResidual<par("balance_tolerance")&&transport.energyResidual<par("balance_tolerance");
  stable=good?stable+1:0;converged=stable>=5;
  if(report||converged)Info<<"Iteration = "<<iteration<<nl<<"P008: V="<<applied<<"; I="<<Ia<<"; T="<<minT<<","<<maxT<<"; sg="<<maxS<<"; change="<<change<<"; phase="<<transport.liquidResidual<<","<<transport.gasResidual<<"; momentum="<<transport.momentumResidual<<"; energy="<<transport.energyResidual<<endl;
  runTime.write();
  if(converged||iteration>=par("iterations")){runTime.writeNow();std::ofstream status((out/"status.json").c_str());
   status<<"{\"converged\":"<<(converged?"true":"false")<<",\"iteration\":"<<iteration<<",\"cells\":"<<mesh.nCells()<<",\"processes\":1}\n";
   if(converged)Info<<"CONVERGED: P008 coupled flow, charge and heat qualified."<<endl;
   else Info<<"ITERATION LIMIT: P008 remains unqualified."<<endl;break;}
 }
 Info<<"End"<<endl;return converged?0:2;
}
