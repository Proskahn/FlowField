/* Li et al. tapered-channel PEMFC reconstruction. Steady iterations, not physical time.
 * OpenFOAM mesh/fields; conservative finite volumes and checked Eigen solves.
 */
#include "Charge.H"
#include "Gas.H"
#include "WaterHeat.H"
#include <fstream>
#include <iomanip>
int main(int argc,char* argv[])
{
 #include "setRootCase.H"
 #include "createTime.H"
 #include "createMesh.H"
 if(Pstream::parRun())FatalErrorInFunction<<"P011 is serial; MPI has not been implemented."<<exit(FatalError);
 IOdictionary props(IOobject("modelProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
 #define SCALAR(n) volScalarField n(IOobject(#n,runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh)
 SCALAR(material);SCALAR(p);SCALAR(T);SCALAR(s);SCALAR(lambda);SCALAR(yr);SCALAR(yw);SCALAR(phiS);SCALAR(phiE);
 volVectorField U(IOobject("U",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
 auto aux=[&](const word& name,const dimensionSet& dim){return volScalarField(IOobject(name,runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,dimensionedScalar("zero",dim,0),"zeroGradient");};
 auto iVol=aux("iVol",dimCurrent/dimVolume),heatSource=aux("heatSource",dimPower/dimVolume);
 dimensionSet molRate(0,-3,-1,0,1,0,0),molFlux(0,0,-1,0,1,0,0);
 auto srcGas=aux("srcGas",molRate),srcReact=aux("srcReact",molRate),srcVapor=aux("srcVapor",molRate),srcLiquid=aux("srcLiquid",molRate),srcMembrane=aux("srcMembrane",molRate),srcHeat=aux("srcHeat",dimPower/dimVolume);
 auto surface=[&](const word& name,const dimensionSet& dim){return surfaceScalarField(IOobject(name,runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,dimensionedScalar("zero",dim,0));};
 auto phiGas=surface("phiGas",molFlux),phiReact=surface("phiReact",molFlux),phiVapor=surface("phiVapor",molFlux),phiLiquid=surface("phiLiquid",molFlux),phiWater=surface("phiWater",molFlux),phiHeat=surface("phiHeat",dimPower);
 Model model(mesh,props,material,p,T,s,lambda,yr,yw,phiS,phiE,iVol,heatSource,U);
 auto par=[&](const word& name){return props.get<scalar>(name);};
 fileName out=runTime.path()/"postProcessing"/"p011";mkDir(out);std::ofstream csv((out/"history.csv").c_str());csv<<std::setprecision(14);
 csv<<"iteration,voltage_V,current_anode_A,current_cathode_A,net_power_W,pumping_power_W,pressure_drop_Pa,inlet_flow_m3_s,field_change,gas_balance,species_balance,liquid_balance,membrane_balance,charge_residual,momentum_residual,energy_balance,min_T,max_T,min_s,max_s,min_lambda,max_lambda,min_yr,min_yw,max_species_sum\n";
 scalar voltage=par("open_circuit_voltage");bool converged=false;label stable=0;
 Info<<"P011: "<<mesh.nCells()<<" cells; steady 3D PEMFC. First build/solve diagnostics every 10 iterations."<<endl;
 model.flow(); // Establish carrier mobility before the first water update.
 while(runTime.loop()){
  label iter=runTime.timeIndex();voltage=max(par("cell_voltage"),par("open_circuit_voltage")-(par("open_circuit_voltage")-par("cell_voltage"))*iter/par("voltage_ramp_iterations"));
  scalarField oldS(s.primitiveField()),oldT(T.primitiveField()),oldL(lambda.primitiveField()),oldR(yr.primitiveField()),oldW(yw.primitiveField()),oldI(iVol.primitiveField());vectorField oldU(U.primitiveField());
  model.properties();
  if(iter<=3||iter%10==0)Info<<"P011: iteration="<<iter<<" charge block"<<endl;
  model.charge(voltage);
  if(iter<=3||iter%10==0)Info<<"P011: iteration="<<iter<<" water block"<<endl;
  model.water();
  if(iter<=3||iter%10==0)Info<<"P011: iteration="<<iter<<" flow block"<<endl;
  model.flow();
  if(iter<=3||iter%10==0)Info<<"P011: iteration="<<iter<<" species block"<<endl;
  model.species();
  if(iter<=3||iter%10==0)Info<<"P011: iteration="<<iter<<" heat block"<<endl;
  model.thermal();model.properties();
  scalar Ia=0,Ic=0,change=0,maximumSum=0,minR=1,minW=1;
  forAll(model.mat,c){srcReact[c]=model.cl(c)?-mag(iVol[c])/((model.anode(c)?2:4)*FARADAY):0;
   srcVapor[c]=-model.phaseRate[c]-model.sorbRate[c];srcGas[c]=srcReact[c]+srcVapor[c];srcLiquid[c]=model.phaseRate[c];
   srcMembrane[c]=(model.mat[c]==3?-iVol[c]/(2*FARADAY):0)+model.sorbRate[c];srcHeat[c]=heatSource[c]+model.phaseRate[c]*MW*par("latent_heat");
   if(model.mat[c]==5)Ia+=iVol[c]*mesh.V()[c];if(model.mat[c]==3)Ic-=iVol[c]*mesh.V()[c];
   change=max(change,mag(T[c]-oldT[c])/max(T[c],1.));
   if(model.gas(c)){minR=min(minR,yr[c]);minW=min(minW,yw[c]);maximumSum=max(maximumSum,yr[c]+yw[c]);
    change=max(change,mag(s[c]-oldS[c]));change=max(change,mag(yr[c]-oldR[c]));change=max(change,mag(yw[c]-oldW[c]));}
   if(model.ionic(c))change=max(change,mag(lambda[c]-oldL[c])/max(lambda[c],1.));
   if(model.cl(c))change=max(change,mag(iVol[c]-oldI[c])/max(mag(iVol[c]),1e5));
   if(model.channel(c))change=max(change,mag(U[c]-oldU[c])/max(mag(U[c]),.1));
  }
  if(!std::isfinite(change)||minR< -1e-8||minW< -1e-8||maximumSum>1.00001||min(s.primitiveField())< -1e-10||max(s.primitiveField())>=.98||min(T.primitiveField())<273.15||max(T.primitiveField())>373.15)
   FatalErrorInFunction<<"Fields left physical bounds: reactant="<<minR<<"; vapor="<<minW<<"; species sum="<<maximumSum<<"; saturation="<<min(s.primitiveField())<<","<<max(s.primitiveField())<<exit(FatalError);
  scalar feedScale=0,waterScale=max(Ic/(2*FARADAY),1e-10),heatScale=max(sum(mag(srcHeat.primitiveField())*mesh.V()),1e-10);
  for(const auto& f:model.faces)if(f.kind==1)feedScale+=model.feed(f.o)*f.area/model.inletArea(f.patch);
  auto balance=[&](const scalarField& flux,const volScalarField& source,scalar scale){scalarField residual=-source.primitiveField()*mesh.V();
   for(label f=0;f<label(model.faces.size());f++){const auto& fc=model.faces[f];residual[fc.o]+=flux[f];if(fc.n>=0)residual[fc.n]-=flux[f];}
   return sum(mag(residual))/scale;};
  scalar gasError=balance(model.gasQ,srcGas,feedScale),speciesError=max(balance(model.reactQ,srcReact,feedScale),balance(model.vaporQ,srcVapor,waterScale));
  scalar liquidError=balance(model.liquidQ,srcLiquid,waterScale),membraneError=balance(model.waterQ,srcMembrane,waterScale),energyError=balance(model.heatQ,srcHeat,heatScale);
  scalar pin=0,pout=0,ai=0,ao=0,flow=0;
  for(label f=0;f<label(model.faces.size());f++){const auto& fc=model.faces[f];if(fc.patch<0||model.anode(fc.o))continue;
   if(fc.kind==1){pin+=p.boundaryField()[fc.patch][fc.index]*fc.area;ai+=fc.area;flow-=model.gasQ[f]/(par("outlet_pressure")/(RGAS*par("temperature_K")));}
   if(fc.kind==2){pout+=p.boundaryField()[fc.patch][fc.index]*fc.area;ao+=fc.area;}}
  scalar dp=pin/ai-pout/ao,pump=dp*flow,net=voltage*Ic-pump;
  csv<<iter<<','<<voltage<<','<<Ia<<','<<Ic<<','<<net<<','<<pump<<','<<dp<<','<<flow<<','<<change<<','<<gasError<<','<<speciesError<<','<<liquidError<<','<<membraneError<<','<<model.chargeResidual<<','<<model.momentumResidual<<','<<energyError<<','<<min(T.primitiveField())<<','<<max(T.primitiveField())<<','<<min(s.primitiveField())<<','<<max(s.primitiveField())<<','<<min(lambda.primitiveField())<<','<<max(lambda.primitiveField())<<','<<minR<<','<<minW<<','<<maximumSum<<'\n';csv.flush();
  if(iter==1||iter%10==0)Info<<"P011: iteration="<<iter<<"; V="<<voltage<<"; I="<<Ic<<" A; net power="<<net<<" W; change="<<change<<"; balances gas/species/liquid/membrane/energy="<<gasError<<","<<speciesError<<","<<liquidError<<","<<membraneError<<","<<energyError<<"; momentum="<<model.momentumResidual<<endl;
  scalar worst=max(max(max(gasError,speciesError),max(liquidError,membraneError)),max(energyError,model.momentumResidual));
  if(voltage==par("cell_voltage")&&iter>=par("minimum_iterations")&&change<par("field_tolerance")&&worst<par("balance_tolerance")&&model.chargeResidual<1e-7&&mag(Ia-Ic)/max(Ic,1e-8)<1e-7)stable++;else stable=0;
  if(stable>=5)converged=true;
  model.writeFlux(phiGas,model.gasQ);model.writeFlux(phiReact,model.reactQ);model.writeFlux(phiVapor,model.vaporQ);model.writeFlux(phiLiquid,model.liquidQ);model.writeFlux(phiWater,model.waterQ);model.writeFlux(phiHeat,model.heatQ);
  if(converged||iter>=par("iterations")){runTime.writeNow();break;}runTime.write();
 }
 std::ofstream status((out/"status.json").c_str());status<<"{\"converged\":"<<(converged?"true":"false")<<",\"iteration\":"<<runTime.value()<<",\"cells\":"<<mesh.nCells()<<",\"processes\":1,\"voltage_V\":"<<voltage<<"}\n";
 if(converged)Info<<"CONVERGED: P011 full coupled model"<<endl;
 Info<<"End"<<endl;return converged?0:2;
}
