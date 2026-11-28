#pragma once
#include "Common.H"
inline void Model::water()
{
 label n=mesh.nCells();scalarField diff(n,0),source(n,0),sink(n,0),bc(faces.size(),0),zero(faces.size(),0),drag(faces.size(),0),target(n,0);
 forAll(mat,c)if(ionic(c)){
  diff[c]=cap(c)*par("membrane_diffusivity")*std::exp(2416*(1/303.-1/T[c]));
  if(cl(c)){scalar a=ct[c]*yw[c]/(psat(T[c])/(RGAS*T[c]));target[c]=waterEq(a+2*s[c]);
   sink[c]=cap(c)*par("sorption_rate");source[c]=sink[c]*target[c]+(mat[c]==3?-iv[c]/(2*FARADAY):0);}}
 for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];if(f.n>=0&&ionic(f.o)&&ionic(f.n))
  drag[i]=2.5*(lambda[f.o]+lambda[f.n])/(2*22)*ionicQ[i]/FARADAY;}
 auto membrane=transport(ionNode,nion,lambda,diff,zero,source,sink,bc,par("scalar_relaxation"),false,false,&drag);
 membraneResidual=membrane.residual;waterQ=membrane.flux;
 forAll(mat,c)sorbRate[c]=cl(c)?sink[c]*(target[c]-lambda[c]):0;
 // Liquid Darcy pressure is p+pc in hydrophobic pores; channel water follows
 // the carrier velocity with a declared dispersion coefficient (paper Eq. 13).
 scalar molLiquid=par("liquid_density")/MW;std::vector<Triplet> trip;Vec diagonal=Vec::Zero(ng),rhs=Vec::Zero(ng);
 scalarField kO(faces.size(),0),kN(faces.size(),0),offset(faces.size(),0),adv(faces.size(),0),evap(n,0),cond(n,0);
 forAll(mat,c)if(gas(c)){scalar excess=ct[c]*yw[c]-psat(T[c])/(RGAS*T[c]);
  scalar rate=par("phase_change_rate")*eps[c];cond[c]=rate*max(excess,0.);
  evap[c]=rate*max(-excess,0.)/(s[c]+1e-4);scalar storage=eps[c]*molLiquid/par("liquid_pseudo_time");
  diagonal[gasNode[c]]+=(evap[c]+storage)*mesh.V()[c];rhs[gasNode[c]]+=(cond[c]+storage*s[c])*mesh.V()[c];}
 for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];if(!gas(f.o)||(f.n>=0&&!gas(f.n))||f.kind==3||f.kind==1)continue;
  label a=gasNode[f.o];if(f.n<0){adv[i]=max(gasQ[i],0.)/faceC(f)*molLiquid/max(1-s[f.o],1e-6);diagonal[a]+=adv[i];continue;}
  label b=gasNode[f.n];
  if(channel(f.o)&&channel(f.n)){
   adv[i]=gasQ[i]/faceC(f)*molLiquid/max(1-(gasQ[i]>=0?s[f.o]:s[f.n]),1e-6);
   kO[i]=kN[i]=par("channel_liquid_dispersion")*molLiquid*f.area/(f.dO+f.dN);
  }else{
   scalar drive=p[f.o]-p[f.n]+pc[f.o]-pc[f.n];label donor=drive>=0?f.o:f.n;
   scalar rel=std::pow(max(s[donor],0.),3);
   scalar bo=channel(f.o)?rAU[f.o]*mu[f.o]/par("liquid_viscosity"):K[f.o]/(sqr(eps[f.o])*par("liquid_viscosity"));
   scalar bn=channel(f.n)?rAU[f.n]*mu[f.n]/par("liquid_viscosity"):K[f.n]/(sqr(eps[f.n])*par("liquid_viscosity"));
   scalar mob=molLiquid*f.area*rel/(f.dO/bo+f.dN/bn);
   kO[i]=mob*pcSlope[f.o];kN[i]=mob*pcSlope[f.n];
   offset[i]=mob*(drive-pcSlope[f.o]*s[f.o]+pcSlope[f.n]*s[f.n]);
  }
  diagonal[a]+=kO[i]+max(adv[i],0.);diagonal[b]+=kN[i]+max(-adv[i],0.);
  trip.emplace_back(a,b,-kN[i]+min(adv[i],0.));trip.emplace_back(b,a,-kO[i]-max(adv[i],0.));
  rhs[a]-=offset[i];rhs[b]+=offset[i];
 }
 for(label a=0;a<ng;a++)trip.emplace_back(a,a,diagonal[a]);SparseMatrix A(ng,ng);A.setFromTriplets(trip.begin(),trip.end());LinearBlock solve(A,direct());Vec answer=solve.solve(rhs),actual(ng);
 scalar step=par("scalar_relaxation");
 forAll(mat,c)if(gas(c)){scalar ds=answer[gasNode[c]]-s[c];
  // Compare first: dividing a finite bound by a subnormal update would
  // overflow even though that update needs no limiting.
  if(-step*ds>.8*max(s[c],1e-14))step=.8*max(s[c],1e-14)/(-ds);
  if(step*ds>.8*(.98-s[c]))step=.8*(.98-s[c])/ds;}
 forAll(mat,c)if(gas(c)){s[c]+=step*(answer[gasNode[c]]-s[c]);actual[gasNode[c]]=s[c];phaseRate[c]=cond[c]-evap[c]*s[c];}
 liquidResidual=(A*actual-rhs).lpNorm<1>()/max(par("reference_current_density")*par("reactionArea")/(2*FARADAY),1e-14);
 for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];liquidQ[i]=0;if(!gas(f.o)||(f.n>=0&&!gas(f.n)))continue;
  if(f.n>=0)liquidQ[i]=kO[i]*s[f.o]-kN[i]*s[f.n]+adv[i]*(adv[i]>=0?s[f.o]:s[f.n])+offset[i];
  else if(f.kind==2)liquidQ[i]=adv[i]*s[f.o];}
 s.correctBoundaryConditions();
}

inline void Model::thermal()
{
 label n=mesh.nCells();labelList node(n);scalarField k(n,0),source(n,0),sink(n,0),adv(faces.size(),0),bc(faces.size(),par("temperature_K"));
 const scalar solidK[9]={100,0,1.5,1,.2,1,1.5,0,100};
 forAll(mat,c){node[c]=c;scalar kg=anode(c)?.19:.028;
  k[c]=gas(c)?(1-eps[c])*solidK[mat[c]]+eps[c]*((1-s[c])*kg+s[c]*par("liquid_conductivity")):solidK[mat[c]];
  source[c]=heat[c]+phaseRate[c]*MW*par("latent_heat");}
 for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];label donor=gasQ[i]>=0||f.n<0?f.o:f.n;
  scalar cp=29.+5*yw[donor];adv[i]=gasQ[i]*cp+liquidQ[i]*MW*4180.;
  // Sensible enthalpy reference at imposed feed/wall temperature.
  source[f.o]+=adv[i]*par("temperature_K")/mesh.V()[f.o];if(f.n>=0)source[f.n]-=adv[i]*par("temperature_K")/mesh.V()[f.n];}
 auto result=transport(node,n,T,k,adv,source,sink,bc,par("scalar_relaxation"),true);
 heatQ=result.flux;scalarField net(n,0);
 for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];heatQ[i]-=adv[i]*par("temperature_K");net[f.o]+=heatQ[i];if(f.n>=0)net[f.n]-=heatQ[i];}
 scalar numerator=0,scale=0;forAll(mat,c){scalar q=(heat[c]+phaseRate[c]*MW*par("latent_heat"))*mesh.V()[c];numerator+=mag(net[c]-q);scale+=mag(q);}
 energyResidual=numerator/max(scale,1e-12);
}
