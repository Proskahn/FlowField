#pragma once
#include "Common.H"
#include "ChargeNewton.H"
inline void Model::charge(scalar voltage)
{
 label nc=mesh.nCells(),nn=0;labelList sn(nc,-1),en(nc,-1);scalarField sigmaS(nc,0),sigmaE(nc,0),pref(nc,0),beta(nc,0);
 forAll(mat,c){if(electronic(c))sn[c]=nn++;if(ionic(c))en[c]=nn++;
  sigmaS[c]=par(mat[c]==0||mat[c]==8?"electronic_conductivity_bp":cl(c)?"electronic_conductivity_cl":"electronic_conductivity_gdl");
  if(ionic(c)){sigmaE[c]=(.5139*lambda[c]-.326)*std::exp(1268*(1/303.-1/T[c]))*(mat[c]==4?1:std::pow(par("ionomer_fraction"),1.5));
   if(sigmaE[c]<=0)FatalErrorInFunction<<"Nonpositive proton conductivity; membrane hydration out of range"<<exit(FatalError);}
  if(cl(c)){scalar cref=par("reference_concentration");
   scalar base=par(anode(c)?"exchange_current_anode":"exchange_current_cathode")*std::exp(par(anode(c)?"activation_energy_anode":"activation_energy_cathode")/RGAS*(1/par("kinetic_reference_temperature")-1/T[c]));
   pref[c]=base*std::pow(max(1-s[c],1e-12),1.5)*std::pow(max(ct[c]*yr[c]/cref,1e-16),par("concentration_exponent"));
   beta[c]=par("transfer_coefficient")*FARADAY/(RGAS*T[c]);}
 }
 struct Edge{label a,b,o,n,f;scalar k;};std::vector<Edge> edges;
 Vec diag=Vec::Zero(nn),injection=Vec::Zero(nn),x(nn);
 forAll(mat,c){if(sn[c]>=0)x[sn[c]]=phiS[c];if(en[c]>=0)x[en[c]]=phiE[c];}
 for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];label o=f.o,n=f.n;
  if(n>=0){for(int type=0;type<2;type++){const auto& node=type?en:sn;const auto& k=type?sigmaE:sigmaS;
   if(node[o]>=0&&node[n]>=0)edges.push_back({node[o],node[n],o,n,type?i:-1,f.area/(f.dO/k[o]+f.dN/k[n])});}}
  else if(f.kind==3&&sn[o]>=0){word name=mesh.boundary()[f.patch].name();
   if(name=="anodeTerminal"||name=="cathodeTerminal"){scalar k=sigmaS[o]*f.area/f.dO;diag[sn[o]]+=k;injection[sn[o]]+=k*(name=="cathodeTerminal"?voltage:0);}}
 }
 auto assemble=[&](const Vec& v,Vec& residual,std::vector<Triplet>* jac){
  residual=diag.cwiseProduct(v)-injection;
  if(jac)for(label i=0;i<nn;i++)jac->emplace_back(i,i,diag[i]);
  auto link=[&](label a,label b,scalar q,scalar d){residual[a]+=q;residual[b]-=q;
   if(jac){jac->emplace_back(a,a,d);jac->emplace_back(b,b,d);jac->emplace_back(a,b,-d);jac->emplace_back(b,a,-d);}};
  for(const auto& e:edges)link(e.a,e.b,e.k*(v[e.a]-v[e.b]),e.k);
  forAll(mat,c)if(cl(c)){scalar eta=v[sn[c]]-v[en[c]]-(anode(c)?0:par("open_circuit_voltage"));
   scalar arg=min(max(beta[c]*eta,-50.),50.);
   link(sn[c],en[c],p011Charge::rate(pref[c],arg)*mesh.V()[c],p011Charge::slope(pref[c],beta[c],arg)*mesh.V()[c]);}
 };
 const scalar currentScale=par("reference_current_density")*par("reactionArea");
 try{
  auto result=p011Charge::solve(x,currentScale,assemble,[&](const SparseMatrix& A,const Vec& rhs)->Vec{
   LinearBlock lu(A,direct());return lu.solve(rhs);
  });
  chargeResidual=result.residual;
  if(result.stagnated)Info<<"P011: charge Newton stagnated within acceptance; residual="<<chargeResidual<<endl;
 }catch(const std::exception& error){
  FatalErrorInFunction<<error.what()<<exit(FatalError);
 }
 // thermoneutral_voltage is liquid-product HHV. Reaction water uses the
 // vapor enthalpy reference; condensation latent heat is added in thermal().
 iv.primitiveFieldRef()=0;heat.primitiveFieldRef()=0;ionicQ=0;
 forAll(mat,c){if(sn[c]>=0)phiS[c]=x[sn[c]];if(en[c]>=0)phiE[c]=x[en[c]];
  if(cl(c)){scalar eta=phiS[c]-phiE[c]-(anode(c)?0:par("open_circuit_voltage"));
   iv[c]=p011Charge::rate(pref[c],min(max(beta[c]*eta,-50.),50.));heat[c]=iv[c]*eta+mag(iv[c])*.5*(par("thermoneutral_voltage")-par("open_circuit_voltage")-MW*par("latent_heat")/(2*FARADAY));}}
 for(const auto& e:edges){scalar d=x[e.a]-x[e.b],power=e.k*d*d;heat[e.o]+=.5*power/mesh.V()[e.o];heat[e.n]+=.5*power/mesh.V()[e.n];if(e.f>=0)ionicQ[e.f]=e.k*d;}
 for(const auto& f:faces)if(f.n<0&&f.kind==3&&sn[f.o]>=0){word name=mesh.boundary()[f.patch].name();
  if(name=="anodeTerminal"||name=="cathodeTerminal"){scalar v=name=="cathodeTerminal"?voltage:0;
   heat[f.o]+=sigmaS[f.o]*f.area/f.dO*sqr(phiS[f.o]-v)/mesh.V()[f.o];phiS.boundaryFieldRef()[f.patch][f.index]=v;}}
 phiS.correctBoundaryConditions();phiE.correctBoundaryConditions();iv.correctBoundaryConditions();heat.correctBoundaryConditions();
}
