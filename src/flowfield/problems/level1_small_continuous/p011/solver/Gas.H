#pragma once
#include "Common.H"
inline void Model::flow()
{
 label nc=mesh.nCells(),nch=0;labelList node(nc,-1);forAll(mat,c)if(channel(c))node[c]=nch++;
 std::vector<Triplet> trip;Vec diagonal=Vec::Zero(nch);std::array<Vec,3> rhs;
 for(auto& b:rhs)b=Vec::Zero(nch);
 auto inletU=[&](const Face& f){return -f.normal*feed(f.o)/(par("outlet_pressure")/(RGAS*par("temperature_K"))*inletArea(f.patch));};
 for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];bool co=channel(f.o),cn=f.n>=0&&channel(f.n);if(!co&&!cn)continue;
  scalar mass=gasQ[i]/faceC(f)*(gasQ[i]>=0||f.n<0?rho[f.o]:rho[f.n]);
  if(co&&cn){label a=node[f.o],b=node[f.n];scalar k=f.area/(f.dO/mu[f.o]+f.dN/mu[f.n]);
   diagonal[a]+=k+max(mass,0.);diagonal[b]+=k+max(-mass,0.);trip.emplace_back(a,b,-k+min(mass,0.));trip.emplace_back(b,a,-k-max(mass,0.));
  }else{label c=co?f.o:f.n,a=node[c];scalar d=co?f.dO:f.dN,q=co?mass:-mass;
   if(f.n<0&&f.kind==2){diagonal[a]+=max(q,0.);continue;}
   vector bc=vector::zero;if(f.n<0&&f.kind==1)bc=inletU(f);
   else{label other=co?f.n:f.o;if(other>=0&&gas(other))bc=U[other];}
   scalar k=mu[c]*f.area/d;diagonal[a]+=k+max(q,0.);for(int dim=0;dim<3;dim++)rhs[dim][a]+=(k-min(q,0.))*bc[dim];
  }
 }
 for(label a=0;a<nch;a++)trip.emplace_back(a,a,diagonal[a]);SparseMatrix A(nch,nch);A.setFromTriplets(trip.begin(),trip.end());
 scalar relax=par("velocity_relaxation");SparseMatrix relaxed=A;for(label a=0;a<nch;a++)relaxed.coeffRef(a,a)=diagonal[a]/relax;
 LinearBlock momentum(relaxed,direct());H=vector::zero;
 for(int d=0;d<3;d++){Vec b=rhs[d],old(nch);forAll(mat,c)if(channel(c)){label a=node[c];old[a]=U[c][d];b[a]+=(1/relax-1)*diagonal[a]*old[a]-mesh.V()[c]*gradP[c][d];}
  Vec u=momentum.solve(b),off=A*u-diagonal.cwiseProduct(u);
  forAll(mat,c)if(channel(c)){label a=node[c];rAU[c]=mesh.V()[c]*relax/diagonal[a];H[c][d]=(rhs[d][a]+(1/relax-1)*diagonal[a]*old[a]-off[a])*relax/diagonal[a];}}
 forAll(mat,c)if(gas(c)&&!channel(c))rAU[c]=K[c]*std::pow(max(1-s[c],1e-6),3)/(sqr(eps[c])*mu[c]);
 trip.clear();Vec pd=Vec::Zero(ng),pr=Vec::Zero(ng);scalarField coeff(faces.size(),0),hf(faces.size(),0);
 forAll(mat,c)if(gas(c)){scalar reaction=cl(c)?-mag(iv[c])/((anode(c)?2:4)*FARADAY):0;
  pr[gasNode[c]]=(reaction-phaseRate[c]-sorbRate[c])*mesh.V()[c];}
 for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];if(!gas(f.o)||(f.n>=0&&!gas(f.n))||f.kind==3)continue;
  label a=gasNode[f.o];if(f.kind==1){pr[a]+=feed(f.o)*f.area/inletArea(f.patch);continue;}
  scalar ro=f.dO/rAU[f.o],rn=f.n>=0?f.dN/rAU[f.n]:0;
  coeff[i]=faceC(f)*f.area/(ro+rn);hf[i]=coeff[i]*(ro*(H[f.o]&f.normal)+(f.n>=0?rn*(H[f.n]&f.normal):0));
  pd[a]+=coeff[i];pr[a]-=hf[i];if(f.n>=0){label b=gasNode[f.n];pd[b]+=coeff[i];pr[b]+=hf[i];trip.emplace_back(a,b,-coeff[i]);trip.emplace_back(b,a,-coeff[i]);}}
 for(label a=0;a<ng;a++)trip.emplace_back(a,a,pd[a]);SparseMatrix P(ng,ng);P.setFromTriplets(trip.begin(),trip.end());LinearBlock pressure(P,direct());Vec answer=pressure.solve(pr);
 forAll(mat,c)if(gas(c))p[c]=answer[gasNode[c]];
 gasResidual=(P*answer-pr).lpNorm<1>()/max(pr.lpNorm<1>(),1e-12);gradP=vector::zero;vectorField qv(nc,vector::zero);
 for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];gasQ[i]=0;
  bool active=gas(f.o)&&(f.n<0||gas(f.n));
  if(active){if(f.kind==1)gasQ[i]=-feed(f.o)*f.area/inletArea(f.patch);
   else if(f.kind!=3)gasQ[i]=coeff[i]*(p[f.o]-(f.n>=0?p[f.n]:0))+hf[i];}
  for(int side=0;side<2;side++){label c=side?f.n:f.o;if(c<0||!gas(c))continue;
   scalar sign=side?-1:1,d=side?f.dN:f.dO,pf=p[c];
   if(active&&f.kind!=3)pf=p[c]+d*((H[c]&(sign*f.normal))-sign*gasQ[i]/(faceC(f)*f.area))/rAU[c];
   gradP[c]+=sign*f.area*f.normal*pf/mesh.V()[c];qv[c]+=gasQ[i]/faceC(f)*d*f.normal/mesh.V()[c];
   if(f.patch>=0)p.boundaryFieldRef()[f.patch][f.index]=pf;
  }
 }
 forAll(mat,c)U[c]=channel(c)?H[c]-rAU[c]*gradP[c]:gas(c)?qv[c]:vector::zero;
 scalar numerator=0,denominator=0;
 for(int d=0;d<3;d++){Vec u(nch),force(nch);forAll(mat,c)if(channel(c)){u[node[c]]=U[c][d];force[node[c]]=mesh.V()[c]*gradP[c][d];}
  numerator+=(A*u-rhs[d]+force).lpNorm<1>();denominator+=rhs[d].lpNorm<1>()+force.lpNorm<1>()+diagonal.cwiseProduct(u).lpNorm<1>();}
 momentumResidual=numerator/max(denominator,1e-15);U.correctBoundaryConditions();
 for(const auto& f:faces)if(f.kind==1)U.boundaryFieldRef()[f.patch][f.index]=inletU(f);
}

inline void Model::species()
{
 label n=mesh.nCells();scalarField diff(n,0),source(n,0),sink(n,0),bc(faces.size(),0);
 forAll(mat,c)if(gas(c)){
  scalar D=diffusivity(c,false)*std::pow(T[c]/par("diffusion_reference_temperature"),1.5)*101325/par("outlet_pressure");
  diff[c]=ct[c]*D*std::pow(eps[c]*max(1-s[c],1e-6),1.5);
  if(cl(c))sink[c]=mag(iv[c])/((anode(c)?2:4)*FARADAY*max(yr[c],1e-12));}
 for(label i=0;i<label(faces.size());i++)bc[i]=inletReact(faces[i].o);
 auto r=transport(gasNode,ng,yr,diff,gasQ,source,sink,bc,par("scalar_relaxation"));reactQ=r.flux;speciesResidual=r.residual;
 forAll(mat,c)if(gas(c)){
  // The binary H2/H2O pair must share one mutual diffusivity so its two
  // diffusive molar fluxes cancel exactly. N2 closes the cathode mixture.
  scalar D=diffusivity(c,true)*std::pow(T[c]/par("diffusion_reference_temperature"),1.5)*101325/par("outlet_pressure");
  diff[c]=ct[c]*D*std::pow(eps[c]*max(1-s[c],1e-6),1.5);
  scalar sat=psat(T[c])/(RGAS*T[c]),a=ct[c]*yw[c]/sat+2*s[c];
  scalar slope=a<=1?17.81-79.7*a+108*a*a:1.4;
  sink[c]=par("phase_change_rate")*eps[c]*ct[c]*(ct[c]*yw[c]>sat?1:s[c]/(s[c]+1e-4));
  if(cl(c))sink[c]+=cap(c)*par("sorption_rate")*slope*ct[c]/sat;
  // First-order implicit tangent of local vapor sinks. This term cancels
  // exactly at the steady fixed point; it is not a changed physical rate.
  source[c]=-phaseRate[c]-sorbRate[c]+sink[c]*yw[c];}
 forAll(bc,i)bc[i]=inletWater();
 auto w=transport(gasNode,ng,yw,diff,gasQ,source,sink,bc,par("scalar_relaxation"));vaporQ=w.flux;speciesResidual=max(speciesResidual,w.residual);
 // Binary anode mixture: complementary fractions and canceling diffusion.
 forAll(mat,c)if(gas(c)&&anode(c))yr[c]=1-yw[c];
 for(label i=0;i<label(faces.size());i++)if(anode(faces[i].o))reactQ[i]=gasQ[i]-vaporQ[i];
 yr.correctBoundaryConditions();
}
