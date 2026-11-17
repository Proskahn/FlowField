// Conservative orthogonal finite-volume operators. Serial Eigen linear algebra;
// OpenFOAM owns the mesh, fields, boundary data and ParaView output.
#include "fvCFD.H"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <vector>
#include <array>
#include <cmath>
using namespace Foam;
using SparseMatrix=Eigen::SparseMatrix<double>;
using Vec=Eigen::VectorXd;
using Triplet=Eigen::Triplet<double>;

// Equilibrated Krylov solves avoid repeated 3D sparse-direct fill-in. Each
// solution is checked against its scaled algebraic residual; direct LU is a
// bounded fallback, not an alternative physical model or relaxed criterion.
class LinearBlock
{
 SparseMatrix matrix_,scaled_;Vec scale_;
 Eigen::BiCGSTAB<SparseMatrix,Eigen::IncompleteLUT<double>> iterative_;
 Eigen::SparseLU<SparseMatrix> direct_;
 bool useDirect_;
public:
 LinearBlock(const SparseMatrix& A,bool direct=false):matrix_(A),scaled_(A),scale_(Vec::Zero(A.rows())),useDirect_(direct)
 {
  if(useDirect_){direct_.compute(matrix_);return;}
  for(int j=0;j<A.outerSize();j++)for(SparseMatrix::InnerIterator e(A,j);e;++e)scale_[e.row()]=max(scale_[e.row()],mag(e.value()));
  for(int i=0;i<scale_.size();i++)scale_[i]=max(scale_[i],1e-30);
  for(int j=0;j<scaled_.outerSize();j++)for(SparseMatrix::InnerIterator e(scaled_,j);e;++e)e.valueRef()/=scale_[e.row()];
  iterative_.preconditioner().setDroptol(1e-4);iterative_.preconditioner().setFillfactor(12);
  iterative_.setTolerance(1e-11);iterative_.setMaxIterations(3000);iterative_.compute(scaled_);
 }
 Vec solve(const Vec& rhs)
 {
  if(rhs.norm()<1e-280)return Vec::Zero(rhs.size());
  if(!useDirect_){Vec b=rhs.cwiseQuotient(scale_);Vec x=iterative_.solve(b);
   if(iterative_.info()==Eigen::Success&&x.allFinite()&&(scaled_*x-b).norm()<1e-9*b.norm())return x;
   Info<<"P008 linear algebra: using direct fallback for "<<rhs.size()<<" unknowns"<<endl;
   direct_.compute(matrix_);useDirect_=true;}
  if(direct_.info()!=Eigen::Success)FatalErrorInFunction<<"P008 linear factorization failed"<<exit(FatalError);
  Vec x=direct_.solve(rhs);if(direct_.info()!=Eigen::Success||!x.allFinite())FatalErrorInFunction<<"P008 linear solve failed"<<exit(FatalError);
  return x;
 }
};

volScalarField auxiliary(const word& name,const fvMesh& mesh,const dimensionSet& dims,
 scalar value=0,IOobject::writeOption option=IOobject::NO_WRITE)
{
 return volScalarField(IOobject(name,mesh.time().timeName(),mesh,IOobject::NO_READ,option),
 mesh,dimensionedScalar(name,dims,value),"zeroGradient");
}
struct Face {label o,n,patch,index;scalar area,dO,dN;vector normal;int kind;};
class Transport
{
public:
 const fvMesh& mesh;const labelList& mat;const dictionary& props;
 std::vector<Face> faces;labelList node,channelNode;label nf=0,nch=0;
 scalarField porosity,permeability,rhoGas,muGas,pc,dpc,rAU,muMix,rhoMix;
 vectorField H,gradient;scalarField qL,qG,pLface,pGface;
 scalar liquidResidual=GREAT,gasResidual=GREAT,momentumResidual=GREAT,energyResidual=GREAT;
 scalar dt=.001;
 Transport(const fvMesh& m,const labelList& materials,const dictionary& properties)
 :mesh(m),mat(materials),props(properties),node(m.nCells(),-1),channelNode(m.nCells(),-1),
 porosity(m.nCells(),0),permeability(m.nCells(),0),rhoGas(m.nCells(),0),muGas(m.nCells(),0),
 pc(m.nCells(),0),dpc(m.nCells(),0),rAU(m.nCells(),0),muMix(m.nCells(),0),rhoMix(m.nCells(),0),
 H(m.nCells(),vector::zero),gradient(m.nCells(),vector::zero)
 {
  forAll(mat,c){if(fluid(c))node[c]=nf++;if(channel(c))channelNode[c]=nch++;
   porosity[c]=channel(c)?1:(mat[c]==2||mat[c]==6)?par("porosity_ptl"):(mat[c]==3||mat[c]==5)?par("porosity_cl"):0;
   permeability[c]=(mat[c]==2||mat[c]==6)?par("permeability_ptl"):par("permeability_cl");
   rhoGas[c]=par("outlet_pressure")*(mat[c]>=5?.0319988:.00201588)/(8.314462618*par("inlet_temperature"));
   muGas[c]=par(mat[c]>=5?"oxygen_viscosity":"hydrogen_viscosity");}
  forAll(m.neighbour(),f){label o=m.owner()[f],n=m.neighbour()[f];scalar a=m.magSf()[f];
   faces.push_back({o,n,-1,f,a,mag(m.Cf()[f]-m.C()[o]),mag(m.Cf()[f]-m.C()[n]),m.Sf()[f]/a,0});}
  forAll(m.boundary(),p){const auto& patch=m.boundary()[p];
   int kind=patch.name().ends_with("Inlet")?1:patch.name().ends_with("Outlet")?2:3;
   forAll(patch,f){label c=patch.faceCells()[f];scalar a=patch.magSf()[f];
    faces.push_back({c,-1,p,f,a,1/patch.deltaCoeffs()[f],0,patch.Sf()[f]/a,kind});}}
  label sz=faces.size();qL.setSize(sz,0);qG.setSize(sz,0);pLface.setSize(sz,0);pGface.setSize(sz,0);
  for(label f=0;f<sz;f++)if(faces[f].kind==1)qL[f]=-feed()*faces[f].area/inletArea(faces[f].patch);
 }
 scalar par(const word& key)const{return props.get<scalar>(key);}
 bool fluid(label c)const{return mat[c]==1||mat[c]==2||mat[c]==3||mat[c]==5||mat[c]==6||mat[c]==7;}
 bool channel(label c)const{return mat[c]==1||mat[c]==7;}
 scalar feed()const{return par("inlet_flow_ml_min")*1e-6/60;}
 scalar inletArea(label p)const{return sum(mesh.magSf().boundaryField()[p]);}
 void capillary(const scalarField& s)
 {
  forAll(mat,c){pc[c]=0;dpc[c]=0;if(!fluid(c)||channel(c))continue;
   scalar b=par("surface_tension")*std::cos(par("contact_angle_deg")*constant::mathematical::pi/180)*std::sqrt(porosity[c]/permeability[c]);
   scalar g=s[c];pc[c]=b*(1.417*g-2.120*g*g+1.263*g*g*g);dpc[c]=b*(1.417-4.240*g+3.789*g*g);}
 }
 // Return Q and derivatives with respect to pO,pN,sO,sN. Half-cell resistances
 // enforce phase-pressure continuity at channel/PTL and PTL/CL interfaces.
 std::array<scalar,5> flux(const Face& f,bool gas,const scalarField& p,const scalarField& s)
 {
  label o=f.o,n=f.n;std::array<scalar,5> v{{0,0,0,0,0}};
  if(!fluid(o)||(n>=0&&!fluid(n))||f.kind==3)return v;
  if(f.kind==1){v[0]=gas?0:-feed()*f.area/inletArea(f.patch);return v;}
  scalar bo=channel(o)?rAU[o]:permeability[o]/(gas?muGas[o]:par("liquid_viscosity"));
  scalar bn=n>=0?(channel(n)?rAU[n]:permeability[n]/(gas?muGas[n]:par("liquid_viscosity"))):1;
  scalar ho=channel(o)?f.dO*(H[o]&f.normal)/bo:0;
  scalar hn=n>=0&&channel(n)?f.dN*(H[n]&f.normal)/bn:0;
  scalar drive=p[o]-(n>=0?p[n]:0)+(gas?pc[o]-(n>=0?pc[n]:0):0)+ho+hn;
  label donor=(drive>=0||n<0)?o:n;scalar a=gas?s[donor]:1-s[donor];
  if(a<=0){return v;}
  int mo=channel(o)?1:3,mn=n>=0?(channel(n)?1:3):0;
  scalar ro=f.dO/(bo*std::pow(a,mo)),rn=n>=0?f.dN/(bn*std::pow(a,mn)):0;
  scalar mob=f.area/(ro+rn);v[0]=mob*drive;v[1]=mob;v[2]=-mob;
  scalar dm=mob*(mo*ro+mn*rn)/(a*(ro+rn))*(gas?1:-1);
  v[donor==o?3:4]+=dm*drive;
  if(gas){v[3]+=mob*dpc[o];if(n>=0)v[4]-=mob*dpc[n];}
  if(n>=0&&channel(o)&&channel(n)){
   scalar diffusion=par(mat[o]>=5?"oxygen_channel_dispersion":"hydrogen_channel_dispersion")
    *std::pow(par("inlet_temperature")/298.15,1.5)*101325/par("outlet_pressure")*f.area/(f.dO+f.dN);
   scalar sign=gas?1:-1;v[0]+=sign*diffusion*(s[o]-s[n]);v[3]+=sign*diffusion;v[4]-=sign*diffusion;
  }
  return v;
 }
 void project(volScalarField& p,volScalarField& sg,const volScalarField& slSource,const volScalarField& sgSource)
 {
  scalarField oldS(sg.primitiveField()),pv(p.primitiveField()),sv(sg.primitiveField());
  const scalar ps=1000.,qs=feed()/nf;Vec x(2*nf);
  forAll(mat,c)if(fluid(c)){x[2*node[c]]=pv[c]/ps;x[2*node[c]+1]=sv[c];}
  auto assemble=[&](const Vec& xx,Vec& residual,SparseMatrix* jac,bool storage){
   forAll(mat,c)if(fluid(c)){pv[c]=ps*xx[2*node[c]];sv[c]=xx[2*node[c]+1];}capillary(sv);
   residual=Vec::Zero(2*nf);std::vector<Triplet> entries;entries.reserve(24*nf);
   auto add=[&](label i,label j,scalar v){if(jac)entries.emplace_back(i,j,v/qs);};
   forAll(mat,c)if(fluid(c)){label a=2*node[c];scalar volume=mesh.V()[c];
    residual[a]-=(slSource[c]+sgSource[c])*volume;
    residual[a+1]-=sgSource[c]*volume;
    if(storage){scalar capacity=porosity[c]*volume/dt;residual[a+1]+=capacity*(sv[c]-oldS[c]);add(a+1,a+1,capacity);}}
   for(const auto& f:faces){if(!fluid(f.o)||(f.n>=0&&!fluid(f.n)))continue;
    auto l=flux(f,false,pv,sv),g=flux(f,true,pv,sv);label a=2*node[f.o],b=f.n>=0?2*node[f.n]:-1;
    for(int row=0;row<2;row++){auto v=g;if(row==0)for(int k=0;k<5;k++)v[k]+=l[k];
     residual[a+row]+=v[0];if(b>=0)residual[b+row]-=v[0];
     add(a+row,a,v[1]*ps);add(a+row,a+1,v[3]);
     if(b>=0){add(a+row,b,v[2]*ps);add(a+row,b+1,v[4]);
      add(b+row,a,-v[1]*ps);add(b+row,a+1,-v[3]);add(b+row,b,-v[2]*ps);add(b+row,b+1,-v[4]);}}
   }
   residual/=qs;if(jac){jac->resize(2*nf,2*nf);jac->setFromTriplets(entries.begin(),entries.end());}
  };
  bool accepted=false;Vec res;SparseMatrix J;
  for(int retry=0;retry<8&&!accepted;retry++){
   forAll(mat,c)if(fluid(c)){x[2*node[c]]=p[c]/ps;x[2*node[c]+1]=oldS[c];}
   for(int it=0;it<45;it++){
    assemble(x,res,&J,true);scalar norm=res.lpNorm<Eigen::Infinity>();
    if(norm<1e-7){accepted=true;break;}
    LinearBlock lu(J,props.get<word>("linear_solver")=="direct");
    Vec dx=lu.solve(-res);if(!dx.allFinite())break;
    scalar step=1;
    for(label i=0;i<nf;i++){scalar s=x[2*i+1],ds=dx[2*i+1];
     if(ds<0)step=min(step,.99*s/(-ds));if(ds>0)step=min(step,.99*(par("maximum_packing")-1e-6-s)/ds);}
    bool better=false;Vec trial,rr;
    for(int ls=0;ls<22;ls++){trial=x+step*dx;assemble(trial,rr,nullptr,true);
     if(rr.squaredNorm()<res.squaredNorm()){better=true;break;}step*=.5;}
    if(!better)break;x=trial;
   }
   if(!accepted)dt*=.2;
  }
  if(!accepted)FatalErrorInFunction<<"P008 phase projection failed; dt="<<dt<<exit(FatalError);
  assemble(x,res,nullptr,false);liquidResidual=0;gasResidual=0;
  forAll(mat,c)if(fluid(c)){label a=2*node[c];p[c]=pv[c];sg[c]=sv[c];
   liquidResidual+=mag(res[a]-res[a+1])*qs;gasResidual+=mag(res[a+1])*qs;}
  liquidResidual/=2*feed();scalar gasScale=0;
  forAll(mat,c)gasScale+=mag(sgSource[c])*mesh.V()[c];gasResidual/=max(gasScale,1e-12);
  for(label i=0;i<label(faces.size());i++){
   const auto& f=faces[i];auto l=flux(f,false,pv,sv),g=flux(f,true,pv,sv);qL[i]=l[0];qG[i]=g[0];
   pLface[i]=p[f.o];pGface[i]=p[f.o]+pc[f.o];
   if(!fluid(f.o)||(f.n>=0&&!fluid(f.n))||f.kind==3)continue;
   if(f.kind==1){pLface[i]=p[f.o]+f.dO/rAU[f.o]*((H[f.o]&f.normal)-qL[i]/f.area);pGface[i]=pLface[i];continue;}
   if(f.kind==2){pLface[i]=pGface[i]=0;continue;}
   for(int gas=0;gas<2;gas++){
    scalar q=gas?qG[i]:qL[i];
    if(channel(f.o)&&channel(f.n)){
     scalar diffusion=par(mat[f.o]>=5?"oxygen_channel_dispersion":"hydrogen_channel_dispersion")
      *std::pow(par("inlet_temperature")/298.15,1.5)*101325/par("outlet_pressure")*f.area/(f.dO+f.dN);
     q-=(gas?1:-1)*diffusion*(sg[f.o]-sg[f.n]);
    }
    label donor=q>=0?f.o:f.n;scalar a=gas?sg[donor]:1-sg[donor];
    scalar base=channel(f.o)?rAU[f.o]:permeability[f.o]/(gas?muGas[f.o]:par("liquid_viscosity"));
    scalar resistance=f.dO/(base*std::pow(max(a,1e-15),channel(f.o)?1:3));
    scalar facep=p[f.o]+(gas?pc[f.o]:0)+(channel(f.o)?f.dO*(H[f.o]&f.normal)/base:0)-q*resistance/f.area;
    (gas?pGface[i]:pLface[i])=facep;
   }
  }
  dt=min(dt*1.1,.1);p.correctBoundaryConditions();sg.correctBoundaryConditions();
 }
 void pressureGradient(const volScalarField& p,const volScalarField& sg)
 {
  gradient=vector::zero;
  for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];
   for(int side=0;side<2;side++){label c=side?f.n:f.o;if(c<0||!channel(c))continue;
    label other=side?f.o:f.n;scalar pf=p[c];
    if((other>=0&&fluid(other))||f.kind==1||f.kind==2)pf=(1-sg[c])*pLface[i]+sg[c]*pGface[i];
    gradient[c]+=(side?-1:1)*f.area*f.normal*pf/mesh.V()[c];}}
 }
 SparseMatrix momentumMatrix;std::array<Vec,3> momentumRhs;Vec diagonal;
 void momentum(const volScalarField& sg,volVectorField& U)
 {
  scalar muL=par("liquid_viscosity"),rhoL=par("liquid_density"),packing=par("maximum_packing");
  forAll(mat,c){scalar g=sg[c];rhoMix[c]=rhoL*(1-g)+rhoGas[c]*g;
   scalar star=(muGas[c]+.4*muL)/(muGas[c]+muL);
   muMix[c]=muL*std::pow(1-g/packing,-2.5*packing*star);}
  std::vector<Triplet> entries;diagonal=Vec::Zero(nch);for(auto& b:momentumRhs)b=Vec::Zero(nch);
  auto fixed=[&](label c,scalar coeff,const vector& value){label a=channelNode[c];diagonal[a]+=coeff;
   for(int d=0;d<3;d++)momentumRhs[d][a]+=coeff*value[d];};
  // Gauss gradient including internal no-slip solid walls, and continuous
  // velocity traces on the channel/porous interface (Darcy velocity provided).
  tensorField grad(mesh.nCells(),tensor::zero);
  for(const auto& f:faces){vector uf=vector::zero;
   if(f.n>=0){if(fluid(f.o)&&fluid(f.n))uf=(f.dN*U[f.o]+f.dO*U[f.n])/(f.dO+f.dN);}
   else if(f.kind==1)uf=U.boundaryField()[f.patch][f.index];else if(f.kind==2)uf=U[f.o];
   if(channel(f.o))grad[f.o]+=f.area*(f.normal*uf)/mesh.V()[f.o];
   if(f.n>=0&&channel(f.n))grad[f.n]-=f.area*(f.normal*uf)/mesh.V()[f.n];}
  for(label i=0;i<label(faces.size());i++){
   const auto& f=faces[i];bool co=channel(f.o),cn=f.n>=0&&channel(f.n);if(!co&&!cn)continue;
   scalar mass=par("liquid_density")*qL[i]+rhoGas[f.o]*qG[i];
   if(co&&cn){label a=channelNode[f.o],b=channelNode[f.n];
    scalar k=f.area/(f.dO/muMix[f.o]+f.dN/muMix[f.n]);
    diagonal[a]+=k+max(mass,0.);diagonal[b]+=k+max(-mass,0.);
    entries.emplace_back(a,b,-k+min(mass,0.));entries.emplace_back(b,a,-k-max(mass,0.));
    tensor gf=(f.dN*grad[f.o]+f.dO*grad[f.n])/(f.dO+f.dN);
    scalar muf=(f.dO+f.dN)/(f.dO/muMix[f.o]+f.dN/muMix[f.n]);
    vector stress=muf*f.area*(f.normal&(gf.T()-(2./3)*tr(gf)*tensor::I));
    for(int d=0;d<3;d++){momentumRhs[d][a]+=stress[d];momentumRhs[d][b]-=stress[d];}
   }else{
    label c=co?f.o:f.n;scalar distance=co?f.dO:f.dN;scalar signedMass=co?mass:-mass;vector bc=vector::zero;
    if(f.n<0&&f.kind==2){diagonal[channelNode[c]]+=signedMass;continue;}
    if(f.n<0&&f.kind==1)bc=U.boundaryField()[f.patch][f.index];
    else {label other=co?f.n:f.o;if(other>=0&&fluid(other))bc=U[other];}
    fixed(c,muMix[c]*f.area/distance,bc);
    diagonal[channelNode[c]]+=max(signedMass,0.);
    for(int d=0;d<3;d++)momentumRhs[d][channelNode[c]]-=min(signedMass,0.)*bc[d];
   }
  }
  for(label a=0;a<nch;a++)entries.emplace_back(a,a,diagonal[a]);
  momentumMatrix.resize(nch,nch);momentumMatrix.setFromTriplets(entries.begin(),entries.end());
  scalar relax=par("velocity_relaxation");
  SparseMatrix relaxed=momentumMatrix;for(label a=0;a<nch;a++)relaxed.coeffRef(a,a)=diagonal[a]/relax;
  LinearBlock lu(relaxed,props.get<word>("linear_solver")=="direct");
  for(int d=0;d<3;d++){
   Vec old(nch),rhs=momentumRhs[d];forAll(mat,c)if(channel(c)){label a=channelNode[c];old[a]=U[c][d];rhs[a]+=(1/relax-1)*diagonal[a]*old[a]-mesh.V()[c]*gradient[c][d];}
   Vec u=lu.solve(rhs);if(!u.allFinite())FatalErrorInFunction<<"Nonfinite momentum"<<exit(FatalError);
   Vec off=momentumMatrix*u-diagonal.cwiseProduct(u);
   forAll(mat,c)if(channel(c)){label a=channelNode[c];rAU[c]=mesh.V()[c]/(diagonal[a]/relax);
    H[c][d]=(momentumRhs[d][a]+(1/relax-1)*diagonal[a]*old[a]-off[a])/(diagonal[a]/relax);U[c][d]=u[a];}
  }
 }
 void velocities(const volScalarField& p,const volScalarField& sg,volVectorField& U,volVectorField& Ul,volVectorField& Ug)
 {
  pressureGradient(p,sg);vectorField ql(mesh.nCells(),vector::zero),qg(mesh.nCells(),vector::zero);
  // Orthogonal reconstruction: sum Q_f (Cf-Cc) / V. Exact for constant flow.
  for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];
   ql[f.o]+=qL[i]*f.dO*f.normal/mesh.V()[f.o];qg[f.o]+=qG[i]*f.dO*f.normal/mesh.V()[f.o];
   if(f.n>=0){ql[f.n]+=qL[i]*f.dN*f.normal/mesh.V()[f.n];qg[f.n]+=qG[i]*f.dN*f.normal/mesh.V()[f.n];}}
  forAll(mat,c){
   if(channel(c)){U[c]=H[c]-rAU[c]*gradient[c];Ul[c]=Ug[c]=U[c];}
   else if(fluid(c)){Ul[c]=ql[c]/max(porosity[c]*(1-sg[c]),1e-14);Ug[c]=qg[c]/max(porosity[c]*sg[c],1e-14);U[c]=ql[c]+qg[c];}
   else U[c]=Ul[c]=Ug[c]=vector::zero;}
  scalar num=0,den=0;
  for(int d=0;d<3;d++){Vec u(nch),force(nch);
   forAll(mat,c)if(channel(c)){u[channelNode[c]]=U[c][d];force[channelNode[c]]=gradient[c][d]*mesh.V()[c];}
   Vec r=momentumMatrix*u-momentumRhs[d]+force;num+=r.lpNorm<1>();
   den+=force.lpNorm<1>()+diagonal.cwiseProduct(u).lpNorm<1>()+momentumRhs[d].lpNorm<1>();}
  momentumResidual=num/max(den,1e-20);
  for(auto* f:{&U,&Ul,&Ug})f->correctBoundaryConditions();
 }
 void thermal(const volScalarField& sg,const volScalarField& heat,volScalarField& T)
 {
  label n=mesh.nCells();scalarField k(n,0),capacity(n,0);scalar rhoL=par("liquid_density");
  const bool lte=props.get<word>("energy_form")=="lte_phase_advection";
  const scalar ks[9]={200,0,3,1,.95,1,3,0,200};
  const scalar rs[9]={2266,0,450,2145,10927,2145,450,0,2266};
  const scalar cs[9]={830,0,871,3300,833,3300,871,0,830};
  forAll(mat,c){scalar s=sg[c],e=porosity[c];scalar cpG=par(mat[c]>=5?"oxygen_cp":"hydrogen_cp");
   scalar kg=par(mat[c]>=5?"oxygen_conductivity":"hydrogen_conductivity");
   scalar rm=(1-s)*rhoL+s*rhoGas[c],cm=((1-s)*rhoL*par("liquid_cp")+s*rhoGas[c]*cpG)/rm;
   k[c]=(1-e)*ks[mat[c]]+e*((1-s)*par("liquid_conductivity")+s*kg);
   capacity[c]=((1-e)*rs[mat[c]]+e*rm)*((1-e)*cs[mat[c]]+e*cm);}
  std::vector<Triplet> entries;Vec diag=Vec::Zero(n),rhs(n);forAll(mat,c)rhs[c]=heat[c]*mesh.V()[c];
  for(label i=0;i<label(faces.size());i++){
   const auto& f=faces[i];label o=f.o,nb=f.n;scalar q=qL[i]+qG[i];
   if(nb>=0){scalar conduct=f.area/(f.dO/k[o]+f.dN/k[nb]);
    diag[o]+=conduct;diag[nb]+=conduct;entries.emplace_back(o,nb,-conduct);entries.emplace_back(nb,o,-conduct);
    // Literal Eq45 is available for discrepancy studies. Default: COMSOL LTE
    // phase-fluid sensible heat advection; immobile solid heat capacity is
    // storage only (zero at steady state), never a transported fluid property.
    scalar adv=lte?rhoL*par("liquid_cp")*qL[i]+rhoGas[o]*par(mat[o]>=5?"oxygen_cp":"hydrogen_cp")*qG[i]:q*(q>=0?capacity[o]:capacity[nb]);
    diag[o]+=max(adv,0.);entries.emplace_back(o,nb,min(adv,0.));
    diag[nb]+=max(-adv,0.);entries.emplace_back(nb,o,-max(adv,0.));
    if(lte){diag[o]-=adv;diag[nb]+=adv;} // u.grad(T) = div(uT)-T div(u).
   }else if(f.kind==1){scalar conduct=k[o]*f.area/f.dO;
    diag[o]+=conduct;rhs[o]+=conduct*par("inlet_temperature");
    rhs[o]-=q*rhoL*par("liquid_cp")*par("inlet_temperature");
    if(lte)diag[o]-=q*rhoL*par("liquid_cp");
   }else if(f.kind==2){if(!lte)diag[o]+=q*capacity[o];}
   else{scalar conduct=f.area/(f.dO/k[o]+1/par("external_heat_transfer"));
    diag[o]+=conduct;rhs[o]+=conduct*par("ambient_temperature");}
  }
  for(label c=0;c<n;c++)entries.emplace_back(c,c,diag[c]);SparseMatrix A(n,n);A.setFromTriplets(entries.begin(),entries.end());
  LinearBlock lu(A,props.get<word>("linear_solver")=="direct");
  Vec solution=lu.solve(rhs),t(n);if(!solution.allFinite())FatalErrorInFunction<<"Nonfinite temperature"<<exit(FatalError);
  forAll(mat,c){T[c]+=par("thermal_relaxation")*(solution[c]-T[c]);t[c]=T[c];}
  energyResidual=(A*t-rhs).lpNorm<1>()/max(sum(mag(heat.primitiveField())*mesh.V()),1e-12);
  T.correctBoundaryConditions();
  for(const auto& face:faces)if(face.kind==3){label c=face.o;scalar conduct=k[c]/face.dO;
   T.boundaryFieldRef()[face.patch][face.index]=(conduct*T[c]+par("external_heat_transfer")*par("ambient_temperature"))/(conduct+par("external_heat_transfer"));}
 }
 void writeFlux(surfaceScalarField& l,surfaceScalarField& g,volScalarField& p)
 {
  for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];if(f.patch<0){l[f.index]=qL[i];g[f.index]=qG[i];}
   else{l.boundaryFieldRef()[f.patch][f.index]=qL[i];g.boundaryFieldRef()[f.patch][f.index]=qG[i];p.boundaryFieldRef()[f.patch][f.index]=pLface[i];}}
 }
};
