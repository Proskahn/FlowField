#pragma once
// Conservative orthogonal finite-volume operators. Serial Eigen linear algebra;
// OpenFOAM owns the mesh, fields, boundary data and ParaView output.
#include "fvCFD.H"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <vector>
#include <array>
#include <cmath>
#include <cfenv>
using namespace Foam;
using SparseMatrix=Eigen::SparseMatrix<double>;
using Vec=Eigen::VectorXd;
using Triplet=Eigen::Triplet<double>;

// Krylov breakdown can produce an IEEE NaN before Eigen reports failure.
// Hold traps only inside the algebra package, restore them immediately, then
// require finite values and a verified residual or use checked direct fallback.
struct AlgebraEnvironment{fenv_t saved;AlgebraEnvironment(){std::feholdexcept(&saved);}~AlgebraEnvironment(){std::fesetenv(&saved);}};

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
  if(useDirect_){AlgebraEnvironment guard;direct_.compute(matrix_);return;}
  for(int j=0;j<A.outerSize();j++)for(SparseMatrix::InnerIterator e(A,j);e;++e)scale_[e.row()]=max(scale_[e.row()],mag(e.value()));
  for(int i=0;i<scale_.size();i++)scale_[i]=max(scale_[i],1e-30);
  for(int j=0;j<scaled_.outerSize();j++)for(SparseMatrix::InnerIterator e(scaled_,j);e;++e)e.valueRef()/=scale_[e.row()];
  iterative_.preconditioner().setDroptol(1e-4);iterative_.preconditioner().setFillfactor(12);
  iterative_.setTolerance(1e-11);iterative_.setMaxIterations(3000);
  {AlgebraEnvironment guard;iterative_.compute(scaled_);}
 }
 Vec solve(const Vec& rhs)
 {
  AlgebraEnvironment checkedAlgebra;
  if(rhs.norm()<1e-280)return Vec::Zero(rhs.size());
  if(!rhs.allFinite())FatalErrorInFunction<<"Nonfinite linear right-hand side"<<exit(FatalError);
  if(!useDirect_){Vec b=rhs.cwiseQuotient(scale_);Vec x;
   {AlgebraEnvironment guard;x=iterative_.solve(b);}
   if(iterative_.info()==Eigen::Success&&x.allFinite()&&(scaled_*x-b).norm()<1e-9*b.norm())return x;
   Info<<"P011 linear algebra: using direct fallback for "<<rhs.size()<<" unknowns"<<endl;
   {AlgebraEnvironment guard;direct_.compute(matrix_);}useDirect_=true;}
  if(direct_.info()!=Eigen::Success)FatalErrorInFunction<<"P011 linear factorization failed"<<exit(FatalError);
  Vec x;{AlgebraEnvironment guard;x=direct_.solve(rhs);}
  if(direct_.info()!=Eigen::Success||!x.allFinite()||(matrix_*x-rhs).norm()>1e-8*rhs.norm())FatalErrorInFunction<<"P011 linear solve failed"<<exit(FatalError);
  return x;
 }
};

constexpr scalar FARADAY=96485.33212, RGAS=8.314462618, MW=.018;
inline scalar psat(scalar T){scalar t=T-273.15;return 101325*std::pow(10.,-2.1794+.02953*t-9.1837e-5*t*t+1.4454e-7*t*t*t);}
inline scalar waterEq(scalar activity){scalar a=max(activity,0.);return a<=1?.043+17.81*a-39.85*a*a+36*a*a*a:14.003+1.4*(a-1);}
struct Face{label o,n,patch,index;scalar area,dO,dN;vector normal;int kind;};
struct BlockResult{scalar residual;scalarField flux;};

class Model
{
public:
 const fvMesh& mesh;const dictionary& props;labelList mat,gasNode,ionNode;
 label ng=0,nion=0;std::vector<Face> faces;
 volScalarField &p,&T,&s,&lambda,&yr,&yw,&phiS,&phiE,&iv,&heat;
 volVectorField& U;
 scalarField eps,K,mu,rho,ct,rAU,pc,pcSlope,gasQ,liquidQ,reactQ,vaporQ,waterQ,heatQ,ionicQ,phaseRate,sorbRate;
 vectorField H,gradP;
 scalar chargeResidual=GREAT,momentumResidual=GREAT,gasResidual=GREAT,speciesResidual=GREAT;
 scalar liquidResidual=GREAT,membraneResidual=GREAT,energyResidual=GREAT;
 Model(const fvMesh& mesh_,const dictionary& d,volScalarField& material,
       volScalarField& p_,volScalarField& t_,volScalarField& s_,volScalarField& l_,
       volScalarField& r_,volScalarField& w_,volScalarField& ps_,volScalarField& pe_,
       volScalarField& i_,volScalarField& h_,volVectorField& u_)
 :mesh(mesh_),props(d),mat(mesh.nCells()),gasNode(mesh.nCells(),-1),ionNode(mesh.nCells(),-1),
 p(p_),T(t_),s(s_),lambda(l_),yr(r_),yw(w_),phiS(ps_),phiE(pe_),iv(i_),heat(h_),U(u_),
 eps(mesh.nCells(),0),K(mesh.nCells(),0),mu(mesh.nCells(),0),rho(mesh.nCells(),0),ct(mesh.nCells(),0),
 rAU(mesh.nCells(),0),pc(mesh.nCells(),0),pcSlope(mesh.nCells(),0),
 phaseRate(mesh.nCells(),0),sorbRate(mesh.nCells(),0),H(mesh.nCells(),vector::zero),gradP(mesh.nCells(),vector::zero)
 {
  forAll(mat,c){mat[c]=label(material[c]+.5);if(gas(c))gasNode[c]=ng++;if(ionic(c))ionNode[c]=nion++;
   eps[c]=channel(c)?1:(mat[c]==2||mat[c]==6)?par("gdl_porosity"):cl(c)?par("cl_porosity"):0;
   K[c]=mat[c]==2||mat[c]==6?par("permeability_gdl"):par("permeability_cl");
   mu[c]=par(anode(c)?"hydrogen_viscosity":"air_viscosity");}
  forAll(mesh.neighbour(),f){label o=mesh.owner()[f],n=mesh.neighbour()[f];scalar a=mesh.magSf()[f];
   faces.push_back({o,n,-1,f,a,mag(mesh.Cf()[f]-mesh.C()[o]),mag(mesh.Cf()[f]-mesh.C()[n]),mesh.Sf()[f]/a,0});}
  forAll(mesh.boundary(),b){const auto& patch=mesh.boundary()[b];
   int kind=patch.name().ends_with("Inlet")?1:patch.name().ends_with("Outlet")?2:3;
   forAll(patch,f){scalar a=patch.magSf()[f];faces.push_back({patch.faceCells()[f],-1,b,f,a,1/patch.deltaCoeffs()[f],0,patch.Sf()[f]/a,kind});}}
  label nf=faces.size();for(auto* v:{&gasQ,&liquidQ,&reactQ,&vaporQ,&waterQ,&heatQ,&ionicQ})v->setSize(nf,0);
  properties();for(label f=0;f<nf;f++)if(faces[f].kind==1)gasQ[f]=-feed(faces[f].o)*faces[f].area/inletArea(faces[f].patch);
 }
 scalar par(const word& k)const{return props.get<scalar>(k);}
 bool gas(label c)const{return mat[c]==1||mat[c]==2||mat[c]==3||mat[c]==5||mat[c]==6||mat[c]==7;}
 bool channel(label c)const{return mat[c]==1||mat[c]==7;}
 bool cl(label c)const{return mat[c]==3||mat[c]==5;}
 bool ionic(label c)const{return mat[c]>=3&&mat[c]<=5;}
 bool electronic(label c)const{return mat[c]!=1&&mat[c]!=4&&mat[c]!=7;}
 bool anode(label c)const{return mat[c]>=5;}
 // Mixture-averaged binary diffusivities from Table 2, before Bruggeman scaling.
 scalar diffusivity(label c,bool vapor)const{
  if(anode(c))return par("hydrogen_diffusivity");
  scalar oxygen=max(yr[c],0.),water=max(yw[c],0.),nitrogen=max(1-oxygen-water,0.);
  scalar numerator=vapor?oxygen+nitrogen:water+nitrogen;
  scalar denominator=vapor?oxygen/par("oxygen_water_diffusivity")+nitrogen/par("nitrogen_water_diffusivity"):
      water/par("oxygen_water_diffusivity")+nitrogen/par("oxygen_nitrogen_diffusivity");
  return denominator>1e-20?numerator/denominator:par("oxygen_water_diffusivity");
 }
 scalar inletWater()const{return par("humidity_percent")*.01*psat(par("temperature_K"))/par("outlet_pressure");}
 scalar inletReact(label c)const{return (1-inletWater())*(anode(c)?1:.21);}
 scalar feed(label c)const{return par("reference_current_density")*par("reactionArea")*par(anode(c)?"anode_stoichiometry":"cathode_stoichiometry")/((anode(c)?2:4)*FARADAY*inletReact(c));}
 scalar inletArea(label patch)const{return sum(mesh.magSf().boundaryField()[patch]);}
 scalar faceC(const Face& f)const{return f.n<0?ct[f.o]:2/(1/ct[f.o]+1/ct[f.n]);}
 scalar cap(label c)const{return par("membrane_dry_density")/par("membrane_equivalent_weight")*(mat[c]==4?1:par("ionomer_fraction"));}
 void properties(){forAll(mat,c){ct[c]=par("outlet_pressure")/(RGAS*T[c]);
   scalar molar=anode(c)?yr[c]*.002+yw[c]*MW:yr[c]*.032+yw[c]*MW+(1-yr[c]-yw[c])*.028;
   rho[c]=ct[c]*max(molar,.002);
   scalar b=gas(c)&&!channel(c)?-par("surface_tension")*std::cos(par("contact_angle_deg")*constant::mathematical::pi/180)*std::sqrt(eps[c]/K[c]):0;
   scalar v=s[c];pc[c]=b*(1.417*v-2.120*v*v+1.263*v*v*v);pcSlope[c]=b*(1.417-4.240*v+3.789*v*v);
  }}
 bool direct()const{return props.get<word>("linear_solver")=="direct";}
 // Conservative scalar diffusion+upwind advection. Inactive cells are sealed.
 // Danckwerts inlet: prescribed total species flux, zero diffusive contribution.
 BlockResult transport(const labelList& node,label count,volScalarField& value,
   const scalarField& diff,const scalarField& adv,const scalarField& source,const scalarField& sink,
   const scalarField& bc,scalar relaxation,bool wallFixed=false,bool outletFixed=false,
   const scalarField* extra=nullptr)
 {
  std::vector<Triplet> trip;Vec diag=Vec::Zero(count),rhs=Vec::Zero(count);
  forAll(node,c)if(node[c]>=0){rhs[node[c]]=source[c]*mesh.V()[c];diag[node[c]]=sink[c]*mesh.V()[c];}
  for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];label a=node[f.o],b=f.n>=0?node[f.n]:-1;if(a<0)continue;
   scalar q=adv[i],e=extra?(*extra)[i]:0;
   if(b>=0){scalar k=diff[f.o]>0&&diff[f.n]>0?f.area/(f.dO/diff[f.o]+f.dN/diff[f.n]):0;
    diag[a]+=k+max(q,0.);diag[b]+=k+max(-q,0.);trip.emplace_back(a,b,-k+min(q,0.));trip.emplace_back(b,a,-k-max(q,0.));rhs[a]-=e;rhs[b]+=e;
   }else if(f.n<0){
    if(f.kind==1){rhs[a]-=q*bc[i];}
    else if(f.kind==2){if(q<0)rhs[a]-=q*bc[i];else diag[a]+=q;}
    if((wallFixed&&f.kind==3)||(outletFixed&&f.kind==2)){
     scalar k=diff[f.o]*f.area/f.dO;diag[a]+=k;rhs[a]+=k*bc[i];}
   }
  }
  for(label i=0;i<count;i++)trip.emplace_back(i,i,diag[i]);SparseMatrix A(count,count);A.setFromTriplets(trip.begin(),trip.end());
  LinearBlock solve(A,direct());Vec answer=solve.solve(rhs),v(count);
  forAll(node,c)if(node[c]>=0){value[c]+=relaxation*(answer[node[c]]-value[c]);v[node[c]]=value[c];}
  scalar norm=(A*v-rhs).lpNorm<1>()/max(rhs.lpNorm<1>(),1e-14);scalarField flux(faces.size(),0);
  for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];if(node[f.o]<0)continue;scalar q=adv[i];
   if(f.n>=0&&node[f.n]>=0){scalar k=diff[f.o]>0&&diff[f.n]>0?f.area/(f.dO/diff[f.o]+f.dN/diff[f.n]):0;
    flux[i]=q*(q>=0?value[f.o]:value[f.n])+k*(value[f.o]-value[f.n])+(extra?(*extra)[i]:0);}
   else if(f.n<0){if(f.kind==1)flux[i]=q*bc[i];else if(f.kind==2)flux[i]=q*(q>=0?value[f.o]:bc[i]);
    if((wallFixed&&f.kind==3)||(outletFixed&&f.kind==2))flux[i]+=diff[f.o]*f.area/f.dO*(value[f.o]-bc[i]);}
  }
  value.correctBoundaryConditions();return {norm,flux};
 }
 void charge(scalar voltage);
 void flow();
 void water();
 void species();
 void thermal();
 void writeFlux(surfaceScalarField& field,const scalarField& flux){for(label i=0;i<label(faces.size());i++){const auto& f=faces[i];
  if(f.patch<0)field[f.index]=flux[i];else field.boundaryFieldRef()[f.patch][f.index]=flux[i];}}
};

