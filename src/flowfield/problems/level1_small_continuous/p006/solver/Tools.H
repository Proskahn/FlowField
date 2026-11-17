#include "fvCFD.H"
#include "syncTools.H"
using namespace Foam;

static volScalarField auxiliary(const word& name, const fvMesh& mesh,
                               const dimensionSet& dims, scalar value=0,
                               IOobject::writeOption write=IOobject::NO_WRITE)
{
    wordList patchTypes(mesh.boundary().size(),"zeroGradient");
    forAll(mesh.boundary(), p)
        if (mesh.boundary()[p].coupled()) patchTypes[p]=mesh.boundary()[p].type();
    return volScalarField(IOobject(name,mesh.time().timeName(),mesh,
        IOobject::NO_READ,write),mesh,dimensionedScalar(name,dims,value),patchTypes);
}

// Internal solid/fluid boundaries are geometric walls, although represented
// on a single mesh. Use their prescribed zero face velocity when forming the
// phase stress gradient, rather than interpolating through a solid cell.
static tmp<volTensorField> phaseGradient(const volVectorField& q,
    const boolList& fluid, const boolList& neighbourFluid)
{
    const fvMesh& mesh=q.mesh();
    surfaceVectorField qf(fvc::interpolate(q));
    forAll(mesh.neighbour(),f)
        if (!fluid[mesh.owner()[f]] || !fluid[mesh.neighbour()[f]]) qf[f]=vector::zero;
    forAll(mesh.boundary(),pi)
    {
        const auto& patch=mesh.boundary()[pi];
        if (!patch.coupled()) continue;
        forAll(patch,f)
        {
            label c=patch.faceCells()[f],b=patch.start()-mesh.nInternalFaces()+f;
            if (!fluid[c] || !neighbourFluid[b]) qf.boundaryFieldRef()[pi][f]=vector::zero;
        }
    }
    return fvc::surfaceIntegrate(mesh.Sf()*qf);
}

// OpenFOAM stores grad(q)_ij = d(q_j)/d(x_i). The divergence contracts
// the face normal on the LEFT. Reversing this contraction duplicates the
// shear Laplacian instead of adding the transpose/deviatoric stress.
static tmp<surfaceVectorField> phaseTransposeStressFlux(
    const surfaceScalarField& viscosity, const volVectorField& q,
    const boolList& fluid, const boolList& neighbourFluid)
{
    return viscosity*(q.mesh().Sf()
        & fvc::interpolate(dev2(Foam::T(phaseGradient(q,fluid,neighbourFluid)))));
}

// Extremely large dry-phase/solid diagonals can dominate OpenFOAM's global
// residual normalization and make unresolved channel equations appear solved.
// Equilibrate a COPY, including its RHS and processor coefficients. The
// original matrix still supplies the physical SIMPLE diagonal and predictor.
static void solveRowScaled(fvVectorMatrix& equation, const volVectorField& force)
{
    volScalarField inverse("momentumInverse",1.0/equation.A());
    fvVectorMatrix scaled(equation==force);
    scaled*=inverse;
    scaled.solve();
}

// Distance-weighted harmonic coefficient. Exactly zero across an inactive
// material interface; there is no small artificial conductivity or leakage.
static void harmonic(surfaceScalarField& face, const volScalarField& cell,
                     const vectorField& neighbourCentres,
                     const boolList* active=nullptr)
{
    const fvMesh& mesh=cell.mesh();
    scalarField neighbourValue;
    syncTools::swapBoundaryCellList(mesh,cell.primitiveField(),neighbourValue);
    boolList neighbourActive;
    if (active) syncTools::swapBoundaryCellList(mesh,*active,neighbourActive);
    forAll(mesh.neighbour(),f)
    {
        label o=mesh.owner()[f], n=mesh.neighbour()[f];
        if ((active && (!(*active)[o] || !(*active)[n])) || cell[o]<=0 || cell[n]<=0)
        { face[f]=0; continue; }
        scalar dO=mag(mesh.Cf()[f]-mesh.C()[o]), dN=mag(mesh.Cf()[f]-mesh.C()[n]);
        face[f]=(dO+dN)/(dO/cell[o]+dN/cell[n]);
    }
    forAll(mesh.boundary(),p)
    {
        const auto& patch=mesh.boundary()[p];
        forAll(patch,f)
        {
            label c=patch.faceCells()[f], b=patch.start()-mesh.nInternalFaces()+f;
            scalar value=(active && !(*active)[c]) ? 0 : cell[c];
            if (patch.coupled())
            {
                if (value<=0 || neighbourValue[b]<=0 || (active && !neighbourActive[b])) value=0;
                else
                {
                    scalar dO=mag(patch.Cf()[f]-mesh.C()[c]);
                    scalar dN=mag(patch.Cf()[f]-neighbourCentres[b]);
                    value=(dO+dN)/(dO/value+dN/neighbourValue[b]);
                }
            }
            face.boundaryFieldRef()[p][f]=value;
        }
    }
}


// Reconstruct the same physical pressure trace used by FlowNetwork, including
// material-interface Darcy resistance and the gas capillary pressure.
static void refreshPhasePressureForces(const volScalarField& p,const volScalarField& pc,
    const volScalarField& dragL,const volScalarField& dragG,
    const labelList& material,const boolList& fluid,const labelList& neighbourMaterial,
    const boolList& neighbourFluid,const vectorField& neighbourCentres,
    volVectorField& forceL,volVectorField& forceG)
{
    const fvMesh& mesh=p.mesh();
    scalarField neighbourP,neighbourPc,neighbourDL,neighbourDG;
    syncTools::swapBoundaryCellList(mesh,p.primitiveField(),neighbourP);
    syncTools::swapBoundaryCellList(mesh,pc.primitiveField(),neighbourPc);
    syncTools::swapBoundaryCellList(mesh,dragL.primitiveField(),neighbourDL);
    syncTools::swapBoundaryCellList(mesh,dragG.primitiveField(),neighbourDG);
    surfaceScalarField faceL("currentPressureFaceL",fvc::interpolate(p));
    surfaceScalarField faceG("currentPressureFaceG",faceL);
    auto trace=[](scalar po,scalar pn,scalar do_,scalar dn,scalar dragO,scalar dragN,bool different)
    {scalar wo=different?dn*dragN:dn,wn=different?do_*dragO:do_;return (wo*po+wn*pn)/(wo+wn);};
    forAll(mesh.neighbour(),f)
    {
        label o=mesh.owner()[f],n=mesh.neighbour()[f];
        if(fluid[o]&&fluid[n])
        {
            scalar do_=mag(mesh.Cf()[f]-mesh.C()[o]),dn=mag(mesh.Cf()[f]-mesh.C()[n]);
            faceL[f]=trace(p[o],p[n],do_,dn,dragL[o],dragL[n],material[o]!=material[n]);
            faceG[f]=trace(p[o]+pc[o],p[n]+pc[n],do_,dn,dragG[o],dragG[n],material[o]!=material[n]);
        }
        else if(fluid[o]||fluid[n]){label c=fluid[o]?o:n;faceL[f]=p[c];faceG[f]=p[c]+pc[c];}
        else{faceL[f]=0;faceG[f]=0;}
    }
    forAll(mesh.boundary(),pi)
    {
        const auto& patch=mesh.boundary()[pi];
        forAll(patch,f)
        {
            label c=patch.faceCells()[f],b=patch.start()-mesh.nInternalFaces()+f;
            scalar l=fluid[c]?p[c]:0,g=fluid[c]?p[c]+pc[c]:0;
            if(patch.coupled()&&fluid[c]&&neighbourFluid[b])
            {
                scalar do_=mag(patch.Cf()[f]-mesh.C()[c]),dn=mag(patch.Cf()[f]-neighbourCentres[b]);
                l=trace(p[c],neighbourP[b],do_,dn,dragL[c],neighbourDL[b],material[c]!=neighbourMaterial[b]);
                g=trace(p[c]+pc[c],neighbourP[b]+neighbourPc[b],do_,dn,dragG[c],neighbourDG[b],material[c]!=neighbourMaterial[b]);
            }
            else if(!patch.coupled()&&patch.name().ends_with("Outlet")){l=0;g=0;}
            else if(!patch.coupled()&&patch.name().ends_with("Inlet")){g=l;}
            faceL.boundaryFieldRef()[pi][f]=l;faceG.boundaryFieldRef()[pi][f]=g;
        }
    }
    forceL=-fvc::surfaceIntegrate(faceL*mesh.Sf());forceG=-fvc::surfaceIntegrate(faceG*mesh.Sf());
    forAll(fluid,c)if(!fluid[c]){forceL[c]=vector::zero;forceG[c]=vector::zero;}
}

// Algebraically equivalent column/row scaling for extreme phase mobility.
static void solveColumnScaled(fvVectorMatrix& equation,const volVectorField& force,
    const volScalarField& drag,scalar referenceDrag)
{
    auto& velocity=const_cast<volVectorField&>(equation.psi());
    const fvMesh& mesh=velocity.mesh();
    volScalarField weight(IOobject("momentumColumnScale",mesh.time().timeName(),mesh),mesh,
        dimensionedScalar("one",dimless,1.));
    forAll(weight,c)weight[c]=drag[c]>0 ? referenceDrag/drag[c] : 1.;
    weight.correctBoundaryConditions();
    scalarField neighbourWeight;syncTools::swapBoundaryCellList(mesh,weight.primitiveField(),neighbourWeight);
    fvVectorMatrix scaled(equation==force);
    scaled.diag()*=weight.primitiveField();
    scalarField& lower=scaled.lower();scalarField& upper=scaled.upper();
    forAll(mesh.neighbour(),f){upper[f]*=weight[mesh.neighbour()[f]];lower[f]*=weight[mesh.owner()[f]];}
    forAll(mesh.boundary(),pi)
    {
        const auto& patch=mesh.boundary()[pi];
        forAll(patch,f)
        {
            scaled.internalCoeffs()[pi][f]*=weight[patch.faceCells()[f]];
            if(patch.coupled())scaled.boundaryCoeffs()[pi][f]*=neighbourWeight[patch.start()-mesh.nInternalFaces()+f];
        }
    }
    volScalarField inverse("momentumScaledInverse",1./(equation.A()*weight));
    scaled*=inverse;
    velocity.primitiveFieldRef()/=weight.primitiveField();velocity.correctBoundaryConditions();
    scaled.solve();
    velocity.primitiveFieldRef()*=weight.primitiveField();velocity.correctBoundaryConditions();

}
