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

