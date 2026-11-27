// Analytic two-phase Darcy flow through uniform and high-contrast media.
// Exercise the production Newton/GMRES implementation, including empty ranks.
#include "Tools.H"
#include "ChargeNetwork.H"
#include "FlowNetwork.H"

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    IOdictionary props(IOobject("modelProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
    const bool solidHalf=props.get<bool>("solid_half");
    labelList material(mesh.nCells(),1),neighbourMaterial;
    forAll(material,c)if(solidHalf && mesh.C()[c].y()>.5)material[c]=0;
    syncTools::swapBoundaryCellList(mesh,material,neighbourMaterial);
    label active=0;forAll(material,c)if(material[c])++active;
    label emptyRanks=returnReduce(label(active==0),sumOp<label>());
    Info<<"PROJECTION_RANKS empty="<<emptyRanks<<nl;
    if(solidHalf && Pstream::parRun() && emptyRanks!=1)return 1;
    bool passed=true;
    for(label mode=0;mode<2;++mode)
    {
        FlowNetwork network(mesh,material);
        const scalar contrast=mode?1e4:1.,targetS=.3,initialS=.35,base=1e-6;
        const scalar liquidFlux=base*pow3(1-targetS),gasFlux=base*pow3(targetS);
        auto exactPressure=[&](scalar x){return x<.5?(.5-x)+.5*contrast:(1-x)*contrast;};
        auto eps=auxiliary("eps",mesh,dimless,1);
        auto rL=auxiliary("rL",mesh,dimTime/dimDensity,0);
        auto rG=auxiliary("rG",mesh,dimTime/dimDensity,0);
        auto p=auxiliary("p",mesh,dimPressure,0);
        auto sg=auxiliary("sg",mesh,dimless,initialS);
        auto pc=auxiliary("pc",mesh,dimPressure,0);
        auto Sl=auxiliary("Sl",mesh,dimless/dimTime,0);
        auto Sg=auxiliary("Sg",mesh,dimless/dimTime,0);
        wordList types(mesh.boundary().size(),"fixedValue");
        forAll(mesh.boundary(),pi)if(mesh.boundary()[pi].coupled())types[pi]=mesh.boundary()[pi].type();
        volVectorField HL(IOobject("HL",runTime.timeName(),mesh),mesh,
            dimensionedVector("zero",dimVelocity,vector::zero),types);
        volVectorField HG(IOobject("HG",runTime.timeName(),mesh),mesh,
            dimensionedVector("zero",dimVelocity,vector::zero),types);
        forAll(material,c)
        {
            if(!material[c]){eps[c]=0;continue;}
            scalar mobility=base/(mesh.C()[c].x()<.5?1.:contrast);
            rL[c]=mobility*(1-initialS);rG[c]=mobility*initialS;
            p[c]=.8*exactPressure(mesh.C()[c].x());
        }
        forAll(mesh.boundary(),pi)if(mesh.boundary()[pi].name().ends_with("Inlet"))
            forAll(mesh.boundary()[pi],f)if(material[mesh.boundary()[pi].faceCells()[f]])
            {
                HL.boundaryFieldRef()[pi][f]=vector(liquidFlux/(1-targetS),0,0);
                HG.boundaryFieldRef()[pi][f]=vector(gasFlux/targetS,0,0);
            }
        eps.correctBoundaryConditions();rL.correctBoundaryConditions();rG.correctBoundaryConditions();
        p.correctBoundaryConditions();HL.correctBoundaryConditions();HG.correctBoundaryConditions();
        surfaceScalarField phiL(IOobject("phiL",runTime.timeName(),mesh),mesh,dimensionedScalar("zero",dimVolume/dimTime,0));
        surfaceScalarField phiG(IOobject("phiG",runTime.timeName(),mesh),mesh,dimensionedScalar("zero",dimVolume/dimTime,0));
        surfaceScalarField faceL(IOobject("faceL",runTime.timeName(),mesh),mesh,dimensionedScalar("zero",dimPressure,0));
        surfaceScalarField faceG(IOobject("faceG",runTime.timeName(),mesh),mesh,dimensionedScalar("zero",dimPressure,0));
        network.solve(props,eps,rL,rG,HL,HG,Sl,Sg,1.,1.,p,sg,pc,phiL,phiG,faceL,faceG);
        scalar pressureError=0,saturationError=0,fluxError=0;
        forAll(material,c)if(material[c])
        {
            pressureError=max(pressureError,mag(p[c]-exactPressure(mesh.C()[c].x()))/contrast);
            saturationError=max(saturationError,mag(sg[c]-targetS));
        }
        auto checkFlux=[&](scalar l,scalar g,const vector& area,bool fluid)
        {
            const scalar direction=fluid?area.x():0;
            fluxError=max(fluxError,max(mag(l-liquidFlux*direction)/liquidFlux,
                mag(g-gasFlux*direction)/gasFlux));
        };
        forAll(mesh.neighbour(),f)
            checkFlux(phiL[f],phiG[f],mesh.Sf()[f],material[mesh.owner()[f]] && material[mesh.neighbour()[f]]);
        forAll(mesh.boundary(),pi)forAll(mesh.boundary()[pi],f)
        {
            const auto& patch=mesh.boundary()[pi];
            bool fluid=material[patch.faceCells()[f]];
            if(patch.coupled())fluid=fluid && neighbourMaterial[patch.start()-mesh.nInternalFaces()+f];
            checkFlux(phiL.boundaryField()[pi][f],phiG.boundaryField()[pi][f],patch.Sf()[f],fluid);
        }
        reduce(pressureError,maxOp<scalar>());reduce(saturationError,maxOp<scalar>());reduce(fluxError,maxOp<scalar>());
        Info<<"PROJECTION_CHECK contrast="<<contrast<<" pressure_error="<<pressureError
            <<" saturation_error="<<saturationError<<" flux_error="<<fluxError<<nl;
        passed=passed && pressureError<1e-7 && saturationError<1e-7 && fluxError<1e-7;
    }
    Info<<(passed?"PROJECTION_CHECK PASSED":"PROJECTION_CHECK FAILED")<<nl;
    return passed?0:1;
}
