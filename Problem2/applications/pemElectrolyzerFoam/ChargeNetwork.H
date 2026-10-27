// Conservative, monolithic Newton solve for electronic and ionic potentials.
// Each active potential is a graph node; CL cells link the two graphs through
// Butler-Volmer current. OpenFOAM's lduMatrix/PCG/DIC solves the Jacobian.
#include "lduPrimitiveMesh.H"
#include <vector>
#include <algorithm>

struct ChargeEdge
{
    Foam::label a, b, owner, neighbour, face, type;
};

class ChargeNetwork
{
    const Foam::fvMesh& mesh_;
    const Foam::labelList& material_;
    Foam::labelList s_, e_;
    std::vector<ChargeEdge> edges_;
    Foam::autoPtr<Foam::lduPrimitiveMesh> graph_;
    Foam::label nodes_;
    Foam::dictionary controls_;

public:
    ChargeNetwork(const Foam::fvMesh& mesh, const Foam::labelList& material)
    : mesh_(mesh), material_(material), s_(mesh.nCells(), -1),
      e_(mesh.nCells(), -1), nodes_(0)
    {
        using namespace Foam;
        forAll(material_, c)
            if (material_[c] != 1 && material_[c] != 4 && material_[c] != 7) s_[c] = nodes_++;
        forAll(material_, c)
            if (material_[c] >= 3 && material_[c] <= 5) e_[c] = nodes_++;
        forAll(mesh_.neighbour(), f)
        {
            label o = mesh_.owner()[f], n = mesh_.neighbour()[f];
            if (s_[o] >= 0 && s_[n] >= 0)
                edges_.push_back({s_[o], s_[n], o, n, f, 0});
            if (e_[o] >= 0 && e_[n] >= 0)
                edges_.push_back({e_[o], e_[n], o, n, f, 1});
        }
        forAll(material_, c)
            if (material_[c] == 3 || material_[c] == 5)
                edges_.push_back({s_[c], e_[c], c, c, -1, 2});
        for (auto& edge : edges_) if (edge.a > edge.b) std::swap(edge.a, edge.b);
        std::sort(edges_.begin(), edges_.end(), [](const ChargeEdge& a, const ChargeEdge& b)
        { return a.a == b.a ? a.b < b.b : a.a < b.a; });
        labelList lower(edges_.size()), upper(edges_.size());
        forAll(lower, f) { lower[f] = edges_[f].a; upper[f] = edges_[f].b; }
        graph_.reset(new lduPrimitiveMesh(nodes_, lower, upper, UPstream::worldComm, true));
        controls_.add("solver", word("PCG"));
        controls_.add("preconditioner", word("DIC"));
        controls_.add("tolerance", 1e-11);
        controls_.add("relTol", 0.0);
        controls_.add("maxIter", 6000);
    }

    Foam::scalar solve
    (
        const Foam::dictionary& props,
        const Foam::volScalarField& T, const Foam::volScalarField& C,
        const Foam::volScalarField& sigmaS, const Foam::volScalarField& sigmaE,
        Foam::volScalarField& phiS, Foam::volScalarField& phiE,
        Foam::volScalarField& iv, Foam::volScalarField& heat,
        Foam::volScalarField& coverage, Foam::scalar& chargeError
    )
    {
        using namespace Foam;
        const scalar F=96485.33212, R=8.314462618;
        const scalar j=props.get<scalar>("currentDensity"), area=props.get<scalar>("reactionArea");
        const scalar current=max(j*area, SMALL), pa=props.get<scalar>("operatingPressure");
        const scalar tr=props.get<scalar>("referenceTemperature");
        const scalar aa=props.get<scalar>("alphaAnodic"), ac=props.get<scalar>("alphaCathodic");
        const scalar floor=props.get<scalar>("concentrationFloorRatio");
        const scalar dcl=props.get<scalar>("catalystThickness");
        const bool nernst=props.get<bool>("includeNernstConcentration");
        const word coverageMode=props.get<word>("coveragePressureMode");
        const scalar heatSign=props.get<scalar>("reversibleHeatSign");
        scalarField potential(nodes_, 0), eq(mesh_.nCells(),0), pref(mesh_.nCells(),0);
        scalarField conductance(edges_.size(), 0), fixedDiag(nodes_,0), injection(nodes_,0);
        forAll(material_, c)
        {
            if (s_[c] >= 0) potential[s_[c]]=phiS[c];
            if (e_[c] >= 0) potential[e_[c]]=phiE[c];
            if (material_[c] != 3 && material_[c] != 5) continue;
            bool anode=material_[c]==5;
            scalar cref=props.get<scalar>(anode ? "referenceConcentrationO2" : "referenceConcentrationH2");
            scalar ratio=max(C[c]/cref, floor);
            scalar ratioT=T[c]/tr;
            // Antoine saturation pressure (Pa); valid only for liquid-water temperatures.
            scalar psat=133.322368*std::pow(10.0,8.07131-1730.63/(233.426+T[c]-273.15));
            scalar pressureFactor=coverageMode=="paperSI" ? 1.0/max(pa-psat,1.0) : pa/max(pa-psat,1.0);
            scalar localJ=mag(iv[c])>SMALL ? mag(iv[c])*dcl : j;
            scalar theta=(-97.25+182*ratioT-84*sqr(ratioT))*std::pow(localJ/300000.0,0.3)*pressureFactor;
            coverage[c]=min(max(theta,0.0),0.95);
            pref[c]=props.get<scalar>(anode ? "activeAreaAnode" : "activeAreaCathode")
                   *props.get<scalar>(anode ? "exchangeCurrentAnode" : "exchangeCurrentCathode")
                   *(1-coverage[c])*ratio;
            eq[c]=anode ? 1.229-9e-4*(T[c]-298.15) : 0.0;
            if (nernst) eq[c]+=(anode ? 1.0/4 : -1.0/2)*R*T[c]/F*std::log(ratio);
        }
        for (label f=0; f<label(edges_.size()); ++f)
        {
            const auto& edge=edges_[f];
            if (edge.type==2) continue;
            const auto& sigma=edge.type==0 ? sigmaS : sigmaE;
            scalar dO=mag(mesh_.Cf()[edge.face]-mesh_.C()[edge.owner]);
            scalar dN=mag(mesh_.Cf()[edge.face]-mesh_.C()[edge.neighbour]);
            conductance[f]=mesh_.magSf()[edge.face]/(dO/sigma[edge.owner]+dN/sigma[edge.neighbour]);
        }
        forAll(mesh_.boundary(), patchi)
        {
            const word& name=mesh_.boundary()[patchi].name();
            if (name!="anodeTerminal" && name!="cathodeTerminal") continue;
            const auto& patch=mesh_.boundary()[patchi];
            forAll(patch, f)
            {
                label c=patch.faceCells()[f], node=s_[c];
                scalar A=mesh_.magSf().boundaryField()[patchi][f];
                if (name=="anodeTerminal") injection[node]+=j*A;
                else fixedDiag[node]+=sigmaS[c]*A*patch.deltaCoeffs()[f];
            }
        }
        auto assemble = [&](const scalarField& x, scalarField& residual, scalarField& deriv)
        {
            residual=fixedDiag*x-injection;
            for (label f=0; f<label(edges_.size()); ++f)
            {
                const auto& edge=edges_[f];
                scalar flux;
                if (edge.type==2)
                {
                    label c=edge.owner;
                    scalar eta=x[s_[c]]-x[e_[c]]-eq[c], beta=F/(R*T[c]);
                    scalar ep=std::exp(min(max(aa*beta*eta,-60.0),60.0));
                    scalar em=std::exp(min(max(-ac*beta*eta,-60.0),60.0));
                    flux=pref[c]*(ep-em)*mesh_.V()[c];
                    deriv[f]=pref[c]*beta*(aa*ep+ac*em)*mesh_.V()[c];
                }
                else { flux=conductance[f]*(x[edge.a]-x[edge.b]); deriv[f]=conductance[f]; }
                residual[edge.a]+=flux;
                residual[edge.b]-=flux;
            }
        };
        scalarField residual(nodes_,0), deriv(edges_.size(),0);
        chargeError=GREAT;
        for (label iter=0; iter<props.get<label>("maxChargeNewton"); ++iter)
        {
            assemble(potential,residual,deriv);
            chargeError=sum(mag(residual))/current;
            if (chargeError<1e-7) break;
            lduMatrix A(graph_());
            A.diag()=fixedDiag;
            A.upper()=-deriv;
            for (label f=0; f<label(edges_.size()); ++f)
            {
                A.diag()[edges_[f].a]+=deriv[f];
                A.diag()[edges_[f].b]+=deriv[f];
            }
            scalarField delta(nodes_,0);
            FieldField<Field,scalar> coeffs(0);
            lduInterfaceFieldPtrsList interfaces(0);
            auto solver=lduMatrix::solver::New("charge",A,coeffs,coeffs,interfaces,controls_);
            solver->solve(delta,-residual);
            scalar step=min(1.0,0.2/max(gMax(mag(delta)),SMALL));
            scalarField trial(nodes_), trialRes(nodes_), trialDeriv(edges_.size());
            scalar norm=sum(sqr(residual));
            for (label ls=0; ls<16; ++ls)
            {
                trial=potential+step*delta;
                assemble(trial,trialRes,trialDeriv);
                if (sum(sqr(trialRes))<norm) break;
                step*=0.5;
            }
            potential=trial;
        }
        assemble(potential,residual,deriv);
        chargeError=sum(mag(residual))/current;
        iv.primitiveFieldRef()=0;
        heat.primitiveFieldRef()=0;
        forAll(material_, c)
        {
            if (s_[c]>=0) phiS[c]=potential[s_[c]];
            if (e_[c]>=0) phiE[c]=potential[e_[c]];
        }
        for (label f=0; f<label(edges_.size()); ++f)
        {
            const auto& edge=edges_[f];
            if (edge.type==2)
            {
                label c=edge.owner;
                scalar eta=phiS[c]-phiE[c]-eq[c], beta=F/(R*T[c]);
                iv[c]=pref[c]*(std::exp(min(max(aa*beta*eta,-60.0),60.0))
                                -std::exp(min(max(-ac*beta*eta,-60.0),60.0)));
                heat[c]+=iv[c]*eta;
                if (material_[c]==5) heat[c]+=heatSign*iv[c]*T[c]*(-9e-4);
            }
            else
            {
                scalar power=conductance[f]*sqr(potential[edge.a]-potential[edge.b]);
                heat[edge.owner]+=0.5*power/mesh_.V()[edge.owner];
                heat[edge.neighbour]+=0.5*power/mesh_.V()[edge.neighbour];
            }
        }
        scalar voltageIntegral=0, terminalArea=0;
        forAll(mesh_.boundary(), patchi)
        {
            const auto& patch=mesh_.boundary()[patchi];
            if (patch.name()!="anodeTerminal" && patch.name()!="cathodeTerminal") continue;
            forAll(patch,f)
            {
                label c=patch.faceCells()[f];
                scalar A=mesh_.magSf().boundaryField()[patchi][f], d=1/patch.deltaCoeffs()[f];
                if (patch.name()=="anodeTerminal")
                {
                    voltageIntegral+=A*(phiS[c]+j*d/sigmaS[c]); terminalArea+=A;
                    heat[c]+=sqr(j)*A*d/sigmaS[c]/mesh_.V()[c];
                    phiS.boundaryFieldRef()[patchi][f]=phiS[c]+j*d/sigmaS[c];
                }
                else
                {
                    heat[c]+=sigmaS[c]*A/d*sqr(phiS[c])/mesh_.V()[c];
                    phiS.boundaryFieldRef()[patchi][f]=0;
                }
            }
        }
        return voltageIntegral/terminalArea;
    }
};
