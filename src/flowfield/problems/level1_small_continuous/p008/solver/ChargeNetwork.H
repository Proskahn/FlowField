// Conservative, monolithic Newton solve for electronic and ionic potentials.
// Each active potential is a graph node; CL cells link the two graphs through
// Butler-Volmer current. OpenFOAM's lduMatrix/PCG/DIC solves the Jacobian.
#include "lduPrimitiveMesh.H"
#include "processorGAMGInterface.H"
#include "processorGAMGInterfaceField.H"
#include "processorFvPatch.H"
#include "syncTools.H"
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
    Foam::lduInterfacePtrsList interfaces_;
    Foam::autoPtr<Foam::lduPrimitiveMesh> graph_;
    Foam::PtrList<Foam::lduInterfaceField> ownedInterfaceFields_;
    Foam::lduInterfaceFieldPtrsList interfaceFields_;
    Foam::List<Foam::labelList> interfaceFaces_, interfaceTypes_;
    Foam::vectorField neighbourCentres_;
    Foam::label nodes_;
    Foam::dictionary controls_;

public:
    ChargeNetwork(const Foam::fvMesh& mesh, const Foam::labelList& material)
    : mesh_(mesh), material_(material), s_(mesh.nCells(), -1),
      e_(mesh.nCells(), -1), interfaces_(mesh.boundary().size()),
      ownedInterfaceFields_(mesh.boundary().size()),
      interfaceFields_(mesh.boundary().size()),
      interfaceFaces_(mesh.boundary().size()), interfaceTypes_(mesh.boundary().size()), nodes_(0)
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
        // Processor faces carry one edge for each potential that conducts on
        // BOTH sides. A CL face may carry two edges. Both ranks use face order,
        // then electronic/ionic order, so their interface addressing agrees.
        labelList neighbourMaterial;
        syncTools::swapBoundaryCellList(mesh_, material_, neighbourMaterial);
        syncTools::swapBoundaryCellPositions(mesh_, mesh_.C().primitiveField(), neighbourCentres_);
        forAll(mesh_.boundary(), p)
        {
            const auto& patch=mesh_.boundary()[p];
            if (!patch.coupled()) continue;
            if (!isA<processorFvPatch>(patch))
                FatalErrorInFunction << "Only processor coupled patches are supported." << exit(FatalError);
            const auto& proc=refCast<const processorFvPatch>(patch);
            DynamicList<label> cells, faces, types;
            forAll(patch, f)
            {
                label c=patch.faceCells()[f], m=neighbourMaterial[patch.start()-mesh_.nInternalFaces()+f];
                if (s_[c]>=0 && m!=1 && m!=4 && m!=7)
                { cells.append(s_[c]); faces.append(f); types.append(0); }
                if (e_[c]>=0 && m>=3 && m<=5)
                { cells.append(e_[c]); faces.append(f); types.append(1); }
            }
            interfaceFaces_[p]=labelList(faces);
            interfaceTypes_[p]=labelList(types);
            labelList addressing(cells), identity(cells.size());
            forAll(identity, i) identity[i]=i;
            auto* interface=new processorGAMGInterface
            (
                p, interfaces_, addressing, identity, UPstream::worldComm,
                proc.myProcNo(), proc.neighbProcNo(), tensorField(), UPstream::msgType()
            );
            interfaces_.set(p,interface);
            ownedInterfaceFields_.set(p,new processorGAMGInterfaceField(*interface,false,0));
            interfaceFields_.set(p,&ownedInterfaceFields_[p]);
        }
        // addInterfaces transfers ownership of the interface objects to graph_.
        graph_->addInterfaces(interfaces_,lduPrimitiveMesh::nonBlockingSchedule<processorGAMGInterface>(interfaces_));
        controls_.add("solver", word("PCG"));
        controls_.add("preconditioner", word("DIC"));
        controls_.add("tolerance", 1e-13);
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
        const scalar area=props.get<scalar>("reactionArea"), applied=props.get<scalar>("cell_voltage");
        const scalar current=10000*area; // Residual normalization: fixed 1 A/cm2 reference, not imposed current.
        const scalar tr=props.get<scalar>("kinetic_reference_temperature");

        scalarField potential(nodes_, 0), eq(mesh_.nCells(),0), pref(mesh_.nCells(),0);
        scalarField conductance(edges_.size(), 0), fixedDiag(nodes_,0), injection(nodes_,0);
        FieldField<Field,scalar> interfaceCoeffs(interfaces_.size());
        scalarField neighbourSigmaS, neighbourSigmaE;
        syncTools::swapBoundaryCellList(mesh_,sigmaS.primitiveField(),neighbourSigmaS);
        syncTools::swapBoundaryCellList(mesh_,sigmaE.primitiveField(),neighbourSigmaE);
        forAll(interfaces_, p)
        {
            interfaceCoeffs.set(p,new scalarField(interfaceFaces_[p].size(),0));
            if (!interfaces_.set(p)) continue;
            const auto& patch=mesh_.boundary()[p];
            forAll(interfaceFaces_[p], i)
            {
                label f=interfaceFaces_[p][i], c=patch.faceCells()[f];
                label b=patch.start()-mesh_.nInternalFaces()+f;
                const auto& sigma=interfaceTypes_[p][i]==0 ? sigmaS : sigmaE;
                scalar sigmaN=interfaceTypes_[p][i]==0 ? neighbourSigmaS[b] : neighbourSigmaE[b];
                scalar dO=mag(patch.Cf()[f]-mesh_.C()[c]);
                scalar dN=mag(patch.Cf()[f]-neighbourCentres_[b]);
                interfaceCoeffs[p][i]=patch.magSf()[f]/(dO/sigma[c]+dN/sigmaN);
            }
        }
        forAll(material_, c)
        {
            if (s_[c] >= 0) potential[s_[c]]=phiS[c];
            if (e_[c] >= 0) potential[e_[c]]=phiE[c];
            if (material_[c] != 3 && material_[c] != 5) continue;
            bool anode=material_[c]==5;
            coverage[c]=C[c];
            scalar activation=props.get<scalar>(anode ? "activation_energy_anode" : "activation_energy_cathode");
            pref[c]=props.get<scalar>(anode ? "active_area_anode" : "active_area_cathode")
                   *props.get<scalar>(anode ? "exchange_current_anode" : "exchange_current_cathode")
                   *(1-C[c])*std::exp(activation/R*(1/tr-1/T[c]));
            // Pure-gas and liquid activities at Pref=1 atm: Nernst logarithm zero.
            eq[c]=anode ? 1.481-.000846*T[c] : 0.0;
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
                scalar conduct=sigmaS[c]*A*patch.deltaCoeffs()[f];
                fixedDiag[node]+=conduct;
                if (name=="anodeTerminal") injection[node]+=conduct*applied;
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
                    scalar aa=.0675+.00095*T[c], ac=.1175+.00095*T[c];
                    scalar ep=std::exp(min(max(aa*beta*eta,-60.0),60.0));
                    scalar em=std::exp(min(max(-ac*beta*eta,-60.0),60.0));
                    flux=pref[c]*(ep-em)*mesh_.V()[c];
                    deriv[f]=pref[c]*beta*(aa*ep+ac*em)*mesh_.V()[c];
                }
                else { flux=conductance[f]*(x[edge.a]-x[edge.b]); deriv[f]=conductance[f]; }
                residual[edge.a]+=flux;
                residual[edge.b]-=flux;
            }
            scalarField cellS(mesh_.nCells(),0), cellE(mesh_.nCells(),0), nbrS, nbrE;
            forAll(material_, c)
            {
                if (s_[c]>=0) cellS[c]=x[s_[c]];
                if (e_[c]>=0) cellE[c]=x[e_[c]];
            }
            syncTools::swapBoundaryCellList(mesh_,cellS,nbrS);
            syncTools::swapBoundaryCellList(mesh_,cellE,nbrE);
            forAll(interfaces_, p) if (interfaces_.set(p))
            {
                const auto& patch=mesh_.boundary()[p];
                const auto& cells=interfaces_[p].faceCells();
                forAll(interfaceFaces_[p], i)
                {
                    label b=patch.start()-mesh_.nInternalFaces()+interfaceFaces_[p][i];
                    scalar neighbour=interfaceTypes_[p][i]==0 ? nbrS[b] : nbrE[b];
                    residual[cells[i]]+=interfaceCoeffs[p][i]*(x[cells[i]]-neighbour);
                }
            }
        };
        scalarField residual(nodes_,0), deriv(edges_.size(),0);
        chargeError=GREAT;
        for (label iter=0; iter<props.get<label>("maxChargeNewton"); ++iter)
        {
            assemble(potential,residual,deriv);
            chargeError=gSum(mag(residual))/current;
            if (chargeError<1e-11) break;
            lduMatrix A(graph_());
            A.diag()=fixedDiag;
            A.upper()=-deriv;
            for (label f=0; f<label(edges_.size()); ++f)
            {
                A.diag()[edges_[f].a]+=deriv[f];
                A.diag()[edges_[f].b]+=deriv[f];
            }
            scalarField delta(nodes_,0);
            forAll(interfaces_, p) if (interfaces_.set(p))
                forAll(interfaceCoeffs[p], i)
                    A.diag()[interfaces_[p].faceCells()[i]]+=interfaceCoeffs[p][i];
            auto solver=lduMatrix::solver::New("charge",A,interfaceCoeffs,interfaceCoeffs,interfaceFields_,controls_);
            solver->solve(delta,-residual);
            scalar step=min(1.0,0.2/max(gMax(mag(delta)),SMALL));
            scalarField trial(nodes_), trialRes(nodes_), trialDeriv(edges_.size());
            scalar norm=gSum(sqr(residual));
            for (label ls=0; ls<16; ++ls)
            {
                trial=potential+step*delta;
                assemble(trial,trialRes,trialDeriv);
                if (gSum(sqr(trialRes))<norm) break;
                step*=0.5;
            }
            potential=trial;
        }
        assemble(potential,residual,deriv);
        chargeError=gSum(mag(residual))/current;
        iv.primitiveFieldRef()=0;
        heat.primitiveFieldRef()=0;
        forAll(material_, c)
        {
            if (s_[c]>=0) phiS[c]=potential[s_[c]];
            if (e_[c]>=0) phiE[c]=potential[e_[c]];
        }
        phiS.correctBoundaryConditions(); phiE.correctBoundaryConditions();
        forAll(interfaces_, p) if (interfaces_.set(p))
        {
            const auto& patch=mesh_.boundary()[p];
            forAll(interfaceFaces_[p], i)
            {
                label f=interfaceFaces_[p][i], c=patch.faceCells()[f];
                const auto& potentialField=interfaceTypes_[p][i]==0 ? phiS : phiE;
                scalar jump=potentialField[c]-potentialField.boundaryField()[p][f];
                // Each rank receives half of this shared face's Joule loss.
                heat[c]+=0.5*interfaceCoeffs[p][i]*sqr(jump)/mesh_.V()[c];
            }
        }
        for (label f=0; f<label(edges_.size()); ++f)
        {
            const auto& edge=edges_[f];
            if (edge.type==2)
            {
                label c=edge.owner;
                scalar eta=phiS[c]-phiE[c]-eq[c], beta=F/(R*T[c]);
                scalar aa=.0675+.00095*T[c], ac=.1175+.00095*T[c];
                iv[c]=pref[c]*(std::exp(min(max(aa*beta*eta,-60.0),60.0))
                                -std::exp(min(max(-ac*beta*eta,-60.0),60.0)));
                heat[c]+=iv[c]*eta;
                if (material_[c]==5) heat[c]+=iv[c]*T[c]*(-.000846);
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
                    voltageIntegral+=A*applied; terminalArea+=A;
                    heat[c]+=sigmaS[c]*A/d*sqr(applied-phiS[c])/mesh_.V()[c];
                    phiS.boundaryFieldRef()[patchi][f]=applied;
                }
                else
                {
                    heat[c]+=sigmaS[c]*A/d*sqr(phiS[c])/mesh_.V()[c];
                    phiS.boundaryFieldRef()[patchi][f]=0;
                }
            }
        }
        reduce(voltageIntegral,sumOp<scalar>());
        reduce(terminalArea,sumOp<scalar>());
        iv.correctBoundaryConditions(); heat.correctBoundaryConditions(); coverage.correctBoundaryConditions();
        return voltageIntegral/terminalArea;
    }
};
