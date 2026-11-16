// Monolithic pressure/saturation Newton projection. Solves total volume and
// gas continuity together, with bounded line searches and conservative fluxes.
// Phase momentum predictors remain distributed OpenFOAM finite-volume matrices.
#include <map>
#include <array>
#include <Eigen/SparseLU>

class FlowNetwork
{
    const Foam::fvMesh& mesh_;
    const Foam::labelList& mat_;
    Foam::labelList index_;
    Foam::label count_;
    std::map<std::pair<Foam::label,Foam::label>,Foam::label> edgeIndex_;
    Foam::labelList lower_,upper_;
    Foam::lduInterfacePtrsList interfaces_;
    Foam::autoPtr<Foam::lduPrimitiveMesh> graph_;
    Foam::PtrList<Foam::lduInterfaceField> owned_;
    Foam::lduInterfaceFieldPtrsList interfaceFields_;
    Foam::List<Foam::labelList> faces_,rows_,cols_;
    Foam::labelList nbrMat_;
    Foam::vectorField nbrC_;
    Foam::dictionary controls_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> factorization_;
    bool patternAnalyzed_=false;

public:
    FlowNetwork(const Foam::fvMesh& mesh,const Foam::labelList& mat)
    :mesh_(mesh),mat_(mat),index_(mesh.nCells(),-1),count_(0),
     interfaces_(mesh.boundary().size()),owned_(mesh.boundary().size()),
     interfaceFields_(mesh.boundary().size()),faces_(mesh.boundary().size()),
     rows_(mesh.boundary().size()),cols_(mesh.boundary().size())
    {
        using namespace Foam;
        auto active=[](label m){return m==1||m==2||m==3||m==5||m==6||m==7;};
        // Group the strongly coupled through-thickness column before moving
        // laterally. Thin porous layers otherwise defeat scalar ILU ordering.
        std::vector<label> cells;
        forAll(mat_,c)if(active(mat_[c]))cells.push_back(c);
        std::sort(cells.begin(),cells.end(),[&](label a,label b)
        {const auto& ca=mesh_.C()[a];const auto& cb=mesh_.C()[b];
         if(mag(ca.x()-cb.x())>1e-10)return ca.x()<cb.x();
         if(mag(ca.y()-cb.y())>1e-10)return ca.y()<cb.y();
         return ca.z()<cb.z();});
        for(label c:cells)index_[c]=2*count_++;
        auto edge=[&](label a,label b){if(a>b)std::swap(a,b);edgeIndex_[{a,b}]=0;};
        forAll(index_,c)if(index_[c]>=0)edge(index_[c],index_[c]+1);
        forAll(mesh_.neighbour(),f)
        {label a=index_[mesh_.owner()[f]],b=index_[mesh_.neighbour()[f]];if(a<0||b<0)continue;
         for(label i=0;i<2;++i)for(label j=0;j<2;++j)edge(a+i,b+j);}
        lower_.setSize(edgeIndex_.size());upper_.setSize(edgeIndex_.size());label e=0;
        for(auto& pair:edgeIndex_){pair.second=e;lower_[e]=pair.first.first;upper_[e]=pair.first.second;++e;}
        graph_.reset(new lduPrimitiveMesh(2*count_,lower_,upper_,UPstream::worldComm,true));
        syncTools::swapBoundaryCellList(mesh_,mat_,nbrMat_);
        syncTools::swapBoundaryCellPositions(mesh_,mesh_.C().primitiveField(),nbrC_);
        forAll(mesh_.boundary(),pi)
        {
            const auto& patch=mesh_.boundary()[pi];if(!patch.coupled())continue;
            const auto& proc=refCast<const processorFvPatch>(patch);
            DynamicList<label> cells,faces,rows,cols;
            forAll(patch,f)
            {
                label c=patch.faceCells()[f],b=patch.start()-mesh_.nInternalFaces()+f;
                if(index_[c]<0||!active(nbrMat_[b]))continue;
                // Transpose cross-field entries on the higher rank so exchanged
                // node values arrive at the matching off-diagonal column.
                for(label k=0;k<4;++k)
                {label row=proc.myProcNo()<proc.neighbProcNo()?k/2:k%2;
                 label col=proc.myProcNo()<proc.neighbProcNo()?k%2:k/2;
                 cells.append(index_[c]+row);faces.append(f);rows.append(row);cols.append(col);}
            }
            faces_[pi]=labelList(faces);rows_[pi]=labelList(rows);cols_[pi]=labelList(cols);
            labelList addr(cells),identity(cells.size());forAll(identity,i)identity[i]=i;
            auto* iface=new processorGAMGInterface(pi,interfaces_,addr,identity,UPstream::worldComm,
                proc.myProcNo(),proc.neighbProcNo(),tensorField(),UPstream::msgType());
            interfaces_.set(pi,iface);owned_.set(pi,new processorGAMGInterfaceField(*iface,false,0));
            interfaceFields_.set(pi,&owned_[pi]);
        }
        graph_->addInterfaces(interfaces_,lduPrimitiveMesh::nonBlockingSchedule<processorGAMGInterface>(interfaces_));
        controls_.add("solver",word("PBiCGStab"));controls_.add("preconditioner",word("DILU"));
        controls_.add("tolerance",1e-10);controls_.add("relTol",0.0);controls_.add("maxIter",10000);
    }

    // Sparse LU in serial; exact local-block LU preconditions distributed GMRES.
    // All processor couplings remain in the global matrix-vector product.
    Foam::label linearSolve(Foam::lduMatrix& A,const Foam::scalarField& rhs,
        Foam::scalarField& x,const Foam::FieldField<Foam::Field,Foam::scalar>& coeff,
        Foam::scalar& error)
    {
        using namespace Foam;
        const label restart=80,maximum=800,n=rhs.size();
        std::vector<Eigen::Triplet<double>> entries;entries.reserve(n+2*edgeIndex_.size());
        forAll(rhs,i)entries.emplace_back(i,i,A.diag()[i]);
        for(const auto& pair:edgeIndex_)
        {label a=pair.first.first,b=pair.first.second,e=pair.second;
         entries.emplace_back(a,b,A.upper()[e]);entries.emplace_back(b,a,A.lower()[e]);}
        Eigen::SparseMatrix<double> matrix(n,n);matrix.setFromTriplets(entries.begin(),entries.end());matrix.makeCompressed();
        // A rank may own only solid cells. It still participates in every
        // distributed reduction and matrix exchange, but has no local LU block.
        if(n)
        {
            if(!patternAnalyzed_){factorization_.analyzePattern(matrix);patternAnalyzed_=true;}
            factorization_.factorize(matrix);
            if(factorization_.info()!=Eigen::Success)FatalErrorInFunction<<"Phase Jacobian factorization failed"<<exit(FatalError);
        }
        if(!Pstream::parRun())
        {
            if(!n){error=0;return 0;}
            Eigen::Map<const Eigen::VectorXd> b(rhs.cdata(),n);
            Eigen::VectorXd solution=factorization_.solve(b);
            forAll(x,i)x[i]=solution[i];
            error=(matrix*solution-b).norm()/max(b.norm(),VSMALL);
            return 1;
        }
        const scalar reference=max(std::sqrt(gSum(sqr(rhs))),VSMALL);
        scalarField residual(rhs),product(n,0);label iterations=0;
        while(iterations<maximum)
        {
            A.Amul(product,x,coeff,interfaceFields_,0);residual=rhs-product;
            scalar beta=std::sqrt(gSum(sqr(residual)));error=beta/reference;
            if(error<1e-6)return iterations;
            List<scalarField> v(restart+1),z(restart);
            forAll(v,i){v[i].setSize(n,0);}
            forAll(z,i){z[i].setSize(n,0);}
            v[0]=residual/beta;
            List<scalarField> h(restart+1);forAll(h,i)h[i].setSize(restart,0);
            scalarField cosine(restart,0),sine(restart,0),g(restart+1,0);g[0]=beta;
            label used=0;
            for(label j=0;j<restart&&iterations<maximum;++j,++iterations)
            {
                if(n)
                {Eigen::Map<const Eigen::VectorXd> b(v[j].cdata(),n);
                 Eigen::VectorXd solution=factorization_.solve(b);
                 forAll(z[j],i)z[j][i]=solution[i];}
                A.Amul(product,z[j],coeff,interfaceFields_,0);
                // Reorthogonalization protects the thin-layer, high-contrast system.
                for(label pass=0;pass<2;++pass)for(label i=0;i<=j;++i)
                {scalar a=gSum(product*v[i]);h[i][j]+=a;product-=a*v[i];}
                h[j+1][j]=std::sqrt(gSum(sqr(product)));
                if(h[j+1][j]>VSMALL)v[j+1]=product/h[j+1][j];
                for(label i=0;i<j;++i)
                {scalar a=cosine[i]*h[i][j]+sine[i]*h[i+1][j];h[i+1][j]=-sine[i]*h[i][j]+cosine[i]*h[i+1][j];h[i][j]=a;}
                scalar norm=std::hypot(h[j][j],h[j+1][j]);
                if(norm<VSMALL)break;
                cosine[j]=h[j][j]/norm;sine[j]=h[j+1][j]/norm;
                h[j][j]=norm;h[j+1][j]=0;
                g[j+1]=-sine[j]*g[j];g[j]*=cosine[j];used=j+1;
                if(mag(g[j+1])/reference<1e-6){++iterations;break;}
            }
            if(!used)break;
            scalarField y(used,0);
            for(label i=used-1;i>=0;--i)
            {scalar value=g[i];for(label j=i+1;j<used;++j)value-=h[i][j]*y[j];y[i]=value/h[i][i];}
            for(label i=0;i<used;++i)x+=y[i]*z[i];
        }
        A.Amul(product,x,coeff,interfaceFields_,0);error=std::sqrt(gSum(sqr(rhs-product)))/reference;
        return iterations;
    }

    void solve(const Foam::dictionary& props,const Foam::volScalarField& eps,
        const Foam::volScalarField& rL,const Foam::volScalarField& rG,
        const Foam::volVectorField& HL,const Foam::volVectorField& HG,
        const Foam::volScalarField& Sl,const Foam::volScalarField& Sg,Foam::scalar dt,Foam::scalar pseudoCo,
        Foam::volScalarField& p,Foam::volScalarField& sg,Foam::volScalarField& pc,
        Foam::surfaceScalarField& phiL,Foam::surfaceScalarField& phiG,
        Foam::surfaceScalarField& pressureFaceL,Foam::surfaceScalarField& pressureFaceG)
    {
        using namespace Foam;
        const scalar pressureScale=1000.,satFloor=1e-6;
        scalarField x(2*count_,0),oldS(sg.primitiveField()),initialS(count_,0);
        // Solve for the saturation increment, not the absolute saturation.
        // Retry capacities can be large enough that s_new-s_old loses the
        // entire increment to roundoff. capacity*deltaS preserves that term.
        forAll(index_,c)if(index_[c]>=0)
        {x[index_[c]]=p[c]/pressureScale;initialS[index_[c]/2]=sg[c];}
        surfaceScalarField hL("hL",fvc::flux(HL)),hG("hG",fvc::flux(HG));
        scalarField nbrE;syncTools::swapBoundaryCellList(mesh_,eps.primitiveField(),nbrE);
        // Local pseudo-time limits the explicit momentum/saturation coupling
        // where intrinsic gas velocities become large in nearly empty cells.
        // This capacity vanishes from the equations at a steady fixed point.
        scalarField capacity(eps.primitiveField()*mesh_.V()/dt);
        auto addCapacity=[&](label c,scalar l,scalar g)
        {capacity[c]+=(mag(l)/max(1-oldS[c],satFloor)+mag(g)/max(oldS[c],satFloor))/pseudoCo;};
        forAll(mesh_.neighbour(),f)
        {
            label o=mesh_.owner()[f],n=mesh_.neighbour()[f];
            if(index_[o]<0||index_[n]<0)continue;
            addCapacity(o,phiL[f],phiG[f]);addCapacity(n,phiL[f],phiG[f]);
        }
        forAll(mesh_.boundary(),pi)
        {
            const auto& patch=mesh_.boundary()[pi];
            forAll(patch,f)
            {
                label c=patch.faceCells()[f],b=patch.start()-mesh_.nInternalFaces()+f;
                if(index_[c]<0)continue;
                if(patch.coupled())
                {if(nbrE[b]<=0)continue;}
                addCapacity(c,phiL.boundaryField()[pi][f],phiG.boundaryField()[pi][f]);
            }
        }
        scalarField baseL(mesh_.nCells()),baseG(mesh_.nCells());
        // Factor out the entire cubic relative permeability before harmonic
        // interpolation. In the Darcy limit the base is exactly K/mu, so an
        // empty downstream cell cannot block inflow from an occupied donor.
        forAll(baseL,c){baseL[c]=rL[c]*sqr(eps[c])/max(1-oldS[c],1e-6);baseG[c]=rG[c]*sqr(eps[c])/max(oldS[c],1e-6);}
        scalarField nbrBL,nbrBG;
        syncTools::swapBoundaryCellList(mesh_,baseL,nbrBL);syncTools::swapBoundaryCellList(mesh_,baseG,nbrBG);
        auto cap=[&](label material,scalar s)
        {
            if(material==1||material==7)return std::array<scalar,2>{{0,0}};
            scalar porosity=props.get<scalar>(material==2||material==6?"porosity_ptl":"porosity_cl");
            scalar perm=props.get<scalar>(material==2||material==6?"permeability_ptl":"permeability_cl");
            scalar cosine=std::cos(props.get<scalar>("contact_angle_deg")*constant::mathematical::pi/180);
            scalar z=cosine>0?s:1-s,pref=props.get<scalar>("surface_tension")*cosine*std::sqrt(porosity/perm);
            return std::array<scalar,2>{{pref*(1.417*z-2.120*sqr(z)+1.263*pow3(z)),
                pref*(1.417-4.240*z+3.789*sqr(z))*(cosine>0?1:-1)}};
        };
        auto physicalMobility=[&](label material,scalar fraction,bool gas)
        {
            scalar perm=props.get<scalar>(material==1||material==7?"permeability_channel":
                material==2||material==6?"permeability_ptl":"permeability_cl");
            scalar viscosity=props.get<scalar>(gas?(material>=5?"oxygen_viscosity":"hydrogen_viscosity"):"liquid_viscosity");
            return perm*pow3(max(fraction,satFloor))/viscosity;
        };
        bool fixedDonors=false;
        boolList pressureDonorL(mesh_.nFaces(),true),pressureDonorG(mesh_.nFaces(),true);
        // Returns phase flux and derivatives w.r.t. [pO*,sO,pN*,sN].
        auto phaseFlux=[&](bool gas,label face,scalar po,scalar so,scalar pn,scalar sn,
            scalar bo,scalar bn,scalar do_,scalar dn,scalar area,scalar prediction,
            scalar porosity,label mo,label mn)
        {
            scalar ao=gas?so:1-so,an=gas?sn:1-sn,sign=gas?1:-1;
            auto co=gas?cap(mo,so):std::array<scalar,2>{{0,0}};
            auto cn=gas?cap(mn,sn):std::array<scalar,2>{{0,0}};
            scalar jump=pressureScale*(po-pn)+co[0]-cn[0];
            // Upwind each transport contribution by its own driving direction.
            // Harmonic phase fractions would disconnect a newly emptied cell,
            // preventing its refill and defeating positivity of the transport.
            scalar resistance=do_/max(bo,VSMALL)+dn/max(bn,VSMALL);
            scalar base=area/resistance;
            bool pressureOwner=jump>=0,upOwner=prediction>=0;
            auto& donors=gas?pressureDonorG:pressureDonorL;
            if(fixedDonors)pressureOwner=donors[face];else donors[face]=pressureOwner;
            scalar alpha=pressureOwner?ao:an,mobility=base*pow3(alpha);
            scalar dmo=pressureOwner?3*base*sqr(alpha)*sign:0;
            scalar dmn=pressureOwner?0:3*base*sqr(alpha)*sign;
            scalar flux=prediction*porosity*(upOwner?ao:an)+mobility*jump;
            // A material interface has continuous phase pressure but a jump
            // in its normal gradient. Use physical half-cell Darcy resistance
            // for that pressure trace; never numerical momentum diagonals.
            scalar wo=dn,wn=do_;
            if(mo!=mn){wo=dn/physicalMobility(mn,an,gas);wn=do_/physicalMobility(mo,ao,gas);}
            scalar facePressure=(wo*(pressureScale*po+co[0])+wn*(pressureScale*pn+cn[0]))/(wo+wn);
            std::array<scalar,6> result{{flux,mobility*pressureScale,
                dmo*jump+mobility*co[1]+(upOwner?prediction*porosity*sign:0),
                -mobility*pressureScale,dmn*jump-mobility*cn[1]+(!upOwner?prediction*porosity*sign:0),facePressure}};
            return result;
        };
        // lduPrimitiveMesh owns (and transfers) lower_/upper_; use the retained
        // edge map for coefficient sizes after constructing the graph.
        lduMatrix A(graph_());A.diag().setSize(2*count_);A.upper().setSize(edgeIndex_.size());A.lower().setSize(edgeIndex_.size());
        FieldField<Field,scalar> interfaceCoeffs(interfaces_.size());
        forAll(interfaces_,pi)interfaceCoeffs.set(pi,new scalarField(faces_[pi].size(),0));
        auto assemble=[&](const scalarField& trial,scalarField& res,bool jacobian,bool saveFlux)
        {
            res=0;if(jacobian){A.diag()=0;A.upper()=0;A.lower()=0;forAll(interfaceCoeffs,pi)interfaceCoeffs[pi]=0;}
            auto add=[&](label row,label col,scalar value)
            {if(!jacobian)return;if(row==col)A.diag()[row]+=value;
             else{label a=min(row,col),b=max(row,col),e=edgeIndex_.at({a,b});if(row<col)A.upper()[e]+=value;else A.lower()[e]+=value;}};
            scalarField pv(mesh_.nCells(),0),sv(mesh_.nCells(),0),nbrP,nbrS;
            forAll(index_,c)if(index_[c]>=0)
            {
                label a=index_[c];pv[c]=trial[a];sv[c]=oldS[c]+trial[a+1];scalar V=mesh_.V()[c];
                res[a]=-(Sl[c]+Sg[c])*V;
                res[a+1]=capacity[c]*trial[a+1]-Sg[c]*V;
                add(a+1,a+1,capacity[c]);
            }
            syncTools::swapBoundaryCellList(mesh_,pv,nbrP);syncTools::swapBoundaryCellList(mesh_,sv,nbrS);
            forAll(mesh_.neighbour(),f)
            {
                label o=mesh_.owner()[f],n=mesh_.neighbour()[f],a=index_[o],b=index_[n];
                if(a<0||b<0)
                {if(saveFlux){phiL[f]=0;phiG[f]=0;label c=a>=0?o:n;
                 pressureFaceL[f]=(a>=0||b>=0)?pressureScale*pv[c]:0;
                 pressureFaceG[f]=(a>=0||b>=0)?pressureScale*pv[c]+cap(mat_[c],sv[c])[0]:0;}continue;}
                scalar do_=mag(mesh_.Cf()[f]-mesh_.C()[o]),dn=mag(mesh_.Cf()[f]-mesh_.C()[n]);
                scalar ef=(do_+dn)/(do_/eps[o]+dn/eps[n]);
                auto L=phaseFlux(false,f,pv[o],sv[o],pv[n],sv[n],baseL[o],baseL[n],do_,dn,mesh_.magSf()[f],hL[f],ef,mat_[o],mat_[n]);
                auto G=phaseFlux(true,f,pv[o],sv[o],pv[n],sv[n],baseG[o],baseG[n],do_,dn,mesh_.magSf()[f],hG[f],ef,mat_[o],mat_[n]);
                res[a]+=L[0]+G[0];res[b]-=L[0]+G[0];res[a+1]+=G[0];res[b+1]-=G[0];
                label columns[4]={a,a+1,b,b+1};
                for(label k=0;k<4;++k){add(a,columns[k],L[k+1]+G[k+1]);add(b,columns[k],-L[k+1]-G[k+1]);add(a+1,columns[k],G[k+1]);add(b+1,columns[k],-G[k+1]);}
                if(saveFlux){phiL[f]=L[0];phiG[f]=G[0];
                    pressureFaceL[f]=L[5];pressureFaceG[f]=G[5];}
            }
            forAll(mesh_.boundary(),pi)
            {
                const auto& patch=mesh_.boundary()[pi];
                label interfaceEntry=0;
                forAll(patch,f)
                {
                    label c=patch.faceCells()[f],a=index_[c],bd=patch.start()-mesh_.nInternalFaces()+f;
                    if(a<0){if(saveFlux){phiL.boundaryFieldRef()[pi][f]=0;phiG.boundaryFieldRef()[pi][f]=0;pressureFaceL.boundaryFieldRef()[pi][f]=0;pressureFaceG.boundaryFieldRef()[pi][f]=0;}continue;}
                    if(patch.coupled())
                    {
                        label m=nbrMat_[bd];if(m==0||m==4||m==8){if(saveFlux){phiL.boundaryFieldRef()[pi][f]=0;phiG.boundaryFieldRef()[pi][f]=0;pressureFaceL.boundaryFieldRef()[pi][f]=pressureScale*pv[c];pressureFaceG.boundaryFieldRef()[pi][f]=pressureScale*pv[c]+cap(mat_[c],sv[c])[0];}continue;}
                        scalar do_=mag(patch.Cf()[f]-mesh_.C()[c]),dn=mag(patch.Cf()[f]-nbrC_[bd]);
                        scalar ef=(do_+dn)/(do_/eps[c]+dn/nbrE[bd]);
                        auto L=phaseFlux(false,patch.start()+f,pv[c],sv[c],nbrP[bd],nbrS[bd],baseL[c],nbrBL[bd],do_,dn,patch.magSf()[f],hL.boundaryField()[pi][f],ef,mat_[c],m);
                        auto G=phaseFlux(true,patch.start()+f,pv[c],sv[c],nbrP[bd],nbrS[bd],baseG[c],nbrBG[bd],do_,dn,patch.magSf()[f],hG.boundaryField()[pi][f],ef,mat_[c],m);
                        res[a]+=L[0]+G[0];res[a+1]+=G[0];
                        for(label k=0;k<2;++k){add(a,a+k,L[k+1]+G[k+1]);add(a+1,a+k,G[k+1]);}
                        for(label k=0;k<4;++k)
                        {label i=interfaceEntry++,row=rows_[pi][i],col=cols_[pi][i];
                         if(jacobian)interfaceCoeffs[pi][i]=-(row==0?L[3+col]+G[3+col]:G[3+col]);}
                        if(saveFlux){phiL.boundaryFieldRef()[pi][f]=L[0];phiG.boundaryFieldRef()[pi][f]=G[0];
                            pressureFaceL.boundaryFieldRef()[pi][f]=L[5];
                            pressureFaceG.boundaryFieldRef()[pi][f]=G[5];}
                    }
                    else if(patch.name().ends_with("Inlet"))
                    {
                        scalar sl=props.get<scalar>("inlet_liquid_saturation");
                        scalar L=hL.boundaryField()[pi][f]*sl,G=hG.boundaryField()[pi][f]*(1-sl);
                        res[a]+=L+G;res[a+1]+=G;
                        if(saveFlux){phiL.boundaryFieldRef()[pi][f]=L;phiG.boundaryFieldRef()[pi][f]=G;
                            pressureFaceL.boundaryFieldRef()[pi][f]=pressureScale*pv[c];pressureFaceG.boundaryFieldRef()[pi][f]=pressureScale*pv[c];}
                    }
                    else if(patch.name().ends_with("Outlet"))
                    {
                        scalar d=1/patch.deltaCoeffs()[f],area=patch.magSf()[f],s=sv[c];
                        scalar ml=baseL[c]*pow3(1-s)*area/d;
                        scalar mg=baseG[c]*pow3(s)*area/d;
                        scalar L=hL.boundaryField()[pi][f]*(1-s)+ml*pv[c]*pressureScale;
                        scalar G=hG.boundaryField()[pi][f]*s+mg*pv[c]*pressureScale;
                        scalar dl=-hL.boundaryField()[pi][f]-3*baseL[c]*sqr(1-s)*area/d*pv[c]*pressureScale;
                        scalar dg=hG.boundaryField()[pi][f]+3*baseG[c]*sqr(s)*area/d*pv[c]*pressureScale;
                        res[a]+=L+G;res[a+1]+=G;
                        add(a,a,(ml+mg)*pressureScale);add(a,a+1,dl+dg);
                        add(a+1,a,mg*pressureScale);add(a+1,a+1,dg);
                        if(saveFlux){phiL.boundaryFieldRef()[pi][f]=L;phiG.boundaryFieldRef()[pi][f]=G;
                            pressureFaceL.boundaryFieldRef()[pi][f]=0;pressureFaceG.boundaryFieldRef()[pi][f]=0;}
                    }
                    else if(saveFlux){phiL.boundaryFieldRef()[pi][f]=0;phiG.boundaryFieldRef()[pi][f]=0;pressureFaceL.boundaryFieldRef()[pi][f]=pressureScale*pv[c];pressureFaceG.boundaryFieldRef()[pi][f]=pressureScale*pv[c]+cap(mat_[c],sv[c])[0];}
                }
            }
        };
        scalarField res(2*count_),trial(2*count_),trialRes(2*count_);
        scalar flow=2*props.get<scalar>("inlet_flow_ml_min")*1e-6/60+gSum(Sg.primitiveField()*mesh_.V());
        scalar error=GREAT;
        const scalarField initialX(x);
        bool projectionAccepted=false;
        // Momentum coefficients lag saturation within a segregated iteration.
        // Reduce local pseudo-time if a phase would lose over 20% of its
        // inventory in one update; re-solve conservation, never clip the field.
        for(label retry=0;retry<12;++retry)
        {
        for(label iteration=0;iteration<40;++iteration)
        {
            assemble(x,res,true,false);error=gSum(mag(res))/flow;
            if(mesh_.time().value()==1 && iteration==0 && retry==0)
            {
                scalarField direction(x.size()),product(x.size());
                scalar errors[2]={0,0};const scalar h=1e-6;
                // The flux is piecewise differentiable at donor switches.
                // Verify each analytic block on its current branch, with a
                // relative saturation perturbation even near a vanished phase.
                fixedDonors=true;
                for(label field=0;field<2;++field)
                {
                    direction=0;
                    forAll(direction,i)if(i%2==field)
                    {scalar s=initialS[i/2]+x[i];
                     scalar scale=field?max(satFloor,min(s,1-s)):max(1.,mag(x[i]));
                     direction[i]=scale*std::sin(1.2345*(i+1+Pstream::myProcNo()));}
                    trial=x+h*direction;assemble(trial,trialRes,false,false);
                    A.Amul(product,direction,interfaceCoeffs,interfaceFields_,0);
                    errors[field]=std::sqrt(gSum(sqr((trialRes-res)/h-product))/max(gSum(sqr(product)),VSMALL));
                }
                fixedDonors=false;
                scalar jacobianError=max(errors[0],errors[1]);
                Info<<"Phase Jacobian finite-difference relative error = "<<jacobianError
                    <<" (pressure="<<errors[0]<<"; saturation="<<errors[1]<<"; fixed upwind donors)"<<endl;
                if(!std::isfinite(jacobianError)||jacobianError>1e-4)
                    FatalErrorInFunction<<"Phase Jacobian verification failed"<<exit(FatalError);
            }
            if(error<1e-7)break;
            scalarField delta(2*count_,0);
            scalar linearError=GREAT;label linearIterations=linearSolve(A,-res,delta,interfaceCoeffs,linearError);
            if(mesh_.time().value()<=2 || linearError>1e-4) Info<<"Projection Newton "<<iteration<<": balance="<<error<<"; linear iterations="<<linearIterations<<"; residual="<<linearError<<endl;
            if(!std::isfinite(linearError)||linearError>.05)break;
            // A numerically absent phase must not force the global Newton
            // fraction-to-boundary step to zero. Freeze only an outward search
            // direction at an active bound; acceptance still checks the full,
            // unmodified conservation residual, including these cells.
            for(label i=0;i<count_;++i)
            {
                scalar s=initialS[i]+x[2*i+1];
                if((s<=satFloor*(1+1e-5)&&delta[2*i+1]<0)
                   ||(s>=1-satFloor*(1+1e-5)&&delta[2*i+1]>0))delta[2*i+1]=0;
            }
            scalar step=1;
            for(label i=0;i<count_;++i)
            {scalar ds=delta[2*i+1],s=initialS[i]+x[2*i+1];if(ds>0)step=min(step,.99*(1-satFloor-s)/ds);if(ds<0)step=min(step,.99*(satFloor-s)/ds);}
            reduce(step,minOp<scalar>());scalar norm=gSum(sqr(res));bool accepted=false;
            for(label ls=0;ls<20;++ls)
            {trial=x+step*delta;assemble(trial,trialRes,false,false);if(gSum(sqr(trialRes))<norm){accepted=true;break;}step*=.5;}
            if(!accepted)break;
            if(mesh_.time().value()<=2) Info<<"  accepted step="<<step<<endl;
            x=trial;
        }
        assemble(x,res,false,false);error=gSum(mag(res))/flow;
        scalar depletion=0;
        forAll(index_,c)if(index_[c]>=0)
        {
            scalar change=x[index_[c]+1];
            scalar loss=change<0?-change/max(oldS[c],satFloor):change/max(1-oldS[c],satFloor);
            depletion=max(depletion,loss);
        }
        reduce(depletion,maxOp<scalar>());
        if(std::isfinite(error)&&error<1e-5&&depletion<=.2)
        {projectionAccepted=true;break;}
        // Preserve the final attempted state and its capacity for the failure
        // diagnostic; resetting here would report the unevaluated start state.
        if(retry==11)break;
        forAll(index_,c)if(index_[c]>=0)
        {
            scalar change=x[index_[c]+1];
            scalar loss=change<0?-change/max(oldS[c],satFloor):change/max(1-oldS[c],satFloor);
            scalar factor=max(1.,loss/.15);
            if(!std::isfinite(error)||error>=1e-5)factor=max(factor,10.);
            capacity[c]*=factor;
        }
        if(retry>0 || mesh_.time().value()<=2)
            Info<<"Projection pseudo-time retry "<<retry+1<<": balance="<<error<<"; maximum phase depletion="<<depletion<<endl;
        x=initialX;
        }
        assemble(x,res,false,true);error=gSum(mag(res))/flow;
        if(!projectionAccepted||!std::isfinite(error)||error>1e-5)
            FatalErrorInFunction<<"Pressure/saturation projection did not converge: "<<error<<exit(FatalError);
        // Steady SIMPLE pressure relaxation: retain the conservative face
        // fluxes from the full projection, but use relaxed cell pressure for
        // momentum correction. Recompute its physical pressure trace at the
        // new saturation; never scale the pressure force by a momentum diagonal.
        surfaceScalarField projectedPhiL("projectedPhiL",phiL),projectedPhiG("projectedPhiG",phiG);
        for(label c=0;c<count_;++c)x[2*c]=initialX[2*c]+.3*(x[2*c]-initialX[2*c]);
        assemble(x,trialRes,false,true);
        phiL=projectedPhiL;phiG=projectedPhiG;
        forAll(index_,c)if(index_[c]>=0)
        {p[c]=pressureScale*x[index_[c]];sg[c]=oldS[c]+x[index_[c]+1];pc[c]=cap(mat_[c],sg[c])[0];}
        p.correctBoundaryConditions();sg.correctBoundaryConditions();pc.correctBoundaryConditions();
    }
};
