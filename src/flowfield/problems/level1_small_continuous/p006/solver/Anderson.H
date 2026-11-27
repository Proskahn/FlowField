// Anderson acceleration of the nonlinear block-preconditioned equations.
#include <Eigen/QR>
#include <deque>
class OuterAcceleration
{
    Foam::labelList material_;
    Foam::scalarField porosity_,input_,candidate_,previousInput_,previousResidual_;
    std::deque<Foam::scalarField> dx_,df_;
    bool pending_=false,havePrevious_=false;
    Foam::label iteration_=0;
    static double coordinate(double s){return s;}
    static double saturation(double s){return std::min(std::max(s,1e-6),1-1e-6);}
    Foam::scalarField pack(const Foam::volVectorField& l,const Foam::volVectorField& g,
        const Foam::volScalarField& p,const Foam::volScalarField& s,const Foam::volScalarField& t,const Foam::volScalarField& water,
        const Foam::surfaceScalarField& fl,const Foam::surfaceScalarField& fg)
    {
        using namespace Foam;scalarField x(10*p.size()+2*p.mesh().nFaces());
        forAll(p,c){
            const scalar gas=saturation(s[c]);
            const scalar alphaL=porosity_[c]>0?porosity_[c]*(1-gas):1.;
            const scalar alphaG=porosity_[c]>0?porosity_[c]*gas:1.;
            for(label d=0;d<3;++d){x[10*c+d]=l[c][d]/(.3*alphaL);x[10*c+3+d]=g[c][d]/(3.*alphaG);}
            x[10*c+6]=p[c]/100.;x[10*c+7]=coordinate(s[c]);x[10*c+8]=t[c]/10.;x[10*c+9]=water[c]/22.;}
        label i=10*p.size();
        forAll(fl,f){x[i++]=fl[f]/(.3*p.mesh().magSf()[f]);x[i++]=fg[f]/(3*p.mesh().magSf()[f]);}
        forAll(fl.boundaryField(),pi)forAll(fl.boundaryField()[pi],f)
        {double a=p.mesh().magSf().boundaryField()[pi][f];x[i++]=fl.boundaryField()[pi][f]/(.3*a);x[i++]=fg.boundaryField()[pi][f]/(3*a);}
        return x;
    }

public:
    explicit OuterAcceleration(const Foam::labelList& m,const Foam::volScalarField& eps)
        :material_(m),porosity_(eps.primitiveField()){}
    void reset(){pending_=false;havePrevious_=false;dx_.clear();df_.clear();}
    void begin(Foam::volVectorField& l,Foam::volVectorField& g,Foam::volScalarField& p,
        Foam::volScalarField& s,Foam::volScalarField& t,Foam::volScalarField& water,Foam::surfaceScalarField& fl,Foam::surfaceScalarField& fg)
    {
        using namespace Foam;
        const bool candidateApplied=pending_;
        if(pending_)
        {
            forAll(p,c){
                if(material_[c]!=0&&material_[c]!=4&&material_[c]!=8)s[c]=saturation(candidate_[10*c+7]);
                const scalar alphaL=porosity_[c]>0?porosity_[c]*(1-s[c]):1.;
                const scalar alphaG=porosity_[c]>0?porosity_[c]*s[c]:1.;
                for(label d=0;d<3;++d){l[c][d]=.3*alphaL*candidate_[10*c+d];g[c][d]=3.*alphaG*candidate_[10*c+3+d];}
                p[c]=100*candidate_[10*c+6];
                t[c]=10*candidate_[10*c+8];water[c]=22*candidate_[10*c+9];}
            label i=10*p.size();
            forAll(fl,f){fl[f]=candidate_[i++]*.3*p.mesh().magSf()[f];fg[f]=candidate_[i++]*3*p.mesh().magSf()[f];}
            forAll(fl.boundaryField(),pi)forAll(fl.boundaryField()[pi],f)
            {double a=p.mesh().magSf().boundaryField()[pi][f];fl.boundaryFieldRef()[pi][f]=candidate_[i++]*.3*a;fg.boundaryFieldRef()[pi][f]=candidate_[i++]*3*a;}
            l.correctBoundaryConditions();g.correctBoundaryConditions();p.correctBoundaryConditions();s.correctBoundaryConditions();t.correctBoundaryConditions();water.correctBoundaryConditions();
            pending_=false;
        }
        input_=pack(l,g,p,s,t,water,fl,fg);
        // Keep the candidate coordinate in the residual when applying bounds;
        // clipping must not create an artificial zero at a saturation limit.
        if(candidateApplied)forAll(p,c)input_[10*c+7]=candidate_[10*c+7];
    }

    void observe(const Foam::volVectorField& l,const Foam::volVectorField& g,const Foam::volScalarField& p,
        const Foam::volScalarField& s,const Foam::volScalarField& t,const Foam::volScalarField& water,
        const Foam::surfaceScalarField& fl,const Foam::surfaceScalarField& fg,const Foam::scalarField& phaseResidual,
        const bool fullyCoupled)
    {
        using namespace Foam;
        scalarField r(pack(l,g,p,s,t,water,fl,fg)-input_);
        forAll(material_,c)if(material_[c]!=0&&material_[c]!=4&&material_[c]!=8)
            r[10*c+7]=phaseResidual[c]+saturation(input_[10*c+7])-input_[10*c+7];
        scalar error=std::sqrt(gSum(sqr(r)));
        if(havePrevious_)
        {
            if(error>2*std::sqrt(gSum(sqr(previousResidual_)))){dx_.clear();df_.clear();}
            else{dx_.emplace_back(input_-previousInput_);df_.emplace_back(r-previousResidual_);}
            if(dx_.size()>40){dx_.pop_front();df_.pop_front();}
        }
        const scalar damping=.2;
        scalarField direction(damping*r);
        const label n=dx_.size();
        if(n)
        {
            Eigen::MatrixXd gram(n,n);Eigen::VectorXd rhs(n);
            for(label i=0;i<n;++i)
            {rhs[i]=gSum(df_[i]*r);for(label j=0;j<n;++j)gram(i,j)=gSum(df_[i]*df_[j]);}
            // Near convergence the update differences can be almost dependent.
            // Excessive regularization then suppresses the remaining slow modes.
            // Retain the startup regularization until momentum AND heat are active;
            // field bounds, coefficient limiting and physical acceptance are unchanged.
            const scalar relativeRegularization=fullyCoupled?1e-14:1e-10;
            const scalar regularization=max(gram.diagonal().maxCoeff()*relativeRegularization,1e-24);
            gram.diagonal().array()+=regularization;
            Eigen::VectorXd gamma=gram.colPivHouseholderQr().solve(rhs);
            const scalar largest=gamma.cwiseAbs().maxCoeff();
            if(largest>10)gamma*=10/largest;
            if(gamma.allFinite())for(label i=0;i<n;++i)direction-=gamma[i]*(dx_[i]+damping*df_[i]);
        }
        scalar step=1;
        forAll(p,c)
        {
            scalar dt=10*direction[10*c+8];
            if(dt>0)step=min(step,.99*(373.15-10*input_[10*c+8])/dt);
            if(dt<0)step=min(step,.99*(273.15-10*input_[10*c+8])/dt);
            scalar dw=22*direction[10*c+9];
            if(dw>0)step=min(step,.99*(30-22*input_[10*c+9])/dw);
            if(dw<0)step=min(step,.99*(.64-22*input_[10*c+9])/dw);
        }
        forAll(direction,i)
        {scalar limit=i<10*p.size()?(i%10==8?.1:i%10==9?.05:i%10==7?.05:5.):5.;
         if(mag(direction[i])>limit)step=min(step,limit/mag(direction[i]));}
        reduce(step,minOp<scalar>());
        candidate_=input_+step*direction;
        forAll(material_,c)if(material_[c]!=0&&material_[c]!=4&&material_[c]!=8)
            candidate_[10*c+7]=min(max(candidate_[10*c+7],1e-6+1e-9),1-1e-6-1e-9);
        Info<<"Anderson block residual "<<++iteration_<<": norm="<<error
            <<"; history="<<n<<"; step="<<step<<endl;
        previousInput_=input_;previousResidual_=r;havePrevious_=true;pending_=true;
    }
};
