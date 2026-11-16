// Safeguarded Anderson acceleration of the steady outer fixed-point iteration.
// Candidates are used only as the next initial guess. Every reported result is
// the unmodified output of a complete conservative CFD iteration.
#include <Eigen/QR>
#include <deque>
class OuterAcceleration
{
    Foam::labelList material_;
    Foam::scalarField input_,previousInput_,previousResidual_,candidate_;
    std::deque<Foam::scalarField> dx_,df_;
    bool havePrevious_=false,pending_=false;
    static double saturationCoordinate(double s)
    {double a=std::min(std::max((s-1e-6)/(1-2e-6),1e-12),1-1e-12);return std::log(a/(1-a))/5.;}
    static double saturation(double a)
    {a=std::min(std::max(5*a,-27.),27.);return 1e-6+(1-2e-6)/(1+std::exp(-a));}
    static Foam::scalarField pack(const Foam::volVectorField& l,const Foam::volVectorField& g,
        const Foam::volScalarField& p,const Foam::volScalarField& s,const Foam::volScalarField& t,
        const Foam::surfaceScalarField& fl,const Foam::surfaceScalarField& fg)
    {
        Foam::scalarField x(9*p.size()+2*p.mesh().nFaces());
        forAll(p,c){for(Foam::label d=0;d<3;++d){x[9*c+d]=l[c][d]/.3;x[9*c+3+d]=g[c][d]/3.;}
            x[9*c+6]=p[c]/100.;x[9*c+7]=saturationCoordinate(s[c]);x[9*c+8]=t[c]/10.;}
        Foam::label i=9*p.size();
        forAll(fl,f){x[i++]=fl[f]/(.3*p.mesh().magSf()[f]);x[i++]=fg[f]/(3*p.mesh().magSf()[f]);}
        forAll(fl.boundaryField(),pi)forAll(fl.boundaryField()[pi],f)
        {double area=p.mesh().magSf().boundaryField()[pi][f];x[i++]=fl.boundaryField()[pi][f]/(.3*area);x[i++]=fg.boundaryField()[pi][f]/(3*area);}
        return x;
    }
public:
    explicit OuterAcceleration(const Foam::labelList& material):material_(material){}
    void reset(){dx_.clear();df_.clear();havePrevious_=false;pending_=false;}
    void begin(Foam::volVectorField& l,Foam::volVectorField& g,
        Foam::volScalarField& p,Foam::volScalarField& s,Foam::volScalarField& t,
        Foam::surfaceScalarField& fl,Foam::surfaceScalarField& fg)
    {
        using namespace Foam;
        input_=pack(l,g,p,s,t,fl,fg);
        if(pending_)
        {
            scalar step=1;
            forAll(p,c)
            {
                scalar dt=10*(candidate_[9*c+8]-input_[9*c+8]);
                if(dt>0)step=min(step,.99*(373.15-t[c])/dt);
                if(dt<0)step=min(step,.99*(273.15-t[c])/dt);
            }
            // Limit extrapolation, not the physical solution or field bounds.
            forAll(input_,i)
            // A 0.04 log-odds-coordinate step changes phase odds by at most
            // exp(0.2), limiting loss of either phase to less than 19%.
            {scalar change=mag(candidate_[i]-input_[i]);scalar limit=i<9*p.size()?(i%9==8?.1:i%9==7?.04:5.):5.;
             if(change>limit)step=min(step,limit/change);}
            reduce(step,minOp<scalar>());
            if(step>.001)
            {
                input_+=step*(candidate_-input_);
                forAll(p,c){for(label d=0;d<3;++d){l[c][d]=.3*input_[9*c+d];g[c][d]=3*input_[9*c+3+d];}
                    p[c]=100*input_[9*c+6];
                    if(material_[c]!=0&&material_[c]!=4&&material_[c]!=8)s[c]=saturation(input_[9*c+7]);
                    t[c]=10*input_[9*c+8];}
                label i=9*p.size();
                forAll(fl,f){fl[f]=input_[i++]*.3*p.mesh().magSf()[f];fg[f]=input_[i++]*3*p.mesh().magSf()[f];}
                forAll(fl.boundaryField(),pi)forAll(fl.boundaryField()[pi],f)
                {scalar area=p.mesh().magSf().boundaryField()[pi][f];fl.boundaryFieldRef()[pi][f]=input_[i++]*.3*area;fg.boundaryFieldRef()[pi][f]=input_[i++]*3*area;}
                l.correctBoundaryConditions();g.correctBoundaryConditions();p.correctBoundaryConditions();s.correctBoundaryConditions();t.correctBoundaryConditions();
            }
            else reset();
            pending_=false;
        }
    }
    void observe(const Foam::volVectorField& l,const Foam::volVectorField& g,
        const Foam::volScalarField& p,const Foam::volScalarField& s,const Foam::volScalarField& t,
        const Foam::surfaceScalarField& fl,const Foam::surfaceScalarField& fg)
    {
        using namespace Foam;
        scalarField result=pack(l,g,p,s,t,fl,fg),f(result-input_);
        if(havePrevious_)
        {
            scalar now=gSum(sqr(f)),before=gSum(sqr(previousResidual_));
            if(now>4*max(before,VSMALL)){dx_.clear();df_.clear();}
            else {dx_.push_back(scalarField(input_-previousInput_));df_.push_back(scalarField(f-previousResidual_));}
            if(dx_.size()>30){dx_.pop_front();df_.pop_front();}
        }
        candidate_=input_+.5*f;
        const label n=dx_.size();
        if(n)
        {
            Eigen::MatrixXd gram(n,n);Eigen::VectorXd rhs(n);
            for(label i=0;i<n;++i){rhs[i]=gSum(df_[i]*f);for(label j=0;j<n;++j)gram(i,j)=gSum(df_[i]*df_[j]);}
            scalar regularization=max(gram.diagonal().maxCoeff()*1e-8,1e-20);
            gram.diagonal().array()+=regularization;
            Eigen::VectorXd gamma=gram.colPivHouseholderQr().solve(rhs);
            scalar largest=gamma.cwiseAbs().maxCoeff();if(largest>10)gamma*=10/largest;
            if(gamma.allFinite())for(label i=0;i<n;++i)candidate_-=gamma[i]*(dx_[i]+.5*df_[i]);
        }
        // Log-odds coordinates preserve saturation bounds during extrapolation.
        // Face fluxes are included because momentum advection uses their lagged values.
        previousInput_=input_;previousResidual_=f;havePrevious_=true;pending_=true;
    }
};
