// FlowField diagnostic function object; the solver remains stock interFoam v2312.
#include "fvCFD.H"
#include "fvMeshFunctionObject.H"
#include "addToRunTimeSelectionTable.H"
#include "Detector.H"
#include <fstream>
#include <iomanip>
#include <cmath>
namespace Foam { namespace functionObjects {
class flowfieldDetachment : public fvMeshFunctionObject {
    double rw_,rg_,threshold_,minimum_,fraction_,persistence_,interval_;
    double lastTime_=0,lastSample_=-1,gasFluxIntegral_=0,injected_=0,continuityIntegral_=0;
    double worstBalance_=0,worstContinuity_=0,minAlpha_=1,maxAlpha_=0;
    double gasTolerance_,continuityTolerance_,alphaTolerance_;
    std::ofstream stream_;
    std::vector<std::vector<int>> neighbors_;
    std::vector<bool> electrode_,port_;
    std::vector<double> volumes_;
    flowfieldP005::Event event_;
public:
    TypeName("flowfieldDetachment");
    flowfieldDetachment(const word& name,const Time& t,const dictionary& d)
    : fvMeshFunctionObject(name,t,d),rw_(d.get<scalar>("waterDensity")),rg_(d.get<scalar>("oxygenDensity")),
      threshold_(d.get<scalar>("gasThreshold")),minimum_(d.get<scalar>("minimumBubbleVolume")),
      fraction_(d.get<scalar>("detachedFraction")),persistence_(d.get<scalar>("persistence")),
      interval_(d.get<scalar>("sampleInterval")),gasTolerance_(d.get<scalar>("gasTolerance")),
      continuityTolerance_(d.get<scalar>("continuityTolerance")),alphaTolerance_(d.get<scalar>("alphaTolerance"))
    {
        if (Pstream::parRun()) FatalErrorInFunction << "P005 monitor currently supports serial cases only" << exit(FatalError);
        const auto& cc=mesh_.cellCells();neighbors_.resize(mesh_.nCells());
        electrode_.resize(mesh_.nCells(),false);port_.resize(mesh_.nCells(),false);
        forAll(cc,i) { for(label n:cc[i])neighbors_[i].push_back(n);volumes_.push_back(mesh_.V()[i]); }
        forAll(mesh_.boundary(),p) {
            const word& nm=mesh_.boundary()[p].name();
            for(label c:mesh_.boundary()[p].faceCells()) {
                if(nm=="electrode" || nm=="gasInlet") electrode_[c]=true;
                if(nm=="waterInlet" || nm=="outlet") port_[c]=true;
            }
        }
        mkDir(t.path()/"postProcessing");
        stream_.open((t.path()/"postProcessing/detachment.csv").c_str());
        stream_ << std::setprecision(16);
        stream_ << "time_s,gas_volume_m3,attached_volume_m3,detached_volume_m3,injected_volume_m3,gas_balance_relative,continuity_relative,min_alpha_water,max_alpha_water,armed,confirmed,event_time_s,event_bracket_start_s,max_gas_balance_relative,max_continuity_relative\n";
    }
    bool execute() override { return sample(false); }
    bool write() override { return true; }
    bool end() override { return sample(true); }
    bool sample(bool final) {
        if(!mesh_.foundObject<volScalarField>("alpha.water"))return true;
        const auto& a=mesh_.lookupObject<volScalarField>("alpha.water");
        const auto& phi=mesh_.lookupObject<surfaceScalarField>("phi");
        const auto& rhoPhi=mesh_.lookupObject<surfaceScalarField>("rhoPhi");
        double t=mesh_.time().value(),dt=t-lastTime_,net=0,inlet=0,totalFlux=0;
        forAll(phi.boundaryField(),p) forAll(phi.boundaryField()[p],f) {
            const double q=phi.boundaryField()[p][f];
            const double gasQ=(rw_*q-rhoPhi.boundaryField()[p][f])/(rw_-rg_);
            net+=gasQ;totalFlux+=q;
            if(mesh_.boundary()[p].name()=="gasInlet")inlet-=gasQ;
        }
        gasFluxIntegral_+=dt*net;injected_+=dt*inlet;continuityIntegral_+=dt*std::abs(totalFlux);lastTime_=t;
        std::vector<double> gas(a.size());double volume=0;
        forAll(a,i){
            double value=a[i];
            if(!std::isfinite(value))FatalErrorInFunction<<"Nonfinite alpha"<<exit(FatalError);
            minAlpha_=std::min(minAlpha_,value);maxAlpha_=std::max(maxAlpha_,value);
            gas[i]=1-value;volume+=gas[i]*volumes_[i];
        }
        const double balance=std::abs(volume+gasFluxIntegral_)/std::max(injected_,1e-30);
        const double continuity=continuityIntegral_/std::max(injected_,1e-30);
        if(injected_>1e-20) {worstBalance_=std::max(worstBalance_,balance);worstContinuity_=std::max(worstContinuity_,continuity);}
        if(minAlpha_<-alphaTolerance_ || maxAlpha_>1+alphaTolerance_ || balance>gasTolerance_ || continuity>continuityTolerance_)
            FatalErrorInFunction<<"P005 phase boundedness/conservation check failed: "<<minAlpha_<<' '<<maxAlpha_<<' '<<balance<<' '<<continuity<<exit(FatalError);
        if(!final && lastSample_>=0 && t-lastSample_<interval_ && t<mesh_.time().endTime().value()-1e-12)return true;
        if(t==lastSample_)return true;
        auto c=flowfieldP005::components(gas,volumes_,neighbors_,electrode_,port_,threshold_);
        event_.update(t,c,minimum_,fraction_,persistence_);lastSample_=t;
        stream_<<t<<','<<c.total<<','<<c.attached<<','<<c.detached<<','<<injected_<<','<<balance<<','<<continuity<<','<<minAlpha_<<','<<maxAlpha_<<','<<event_.armed<<','<<event_.confirmed<<','<<event_.eventTime<<','<<event_.bracketStart<<','<<worstBalance_<<','<<worstContinuity_<<'\n';stream_.flush();
        if(event_.confirmed && !final) {
            Info<<"FLOWFIELD_P005_DETACHED "<<event_.eventTime<<endl;
            const_cast<Time&>(mesh_.time()).writeAndEnd();
        }
        return true;
    }
};
defineTypeNameAndDebug(flowfieldDetachment,0);
addToRunTimeSelectionTable(functionObject,flowfieldDetachment,dictionary);
}}
