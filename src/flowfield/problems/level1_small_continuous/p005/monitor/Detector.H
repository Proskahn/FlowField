// Serial connected-component detector. Independent of OpenFOAM for unit testing.
#ifndef FLOWFIELD_P005_DETECTOR_H
#define FLOWFIELD_P005_DETECTOR_H
#include <vector>
#include <algorithm>
namespace flowfieldP005 {
struct Components { double total=0, attached=0, detached=0; };
inline Components components(const std::vector<double>& gas,
    const std::vector<double>& volume, const std::vector<std::vector<int>>& neighbors,
    const std::vector<bool>& electrode, const std::vector<bool>& port, double threshold)
{
    Components out; std::vector<bool> seen(gas.size(),false); std::vector<int> queue;
    for (unsigned i=0;i<gas.size();++i) out.total+=gas[i]*volume[i];
    for (unsigned i=0;i<gas.size();++i) {
        if (seen[i] || gas[i]<threshold) continue;
        queue.clear();queue.push_back(i);seen[i]=true;
        double v=0;bool touches=false, exits=false;
        for (unsigned q=0;q<queue.size();++q) {
            int cell=queue[q];v+=gas[cell]*volume[cell];
            touches=touches||electrode[cell];exits=exits||port[cell];
            for (int n: neighbors[cell]) if (!seen[n] && gas[n]>=threshold) {
                seen[n]=true;queue.push_back(n);
            }
        }
        if (touches) out.attached=std::max(out.attached,v);
        if (!touches && !exits) out.detached=std::max(out.detached,v);
    }
    return out;
}
struct Event {
    bool armed=false,confirmed=false;
    double candidate=-1,previous=-1,bracketStart=-1,eventTime=-1;
    void update(double t, const Components& c, double minimum, double fraction, double persistence) {
        if (confirmed) return;
        // Must have observed an electrode-connected resolved bubble before detachment.
        bool qualifies=armed && c.detached>=minimum && c.detached>=fraction*c.total;
        if (qualifies) {
            if (candidate<0) {candidate=t;bracketStart=previous;}
            if (t-candidate>=persistence*(1-1e-10)) {confirmed=true;eventTime=candidate;}
        } else candidate=-1;
        if (c.attached>=minimum) armed=true;
        previous=t;
    }
};
}
#endif
