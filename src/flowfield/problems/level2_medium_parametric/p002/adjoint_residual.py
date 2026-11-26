"""Serial, orthogonal finite-volume steady residual of flowfieldP002Foam.

The OpenFOAM solver remains the primal evaluator. This module linearizes its
fixed point, including the SIMPLE/Rhie--Chow face flux and the lagged current
in the bubble coverage closure. It is deliberately restricted to generated
P002 meshes and the pinned discretization in geometry.case_files.
"""

import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse

from ._sparse_ad import Dual, concatenate, linear, scatter, select
from .config import parameters_for, reference
from .geometry import FLUID, PATCHES, geometry
from .verification import values


class NondifferentiableMesh(ValueError):
    pass


def design_mesh(x, config):
    from . import P002Problem
    return geometry(parameters_for(P002Problem(config=config).decode(x), config)['geometry'],
                    'parallel', config.mesh)


def mesh_derivatives(x, config, step=1e-5):
    """Differentiate coordinates only, holding connectivity/materials fixed.

    Both h and h/2 must give the same topology and coordinate derivative. At
    box boundaries use a second-order one-sided stencil. No CFD probes occur.
    """
    x = np.asarray(x, dtype=float)
    base = design_mesh(x, config)
    signature = lambda m: ([len(m[a]) for a in 'xyz'], m['ids'])
    def coordinates(m):
        return np.concatenate([m[a] for a in 'xyz'])*1e-3
    center = coordinates(base)
    slopes = []
    for i in range(7):
        estimates = []
        for h in (step, step/2):
            if x[i]-h >= 0 and x[i]+h <= 1:
                offsets, weights = (-h, h), (-1/(2*h), 1/(2*h))
            else:
                sign = 1 if x[i] < h else -1
                offsets = (0., sign*h, sign*2*h)
                weights = (-3/(2*sign*h), 4/(2*sign*h), -1/(2*sign*h))
            slope = np.zeros_like(center)
            for offset, weight in zip(offsets, weights):
                probe = x.copy(); probe[i] += offset
                m = design_mesh(probe, config)
                if signature(m) != signature(base):
                    raise NondifferentiableMesh(
                        f'P002 mesh connectivity/materials change in coordinate {i} at step {h:g}; '
                        'choose an initial design inside a smooth mesh region (e.g. --initial paper)')
                slope += weight*coordinates(m)
            estimates.append(slope)
        if not np.allclose(*estimates, rtol=1e-4, atol=2e-8):
            raise NondifferentiableMesh(f'P002 coordinate derivative is not stable for variable {i}')
        slopes.append(estimates[-1])
    return base, np.column_stack(slopes)


def face_addressing(m):
    """Match geometry.mesh's exact face order, without writing a mesh."""
    nx, ny, nz = [len(m[a])-1 for a in 'xyz']
    internal, boundary = [], {p: [] for p in PATCHES}
    for c, (i, j, k) in enumerate(m['cells']):
        for axis, sign, di, dj, dk in ((0,-1,-1,0,0),(0,1,1,0,0),
                                      (1,-1,0,-1,0),(1,1,0,1,0),
                                      (2,-1,0,0,-1),(2,1,0,0,1)):
            ii, jj, kk = i+di, j+dj, k+dk
            n = (kk*ny+jj)*nx+ii if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz else -1
            if 0 <= n < c:
                continue
            if n >= 0:
                internal.append((c,n,axis,sign))
            else:
                patch = 'cathodeTerminal' if axis == 2 and sign < 0 else (
                        'anodeTerminal' if axis == 2 and sign > 0 else 'externalWalls')
                if m['ids'][c] in (1,7) and axis == 0:
                    patch = ('anode' if m['ids'][c] == 7 else 'cathode')+('Inlet' if sign < 0 else 'Outlet')
                boundary[patch].append((c,c,axis,sign))
    all_faces = internal + [face for group in boundary.values() for face in group]
    patch_ids = np.full(len(all_faces), -1, dtype=int)
    start = len(internal)
    for p, group in enumerate(boundary.values()):
        patch_ids[start:start+len(group)] = p; start += len(group)
    return np.asarray(all_faces, dtype=int).T, patch_ids, len(internal)


class Residual:
    def __init__(self, x, config, *, geometry_step=1e-5, differentiate_geometry=True):
        self.config, self.x = config, np.asarray(x, dtype=float)
        if differentiate_geometry:
            self.mesh, self.coordinate_jacobian = mesh_derivatives(x, config, geometry_step)
        else:
            self.mesh = design_mesh(x, config)
            self.coordinate_jacobian = np.zeros((sum(len(self.mesh[a]) for a in 'xyz'),7))
        m = self.mesh
        self.material = np.asarray(m['ids'])
        self.nc = len(self.material)
        self.fluid = np.isin(self.material, list(FLUID))
        self.electronic = ~np.isin(self.material, [1,4,7])
        self.ionic = np.isin(self.material, [3,4,5])
        self.catalyst = np.isin(self.material, [3,5])
        (self.o,self.n,self.axis,self.sign), self.patch, self.ni = face_addressing(m)
        self.nf = len(self.o)
        self.internal = self.patch < 0
        self.inlet = np.isin(self.patch, [PATCHES.index(p) for p in ('anodeInlet','cathodeInlet')])
        self.outlet = np.isin(self.patch, [PATCHES.index(p) for p in ('anodeOutlet','cathodeOutlet')])
        self.flow_faces = (self.internal & self.fluid[self.o] & self.fluid[self.n]) | self.outlet
        ids = np.arange(self.nf)
        self.div = sparse.csr_matrix((np.r_[np.ones(self.nf),-np.ones(self.ni)],
            (np.r_[self.o,self.n[:self.ni]],np.r_[ids,ids[:self.ni]])), shape=(self.nc,self.nf))
        self.owners = sparse.csr_matrix((np.ones(self.nf),(self.o,ids)),shape=(self.nc,self.nf))
        self.neighbours = self.owners-self.div
        self.indices, self.slices, self.scales = {}, {}, []
        start = 0
        # All variables are physical quantities. Scales only equilibrate linear algebra.
        for name, mask, scale in [('Ux',self.fluid,.1),('Uy',self.fluid,.1),('Uz',self.fluid,.1),
            ('p',self.fluid,100.),('Cgas',self.fluid,10.),('T',np.ones(self.nc,bool),1.),
            ('phiS',self.electronic,1.),('phiE',self.ionic,1.),('iVol',self.catalyst,1e8),
            ('massPhi',self.flow_faces,1e-6)]:
            self.indices[name] = np.flatnonzero(mask)
            count = len(self.indices[name])
            self.slices[name] = slice(start,start+count)
            self.scales.extend([scale]*count); start += count
        self.size, self.scales = start, np.asarray(self.scales)

    def read_state(self, case):
        case = Path(case)
        status = json.loads((case/'postProcessing/pem/status.json').read_text())
        if not status.get('converged') or status.get('processes') != 1:
            raise ValueError('Adjoint requires a converged serial P002 primal case')
        directory = case/str(int(status['iteration']))
        fields = {}
        for name in self.slices:
            if name in ('Ux','Uy','Uz','massPhi'):
                continue
            fields[name] = np.asarray(values((directory/name).read_text(),'internalField',self.nc))
        text = (directory/'U').read_text()
        match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
        if not match or int(match[1]) != self.nc:
            raise ValueError('Expected saved nonuniform P002 velocity field')
        velocity = np.fromstring(re.sub(r'[()]',' ',match[2]),sep=' ').reshape(self.nc,3)
        fields.update({name:velocity[:,i] for i,name in enumerate(('Ux','Uy','Uz'))})
        text = (directory/'massPhi').read_text()
        flux = list(values(text,'internalField',self.ni))
        for p, name in enumerate(PATCHES):
            block = re.search(r'\b'+name+r'\s*\{(.*?)\}',text,re.S)
            flux.extend(values(block[1],'value',int(np.sum(self.patch == p))))
        fields['massPhi'] = np.asarray(flux)
        return np.concatenate([fields[name][self.indices[name]] for name in self.slices])

    def evaluate(self, state, *, derivatives=True):
        """Return (residual, [voltage, pressure sum]) and their sparse derivatives."""
        state = np.asarray(state)
        if state.shape != (self.size,) or not np.all(np.isfinite(state)):
            raise ValueError('Invalid P002 adjoint state')
        identity = sparse.hstack([sparse.eye(self.size), sparse.csr_matrix((self.size,7))],format='csr') if derivatives else sparse.csr_matrix((self.size,0))
        independent = Dual(state, identity)
        fields = {name:scatter(independent[part],self.indices[name],self.nf if name=='massPhi' else self.nc)
                  for name,part in self.slices.items()}
        T,C,S,E,iv = [fields[name] for name in ('T','Cgas','phiS','phiE','iVol')]
        U = [fields[name] for name in ('Ux','Uy','Uz')]
        p = fields['p']
        constant = T.constant
        coordinates = np.concatenate([self.mesh[a] for a in 'xyz'])*1e-3
        coord_j = sparse.hstack([sparse.csr_matrix((len(coordinates),self.size)), sparse.csr_matrix(self.coordinate_jacobian)],format='csr') if derivatives else sparse.csr_matrix((len(coordinates),0))
        xyz = Dual(coordinates, coord_j)
        index = np.asarray(self.mesh['cells'])
        lengths, start = [], 0
        for a, name in enumerate('xyz'):
            lengths.append(xyz[start+index[:,a]+1]-xyz[start+index[:,a]])
            start += len(self.mesh[name])
        V = lengths[0]*lengths[1]*lengths[2]
        axis_length = sum(lengths[a][self.o]*(self.axis == a) for a in range(3))
        do = axis_length/2
        dn = sum(lengths[a][self.n]*(self.axis == a) for a in range(3))/2
        area = V[self.o]/axis_length
        distance = do+dn
        weight = dn/distance
        def interpolate(v):
            return select(self.internal,weight*v[self.o]+(1-weight)*v[self.n],v[self.o])
        def harmonic(v, active=None):
            mask = self.internal & (v.v[self.o] > 0) & (v.v[self.n] > 0)
            if active is not None:
                mask &= active[self.o] & active[self.n]
            # Safe denominator on inactive faces, whose derivative is exactly zero.
            vo = select(v.v[self.o] > 0,v[self.o],1.)
            vn = select(v.v[self.n] > 0,v[self.n],1.)
            return select(self.internal,select(mask,area/(do/vo+dn/vn),0.),area*v[self.o]/do)
        def conduction(coef,v):
            return linear(self.div,coef*(v[self.o]-v[self.n])*self.internal)
        def advect(flux,v,bc):
            face = select(flux.v >= 0,v[self.o],v[self.n])
            face = select(self.inlet,bc,face)
            return linear(self.div,flux*face)
        mat, model = self.material, self.config.model
        F,R,MO,MH,MW = 96485.33212,8.314462618,.0319988,.00201588,.01801528
        anode = mat >= 5
        eps = np.where(np.isin(mat,[1,7]),1.,np.where(np.isin(mat,[2,6]),model['porosityPTL'],np.where(self.catalyst,model['porosityCL'],0.)))
        eps_d = constant(eps)
        epsf = harmonic(eps_d)*distance/area
        epsf = select(self.internal,epsf,eps_d[self.o])
        rg = model['operatingPressure']*np.where(anode,MO,MH)/(R*T)
        fg = (C*R*T/model['operatingPressure']).clip(0.,.995)*self.fluid
        rho = model['liquidDensity']*(1-fg)+rg*fg
        mu = model['liquidViscosity']*(1-fg)+np.where(anode,model['oxygenViscosity'],model['hydrogenViscosity'])*fg
        cp = (model['liquidDensity']*(1-fg)*model['liquidCp']+rg*fg*np.where(anode,model['oxygenCp'],model['hydrogenCp']))/rho
        kfluid = model['liquidConductivity']*(1-fg)+np.where(anode,model['oxygenConductivity'],model['hydrogenConductivity'])*fg
        ksolid = np.array([model['kTitanium'] if v in (0,6,8) else model['kCarbon'] if v==2 else model['kCathodeCL'] if v==3 else model['kAnodeCL'] if v==5 else model['kMembrane'] for v in mat])
        k = (1-eps)*ksolid+eps*kfluid
        D = eps**1.5*np.where(anode,model['diffusivityO2'],model['diffusivityH2'])*(T/model['referenceTemperature'])*(101325/model['operatingPressure'])
        drag = mu*eps**2/np.where(np.isin(mat,[2,6]),model['permeabilityPTL'],model['permeabilityCL'])*(self.fluid & (eps < 1))
        sigmaS = constant(self.electronic*model['electronConductivity'])
        sigmaE = 6.94*(1268*(1/303.-1/T)).exp()*self.ionic
        cl = np.flatnonzero(self.catalyst)
        acl = mat[cl] == 5
        tc, cc, ic = T[cl],C[cl],iv[cl]
        ratio = (cc/np.where(acl,model['referenceConcentrationO2'],model['referenceConcentrationH2'])).clip(model['concentrationFloorRatio'])
        psat = 133.322368*((8.07131-1730.63/(233.426+tc-273.15))*np.log(10.)).exp()
        factor = (1. if model['coveragePressureMode']=='paperSI' else model['operatingPressure'])/(model['operatingPressure']-psat).clip(1.)
        dcl = reference()['geometry']['cl_thickness_mm']*1e-3
        theta = ((-97.25+182*tc/model['referenceTemperature']-84*(tc/model['referenceTemperature'])**2)*(ic.absolute()*dcl/300000.)**.3*factor).clip(0.,.95)
        pref = np.where(acl,model['activeAreaAnode']*model['exchangeCurrentAnode'],model['activeAreaCathode']*model['exchangeCurrentCathode'])*(1-theta)*ratio
        eq = (1.229-.0009*(tc-298.15))*acl
        if model['includeNernstConcentration']:
            eq += np.where(acl,.25,-.5)*R*tc/F*ratio.log()
        eta = S[cl]-E[cl]-eq
        reaction = pref*((model['alphaAnodic']*F*eta/(R*tc)).clip(-60,60).exp()-(-model['alphaCathodic']*F*eta/(R*tc)).clip(-60,60).exp())
        gs,ge = harmonic(sigmaS),harmonic(sigmaE)
        rs,re_ = conduction(gs,S)+iv*V,conduction(ge,E)-iv*V
        heat = scatter(ic*eta+model['reversibleHeatSign']*ic*tc*(-.0009)*acl,cl,self.nc)*V
        joule = (gs*(S[self.o]-S[self.n])**2+ge*(E[self.o]-E[self.n])**2)*self.internal
        heat += linear(self.owners+self.neighbours,joule/2)
        atop = self.patch == PATCHES.index('anodeTerminal')
        cbottom = self.patch == PATCHES.index('cathodeTerminal')
        current = model['currentDensity']
        rs += linear(self.owners,gs*S[self.o]*cbottom-current*area*atop)
        heat += linear(self.owners,gs*S[self.o]**2*cbottom+current**2*area*do/model['electronConductivity']*atop)
        mass_source = iv*np.where(mat==5,(MO/4-MW/2)/F,np.where(mat==3,-MH/(2*F),0.))
        gas_source = iv*np.where(mat==5,1/(4*F),np.where(mat==3,-1/(2*F),0.))
        # Boundary inflow is held at the specified volumetric rate at each port.
        inlet_mass = constant(np.zeros(self.nf))
        for name in ('anodeInlet','cathodeInlet'):
            mask = self.patch == PATCHES.index(name)
            inlet_mass += -model['liquidDensity']*model['inletFlow_mL_min']*1e-6/60*area*mask/(area*mask).sum()
        M = fields['massPhi']+inlet_mass
        rhof = select(self.inlet,model['liquidDensity'],interpolate(rho))
        cpf = select(self.inlet,model['liquidCp'],interpolate(cp))
        # Pressure gradient uses the fluid pressure at an internal solid wall.
        pf = interpolate(p)
        pf = select(self.internal & self.fluid[self.o] & ~self.fluid[self.n],p[self.o],pf)
        pf = select(self.internal & ~self.fluid[self.o] & self.fluid[self.n],p[self.n],pf)
        pf = select(self.outlet,0.,pf)
        grad = [linear(self.div,pf*area*self.sign*(self.axis==a))/V*self.fluid for a in range(3)]
        muf = harmonic(eps_d*mu,self.fluid)
        wall = self.internal & (self.fluid[self.o] != self.fluid[self.n])
        wallcoef = select(self.fluid[self.o],eps_d[self.o]*mu[self.o]*area/do,eps_d[self.n]*mu[self.n]*area/dn)*wall
        wall_diag = linear(self.owners,wallcoef*self.fluid[self.o])+linear(self.neighbours,wallcoef*self.fluid[self.n])
        fixed_u = ~self.internal & ~self.outlet
        boundary_diag = linear(self.owners,muf*fixed_u)
        diffusion_diag = linear(self.owners+self.neighbours,muf*self.internal)+wall_diag+boundary_diag
        adv_diag = linear(self.owners,M.clip(0.)*self.internal+M*self.outlet)+linear(self.neighbours,(-M).clip(0.)*self.internal)
        diag = (diffusion_diag+adv_diag+drag*V+eps_d*rho*V/model['flowPseudoDt'])/.7
        # Avoid dividing by solid-cell diagonals; their rAU*eps is exactly zero.
        rau_eps = select(self.fluid,V/(select(self.fluid,diag,1.))*eps_d,0.)
        ru = []
        for a in range(3):
            # Only the x velocity is nonzero at the two inlet ports.
            ubc = -inlet_mass/(model['liquidDensity']*area) if a==0 else constant(np.zeros(self.nf))
            visc = conduction(muf,U[a])+wall_diag*U[a]+linear(self.owners,muf*(U[a][self.o]-ubc)*fixed_u)
            ru.append(advect(M,U[a],ubc)+visc+drag*V*U[a]+eps_d*V*grad[a])
        flux_h = sum(interpolate(U[a]+rau_eps*grad[a])*self.sign*(self.axis==a) for a in range(3))*area*rhof*epsf
        pressure_flux = rhof*epsf*interpolate(rau_eps)*(p[self.n]-p[self.o])/distance*area
        pressure_flux = select(self.outlet,rhof*epsf*rau_eps[self.o]*(-p[self.o])/do*area,pressure_flux)
        rflux = M-flux_h+pressure_flux
        # Equivalent row operation: subtract the divergence of the face-flux
        # closure. It exposes the pressure Laplacian instead of a zero diagonal
        # continuity block, greatly improving sparse factorization/pivoting.
        continuity = linear(self.div,M-rflux*self.flow_faces)-mass_source*V
        df = harmonic(D,self.fluid)*self.internal
        gas = advect(M/rhof,C,0.)+conduction(df,C)-gas_source*V
        kf = harmonic(k)
        thermal = advect(M*cpf,T,model['inletTemperature'])-linear(self.div,M*cpf)*T+conduction(kf,T)
        thermal += linear(self.owners,kf*(T[self.o]-model['inletTemperature'])*self.inlet)-heat
        all_res = ru+[continuity,gas,thermal,rs,re_,scatter(ic-reaction,cl,self.nc),rflux]
        residual = concatenate([r[self.indices[name]] for name,r in zip(self.slices,all_res)])
        voltage = (area*(S[self.o]+current*do/model['electronConductivity'])*atop).sum()/(area*atop).sum()
        pressure = sum((area*p[self.o]*(self.patch==PATCHES.index(name))).sum()/(area*(self.patch==PATCHES.index(name))).sum() for name in ('anodeInlet','cathodeInlet'))
        return residual,concatenate([voltage,pressure])
