"""Physical and geometric invariants; OpenFOAM checkMesh is the topology test."""
import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from generate_cases import build_mesh, properties, validate
from analyze import weighted_mean


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=json.loads((ROOT/'config/reference.json').read_text())

    def test_equal_volume_and_feed_across_all_channel_arrangements(self):
        flows=[]
        for w,n in [(2,1),(1,2),(0.5,4)]:
            m=build_mesh(self.p,w,0.1)
            self.assertEqual(len(m['channels']),n)
            self.assertEqual(m['pore_count'],256)
            x,y,z=(m[k] for k in ['x','y','z'])
            volumes=[(x[i+1]-x[i])*(y[j+1]-y[j])*(z[k+1]-z[k]) for i,j,k in m['cells']]
            self.assertAlmostEqual(sum(volumes),24.256,places=8)
            self.assertAlmostEqual(sum(volumes[c] for c in m['zones']['channels']),16,places=8)
            self.assertAlmostEqual(sum(volumes[c] for c in m['zones']['poreStems']),0.256,places=8)
            for c in m['zones']['poreStems']:
                i,j,k=m['cells'][c]
                self.assertGreaterEqual(y[j],0)
                self.assertLessEqual(y[j+1],8)
                self.assertLessEqual(x[i+1]-x[i],0.05+1e-10)
                self.assertLessEqual(y[j+1]-y[j],0.05+1e-10)
            d=properties(self.p,m)
            self.assertAlmostEqual(d['current_A'],0.6464,places=10)
            self.assertAlmostEqual(d['oxygen_velocity_m_s']*d['pore_area_m2'],d['oxygen_flow_m3_s'],places=16)
            flows.append(d['oxygen_flow_m3_s'])
        self.assertEqual(flows[0],flows[1])
        self.assertEqual(flows[1],flows[2])

    def test_mesh_refinement_preserves_pores_and_geometry(self):
        coarse=build_mesh(self.p,1,0.1)
        medium=build_mesh(self.p,1,0.05)
        self.assertGreater(len(medium['cells']),len(coarse['cells']))
        self.assertEqual(coarse['pore_count'],medium['pore_count'])
        self.assertEqual(properties(self.p,coarse),properties(self.p,medium))

    def test_faraday_law_scaling(self):
        m=build_mesh(self.p,1,0.1)
        q=properties(self.p,m)['oxygen_flow_m3_s']
        p={**self.p,'current_density_A_cm2':self.p['current_density_A_cm2']*2}
        self.assertAlmostEqual(properties(p,m)['oxygen_flow_m3_s'],2*q,places=16)

    def test_invalid_pores_and_phase_fractions_are_rejected(self):
        for change in [{'pore_width_mm':0.3},{'water_inlet_alpha':1.1},
                       {'pore_pitch_mm':0.3},{'current_density_A_cm2':-1}]:
            with self.assertRaises(ValueError): validate({**self.p,**change},1,0.05)

    def test_variable_time_step_statistics_are_time_weighted(self):
        rows=[{'time_s':t,'v':2*t+1} for t in [0,0.01,0.04,0.5,1]]
        self.assertAlmostEqual(weighted_mean(rows,0.2,0.8,'v'),2.0)
        self.assertAlmostEqual(weighted_mean(rows,0,1,'v'),2.0)


if __name__=='__main__': unittest.main()
