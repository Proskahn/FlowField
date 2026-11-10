"""Paper transcription, scientific metric guards, and public integration."""

from dataclasses import replace
from itertools import product
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield import FlowFieldProblem, get_problem, list_problems
from flowfield.problems.level2_medium_parametric.p001 import BASELINE_X, P001Problem, PPFCConfig
from flowfield.problems.level2_medium_parametric.p001.case import generate_case
from flowfield.problems.level2_medium_parametric.p001.config import uniformity
from flowfield.problems.level2_medium_parametric.p001.evaluation import run_case
from flowfield.problems.level2_medium_parametric.p001.geometry import unit_layout, split_tetrahedra
from flowfield.problems.level2_medium_parametric.p001.metrics import extract_metrics
from flowfield.runtime import SimulationError


class DefinitionTests(unittest.TestCase):
    def test_table6_bounds_units_and_baseline(self):
        p = get_problem('P001')
        self.assertIsInstance(p, FlowFieldProblem)
        self.assertEqual(p.dimension, 6)
        lo, hi = p.decode([0]*6), p.decode([1]*6)
        self.assertEqual(lo['spacings_mm'], dict(zip(('x1_1','x1_2','x2','x3','h1_1','h1_2'), (31,28,27,28,13,13))))
        self.assertEqual(list(hi['spacings_mm'].values()), [37,33,32,31,17,17])
        self.assertAlmostEqual(lo['spacings_m']['x1_1'], .031)
        self.assertEqual(list(p.decode(BASELINE_X)['spacings_mm'].values()), [35,32,30,30,15,15])
        self.assertEqual(lo['diameter_m'], .4)
        self.assertIn('reconstruction', p.ioh_name)

    def test_all_six_parameters_change_layout_with_fixed_population(self):
        p = P001Problem()
        baseline = unit_layout(p.decode([.5]*6))
        for i in range(6):
            x = [.5]*6
            x[i] = 1.
            self.assertNotEqual(baseline, unit_layout(p.decode(x)))
        for x in product((0., 1.), repeat=6):
            units = unit_layout(p.decode(x))
            self.assertEqual(len(units), len(baseline))
            for unit in units:
                radius = .0075 if unit['kind'] in ('convex','concave') else .013/math.sqrt(2)
                self.assertLess(math.hypot(unit['x'], unit['y'])+radius, .2)

    def test_invalid_coordinates_and_config(self):
        p = P001Problem()
        for x in ([.5]*5, [.5]*7, [math.nan]*6, [-1]*6, [True]*6, ['0']*6):
            with self.subTest(x=x), self.assertRaises(ValueError):
                p.decode(x)
        for kwargs in ({'mesh':'paper-mesh4'}, {'inlet_velocity_m_s':1}, {'sphere_cap_height_m':.008},
                       {'residual_tolerance':math.nan}, {'max_iterations':20.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PPFCConfig(**kwargs)
        with self.assertRaises(TypeError):
            P001Problem(config={})
        c = PPFCConfig()
        self.assertAlmostEqual(c.kinematic_viscosity_m2_s, .0025491/1290.4769)
        self.assertNotEqual(c.fingerprint, replace(c, mesh='medium').fingerprint)

    def test_conforming_hex_subdivision_preserves_shared_faces_and_volume(self):
        nodes = {1:(0.,0.,0.), 2:(1.,0.,0.), 3:(0.,1.,0.), 4:(0.,0.,1.), 5:(0.,0.,-1.)}
        out, quads, hexes = split_tetrahedra(nodes, [(1,(1,2,3))], [(2,(1,2,3,4)),(2,(1,3,2,5))])
        self.assertEqual(len(quads), 3)
        self.assertEqual(len(hexes), 8)
        # The shared face centroid is created once, and the subdivisions share it.
        common = [i for i,v in out.items() if v == (1/3,1/3,0)]
        self.assertEqual(len(common), 1)
        self.assertEqual(sum(common[0] in ids for _,ids in hexes), 6)
        # Every cell centroid is the mean of the original tetrahedron vertices.
        self.assertIn((.25,.25,.25), out.values())
        self.assertIn((.25,.25,-.25), out.values())
        # Independent oriented surface integral for each generated hexahedron.
        faces = ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
        for _,ids in hexes:
            volume = 0.
            for face in faces:
                for tri in ((face[0],face[1],face[2]),(face[0],face[2],face[3])):
                    a,b,c = (out[ids[i]] for i in tri)
                    volume += (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0])
                               + a[2]*(b[0]*c[1]-b[1]*c[0]))/6
            self.assertAlmostEqual(volume, 1/24)
        with self.assertRaisesRegex(ValueError,'nonpositive'):
            split_tetrahedra(nodes, [], [(2,(1,3,2,4))])

    def test_equations5_and6_area_weights_and_zero_flow(self):
        self.assertEqual(uniformity([2,2], [1,3]), 1.)
        self.assertAlmostEqual(uniformity([0,2], [1,3]), .75)
        self.assertEqual(uniformity([0,2], [1,1]), .5)
        self.assertAlmostEqual(uniformity([0,20], [2,6]), .75)
        for u,a in (([0,0],[1,1]), ([],[]), ([1],[0]), ([1,2],[1]), ([math.nan],[1]), ([-1],[1])):
            with self.subTest(u=u,a=a), self.assertRaises(ValueError):
                uniformity(u,a)

    def test_generated_case_no_overwrite_and_tamper_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = P001Problem()
            case = generate_case(Path(tmp)/'case', p.decode(BASELINE_X), p.config)
            manifest = json.loads((case/'case.json').read_text())
            self.assertEqual(manifest['replication_status'], 'provisional_geometry_reconstruction')
            self.assertIn('RNGkEpsilon', (case/'constant/turbulenceProperties').read_text())
            with self.assertRaises(FileExistsError):
                generate_case(case, p.decode(BASELINE_X), p.config)
            (case/'0/U').write_text('altered')
            with self.assertRaisesRegex(ValueError, 'input changed'), patch('flowfield.runtime.Runtime.resolve') as resolve:
                run_case(case)
            resolve.assert_not_called()

    def test_raw_contract_and_failure_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = P001Problem(tmp)
            cfd = {'status':'ok', 'objective':-.7, 'objectives':{'negative_uniformity':-.7},
                   'constraints':{}, 'case_directory':tmp, 'runtime':{}, 'metrics':{'uniformity':.7}}
            with patch('flowfield.problems.level2_medium_parametric.p001.evaluation.evaluate', return_value=cfd) as run:
                self.assertEqual(p(BASELINE_X), -.7)
            run.assert_called_once()
            self.assertFalse(p.last_result['metadata']['paper_validated'])
            self.assertTrue(p.last_result['metadata']['valid_cfd'])
            self.assertTrue((Path(tmp)/'evaluation.json').is_file())
            with patch('flowfield.problems.level2_medium_parametric.p001.evaluation.evaluate', side_effect=SimulationError('fail')):
                with self.assertRaises(SimulationError):
                    p(BASELINE_X)
                raw = P001Problem(tmp, failure_penalty=10).evaluate_raw(BASELINE_X)
            self.assertEqual(raw['status'], 'penalized')
            self.assertEqual(raw['objectives'], {})
            self.assertFalse(raw['metadata']['valid_cfd'])


def fixture(case, config):
    (case/'log.checkMesh').write_text('cells: 1234\nMesh OK.\n')
    (case/'log.simpleFoam').write_text('Time = 30\nSIMPLE solution converged in 31 iterations\nEnd\n')
    (case/'mesh-summary.json').write_text('{"cells": 5000}')
    columns = {'p_inlet':('areaAverage(p)', .01), 'p_outlet':('areaAverage(p)', 0),
        'q_inlet':('sum(phi)', -config.flow_rate_m3_s), 'q_outlet':('sum(phi)', config.flow_rate_m3_s),
        'uniformity_midplane':('uniformity(speed)', .7), 'speed_midplane':('areaAverage(speed)', .02),
        **{f'uniformity_section{i}':('uniformity(speed)', .6+i*.01) for i in range(1,5)}}
    for name, (column, value) in columns.items():
        directory = case/'postProcessing'/name/'0'
        directory.mkdir(parents=True)
        (directory/'surfaceFieldValue.dat').write_text(f'# Time {column}\n'+''.join(f'{i} {value}\n' for i in range(1,31)))


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.case = Path(self.tmp.name)
        self.config = PPFCConfig()
        fixture(self.case, self.config)

    def rewrite(self, name, column, values):
        path = self.case/'postProcessing'/name/'0'/'surfaceFieldValue.dat'
        path.write_text(f'# Time {column}\n'+''.join(f'{i} {v}\n' for i,v in enumerate(values,1)))

    def test_accepted_metric_and_pressure_units(self):
        m = extract_metrics(self.case, self.config)
        self.assertEqual(m['uniformity'], .7)
        self.assertAlmostEqual(m['pressure_drop_pa'], 12.904769)
        self.assertEqual(m['cell_count'], 1234)
        self.assertFalse(m['paper_validated'])

    def test_unconverged_and_failed_mesh(self):
        for name, value in (('log.simpleFoam','Time = 30\nEnd\n'), ('log.checkMesh','Failed 1 mesh checks.\n')):
            path = self.case/name
            original = path.read_text()
            path.write_text(value)
            with self.subTest(name=name), self.assertRaises(SimulationError):
                extract_metrics(self.case, self.config)
            path.write_text(original)

    def test_stale_and_unstable_uniformity(self):
        self.rewrite('uniformity_midplane','uniformity(speed)', [.7]*29)
        with self.assertRaisesRegex(SimulationError, 'unaligned'):
            extract_metrics(self.case, self.config)
        self.rewrite('uniformity_midplane','uniformity(speed)', [.7]*29+[.71])
        with self.assertRaisesRegex(SimulationError, 'not stable'):
            extract_metrics(self.case, self.config)

    def test_wrong_flux_and_undefined_uniformity_are_rejected(self):
        for name, column, value in (('q_outlet','sum(phi)',0), ('speed_midplane','areaAverage(speed)',0),
                                   ('uniformity_section1','uniformity(speed)',1.1)):
            path = self.case/'postProcessing'/name/'0'/'surfaceFieldValue.dat'
            original = path.read_text()
            self.rewrite(name,column,[value]*30)
            with self.subTest(name=name), self.assertRaises(SimulationError):
                extract_metrics(self.case, self.config)
            path.write_text(original)


if __name__ == '__main__':
    unittest.main()
