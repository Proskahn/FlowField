"""P002 geometry, objective semantics, fresh-case provenance and failure checks."""

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield import get_problem, PEMConfig
from flowfield.runtime import SimulationError
from flowfield.problems.level2_medium_parametric.p002 import BASELINE_X, PAPER_OPTIMUM_X, P002Problem
from flowfield.problems.level2_medium_parametric.p002.case import generate_case, INPUTS
from flowfield.problems.level2_medium_parametric.p002.config import (
    PARAMETERS, PAPER_OPTIMUM_MM, normalize, parameters_for, scalar_objective)
from flowfield.problems.level2_medium_parametric.p002.geometry import geometry, mesh, subdivide, FLUID
from flowfield.problems.level2_medium_parametric.p002.evaluation import run_case
from flowfield.problems.level2_medium_parametric.p002.metrics import read_history, HISTORY_COLUMNS
from flowfield.problems.level2_medium_parametric.p002.metrics import field_pressure_drops
from flowfield.problems.level2_medium_parametric.p002.mesh_checks import assess_mesh
from flowfield.problems.level2_medium_parametric.p002.verification import values


class P002Tests(unittest.TestCase):
    def test_published_design_mapping_and_bounds(self):
        p = get_problem('P002')
        self.assertEqual(p.dimension,7)
        self.assertEqual(list(p.decode(BASELINE_X)['dimensions_mm'].values()),[1.]*7)
        for actual,expected in zip(p.decode(PAPER_OPTIMUM_X)['dimensions_mm'].values(),PAPER_OPTIMUM_MM):
            self.assertAlmostEqual(actual,expected,places=14)
        for bad in ([0.]*6,[0.]*6+[1.01],[0.]*6+[math.nan],[0.]*6+[True]):
            with self.assertRaises(ValueError):
                p.decode(bad)
        with self.assertRaises(ValueError):
            normalize([1.]*6+[2.])

    def check_geometry(self, x):
        g = P002Problem().decode(x)
        params = parameters_for(g,PEMConfig())
        m = geometry(params['geometry'],'parallel','coarse')
        volumes = [0.]*9
        for (i,j,k),zone in zip(m['cells'],m['ids']):
            volumes[zone] += ((m['x'][i+1]-m['x'][i])*(m['y'][j+1]-m['y'][j])
                              *(m['z'][k+1]-m['z'][k]))
        dims = g['dimensions_mm']
        channel_area = 20+8*sum(dims[k] for k in PARAMETERS[:5])
        # Geometry knots are rounded to 1e-10 mm before meshing.
        self.assertAlmostEqual(volumes[1],channel_area*dims['Hc'],delta=1e-8)
        self.assertAlmostEqual(volumes[7],channel_area*dims['Ha'],delta=1e-8)
        for zone in (3,5):
            self.assertAlmostEqual(volumes[zone],121*.02,places=10)
        for zone in (2,6):
            self.assertAlmostEqual(volumes[zone],121*.3,places=10)
        self.assertAlmostEqual(volumes[4],121*.178,places=10)
        self.assertAlmostEqual(sum(volumes),121*3.818,places=8)
        with tempfile.TemporaryDirectory() as tmp:
            mesh(Path(tmp),m)
            self.assertAlmostEqual(m['inlet_areas']['anodeInlet'],dims['Ha']*1e-6,places=14)
            self.assertAlmostEqual(m['inlet_areas']['cathodeInlet'],dims['Hc']*1e-6,places=14)
        # Anode/cathode flow networks are connected internally and separated by PEM.
        nx,ny,nz = (len(m[k])-1 for k in ('x','y','z'))
        fluid = {c for c,z in enumerate(m['ids']) if z in FLUID}
        count = 0
        while fluid:
            count += 1
            pending = [fluid.pop()]
            while pending:
                c = pending.pop(); i,j,k = m['cells'][c]
                for di,dj,dk in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
                    ii,jj,kk = i+di,j+dj,k+dk
                    if 0<=ii<nx and 0<=jj<ny and 0<=kk<nz:
                        n = (kk*ny+jj)*nx+ii
                        if n in fluid:
                            fluid.remove(n); pending.append(n)
        self.assertEqual(count,2)

    def test_baseline_and_asymmetric_geometry(self):
        for x in (BASELINE_X,PAPER_OPTIMUM_X,[0,1,0,1,0,0,1],[1]*7):
            with self.subTest(x=x):
                self.check_geometry(x)

    def test_refinement(self):
        g = parameters_for(P002Problem().decode(PAPER_OPTIMUM_X),PEMConfig())['geometry']
        meshes = [geometry(g,'parallel',level) for level in ('coarse','medium','fine')]
        self.assertLess(len(meshes[0]['cells']),len(meshes[1]['cells']))
        self.assertLess(len(meshes[1]['cells']),len(meshes[2]['cells']))
        self.assertEqual({m['z'][-1] for m in meshes},{3.818})

    def test_grading_preserves_boundaries_and_bounds_adjacent_widths(self):
        knots = [0, .999, 1, 1.001, 2]
        grid = subdivide(knots, .5, max_ratio=8)
        self.assertTrue(set(knots) <= set(grid))
        widths = [b-a for a, b in zip(grid, grid[1:])]
        self.assertGreater(min(widths), 0)
        self.assertLessEqual(max(widths), .5)
        self.assertTrue(all(max(a, b)/min(a, b) <= 8 for a, b in zip(widths, widths[1:])))
        # Already well-spaced geometry, including the baseline, is unchanged.
        self.assertEqual(subdivide([0, 1, 2], .5, max_ratio=8), subdivide([0, 1, 2], .5))

    def test_seed42_candidate_has_safe_face_interpolation_weights(self):
        # First proposal from CMA-ES, seed=42, sigma=.1, population=4.
        x = [.29967141530112335, .23617347112231374, .31476977908601855,
             .402306249311177, .22658399350846703, .5321410236422874, .7134836053062501]
        self.check_geometry(x)
        g = parameters_for(P002Problem().decode(x), PEMConfig())['geometry']
        for level in ('coarse', 'medium', 'fine'):
            m = geometry(g, 'parallel', level)
            for axis in ('x', 'y', 'z'):
                widths = [b-a for a, b in zip(m[axis], m[axis][1:])]
                weights = [min(a, b)/(a+b) for a, b in zip(widths, widths[1:])]
                with self.subTest(level=level, axis=axis):
                    self.assertGreater(min(weights), .05)

    def test_scalarization_units_and_identity(self):
        self.assertAlmostEqual(scalar_objective(2.,200.,PEMConfig()),2.*1.21+200.*2.5e-7)
        self.assertEqual(scalar_objective(2.,200.,PEMConfig(scalarization='voltage_v')),2.)
        self.assertEqual(scalar_objective(2.,200.,PEMConfig(scalarization='pressure_sum_pa')),200.)
        self.assertNotEqual(P002Problem().ioh_name,P002Problem(config=PEMConfig(mesh='medium')).ioh_name)
        self.assertNotEqual(P002Problem().ioh_name,P002Problem(config=PEMConfig(scalarization='voltage_v')).ioh_name)
        with self.assertRaises(ValueError):
            scalar_objective(2.,math.nan,PEMConfig())

    def test_config_is_validated_and_immutable(self):
        overrides = {'inletFlow_mL_min':20.}
        config = PEMConfig(model_overrides=overrides)
        overrides['inletFlow_mL_min'] = -1
        self.assertEqual(config.model['inletFlow_mL_min'],20.)
        for kwargs in (dict(iterations=True),dict(model_overrides={'unknown':1}),
                       dict(model_overrides={'porosityCL':1.}),dict(model_overrides={'operatingPressure':1.}),
                       dict(model_overrides={'outerTolerance':1.}),dict(model_overrides={'currentDensity':True})):
            with self.assertRaises(ValueError):
                PEMConfig(**kwargs)
        self.assertEqual(PEMConfig(**config.to_dict()).fingerprint,config.fingerprint)

    def test_case_provenance_and_reuse_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)/'case'
            generate_case(case,P002Problem().decode(BASELINE_X),PEMConfig())
            manifest = json.loads((case/'case.json').read_text())
            self.assertEqual(set(manifest['input_sha256']),INPUTS)
            self.assertIn('startFrom startTime;', (case/'system/controlDict').read_text())
            self.assertIn('flowfieldP002Foam',(case/'solver/Make/files').read_text())
            with self.assertRaises(FileExistsError):
                generate_case(case,P002Problem().decode(BASELINE_X),PEMConfig())
            (case/'0/p').write_text('modified')
            with self.assertRaisesRegex(ValueError,'Case input changed'):
                run_case(case)

    def test_stale_fields_rejected_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)/'case'
            generate_case(case,P002Problem().decode(BASELINE_X),PEMConfig())
            (case/'100').mkdir()
            with self.assertRaises(FileExistsError):
                run_case(case)

    def test_saved_pressure_uses_pa_and_zero_gradient_owner_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)/'case'
            generate_case(case,P002Problem().decode(PAPER_OPTIMUM_X),PEMConfig())
            manifest = json.loads((case/'case.json').read_text())
            (case/'1').mkdir()
            text = (case/'0/p').read_text().replace('internalField uniform 0;', 'internalField uniform 123;')
            (case/'1/p').write_text(text)
            drops = field_pressure_drops(case,1,manifest)
            for value in drops.values():
                self.assertAlmostEqual(value,123.)
            (case/'1/p').write_text(text.replace('[1 -1 -2 0 0 0 0]','[0 2 -2 0 0 0 0]'))
            with self.assertRaisesRegex(ValueError,'dimensional Pa'):
                field_pressure_drops(case,1,manifest)

    def test_mesh_exception_is_confined_to_orthogonal_catalyst_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)/'case'
            generate_case(case,P002Problem().decode(BASELINE_X),PEMConfig())
            manifest = json.loads((case/'case.json').read_text())
            m = geometry(manifest['parameters']['geometry'],'parallel','coarse')
            cell = m['ids'].index(3)
            target = case/'constant/polyMesh/sets/underdeterminedCells'
            target.parent.mkdir()
            target.write_text(f'FoamFile {{}}\n1\n(\n{cell}\n)\n')
            log = ('Max aspect ratio = 50 OK.\nMesh non-orthogonality Max: 0 average: 0\n'
                   'Max skewness = 2e-14 OK.\nCell determinant (wellposedness) : minimum: 2e-6 average: .3\n'
                   ' ***Cells with small determinant (< 0.001) found\nFailed 1 mesh checks.\n')
            self.assertFalse(assess_mesh(case,log,manifest)['openfoam_all_checks_passed'])
            for bad in (log.replace('Failed 1','Failed 2'),log.replace('Max: 0','Max: 5'),
                        log.replace('50 OK','200 OK'),log+' ***Negative volumes\n'):
                with self.assertRaises(ValueError):
                    assess_mesh(case,bad,manifest)
            target.write_text('FoamFile {}\n1\n(\n0\n)\n')
            with self.assertRaisesRegex(ValueError,'not confined'):
                assess_mesh(case,log,manifest)

    def test_raw_keeps_both_objectives_and_calls_once(self):
        module = 'flowfield.problems.level2_medium_parametric.p002.evaluation.evaluate'
        result = dict(status='ok', objective=2.42, objectives={'voltage_V':2.,'pressure_sum_Pa':200.},
                      constraints={}, case_directory=None, runtime={}, metrics={})
        with patch(module,return_value=result) as run:
            p = P002Problem()
            self.assertEqual(p(BASELINE_X),2.42)
            run.assert_called_once()
            self.assertEqual(p.last_result['objectives'],{'voltage_V':2.,'pressure_sum_Pa':200.})
            self.assertFalse(p.last_result['metadata']['paper_validated'])
        with patch(module,side_effect=SimulationError('failure')):
            with self.assertRaises(SimulationError):
                P002Problem()(BASELINE_X)
            result = P002Problem(failure_penalty=100).evaluate_raw(BASELINE_X)
            self.assertEqual(result['status'],'penalized')
            self.assertFalse(result['metadata']['valid_cfd'])
            self.assertEqual(result['objectives'],{})

    def test_strict_history_and_nonfinite_field_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            history = case/'postProcessing/pem/history_0.csv'
            history.parent.mkdir(parents=True)
            def write_rows(iterations):
                with history.open('w',newline='') as stream:
                    writer = csv.DictWriter(stream,fieldnames=HISTORY_COLUMNS)
                    writer.writeheader()
                    for it in iterations:
                        writer.writerow(dict.fromkeys(HISTORY_COLUMNS,0.) | {'iteration':it})
            write_rows(range(1,6))
            self.assertEqual(len(read_history(case)),5)
            write_rows([1,2,4,5,6])
            with self.assertRaises(ValueError):
                read_history(case)
            write_rows([1,2,3,4,math.nan])
            with self.assertRaises(ValueError):
                read_history(case)
        for text in ('internalField uniform nan;', 'internalField nonuniform List<scalar> 2 (1);'):
            with self.assertRaises(ValueError):
                values(text,'internalField',2)


if __name__ == '__main__':
    unittest.main()
