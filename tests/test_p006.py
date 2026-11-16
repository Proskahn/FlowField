import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import shutil
import re
import unittest

from flowfield import create_problem, CaseEvaluator
from flowfield.problems.level1_small_continuous.p006 import P006Problem, ManifoldConfig
from flowfield.problems.level1_small_continuous.p006.config import spline_widths
from flowfield.problems.level1_small_continuous.p006.geometry import build_geometry, PATCHES
from flowfield.problems.level1_small_continuous.p006.metrics import values, vectors, verify_phase_fluxes, extract_metrics
from flowfield.problems.level1_small_continuous.p006.case import generate_case
from flowfield.problems.level1_small_continuous.p006.warm_start import FIELDS
from flowfield.problems.level1_small_continuous.p006.pressure import pressure_forces,verify_pressure_forces


class P006Tests(unittest.TestCase):
    def test_pressure_trace_matches_layered_darcy_solution(self):
        p=P006Problem();m=build_geometry(p.decode(p.initial_design()),p.config)
        nx,ny=len(m['x'])-1,len(m['y'])-1;stride=nx*ny
        # Analytical constant through-plane Darcy flux: pressure is continuous
        # and its slope changes by the permeability ratio at each interface.
        edges=[0,.5,1.5,1.8,1.85,1.8678,1.9178,2.2178,3.2178,3.7178]
        permeability=[1e-6,1e-6,1e-12,1e-13,1e-6,1e-13,1e-12,1e-6,1e-6]
        flux=1e-6;pressure=[]
        for _,_,k in m['cells']:
            z=(m['z'][k]+m['z'][k+1])/2;drop=0.
            for a,b,perm in zip(edges,edges[1:],permeability):
                drop+=max(0,min(z,b)-a)*.001*flux*p.config.liquid_viscosity/perm
            pressure.append(100-drop)
        actual=pressure_forces(m,p.config,pressure,[0.]*len(pressure))
        checked=0
        for c,mat in enumerate(m['ids']):
            if mat not in (2,6) or any(m['ids'][c+d*stride] not in (1,2,3,5,6,7) for d in (-1,1)):continue
            for phase in ('L','G'):
                self.assertAlmostEqual(actual[phase][c][0],0,places=7)
                self.assertAlmostEqual(actual[phase][c][1],0,places=7)
                self.assertAlmostEqual(actual[phase][c][2],flux*p.config.liquid_viscosity/1e-12,places=7)
            checked+=1
        self.assertGreater(checked,100)

    def test_pressure_force_matches_manufactured_linear_field_and_wall_conditions(self):
        p=P006Problem();m=build_geometry(p.decode(p.initial_design()),p.config)
        nx,ny=len(m['x'])-1,len(m['y'])-1
        gradient=1000.;pressure=[];expected=[]
        for c,((i,j,k),mat) in enumerate(zip(m['cells'],m['ids'])):
            x=(m['x'][i]+m['x'][i+1])*.0005
            pressure.append(gradient*(x-(.018 if mat>=5 else 0)))
            count=0
            if mat in (1,2,3,5,6,7):
                for direction in (-1,1):
                    if 0<=i+direction<nx:
                        count+=m['ids'][c+direction] in (1,2,3,5,6,7)
                    else:
                        y=(m['y'][j]+m['y'][j+1])/2
                        count+=(mat==7 and direction==1 and y>24) or (mat==1 and direction==-1 and y<1)
            expected.append([-gradient*count/2,0.,0.])
        # Zero capillary pressure isolates the pressure-gradient operator.
        saturation=[0.]*len(pressure)
        actual=pressure_forces(m,p.config,pressure,saturation)
        for phase in ('L','G'):
            for got,want in zip(actual[phase],expected):
                for a,b in zip(got,want):self.assertAlmostEqual(a,b,places=7)
        self.assertLess(verify_pressure_forces(m,p.config,pressure,saturation,
            {'L':expected,'G':expected}),1e-10)
        broken=[row[:] for row in expected]
        cell=next(i for i,v in enumerate(m['ids']) if v==7)
        broken[cell][0]+=100
        with self.assertRaisesRegex(ValueError,'physical pressure force'):
            verify_pressure_forces(m,p.config,pressure,saturation,{'L':broken,'G':expected})

    def finished_state(self, root):
        p=P006Problem();case=p.write_case(p.initial_design(),root/'source')
        manifest=json.loads((case/'case.json').read_text())
        saved=case/'10';saved.mkdir()
        for name in FIELDS:shutil.copy2(case/'0'/name,saved/name)
        text=(saved/'T').read_text()
        text=re.sub(r'\binternalField\b[^;]+;','internalField uniform 354;',text,count=1)
        # The guess must not replace the target's prescribed inlet temperature.
        (saved/'T').write_text(text.replace('value uniform 353.0','value uniform 351.0'))
        post=case/'postProcessing/p006';post.mkdir(parents=True)
        (post/'status.json').write_text(json.dumps(dict(converged=False,iteration=10,
            cells=manifest['cells'],processes=1)))
        (case/'log.flowfieldP006Foam').write_text('Iteration = 10\nEnd\n')
        (case/'result.json').write_text(json.dumps(dict(status='failed',input_sha256=manifest['input_sha256'])))
        return p,case

    def test_finished_state_is_a_new_guess_with_fresh_boundaries_and_no_inherited_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.finished_state(root)
            original={name:(source/'10'/name).read_bytes() for name in FIELDS}
            # CLI millimetres-to-normalized conversion differs by one ulp.
            cli_design=[(1.-.7)/.8]*3
            case=generate_case(root/'continued',p.decode(cli_design),
                ManifoldConfig(processes=2,iterations=4000),initial_state=source)
            manifest=json.loads((case/'case.json').read_text())
            self.assertEqual(values((case/'0/T').read_text(),'internalField',manifest['cells']),[354]*manifest['cells'])
            self.assertIn('value uniform 353.0;', (case/'0/T').read_text())
            self.assertNotIn('value uniform 351.0;', (case/'0/T').read_text())
            self.assertFalse(manifest['initial_state']['qualification_inherited'])
            self.assertFalse(manifest['initial_state']['source_converged'])
            self.assertFalse((case/'postProcessing').exists())
            self.assertEqual(original,{name:(source/'10'/name).read_bytes() for name in FIELDS})
            CaseEvaluator(p.case_definition)._load_case(case)

    def test_initial_guess_refuses_running_case_without_creating_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.finished_state(root)
            (source/'result.json').unlink()
            with self.assertRaisesRegex(ValueError,'finished run'):
                generate_case(root/'continued',p.decode(p.initial_design()),p.config,initial_state=source)
            self.assertFalse((root/'continued').exists())

    def test_initial_guess_refuses_changed_geometry_or_invalid_saved_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.finished_state(root)
            with self.assertRaisesRegex(ValueError,'same P006 geometry'):
                generate_case(root/'changed',p.decode([.5,.375,.375]),p.config,initial_state=source)
            (source/'10/T').write_text('internalField uniform nan;')
            with self.assertRaises(ValueError):
                generate_case(root/'invalid',p.decode(p.initial_design()),p.config,initial_state=source)
            self.assertFalse((root/'invalid').exists())

    def test_initial_guess_preserves_face_fluxes_and_fresh_inlet_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.finished_state(root)
            saved=source/'10/phiL'
            text=re.sub(r'\binternalField\b[^;]+;','internalField uniform 1e-12;',saved.read_text(),count=1)
            text=re.sub(r'\banodeInlet\s*\{.*?\}',
                'anodeInlet { type calculated; value uniform -1; }',text,flags=re.S)
            text=re.sub(r'\banodeOutlet\s*\{.*?\}',
                'anodeOutlet { type calculated; value uniform 2e-8; }',text,flags=re.S)
            saved.write_text(text)
            case=generate_case(root/'continued',p.decode(p.initial_design()),p.config,initial_state=source)
            copied=(case/'0/phiL').read_text()
            self.assertIn('internalField uniform 1e-12;',copied)
            outlet=re.search(r'\banodeOutlet\s*\{(.*?)\}',copied,re.S).group(1)
            self.assertIn('value uniform 2e-8;',outlet)
            inlet=re.search(r'\banodeInlet\s*\{(.*?)\}',copied,re.S).group(1)
            original=re.search(r'\banodeInlet\s*\{(.*?)\}',(source/'0/phiL').read_text(),re.S).group(1)
            self.assertEqual(inlet,original)
            self.assertTrue(json.loads((case/'case.json').read_text())['initial_state']['preserves_face_fluxes'])

    def test_initial_guess_refuses_incomplete_face_fluxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.finished_state(root)
            (source/'10/phiG').write_text('internalField nonuniform List<scalar> 2 (0);')
            with self.assertRaises(ValueError):
                generate_case(root/'invalid',p.decode(p.initial_design()),p.config,initial_state=source)
            self.assertFalse((root/'invalid').exists())

    def test_spline_knots_symmetry_and_area(self):
        w=spline_widths([.7,1.5,.7])
        self.assertEqual((w[0],w[4],w[8]),(.7,1.5,.7))
        self.assertEqual(w,list(reversed(w)))
        self.assertEqual(spline_widths([1,1,1]),[1]*9)
        p=create_problem('P006');g=p.decode(p.initial_design())
        self.assertEqual(p.dimension,3)
        self.assertAlmostEqual(g['contact_area_mm2'],207.)
        self.assertAlmostEqual(g['constraints']['contact_area'],-.05)
        self.assertTrue(all(v<=0 for v in g['constraints'].values()))

    def test_natural_spline_matches_independent_scipy(self):
        if not importlib.util.find_spec('scipy'):self.skipTest('optional scipy absent')
        from scipy.interpolate import CubicSpline
        for controls in ([.7,1.5,.7],[1.5,.7,.7],[.7,1.2,1.5]):
            expected=CubicSpline([0,4,8],controls,bc_type='natural')(range(9))
            for a,b in zip(spline_widths(controls),expected):self.assertAlmostEqual(a,b,places=13)

    def test_spline_overshoot_is_a_constraint(self):
        p=P006Problem();g=p.decode([1,0,0])
        self.assertGreater(g['constraints']['minimum_width'],0)
        self.assertGreater(g['constraints']['contact_area'],0)

    def test_layer_volumes_and_counterflow_ports(self):
        p=P006Problem();g=p.decode(p.initial_design());m=build_geometry(g,p.config)
        volumes={i:0. for i in range(9)}
        for (i,j,k),mat in zip(m['cells'],m['ids']):
            volumes[mat]+=(m['x'][i+1]-m['x'][i])*(m['y'][j+1]-m['y'][j])*(m['z'][k+1]-m['z'][k])
        self.assertAlmostEqual(sum(volumes.values()),18*25*3.7178,places=7)
        self.assertAlmostEqual(volumes[4],18*25*.0178,places=8)
        self.assertAlmostEqual(volumes[5],18*25*.05,places=8)
        self.assertAlmostEqual(volumes[7],18*2+9*23,places=8)
        with tempfile.TemporaryDirectory() as tmp:
            case=p.write_case(p.initial_design(),Path(tmp)/'case')
            data=json.loads((case/'case.json').read_text())
            self.assertEqual(data['cells'],len(m['cells']))
            self.assertTrue(all(abs(a-1e-6)<1e-15 for a in data['inlet_areas_m2'].values()))
            velocity=(case/'0/Ul').read_text()
            self.assertIn('uniform (-0.277777777778 0 0)',velocity)
            self.assertIn('uniform (0.277777777778 0 0)',velocity)
            for phase,ratio in [('L',1),('G',1/9)]:
                for patch in ('anodeInlet','cathodeInlet'):
                    block=re.search(r'\b'+patch+r'\s*\{(.*?)\}',(case/'0'/('phi'+phase)).read_text(),re.S).group(1)
                    count=int(re.search(r'List<scalar>\s+(\d+)',block).group(1))
                    self.assertAlmostEqual(sum(values(block,'value',count)),-15e-6/60*ratio,places=16)
            CaseEvaluator(p.case_definition)._load_case(case)
            with self.assertRaises(FileExistsError):p.write_case(p.initial_design(),case)
            (case/'0/T').write_text('altered')
            with self.assertRaises(ValueError):CaseEvaluator(p.case_definition)._load_case(case)

    def test_manifold_sweep_keeps_footprint_and_contact_reference(self):
        for w in (1,1.5,2,2.5,3):
            p=P006Problem(config=ManifoldConfig(manifold_width_mm=w))
            g=p.decode(p.initial_design())
            self.assertEqual(g['footprint_mm'],[18,25])
            self.assertEqual(g['clear_branch_length_mm'],25-2*w)
            self.assertAlmostEqual(g['contact_area_mm2']/g['reference_contact_area_mm2'],1.)

    def test_channel_width_optimization_preserves_mesh_topology(self):
        p=P006Problem()
        meshes=[build_geometry(p.decode(x),p.config) for x in ([.375]*3,[.475,.375,.375],[0,1,0],[1,0,1])]
        self.assertTrue(all(m['cells']==meshes[0]['cells'] for m in meshes))
        self.assertTrue(all(m['ids']==meshes[0]['ids'] for m in meshes))

    def test_config_rejects_invalid_inputs_and_changes_identity(self):
        for kw in (dict(manifold_width_mm=4),dict(processes=0),dict(processes=2.5),
                   dict(contact_tolerance=0),dict(permeability_ptl=-1),dict(cell_voltage=3),
                   dict(field_tolerance=.1),dict(inlet_liquid_saturation=1)):
            with self.subTest(kw=kw),self.assertRaises(ValueError):ManifoldConfig(**kw)
        self.assertNotEqual(ManifoldConfig().fingerprint,ManifoldConfig(manifold_width_mm=2).fingerprint)

    def test_field_parser_refuses_nonfinite_and_truncated_output(self):
        self.assertEqual(values('internalField uniform 353;','internalField',2),[353,353])
        for text in ('internalField uniform nan;','internalField nonuniform List<scalar> 2 (1);'):
            with self.assertRaises(ValueError):values(text,'internalField',2)

    def test_vector_parser_refuses_incomplete_or_nonfinite_velocity(self):
        self.assertEqual(vectors('internalField uniform (1 2 3);','internalField',2),[(1,2,3)]*2)
        self.assertEqual(vectors('internalField nonuniform List<vector> 2 ((1 2 3) (4 5 6));','internalField',2),[(1,2,3),(4,5,6)])
        for text in ('internalField uniform (1 nan 3);',
                     'internalField nonuniform List<vector> 2 ((1 2 3));',
                     'internalField uniform (1 2);'):
            with self.assertRaises(ValueError):vectors(text,'internalField',2)

    def test_saved_flux_verification_detects_local_error_even_when_global_balances(self):
        p=P006Problem();m=build_geometry(p.decode(p.initial_design()),p.config)
        with tempfile.TemporaryDirectory() as tmp:
            case=p.write_case(p.initial_design(),Path(tmp)/'case');out=case/'1';out.mkdir()
            zero='internalField uniform 0;\nboundaryField\n{\n'
            for phase in ('L','G'):
                text=zero
                for patch in PATCHES:
                    phi=-1e-7 if phase=='L' and patch=='anodeInlet' else 1e-7 if phase=='L' and patch=='anodeOutlet' else 0
                    text+=f'{patch} {{ type calculated; value uniform {phi}; }}\n'
                (out/('phi'+phase)).write_text(text+'}\n')
            n=len(m['cells'])
            checks=verify_phase_fluxes(case,out,m['ids'],[1e-12]*n,{'Sl':[0.]*n,'Sg':[0.]*n},p.config)
            self.assertAlmostEqual(checks['saved_global_L_balance'],0.)
            self.assertGreater(checks['saved_local_L_balance'],.1)

    def test_converged_voltage_continuation_stage_is_not_the_requested_case(self):
        p=P006Problem(config=ManifoldConfig(minimum_iterations=1))
        with tempfile.TemporaryDirectory() as tmp:
            case=p.write_case(p.initial_design(),Path(tmp)/'case')
            manifest=json.loads((case/'case.json').read_text())
            post=case/'postProcessing/p006';post.mkdir(parents=True)
            (post/'status.json').write_text(json.dumps(dict(converged=True,iteration=5,
                cells=manifest['cells'],processes=1)))
            (case/'log.flowfieldP006Foam').write_text('CONVERGED: coupled fields and local/global balances passed.\nEnd\n')
            row=dict(iteration=1,charge_residual=0,field_change=0,bounded_cells=0,
                liquid_balance=0,gas_balance=0,local_liquid_balance=0,local_gas_balance=0,
                energy_balance=0,applied_voltage_V=1.4,liquid_momentum_residual=0,
                gas_momentum_residual=0)
            # Small field increments do not prove that the steady momentum
            # equations are satisfied; nor does a converged warm-up voltage.
            for changes in ({},dict(applied_voltage_V=2.2,liquid_momentum_residual=.1),
                            dict(applied_voltage_V=2.2,gas_momentum_residual=.1)):
                with self.subTest(changes=changes):
                    with (post/'history.csv').open('w') as stream:
                        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                        for i in range(1,6):writer.writerow(dict(row,**changes,iteration=i))
                    with self.assertRaisesRegex(ValueError,'convergence or conservation'):
                        extract_metrics(case,p.config)


if __name__=='__main__':unittest.main()
