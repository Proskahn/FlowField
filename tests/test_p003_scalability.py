"""Scalable control fields, unchanged geometry, exact derivatives and CLI use."""

import contextlib
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from flowfield import P003Problem, TopologyConfig
from flowfield.mesh import PolyMesh
from flowfield.problems.level4_high_dimensional_topology.p003.case import generate_case
from flowfield.problems.level4_high_dimensional_topology.p003.config import SCALABLE_MAPPING
from flowfield.problems.level4_high_dimensional_topology.p003.design import decode, grid, stencil
from flowfield.problems.level4_high_dimensional_topology.p003.scalability import scalable_config, mapping_report


class P003ScalabilityTests(unittest.TestCase):
    def test_legacy_config_identity_and_geometry_are_unchanged(self):
        c=TopologyConfig()
        x=[.01+.008*((i%23)/22) for i in range(c.dimension)]
        self.assertNotIn('design_mapping',c.to_dict())
        self.assertEqual(c.fingerprint,'98a7221a4e37214dad9f32a06c3485bb58fa677c2b79c51151182d0886431695')
        digest=hashlib.sha256(json.dumps(decode(x,c),sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest,'cc04723580a037be4cc2e0da74f4f9074181ff6d93c8ba19c3678bfd90668e45')
        self.assertEqual(TopologyConfig(**c.to_dict()),c)

    def test_exact_dimensions_preserve_mesh_ports_and_physical_filter(self):
        reference=None
        for dimension in (500,1000,2000,3000,4000,5000,6000):
            with self.subTest(dimension=dimension):
                c=scalable_config(dimension=dimension)
                report=mapping_report(c)
                self.assertEqual(c.dimension,dimension)
                self.assertEqual(report['cfd_cells'],8320)
                self.assertEqual(report['unused_variables'],0)
                self.assertEqual(report['design_mapping'],SCALABLE_MAPPING)
                self.assertEqual(c.filter_radius_m,.00035)
                self.assertEqual(c.projection_beta,50.)
                self.assertEqual(c.iterations,10000)
                self.assertEqual(c.outer_tolerance,1e-7)
                self.assertEqual(c.balance_tolerance,1e-5)
                self.assertEqual(c.to_dict()['design_mapping'],SCALABLE_MAPPING)
                self.assertEqual(TopologyConfig(**c.to_dict()),c)
                mesh=grid(c)
                if reference is not None:self.assertEqual(mesh,reference)
                reference=mesh
                self.assertIn(c.inlet_width_m,mesh[0])
                self.assertIn(c.width_m-c.outlet_width_m,mesh[0])

    def test_invalid_or_unrepresentable_grids_are_rejected(self):
        for options in ({},{'dimension':500,'shape':(25,20)}, {'dimension':503},
                        {'dimension':True},{'dimension':6000.}, {'dimension':9000},
                        {'shape':(50,120)}, {'shape':(2,250)}, {'shape':(25,True)},
                        {'shape':(25,20,1)}, {'dimension':6000,'mesh':'coarse'}):
            with self.subTest(options=options),self.assertRaises(ValueError):scalable_config(**options)
        self.assertEqual(scalable_config(dimension=500,mesh='coarse').dimension,500)
        self.assertEqual(scalable_config(shape=(32,16)).dimension,512)
        with self.assertRaises(ValueError):TopologyConfig(design_mapping='unknown')
        with self.assertRaisesRegex(ValueError,'coarse for the filter'):
            TopologyConfig(mesh='medium',design_nx=19,design_ny=24)

    def test_composed_mapping_matches_area_weighted_filter_of_bilinear_field(self):
        c=replace(scalable_config(dimension=500),inlet_width_m=.0014,outlet_width_m=.0013)
        def field(x,y):return .1+.2*x+.25*y+.1*x*y
        nodes=[field(i/c.design_nx,j/c.design_ny)
               for j in range(c.design_ny+1) for i in range(c.design_nx+1)]
        geometry=decode(nodes,c)
        x,y=grid(c)
        centers=[((a+b)/2,(d+e)/2) for d,e in zip(y,y[1:]) for a,b in zip(x,x[1:])]
        areas=geometry['cell_areas_m2']
        for k in (0,6,len(x)-2,len(centers)//2,len(centers)-1):
            cx,cy=centers[k]
            weights=[area*max(0,1-math.hypot(xx-cx,yy-cy)/c.filter_radius_m)
                     for (xx,yy),area in zip(centers,areas)]
            reference=sum(w*field(xx/c.width_m,yy/c.length_m)
                          for w,(xx,yy) in zip(weights,centers))/sum(weights)
            self.assertAlmostEqual(geometry['filtered'][k],reference,14)

    def test_constant_preservation_bounds_and_volume_gradient_at_both_endpoints(self):
        for n in (500,6000):
            c=scalable_config(dimension=n);problem=P003Problem(config=c)
            rows,_=stencil(c)
            self.assertTrue(all(abs(sum(w for _,w in row)-1)<1e-14 for row in rows))
            self.assertTrue(all(w>0 for row in rows for _,w in row))
            for name,value in (('all_rib',0.),('all_channel',1.),('uniform',.5)):
                geometry=problem.decode(problem.preset(name))
                self.assertAlmostEqual(geometry['volume_fraction'],value,12)
            x=[.01+.004*math.sin(i*.13) for i in range(n)]
            direction=[math.cos(i*.17) for i in range(n)];h=1e-6
            numeric=(problem.decode([v+h*d for v,d in zip(x,direction)])['volume_fraction']-
                     problem.decode([v-h*d for v,d in zip(x,direction)])['volume_fraction'])/(2*h)
            analytic=sum(g*d for g,d in zip(problem.volume_gradient(x),direction))
            self.assertAlmostEqual(analytic,numeric,8)

    @unittest.skipUnless(importlib.util.find_spec('scipy'),'optional sparse repair dependencies unavailable')
    def test_repair_enforces_volume_at_both_dimension_endpoints(self):
        for n in (500,6000):
            problem=P003Problem(config=scalable_config(dimension=n))
            feasible=[.9*v for v in problem.initial_design()]
            designs=problem.repair_candidates([feasible,[.5+.4*math.sin(i)**2 for i in range(n)]])
            self.assertEqual(designs[0].tolist(),feasible)
            for x in designs:self.assertLessEqual(problem.decode(x)['volume_fraction'],.5+1e-12)

    def test_mesh_files_identical_for_different_control_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            hashes=[]
            for n in (500,6000):
                c=scalable_config(dimension=n);problem=P003Problem(config=c)
                case=generate_case(Path(tmp)/str(n),problem.decode(problem.initial_design()),c)
                hashes.append(PolyMesh.from_case(case).checksums())
                manifest=json.loads((case/'case.json').read_text())
                self.assertEqual(manifest['dimension'],n)
                self.assertEqual(manifest['cells'],8320)
                self.assertEqual(manifest['config']['design_mapping'],SCALABLE_MAPPING)
                self.assertFalse((case/'.execution-started').exists())
            self.assertEqual(hashes[0],hashes[1])

    def test_config_cli_and_both_optimizer_dry_runs(self):
        from flowfield.problems.level4_high_dimensional_topology.p003.cli import main
        from flowfield.optimization.cli import main as optimize_main
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'configs/n500.json'
            args=['scalable-config','--dimension','500','--output',str(config)]
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):self.assertEqual(main([*args,'--dry-run']),0)
            self.assertFalse(config.parent.exists())
            self.assertEqual(json.loads(stream.getvalue())['dimension'],500)
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(args),0)
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(args)
            for optimizer in ('cmaes','turbo'):
                stream=io.StringIO()
                run=Path(tmp)/optimizer
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(optimize_main(['--problem','P003','--optimizer',optimizer,
                        '--config',str(config),'--output',str(run),'--budget','5','--dry-run']),0)
                plan=json.loads(stream.getvalue())['problems'][0]
                self.assertEqual(plan['dimension'],500)
                self.assertFalse(run.exists())


if __name__ == '__main__':unittest.main()
