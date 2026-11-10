"""Exact combinatorial reproduction, honest data boundaries, integer IOH and ML math."""
from dataclasses import replace
import importlib.util,json,math,random
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from flowfield import get_problem,P004Problem,P004HydraulicProblem,HydraulicConfig
from flowfield.experiments import random_search
from flowfield.ioh_adapter import make_ioh_problem,analyzer_logger
from flowfield.problems.level3_discrete_combinatorial.p004.library import *
from flowfield.problems.level3_discrete_combinatorial.p004.geometry import footprint_cells
from flowfield.problems.level3_discrete_combinatorial.p004.case import generate_case
from flowfield.problems.level3_discrete_combinatorial.p004.hydraulics import run_case
from flowfield.problems.level3_discrete_combinatorial.p004.dataset import CurrentDataset,MissingPaperDataError
from flowfield.problems.level3_discrete_combinatorial.p004.cli import presets


def fixture(path,ids):
    # Synthetic fixture only: these values are never saved as paper evidence or bundled labels.
    data=dict(schema_version=1,quantity='mean_current_density_A_m2',voltage_v=.475,
        source=dict(kind='external_cfd',description='SYNTHETIC UNIT TEST FIXTURE, NOT CFD',model_id='unit-test'),
        samples=[dict(library_id=i,current_density_A_m2=10000+features(catalog()[i])['corners']*10) for i in ids])
    path.write_text(json.dumps(data));return data


class P004Tests(unittest.TestCase):
    def test_exact_library_count_unique_paths_and_regeneration(self):
        paths=catalog();self.assertEqual(len(paths),28348);self.assertEqual(tuple(enumerate_paths()),paths)
        self.assertEqual(len(set(paths)),28348)
        for p in paths:self.assertEqual(tuple(validate_path(p)),p)
        # The main paper's 6x6 count cannot use the same opposite-corner Hamiltonian endpoints.
        self.assertEqual(enumerate_paths(6),[])

    def test_four_figure_layouts_map_to_stable_ids(self):
        self.assertEqual({k:path_id(v) for k,v in presets().items()},dict(serpentine=23005,figure4_spiral=27281,figure4_second=27279,figure4_third=27280))
        self.assertEqual(presets()['serpentine'],serpentine())
        for p in presets().values():self.assertEqual(path_from_id(path_id(p)),p)

    def test_integer_domain_and_graph_constraints(self):
        p=get_problem('P004');self.assertEqual(p.variable_type,'integer');self.assertEqual(p.dimension,49)
        for x in ([.5]*49,[True]*49,list(range(49)),serpentine()[:-1],[-1]*49):
            with self.assertRaises(ValueError):p.decode(x)
        with self.assertRaises(ValueError):path_from_id(True)
        with self.assertRaises(ValueError):path_from_id(.5)
        self.assertEqual(len(ml_inputs(serpentine())),98)
        self.assertEqual(ml_inputs(serpentine())[:4],[1,1,1,2])

    def test_source_geometry_area_and_connectivity(self):
        for path in presets().values():
            g=P004Problem().decode(path);self.assertAlmostEqual(g['centerline_length_m'],.098)
            self.assertEqual(g['channel_width_m'],.001)
            for q in (4,8):
                cells=set(footprint_cells(path,q));self.assertEqual(len(cells),98*q*q)
                start=next(iter(cells));seen={start};front=[start]
                while front:
                    x,y=front.pop()
                    for v in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                        if v in cells and v not in seen:seen.add(v);front.append(v)
                self.assertEqual(seen,cells)
                self.assertEqual(sum(y==0 for x,y in cells),q);self.assertEqual(sum(y==14*q-1 for x,y in cells),q)

    def test_open_path_geometric_features_and_p4_conventions(self):
        f=features(serpentine());self.assertEqual(f['corners'],12);self.assertEqual(f['adjacent_length_products_lattice2'],72)
        self.assertEqual(f['straight_lengths_lattice'],[6,1]*6+[6])
        self.assertEqual(f['shortcut_distance_sum_lattice'],336)
        self.assertEqual(f['mean_node_summed_neighbor_distance_lattice'],672/49)
        self.assertEqual(features(presets()['figure4_spiral'])['adjacent_length_products_lattice2'],168)

    def test_missing_current_data_does_not_substitute_hydraulics(self):
        p=P004Problem()
        with self.assertRaises(MissingPaperDataError):p.evaluate_raw(p.initial_design())
        self.assertIsNone(p.last_result)
        self.assertNotEqual(p.ioh_name,P004HydraulicProblem().ioh_name)

    def test_label_lookup_units_provenance_and_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labels.json';fixture(path,[0,4,23005]);p=P004Problem(dataset=path)
            r=p.evaluate_raw(p.design_from_id(0));self.assertEqual(r['objective'],-r['metrics']['mean_current_density_A_m2'])
            self.assertFalse(r['metadata']['executed_cfd']);self.assertIn('tabulated',r['metadata']['evaluation_kind'])
            rng=random.Random(42)
            for _ in range(30):self.assertIn(path_id(p.sample_design(rng)),[0,4,23005])
            with self.assertRaises(MissingPaperDataError):p.evaluate_raw(p.design_from_id(1))
            self.assertIsNone(p.last_result)

    def test_dataset_rejects_wrong_units_duplicates_missing_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labels.json';original=fixture(path,[0,1])
            variants=[]
            for k,v in [('quantity','pressure_drop_Pa'),('voltage_v',.4),('source',{})]:
                data=dict(original);data[k]=v;variants.append(data)
            for value in (None,math.nan,-1,True):
                data=json.loads(json.dumps(original));data['samples'][0]['current_density_A_m2']=value;variants.append(data)
            data=json.loads(json.dumps(original));data['samples'].append(data['samples'][0]);variants.append(data)
            for data in variants:
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):CurrentDataset(path)

    def test_case_generation_and_tamper_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=generate_case(Path(tmp)/'case',serpentine());m=json.loads((case/'case.json').read_text())
            self.assertEqual(m['mesh']['cells'],6272);self.assertAlmostEqual(m['mesh']['expected_volume_m3'],98e-9)
            self.assertFalse(m['paper_current_density_reproduced'])
            with self.assertRaises(FileExistsError):generate_case(case,serpentine())
            with (case/'0/U').open('a') as f:f.write('// altered\n')
            with self.assertRaisesRegex(ValueError,'input changed'):run_case(case)
            self.assertFalse((case/'.execution-started').exists())

    def test_auxiliary_case_contract_uses_one_real_backend_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=P004HydraulicProblem(tmp);raw=dict(status='ok',objective=1.,objectives={'pressure_drop_pa':1},constraints={},case_directory=tmp)
            with patch('flowfield.problems.level3_discrete_combinatorial.p004.hydraulics.evaluate',return_value=raw) as backend:
                self.assertEqual(p(p.initial_design()),1.)
            backend.assert_called_once();self.assertFalse(p.last_result['metadata']['electrochemistry_solved'])
            self.assertEqual(p.last_result['metadata']['objective_unit'],'Pa')
            self.assertNotEqual(p.ioh_name,P004HydraulicProblem(config=HydraulicConfig(mesh='medium')).ioh_name)

    @unittest.skipUnless(importlib.util.find_spec('ioh'),'optional IOH unavailable')
    def test_actual_integer_ioh_and_feasible_random_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labels.json';fixture(path,[0,4,23005]);p=P004Problem(dataset=path);f=make_ioh_problem(p)
            self.assertIn('Integer',type(f).__name__)
            with analyzer_logger(f,Path(tmp)/'ioh',algorithm_name='unit-fixture'):
                f(p.design_from_id(0));f(p.design_from_id(4))
                with self.assertRaises(TypeError):f([.5]*49)
            self.assertEqual(f.state.evaluations,2)
            data=next((Path(tmp)/'ioh').rglob('*.dat')).read_text().splitlines();self.assertEqual(len(data),3);self.assertEqual(len(data[0].split()),51)
            summary=random_search(p,budget=4,output=Path(tmp)/'search',with_ioh=False)
            self.assertEqual(summary['evaluations'],4)
            records=[json.loads(v) for v in (Path(tmp)/'search/evaluations.jsonl').read_text().splitlines()]
            self.assertTrue(all(path_id(v['x']) in [0,4,23005] for v in records))


@unittest.skipUnless(importlib.util.find_spec('numpy'),'optional NumPy unavailable')
class MLPTests(unittest.TestCase):
    def test_batchnorm_backprop_against_finite_difference(self):
        import numpy as np
        from flowfield.problems.level3_discrete_combinatorial.p004.surrogate import MLP,TrainingConfig
        model=MLP(TrainingConfig(hidden1=4,hidden2=3,dropout=0));rng=np.random.default_rng(5);x=rng.normal(size=(7,98));y=rng.normal(size=7)
        _,grads=model.loss_grad(x,y);h=1e-6
        for name,index in [('W0',(2,1)),('g0',(2,)),('t0',(1,)),('W1',(2,0)),('g1',(1,)),('t1',(2,)),('W2',(1,0)),('b2',(0,))]:
            original=model.params[name][index];model.params[name][index]=original+h;plus=model.loss_grad(x,y)[0]
            model.params[name][index]=original-h;minus=model.loss_grad(x,y)[0];model.params[name][index]=original
            self.assertAlmostEqual((plus-minus)/(2*h),grads[name][index],6)

    def test_train_split_scaling_and_model_roundtrip(self):
        import numpy as np
        from flowfield.problems.level3_discrete_combinatorial.p004.surrogate import train,Predictor,TrainingConfig,rank
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.json';fixture(path,list(range(20)));out=Path(tmp)/'model'
            meta=train(path,out,TrainingConfig(hidden1=8,hidden2=4,epochs=10,batch_size=4))
            self.assertEqual(len(meta['train_ids']),16);self.assertEqual(len(meta['test_ids']),4)
            self.assertFalse(set(meta['train_ids'])&set(meta['test_ids']))
            labels=CurrentDataset(path).labels
            self.assertEqual(meta['target_mean'],np.mean([labels[i] for i in meta['train_ids']]))
            model=Predictor(out/'model.npz');p=P004Problem(model=out/'model.npz')
            value=model.predict([catalog()[0]])[0];self.assertAlmostEqual(p(p.design_from_id(0)),-value)
            self.assertIn('surrogate',p.last_result['metadata']['evaluation_kind'])
            self.assertFalse(meta['author_weights_reproduced'])
            ranking=rank(out/'model.npz',Path(tmp)/'ranking')
            self.assertEqual(ranking['designs'],28348);self.assertEqual(len(ranking['top15_ids']),15)
            import csv
            with (Path(tmp)/'ranking/predictions.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len({int(v['library_id']) for v in rows}),28348)
            predicted=[float(v['predicted_current_density_A_m2']) for v in rows]
            self.assertEqual(predicted,sorted(predicted,reverse=True))
            self.assertFalse(ranking['executed_cfd'])
            with self.assertRaises(FileExistsError):train(path,out)

if __name__=='__main__':unittest.main()
