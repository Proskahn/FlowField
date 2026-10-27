"""Independent geometry invariants and checks of real solver outputs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('generate',ROOT/'scripts/generate_cases.py')
gen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
CONFIG=json.loads((ROOT/'config/reference.json').read_text())


class GeometryTests(unittest.TestCase):
    def check_geometry(self, kind):
        m=gen.geometry(CONFIG['geometry'],kind,'coarse')
        volumes=[0.0]*9
        for (i,j,k), zone in zip(m['cells'],m['ids']):
            volumes[zone]+=(m['x'][i+1]-m['x'][i])*(m['y'][j+1]-m['y'][j])*(m['z'][k+1]-m['z'][k])*1e-9
        area=100e-6 if kind=='single' else 121e-6
        self.assertAlmostEqual(m['area'],area,places=14)
        for zone in (3,5):
            self.assertAlmostEqual(volumes[zone],area*20e-6,places=16)
        for zone in (2,6):
            self.assertAlmostEqual(volumes[zone],area*300e-6,places=16)
        self.assertAlmostEqual(volumes[4],area*178e-6,places=16)
        gas_volume=50e-9 if kind=='single' else 60e-9
        for zone in (1,7):
            self.assertAlmostEqual(volumes[zone],gas_volume,places=16)
        self.assertEqual(set(m['ids']),set(range(9)))
        with tempfile.TemporaryDirectory() as tmp:
            gen.mesh(Path(tmp),m)
            self.assertEqual(set(m['inlet_areas']),{'anodeInlet','cathodeInlet'})
            for value in m['inlet_areas'].values():
                self.assertAlmostEqual(value,1e-6,places=14)
        # Exactly two connected fluid domains: no path through the membrane.
        nx,ny,nz=(len(m[k])-1 for k in ('x','y','z'))
        fluid={c for c,z in enumerate(m['ids']) if z in gen.FLUID}
        components=0
        while fluid:
            components+=1
            pending=[fluid.pop()]
            while pending:
                c=pending.pop()
                i=c%nx; j=(c//nx)%ny; k=c//(nx*ny)
                for di,dj,dk in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
                    ii,jj,kk=i+di,j+dj,k+dk
                    if not (0<=ii<nx and 0<=jj<ny and 0<=kk<nz):
                        continue
                    n=(kk*ny+jj)*nx+ii
                    if n in fluid:
                        fluid.remove(n); pending.append(n)
        self.assertEqual(components,2)

    def test_single(self):
        self.check_geometry('single')

    def test_parallel(self):
        self.check_geometry('parallel')

    def test_refinement_preserves_dimensions(self):
        for kind in ('single','parallel'):
            meshes=[gen.geometry(CONFIG['geometry'],kind,level) for level in ('coarse','medium','fine')]
            self.assertTrue(len(meshes[0]['cells'])<len(meshes[1]['cells'])<len(meshes[2]['cells']))
            self.assertEqual({(m['x'][-1],m['y'][-1],m['z'][-1]) for m in meshes},
                             {(meshes[0]['x'][-1],meshes[0]['y'][-1],meshes[0]['z'][-1])})

    def test_impossible_channel_rejected(self):
        g=dict(CONFIG['geometry'],channel_height_mm=2)
        with self.assertRaises(ValueError):
            gen.geometry(g,'single','coarse')

    def test_invalid_physical_input_rejected(self):
        config=json.loads(json.dumps(CONFIG))
        gen.validate_config(config)
        config['model']['porosityCL']=1.2
        with self.assertRaises(ValueError):
            gen.validate_config(config)


if __name__=='__main__':
    unittest.main()
