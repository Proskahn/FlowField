import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from flowfield.problem import FlowFieldProblem
from flowfield.optimization import optimize
from flowfield.optimization.state import ResumeError


class ConstrainedToy(FlowFieldProblem):
    def __init__(self):
        super().__init__('CobylaToy',2,[0,0],[1,1]);self.calls=[]
    def initial_design(self):return [.3,.3]
    def decode(self,x):return self.validate_design(x)
    def geometric_constraints(self,x):return dict(sum=sum(x)-1)
    def evaluate_raw(self,x):
        x=self.decode(x);self.calls.append(x)
        f=(x[0]-.8)**2+(x[1]-.6)**2
        return dict(status='ok',objective=f,objectives={'value':f},constraints=self.geometric_constraints(x),metadata={})


@unittest.skipUnless(importlib.util.find_spec('scipy'),'COBYLA requires scipy')
class COBYLATests(unittest.TestCase):
    def test_real_constrained_optimizer_respects_budget_and_improves(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=ConstrainedToy()
            optimize(p,optimizer='cobyla',budget=30,seed=42,output=Path(tmp)/'run',with_ioh=False)
            self.assertLessEqual(len(p.calls),30)
            self.assertGreater(len(p.calls),4)
            self.assertEqual(len(p.calls),len({tuple(x) for x in p.calls}))
            feasible=[x for x in p.calls if sum(x)<=1+1e-8]
            self.assertLess(min((x[0]-.8)**2+(x[1]-.6)**2 for x in feasible),.1)

    def test_interruption_resume_does_not_repeat_cfd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';first=ConstrainedToy()
            def interrupt(name,index,budget,status):
                if index==3 and status=='ok':raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                optimize(first,optimizer='cobyla',budget=8,seed=42,output=root,with_ioh=False,progress=interrupt)
            resumed=ConstrainedToy()
            manifest=root/'experiment.json';original=manifest.read_text()
            altered=json.loads(original);altered['versions']['scipy']='0.0.0'
            manifest.write_text(json.dumps(altered))
            with self.assertRaises(ResumeError):
                optimize(resumed,optimizer='cobyla',budget=8,seed=42,output=root,with_ioh=False,resume=True)
            self.assertEqual(resumed.calls,[])
            manifest.write_text(original)
            optimize(resumed,optimizer='cobyla',budget=8,seed=42,output=root,with_ioh=False,resume=True)
            whole=ConstrainedToy()
            optimize(whole,optimizer='cobyla',budget=8,seed=42,output=Path(tmp)/'whole',with_ioh=False)
            self.assertEqual(first.calls+resumed.calls,whole.calls)


if __name__=='__main__':unittest.main()
