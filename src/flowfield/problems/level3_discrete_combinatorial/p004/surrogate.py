"""Reproducible NumPy MLP workflow; architecture family from Fig.2(d), explicit new hyperparameters.

No pretrained author weights/data are bundled. This module does not run CFD.
"""
from dataclasses import dataclass,asdict
import hashlib,json,math
from pathlib import Path
from .dataset import CurrentDataset
from .library import ml_inputs,catalog

@dataclass(frozen=True)
class TrainingConfig:
    hidden1:int=128
    hidden2:int=64
    epochs:int=300
    batch_size:int=32
    learning_rate:float=.001
    dropout:float=.1
    seed:int=42
    def __post_init__(self):
        for k in ('hidden1','hidden2','epochs','batch_size','seed'):
            v=getattr(self,k)
            if type(v) is not int or v<(0 if k=='seed' else 1):raise ValueError(f'Invalid {k}')
        if self.batch_size<2:raise ValueError('Batch normalization requires batch size >=2')
        if isinstance(self.learning_rate,bool) or not math.isfinite(self.learning_rate) or self.learning_rate<=0:raise ValueError('Invalid learning rate')
        if isinstance(self.dropout,bool) or not math.isfinite(self.dropout) or not 0<=self.dropout<1:raise ValueError('Invalid dropout')


class MLP:
    def __init__(self,config):
        import numpy as np
        self.config=config;self.rng=np.random.default_rng(config.seed);self.params={};self.stats={}
        sizes=[98,config.hidden1,config.hidden2,1]
        for i in range(3):
            self.params[f'W{i}']=self.rng.normal(0,math.sqrt(2/sizes[i]),(sizes[i],sizes[i+1]))
            self.params[f'b{i}']=np.zeros(sizes[i+1])
            if i<2:self.params[f'g{i}']=np.ones(sizes[i+1]);self.params[f't{i}']=np.zeros(sizes[i+1])
    def forward(self,x,training=False,calibrate=False):
        import numpy as np
        a=x;cache=[]
        for i in range(2):
            z=a@self.params[f'W{i}']+self.params[f'b{i}']
            if training or calibrate:
                mean=z.mean(axis=0);var=z.var(axis=0)
                if calibrate:self.stats[f'm{i}']=mean;self.stats[f'v{i}']=var
            else:mean=self.stats[f'm{i}'];var=self.stats[f'v{i}']
            inv=1/np.sqrt(var+1e-5);normalized=(z-mean)*inv
            h=normalized*self.params[f'g{i}']+self.params[f't{i}']
            mask=(self.rng.random(h.shape)>=self.config.dropout)/(1-self.config.dropout) if training and self.config.dropout else np.ones_like(h)
            cache.append((a,normalized,inv,h,mask));a=np.maximum(h,0)*mask
        y=a@self.params['W2']+self.params['b2'];return y[:,0],(cache,a)
    def loss_grad(self,x,y):
        import numpy as np
        pred,(cache,a)=self.forward(x,training=True);d=(2/len(x)*(pred-y))[:,None]
        grads={'W2':a.T@d,'b2':d.sum(axis=0)};da=d@self.params['W2'].T
        for i in (1,0):
            old,norm,inv,h,mask=cache[i];dh=da*mask*(h>0)
            grads[f'g{i}']=(dh*norm).sum(axis=0);grads[f't{i}']=dh.sum(axis=0)
            dn=dh*self.params[f'g{i}'];dz=inv*(dn-dn.mean(axis=0)-norm*(dn*norm).mean(axis=0))
            grads[f'W{i}']=old.T@dz;grads[f'b{i}']=dz.sum(axis=0);da=dz@self.params[f'W{i}'].T
        return float(np.mean((pred-y)**2)),grads


def train(dataset_path,output,config=None):
    import numpy as np
    config=config or TrainingConfig();dataset=CurrentDataset(dataset_path);ids,x,y=dataset.arrays()
    if len(ids)<10:raise ValueError('Need at least 10 labelled geometries for a separate 80/20 split')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(config.seed);order=rng.permutation(len(ids));n=int(.8*len(ids));tr,te=order[:n],order[n:]
    mean=float(y[tr].mean());scale=float(y[tr].std())
    if scale<=0:raise ValueError('Training current labels must vary')
    x=(x-4)/3;target=(y-mean)/scale;model=MLP(config)
    first={k:np.zeros_like(v) for k,v in model.params.items()};second={k:np.zeros_like(v) for k,v in model.params.items()};step=0;history=[]
    for epoch in range(config.epochs):
        shuffled=rng.permutation(tr)
        batches=list(np.array_split(shuffled,max(1,min(math.ceil(len(shuffled)/config.batch_size),len(shuffled)//2))))
        losses=[]
        for batch in batches:
            loss,grad=model.loss_grad(x[batch],target[batch]);losses.append(loss);step+=1
            for k in model.params:
                first[k]=.9*first[k]+.1*grad[k];second[k]=.999*second[k]+.001*grad[k]**2
                model.params[k]-=config.learning_rate*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
            if not math.isfinite(loss) or not all(np.isfinite(v).all() for v in model.params.values()):raise ValueError('Nonfinite MLP training')
        history.append(dict(epoch=epoch+1,normalized_train_batch_mse=float(np.mean(losses))))
    # Final inference BN moments use training samples only, without dropout.
    model.forward(x[tr],calibrate=True)
    def metrics(which):
        pred=model.forward(x[which])[0]*scale+mean;err=pred-y[which]
        return dict(n=len(which),mae_A_m2=float(np.mean(np.abs(err))),mse_A2_m4=float(np.mean(err**2)),
            mape_percent=float(100*np.mean(np.abs(err)/y[which])),minimum_prediction_A_m2=float(pred.min()))
    metadata=dict(schema_version=1,model_kind='P004_numpy_MLP',model_revision='numpy-mlp-bn-v1',numpy_version=np.__version__,architecture=[98,config.hidden1,config.hidden2,1],
        normalization='coordinates (x-4)/3; target mean/std fit on training split; BN moments from final training activations',
        dataset_sha256=dataset.sha256,source=dataset.source,voltage_v=.475,quantity='mean_current_density_A_m2',
        train_ids=[ids[i] for i in tr],test_ids=[ids[i] for i in te],training_config=asdict(config),
        train_metrics=metrics(tr),test_metrics=metrics(te),target_mean=mean,target_scale=scale,
        author_weights_reproduced=False,executed_cfd=False,hyperparameters_status='explicit reconstruction choices; author settings unavailable')
    np.savez_compressed(output/'model.npz',metadata=np.array(json.dumps(metadata)),**model.params,**model.stats)
    (output/'training.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    (output/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
    return metadata


class Predictor:
    def __init__(self,path):
        import numpy as np
        path=Path(path);self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']))
            if metadata.get('schema_version')!=1 or metadata.get('model_kind')!='P004_numpy_MLP' or metadata.get('voltage_v')!=.475 or metadata.get('quantity')!='mean_current_density_A_m2':
                raise ValueError('Not a compatible P004 current-density model')
            self.model=MLP(TrainingConfig(**metadata['training_config']))
            for k,expected in self.model.params.items():
                value=archive[k]
                if value.shape!=expected.shape or not np.isfinite(value).all():raise ValueError('Invalid model weights')
                self.model.params[k]=value.copy()
            for i,size in enumerate([self.model.config.hidden1,self.model.config.hidden2]):
                for prefix in ['m','v']:
                    value=archive[f'{prefix}{i}']
                    if value.shape!=(size,) or not np.isfinite(value).all() or prefix=='v' and (value<0).any():raise ValueError('Invalid BN statistics')
                    self.model.stats[f'{prefix}{i}']=value.copy()
        if not all(math.isfinite(metadata[k]) for k in ['target_mean','target_scale']) or metadata['target_scale']<=0:raise ValueError('Invalid target scaling')
        self.metadata=metadata
    def predict(self,paths):
        import numpy as np
        x=(np.asarray([ml_inputs(p) for p in paths],dtype=float)-4)/3
        y=self.model.forward(x)[0]*self.metadata['target_scale']+self.metadata['target_mean']
        if not np.isfinite(y).all() or (y<=0).any():raise ValueError('Surrogate predicted nonpositive/nonfinite current; no clipping applied')
        return y


def rank(model_path,output):
    import csv
    model=Predictor(model_path);paths=catalog();pred=[]
    for start in range(0,len(paths),512):pred.extend(float(v) for v in model.predict(paths[start:start+512]))
    order=sorted(range(len(paths)),key=lambda i:(-pred[i],i));output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    with (output/'predictions.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['rank','library_id','predicted_current_density_A_m2'])
        writer.writerows((r+1,i,pred[i]) for r,i in enumerate(order))
    summary=dict(evaluation_kind='trained_surrogate_prediction',model_sha256=model.sha256,dataset_sha256=model.metadata['dataset_sha256'],
        designs=len(paths),top15_ids=order[:15],bottom10_ids=order[-10:],paper_validated=False,executed_cfd=False)
    (output/'ranking.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
