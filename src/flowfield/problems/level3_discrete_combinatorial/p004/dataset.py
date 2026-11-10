"""Strict externally supplied current-density labels; no fabricated training data."""
import hashlib,json,math
from pathlib import Path
from .library import COUNT,catalog,ml_inputs,path_id,validate_path

class MissingPaperDataError(RuntimeError):pass

class CurrentDataset:
    def __init__(self,path):
        self.path=Path(path).resolve();raw=self.path.read_bytes();self.sha256=hashlib.sha256(raw).hexdigest()
        data=json.loads(raw)
        if data.get('schema_version')!=1 or data.get('quantity')!='mean_current_density_A_m2':
            raise ValueError('Expected current-density dataset schema v1 with explicit A/m² units')
        voltage=data.get('voltage_v')
        if isinstance(voltage,bool) or not isinstance(voltage,(int,float)) or not math.isfinite(voltage) or abs(voltage-.475)>1e-12:
            raise ValueError('P004 labels must be at 0.475 V')
        source=data.get('source',{})
        if source.get('kind') not in ('author_data','external_cfd') or not all(isinstance(source.get(k),str) and source[k].strip() for k in ('description','model_id')):
            raise ValueError('Labels require source kind, description and model_id; geometry/hydraulic scores are not current labels')
        samples=data.get('samples',[])
        if not isinstance(samples,list) or not samples:raise ValueError('Dataset must contain samples')
        labels={}
        for row in samples:
            if 'path' in row:
                identifier=path_id(validate_path(row['path']))
                if 'library_id' in row and row['library_id']!=identifier:raise ValueError('Path/ID mismatch')
            else:identifier=row.get('library_id')
            if type(identifier) is not int or not 0<=identifier<COUNT or identifier in labels:raise ValueError('Invalid or duplicate library ID')
            value=row.get('current_density_A_m2')
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('Current density must be positive and finite, in A/m²')
            labels[identifier]=float(value)
        self.labels=labels;self.source=source;self.voltage_v=voltage
    def value(self,identifier):
        try:return self.labels[identifier]
        except KeyError as exc:raise MissingPaperDataError(f'No current-density label for library ID {identifier}; no CFD or surrogate fallback is implied') from exc
    def arrays(self):
        import numpy as np
        ids=sorted(self.labels)
        return ids,np.asarray([ml_inputs(catalog()[i]) for i in ids],dtype=float),np.asarray([self.labels[i] for i in ids],dtype=float)
