"""Fresh independent CFC/GA/IGA CFD cases, with baseline qualification first."""
from pathlib import Path
from ....evaluation import write_json
from . import P011Problem
from .config import REFERENCE_WIDTHS, normalized
from .reference import reference_summary


def comparison_study(destination, config, runtime):
    root = Path(destination).resolve()
    if root.exists():
        raise FileExistsError('Comparison destination already exists')
    p = P011Problem(config=config, runtime=runtime, output_root=root/'cases')
    p.preflight()
    runtime.resolve(p.case_definition.commands)
    root.mkdir(parents=True)
    observations = []
    for name, widths in REFERENCE_WIDTHS.items():
        result = p.evaluate_raw(normalized(widths))
        if result['status'] != 'ok':
            raise ValueError(f'{name} has no qualified CFD result')
        observation = dict(label=name, design=normalized(widths), metrics=result['metrics'],
                           objective=result['objective'], case_directory=result['case_directory'])
        write_json(root/(name+'.json'), observation)
        observations.append(observation)
    result = dict(config=config.to_dict(), cfd_executed=True, paper_validated=False,
                  observations=observations, reference=reference_summary())
    write_json(root/'comparison.json', result)
    return result
