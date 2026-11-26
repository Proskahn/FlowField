"""Keep verified optimizer state in memory until execution continues from it."""

from dataclasses import dataclass

from .state import ResumeError


@dataclass
class VerifiedReplay:
    proposals: object
    state: dict
    history: list
    saved_experiment: dict | None
    current_experiment: dict | None

    def close(self):
        self.proposals.close()

    def validate(self, saved, current, history):
        # The per-run lock can be released between suite validation and execution.
        # Never reuse a sampler if another writer has changed its inputs meanwhile.
        if (saved != self.saved_experiment or current != self.current_experiment
                or history != self.history):
            raise ResumeError('Run changed after history verification; resume again. No new CFD was started.')


def verify(proposals, state, history, name, *, saved=None, current=None, progress=None):
    """Validate each proposal once, leaving the generator at the last saved yield.

    The final saved observation is sent only when execution continues. This
    preserves partially evaluated populations and avoids asking for an expensive
    new BO/TuRBO candidate during a dry run.
    """
    def report(index, stage):
        if progress is not None:
            progress(name, index, len(history), stage)

    try:
        report(0, 'verification_started')
        for index, row in enumerate(history):
            report(index+1, 'verifying')
            request = next(proposals) if index == 0 else proposals.send(history[index-1]['result'])
            x, phase, generation, proposal = request
            if (row['x'] != x or row['phase'] != phase or row['generation'] != generation
                    or row.get('proposal') != proposal):
                raise ResumeError(f'Cannot reproduce saved proposal {index+1}; '
                                  'check original code, dependencies and seed. No new CFD was started.')
            report(index+1, 'verified')
        report(len(history), 'verification_finished')
        return VerifiedReplay(proposals, state, history, saved, current)
    except BaseException as exc:
        proposals.close()
        if isinstance(exc, StopIteration):
            raise ResumeError('Saved history extends past optimizer termination') from exc
        raise
