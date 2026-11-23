"""Reveal P002 channel volumes in ParaView, without changing the mesh.

Run with ParaView's pvpython, or paste this into its Python Shell:

    import runpy
    runpy.run_path('/Users/zhuang/FlowField/scripts/view_p002_channels.py',
                   run_name='__main__')

Set SIDE to 'anode', 'cathode', or 'both'. The output directory contains
a PNG preview and a .pvsm file that can be opened with File > Load State.
"""

from pathlib import Path

from paraview import simple as pv


CASE_FILE = (
    Path(__file__).resolve().parents[1]
    / 'runs/p002/paraview-baseline/paraview-baseline.foam'
)
SIDE = 'both'


def show_channels(case_file=CASE_FILE, side=SIDE):
    """Show material 7 (anode) and/or 1 (cathode), using cell values."""
    if side not in ('anode', 'cathode', 'both'):
        raise ValueError("side must be 'anode', 'cathode', or 'both'")
    case_file = Path(case_file).resolve()
    if not case_file.is_file():
        raise FileNotFoundError(case_file)

    reader = pv.OpenFOAMReader(
        registrationName='P002 full cell (hidden)', FileName=str(case_file)
    )
    # This generated case only has time 0, which ParaView normally skips.
    reader.SkipZeroTime = 0
    reader.MeshRegions = ['internalMesh']
    reader.Createcelltopointfiltereddata = 0
    # Refresh the cached file list after re-enabling time 0. In ParaView 6.1
    # UpdatePipelineInformation alone can leave CellArrays empty.
    reader.Refresh()
    reader.UpdatePipelineInformation()
    if 'material' not in reader.CellArrays.Available:
        raise ValueError(f'No material cell field found in {case_file.parent}/0')
    reader.CellArrays = ['material']
    reader.UpdatePipeline(time=0)

    view = pv.GetActiveViewOrCreate('RenderView')
    # Hide any previously displayed box, retaining its pipeline objects.
    for representation in view.Representations:
        representation.Visibility = 0
    view.ViewTime = 0
    view.ViewSize = [1400, 1000]
    view.BackgroundColorMode = 'Single Color'
    view.Background = [0.96, 0.97, 0.99]
    view.CameraParallelProjection = 1

    channels = {}
    for name, material_id, color in (
        ('anode', 7, [0.95, 0.40, 0.10]),
        ('cathode', 1, [0.10, 0.42, 0.80]),
    ):
        channel = pv.Threshold(
            registrationName=f'P002 {name} channels (material {material_id})',
            Input=reader,
        )
        channel.Scalars = ['CELLS', 'material']
        channel.ThresholdMethod = 'Between'
        channel.LowerThreshold = material_id
        channel.UpperThreshold = material_id
        channel.UpdatePipeline(time=0)
        count = channel.GetDataInformation().GetNumberOfCells()
        if count == 0:
            raise ValueError(f'No {name} cells found for material {material_id}')
        print(f'{name}: {count} channel cells')

        display = pv.Show(channel, view)
        display.ColorArrayName = ['CELLS', '']
        display.Representation = 'Surface'
        display.DiffuseColor = color
        display.AmbientColor = color
        display.Visibility = int(side in (name, 'both'))
        channels[name] = channel

    bounds = reader.GetDataInformation().GetBounds()
    center = [(bounds[2*i] + bounds[2*i+1])/2 for i in range(3)]
    span = max(bounds[2*i+1] - bounds[2*i] for i in range(3))
    view.CameraFocalPoint = center
    view.CameraPosition = [
        center[0] + 1.6*span,
        center[1] - 2.2*span,
        center[2] + 1.8*span,
    ]
    view.CameraViewUp = [0, 0, 1]
    pv.ResetCamera(view)
    pv.SetActiveSource(channels['cathode' if side == 'cathode' else 'anode'])
    pv.Render(view)

    output_dir = case_file.parent / 'visualization'
    output_dir.mkdir(exist_ok=True)
    state_file = output_dir / f'channels-{side}.pvsm'
    image_file = output_dir / f'channels-{side}.png'
    pv.SaveScreenshot(str(image_file), view, ImageResolution=[1400, 1000])
    pv.SaveState(str(state_file))
    print(f'Orange = anode; blue = cathode. Showing: {side}.')
    print(f'Saved {state_file}')
    print(f'Saved {image_file}')
    return reader, channels, view


if __name__ == '__main__':
    show_channels()
