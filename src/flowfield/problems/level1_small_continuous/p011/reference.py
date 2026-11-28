"""Published reference observations; never substituted for CFD objectives."""
from .config import DOI, REFERENCE_WIDTHS, TaperedConfig, normalized, decode


def reference_summary():
    # Fig. 7(f) bar labels, not digitized interpolation of a polarization curve.
    pressure = {'CFC': 14.23, 'GA': 15.88, 'IGA': 16.05}
    power = {'CFC': 480.84, 'GA': 549.16, 'IGA': 572.28}
    return dict(doi=DOI, cfd_executed=False, paper_validated=False,
        source='Table 3 endpoint widths and Figure 7(f), at 0.4 V',
        designs={name: dict(geometry=decode(normalized(widths), TaperedConfig()),
                           pressure_drop_Pa=pressure[name], power_density_mW_cm2=power[name],
                           current_density_A_cm2=power[name]/1000/.4)
                 for name, widths in REFERENCE_WIDTHS.items()},
        caveats=['Rounded published heights replaced by exact reciprocal widths to enforce 1 mm2 ports.',
                 'Unreported inlet reference current prevents exact inlet-flow replication.',
                 'Figure 7 IGA gain is 91.44 mW/cm2; text says 90.42.',
                 'Figure 4 mesh-test current differs from Figure 7 baseline.'])
