# Data attribution and notices

This repository contains derived research tables, not original global source archives. Upstream data rights and citation requirements remain with their providers; this notice does not relicense those data or imply provider endorsement. The repository owner has not selected a blanket licence for the original project code.

## NOAA IBTrACS v04r01

Source: [NOAA NCEI IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive). Dataset citation: Gahtan, J., Knapp, K. R., Schreck, C. J. III, Diamond, H. J., Kossin, J. P. and Kruk, M. C. (2024), *International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4.01*, North Indian Ocean subset. DOI: [10.25921/82ty-9e16](https://doi.org/10.25921/82ty-9e16).

Method reference: Knapp et al. (2010), *The International Best Track Archive for Climate Stewardship (IBTrACS): Unifying tropical cyclone best track data*. DOI: [10.1175/2009BAMS2755.1](https://doi.org/10.1175/2009BAMS2755.1).

Project modifications include main-track/synoptic filtering, six-hour pairing, unit standardization, Bay/NIO selection and derived tendencies. Agency reporting conventions and incomplete historical intensity labels remain limitations.

## NOAA OISST v2.1

Source: [NOAA NCEI Daily OISST](https://www.ncei.noaa.gov/products/optimum-interpolation-sst). Product reference: Huang et al. (2021), *Improvements of the Daily Optimum Interpolation Sea Surface Temperature (DOISST) Version 2.1*. DOI: [10.1175/JCLI-D-20-0166.1](https://doi.org/10.1175/JCLI-D-20-0166.1).

Project modifications include regional cropping, storm-linked feature extraction and explicit SST availability masks. Arabian Sea SST values outside the supplied crop remain unavailable; they have not been fabricated or filled in the data archive.

## ECMWF ERA5 / Copernicus Climate Change Service

Contains modified Copernicus Climate Change Service information 2026.

ERA5 variables are based on ECMWF reanalysis and were accessed through the public Earthmover Icechunk archive, as documented by the data handoff. © European Centre for Medium-Range Weather Forecasts (ECMWF), 2026. The CDS catalogue identifies both source products under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): [single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=documentation), DOI [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47), and [pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=documentation), DOI [10.24381/cds.bd0915c6](https://doi.org/10.24381/cds.bd0915c6).

Project modifications include sampling at cyclone centres, unit conversions, shear/deep-layer proxy calculations and aligned feature/target packaging. The European Commission and ECMWF are not responsible for this repository's interpretation or use of the data. [ECMWF attribution guidance](https://confluence.ecmwf.int/pages/viewpage.action?pageId=673339011).

The source information and catalogue terms above were checked on 28 October 2026. The original acquisition timestamps and detailed construction provenance should be recovered from the data-preparation team's source notebooks; they are not reconstructed here.
