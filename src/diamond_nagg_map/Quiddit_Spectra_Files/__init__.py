# SPDX-FileCopyrightText: 2024-present Henry Towbin <24940778+whtowbin@users.noreply.github.com>
#
# SPDX-License-Identifier: MIT
from .Spectrum_obj import Spectrum

# Spectra are imported from the Quiddit https://github.com/LauraSp/QUIDDIT3?tab=readme-ov-file
"""
L. Speich, S.C. Kohn,
QUIDDIT - QUantification of infrared active Defects in Diamond and Inferred Temperatures,
Computers & Geosciences,
Volume 144,
2020,
104558,
ISSN 0098-3004,
https://doi.org/10.1016/j.cageo.2020.104558.
(https://www.sciencedirect.com/science/article/pii/S0098300420305483)
Abstract: QUIDDIT is a free Python software-package designed to process Fourier Transform Infrared (FTIR) spectra of diamonds automatically and efficiently. Core capabilities include baseline correction, determination of nitrogen concentration, nitrogen aggregation state and model temperature and fitting of both the 3107 cm-1 and platelet (B’) peaks. These capabilities have allowed the authors to study platelet defects and their relationship to nitrogen aggregation in previous studies. Data visualisation, vital to interpreting and evaluating results, is another key component of the software. QUIDDIT can be applied to single spectra as well as linescan and 2-dimensional map data. Recently, additional features such as manual platelet peak and nitrogen fitting, custom batch peak fitting and two-stage aggregation modelling were made available. QUIDDIT has been used successfully for natural diamonds containing aggregated forms of nitrogen in the past and has since been adapted for the study of diamonds containing C-centres as well.
Keywords: Diamond; FTIR; Spectral deconvolution

"""
"""
Y Center Spectrum Component Spectrum provided my Maxwell C Day 
(Dipartimento di Geoscienze, Università degli Studi di Padova, Via Gradenigo 6, I-35131 Padova, Italy)

Y-Center First Reported in:
Thomas Hainschwang, Emmanuel Fritsch, Franck Notari, Benjamin Rondeau,
A new defect center in type Ib diamond inducing one phonon infrared absorption: The Y center,
Diamond and Related Materials,
Volume 21,
2012,
Pages 120-126,
ISSN 0925-9635,
https://doi.org/10.1016/j.diamond.2011.11.002.
(https://www.sciencedirect.com/science/article/pii/S0925963511003578)
Abstract: The infrared spectra of 68 natural and synthetic diamonds with detectable C centers at about 1130cm−1 and 1344cm−1 were recorded. After correction and normalization of the spectra there was an attempt to determine the A, B and C center nitrogen content of the samples using the spectral fitting spreadsheet supplied by David Fisher from the DTC. It became clear that for the majority of our samples the results were erroneous for the C center content. In fact, the calculated C center content based on the 1130cm−1 absorption was significantly higher than what the 1344cm−1 absorption height indicated; in the fitted traces proposed by the spreadsheet the 1344cm−1 absorption was always at least 50% higher than in the original spectrum. The overestimated C center content and the residual trace after spectral fitting indicated that some absorption bands were underlying the well-known, single-substitutional nitrogen absorption at 1130cm−1. For this reason all spectra were decomposed again by progressive subtraction of the A, B, C and X center absorptions, in order to visualize any residual absorption not attributed to any of those four, classically-recognized, nitrogen-related one-phonon absorption systems. In the spectra of 38 diamonds the decomposition work resulted in a consistent residual absorption feature with a relatively broad apparent maximum centered at about 1145 to 1150cm−1. For the sake of convenience we use “1145cm−1” to describe this feature. Two additional samples did not exhibit any 1130cm−1 absorption but instead precisely the same one-phonon absorption features as the newly-found residual. Consequently several hundred nominally type Ib diamonds were screened by infrared spectroscopy, and 22 had a one phonon absorption with a dominant 1145cm−1 absorption. This confirms that this absorption system was not an artifact of the original spectral decomposition work. Further, this new system is apparently a characteristic component of many natural type Ib diamonds. Based on the fact that this newly described center is clearly related to single substitutional nitrogen and that the last such one-phonon IR absorption had been named “X center” [1] (positively charged single nitrogen), we propose to call this defect “Y center”.
Keywords: Natural diamond; Absorption; Defect characterization; Impurity characterization
"""