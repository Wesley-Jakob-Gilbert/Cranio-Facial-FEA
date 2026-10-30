# Mouse in vivo loading + time-lapse micro-CT data for the remodeling-law test (validation step 2)

Prepared 2026-10-29 by the mouse-data finder. Table form: `mouse_data.csv` (one row per dataset). Every factual claim has a URL. Things I checked by opening the files myself are marked **(opened)**. Everything else comes from a fetched page. Items I could not confirm are marked UNVERIFIED. No emails have been sent.

## Bottom line

1. **Best dataset: ETH Zürich, Zenodo record 17256047.** It is the only dataset I found that is fully open and already contains per-voxel strain energy density (SED), per-voxel Young's modulus and the applied force for every animal and time point.
   - Record: [zenodo.org/records/17256047](https://zenodo.org/records/17256047), DOI [10.5281/zenodo.17256047](https://doi.org/10.5281/zenodo.17256047).
   - Licence is CC BY 4.0, which also covers MorphoGenix as a company. The record is open and holds 23.3 GB, per the [Zenodo record API](https://zenodo.org/api/records/17256047).
   - The paper is [Nature Communications, DOI 10.1038/s41467-026-70309-2](https://pmc.ncbi.nlm.nih.gov/articles/PMC13106859/): 105 evaluated OVX mice, 8 N cyclic load (3,000 cycles, 10 Hz, 3 times per week) on caudal vertebra CV6, 10.5 µm voxels, scans at weeks 15, 20, 22 and 24.
2. **Second choice: Sheffield (Dall'Ara) on ORDA, plus a request for the full database.** The public records hold one mouse each, so they are good for pipeline development only. The full set (tibia, OVX, 12 N load, 10.4 µm) is "contact Dr Dall'Ara" per the [ORDA record](https://orda.shef.ac.uk/articles/dataset/Data_for_the_paper_Bone_remodelling_in_the_mouse_tibia_is_spatio-temporally_modulated_by_oestrogen_deficiency_and_external_mechanical_loading_a_combined_in_vivo_in_silico_study_/12927365). Email 1 below.
3. **Email-only datasets:** the ETH frequency series with static vs cyclic loading (the dataset you named; no repository), and the Columbia (Guo) 9-mouse tibia set (no data statement anywhere).
4. **Nothing relevant found** on Dryad, the Zenodo search, or the ORMIR dataset list. I found no open rat time-lapse loading dataset. Details in "Search coverage" at the end.

## Ranked summary

| Rank | Dataset | What is downloadable now | Fit for the remodeling-law test |
|---|---|---|---|
| 1 | **ETH Zenodo 17256047** (Schulte, Marques, Müller). Mouse CV6 trabecular bone, OVX, 0 N vs 8 N, weeks 20/22/24 | Yes, 23.3 GB, CC BY 4.0. **(opened)** Per animal and time point: segmented binary masks, SED field, Young's modulus, applied force. No grey-level images. | High. The vehicle-injected 0 N and 8 N groups are a clean loaded vs unloaded contrast. Observed formation/resorption comes from differencing consecutive masks, and the SED field is the stimulus the original authors used. |
| 2 | **Sheffield ORDA** (Cheong, Roberts, Kadirkamanathan, Dall'Ara). Mouse tibia, OVX, 12 N, 40 cycles/day, 3 days/week | One mouse per record, with registered DICOM at 4 time points. **(opened)** The full cohort (6+ mice) is on request. | Medium now, high if the full set and the micro-FE strain fields are shared. Their SED-driven in silico remodeling is the closest published analogue to your law. |
| 3 | **ETH frequency series** (Scheuren 2020; Marques 2023). CV6, sham / static / 2, 5, 10 Hz, weekly scans | No. Supplementary only; the repository record lists no datasets. | High for the static vs cyclic and frequency questions, but only after a successful request. Email 2. |
| 4 | **Columbia (Guo)** (Shyu, Fagre-Guerriero). 9 mice, 9 N, 5 weeks, weekly scans | No. The ORS abstract has no data statement. A GitHub repo holds scripts and result spreadsheets only. | Medium. Only the proximal tibia is scanned in vivo at 5 µm and the loading cycles are unstated. Email 3. |
| 5 | Others, request only: [Schulte 2013 PLoS ONE](https://pmc.ncbi.nlm.nih.gov/articles/PMC3634859/) (CV6, 9 N), [Sci Adv 2020 tibia](https://pmc.ncbi.nlm.nih.gov/articles/PMC7060058/) (18 loaded mice, scans out to 12 weeks). | No. | Lower priority. See CSV. |

## Recommendation

**Primary: ETH Zenodo 17256047, vehicle groups only (0 N vs 8 N) for the confirmatory test.**
- It is open and its licence is commercial-safe. It is the only candidate with SED already computed.
- Sample size: [VEH n=25 and ML n=26](https://pmc.ncbi.nlm.nih.gov/articles/PMC13106859/). The drug groups (BIS, PTH, SclAB) can serve as an exploratory extension, because they change the biology, not just the load.
- Caveat 1: it is trabecular bone in a vertebra under axial load, not craniofacial cortical or sutural bone. A pass supports the stimulus-to-formation/resorption mapping. It does not validate the maxillary geometry, loads or material properties.
- Caveat 2: the files are binary masks, not grey-level images. Mineralisation density cannot be tested, only formation vs resorption vs quiescence on the surface.
- Caveat 3: the README's wording on time points is garbled ("weeks 20 (m3), 22 (m3) and 24 (m4)"). The folder names m2/m3/m4 most plausibly map to weeks 20/22/24, but that is my inference. Check the mapping before registering ([README in code.zip](https://zenodo.org/records/17256047)). **(opened)**

**Secondary: Sheffield tibia, via the full-database request.** It gives a second species-site and a different loading regime (cortical plus trabecular, 12 N trapezoidal). Using it as an independent test set after fixing the model on ETH data (or the reverse) gives you a cross-dataset replication.

## Dataset details

### 1. ETH Zürich, Zenodo 17256047

| Item | Value |
|---|---|
| Paper | Nature Communications, DOI [10.1038/s41467-026-70309-2](https://pmc.ncbi.nlm.nih.gov/articles/PMC13106859/). Zenodo title: "Combined physical and pharmacological anabolic osteoporosis therapies increase bone response and mechanoregulation in female mice" ([Zenodo](https://zenodo.org/records/17256047)). |
| Species, site | Female C57BL/6 mice, 6th caudal vertebra (CV6) ([PMC13106859](https://pmc.ncbi.nlm.nih.gov/articles/PMC13106859/)) |
| Loading | 8 N cyclic axial load, 3,000 cycles, 10 Hz, 3 times per week, weeks 20 to 24. Control animals get 0 N after identical pin and anaesthesia procedures ([PMC13106859](https://pmc.ncbi.nlm.nih.gov/articles/PMC13106859/)). |
| Scans, voxel size | Weeks 15, 20, 22, 24; 10.5 µm isotropic ([PMC13106859](https://pmc.ncbi.nlm.nih.gov/articles/PMC13106859/)). The Zenodo layer-0 files cover three time points (m2/m3/m4). |
| Sample size | 120 OVX, 105 evaluated. VEH 25, ML 26, BIS 9, BIS+ML 9, PTH 10, PTH+ML 9, SclAB 9, SclAB+ML 8 ([PMC13106859](https://pmc.ncbi.nlm.nih.gov/articles/PMC13106859/)) |
| Files | `data.zip` 23,306,271,555 bytes and `code.zip` 72,222 bytes ([Zenodo API](https://zenodo.org/api/records/17256047)). The zip holds 580 `.h5i` files (24.9 GB uncompressed) and 550 CSVs. **(opened)** I listed the zip's central directory over HTTP range requests and extracted two files without downloading the whole archive. |
| Contents of `.h5i` | Plain `.h5i` file: four boolean masks (trabecular, cropped trabecular, full outer, thresholded) on a 350 x 350 x 421 grid. `_sed.h5i` file: `sed`, `youngs_modulus`, `fe_applied_force`, plus the masks. **(opened)** Voxel size is not stored in the file attributes; use 10.5 µm from the paper. |
| Checksums | data.zip md5:6f7d124464ec80884a3891e55a40f12d; code.zip md5:c72ae11ba167abf66aa411c2d1b5c7ff ([Zenodo API](https://zenodo.org/api/records/17256047)) |
| Layout | `data/results_layer0_microCT/{asc,bis,pth}/{m2,m3,m4}/` with 2 files per animal and time point. **(opened)** The vehicle (`co`) and 0n/8n animals appear inside these treatment folders (for example `sost_ovx_el_co_8n_08_m_04.h5i` sits in `asc/m4`). Whether vehicle animals are duplicated across folders is UNVERIFIED. |
| Layer-1 CSVs | Static and dynamic morphometry, FQR (formation/quiescence/resorption), mechanostat parameters and per-interval mechanoregulation tables ([README in code.zip](https://zenodo.org/records/17256047)). These are published outcomes; see the Registered Report section. |
| Licence | CC BY 4.0 ([Zenodo API](https://zenodo.org/api/records/17256047)) |
| Published FE model / strain field | FE solved on the voxel grid; SED and E per voxel are in the files. The solver mesh is implicit (one hexahedron per voxel); the FE input deck is not separately shared. UNVERIFIED which solver and boundary conditions the files used. For the same lab's earlier CV6 work, [Scheuren 2020](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2020.566346/full) used 8-node hexahedral models solved with ParOSol, E = 14.8 GPa, 1% displacement rescaled to 8 N. Whether the 2025 files used identical settings is not stated on the page I read. |
| Published remodeling analysis | Yes: conditional formation/resorption probability vs SED and mechanostat curves, per group, are in the paper and layer-1 CSVs ([README](https://zenodo.org/records/17256047)). |

### 2. Sheffield (Dall'Ara group), ORDA

Related ORDA records (all CC BY 4.0, [Figshare API](https://api.figshare.com/v2/articles/12927365)):

| ORDA record | Paper | Contents | Size |
|---|---|---|---|
| [10.15131/shef.data.12927365](https://doi.org/10.15131/shef.data.12927365) | [Cheong et al., Acta Biomater 2020, DOI 10.1016/j.actbio.2020.09.011](https://eprints.whiterose.ac.uk/id/eprint/166192/1/1-s2.0-S1742706120305250-main.pdf); same record is cited by [Cheong et al., Front Bioeng Biotechnol 2021, DOI 10.3389/fbioe.2021.676867](https://pmc.ncbi.nlm.nih.gov/articles/PMC8225949/) | One OVX + loaded mouse, 4 registered whole-tibia DICOM stacks (weeks 14, 18, 20, 22): 5,772 slices. **(opened)** Whole database on request. | 1.15 GB ([Figshare API](https://api.figshare.com/v2/articles/12927365)) |
| [10.15131/shef.data.16685359](https://doi.org/10.15131/shef.data.16685359) | Record title "Positive interactions of mechanical loading and PTH treatments on spatio-temporal bone remodelling" (the record text names an Acta Biomaterialia 2021 paper; I did not find its DOI) | One mouse, same design plus PTH. Whole database on request. | 1.19 GB ([Figshare API](https://api.figshare.com/v2/articles/16685359)) |
| [10.15131/shef.data.29994211](https://doi.org/10.15131/shef.data.29994211) | [Farage-O'Reilly et al., Biomech Model Mechanobiol 2025, DOI 10.1007/s10237-025-02011-z](https://pmc.ncbi.nlm.nih.gov/articles/PMC12618303/) | Source DICOM for one mouse (weeks 18 and 20; 2,886 files **(opened)**), loading-direction frequency-curve `.mat` files, four tiny SED heat-map `.mat` summaries (about 3.5 kB each, not per-voxel fields) and figure PNGs. Record text: "examples of the input images for one mouse and the results". | about 0.61 GB ([Figshare API](https://api.figshare.com/v2/articles/29994211)) |

- **Newer Sheffield record:** [10.15131/shef.data.32324652](https://doi.org/10.15131/shef.data.32324652) belongs to [a 2026 Biomech Model Mechanobiol paper, DOI 10.1007/s10237-026-02078-2](https://eprints.whiterose.ac.uk/id/eprint/243006/1/s10237-026-02078-2.pdf) on muscle-force-driven micro-FE of six OVX tibiae (scanned at 18 weeks). It has no loading experiment or time-lapse remodeling outcome, and I did not open the record. UNVERIFIED contents.
- **Species, site, loading:** female C57BL/6J, right tibia, OVX at 14 weeks. Load is a 2 to 12 N trapezoidal pulse, 0.2 s hold, 10 s between cycles, 16 kN/s, 40 cycles per day, 3 times per week on alternate days, at weeks 19 and 21 ([Acta Biomater 2020, accepted manuscript](https://eprints.whiterose.ac.uk/id/eprint/166192/1/1-s2.0-S1742706120305250-main.pdf)). The frequency in Hz is not stated, which matters for a cycle-count based stimulus; I cannot say whether the shared README gives it.
- **Scans:** every 2 weeks, 10.4 µm isotropic. The study has 6 OVX + loaded mice ([Acta Biomater 2020](https://eprints.whiterose.ac.uk/id/eprint/166192/1/1-s2.0-S1742706120305250-main.pdf)). A later paper reports 11 OVX mice (5 unloaded, 6 loaded) scanned every other week from 14 to 24 weeks ([Front Bioeng 2024](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1335955/full)).
- **FE and strain:** micro-FE models use 10.4 µm bricks, E = 14.8 GPa, ν = 0.3 ([Front Bioeng 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8225949/)). ORDA record 12927365 contains DICOM only, no FE models or strain fields **(opened)**. Strain fields can be recomputed from the images.
- **Data statements elsewhere:** the 2024 and 2022 papers say raw data "will be made available by the authors, without undue reservation" ([Front Bioeng 2024](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1335955/full), [Front Endocrinol 2022](https://eprints.whiterose.ac.uk/id/eprint/189206/1/fendo-13-915938.pdf)).
- **Contact:** the ORDA records and the funder's output listing give e.dallara@sheffield.ac.uk ([Gateway to Research](https://gtr.ukri.org/project/CA6AC550-785C-4833-B91A-72FB0C54878A?pn=0&fetchSize=25&selectedSortableField=date&selectedSortOrder=ASC)).
- **Published remodeling analysis:** yes, the Acta Biomater paper is titled as a combined in vivo/in silico study. See the Registered Report section.
- **MouseTibia service:** [mousetibia.insigneo.org](https://mousetibia.insigneo.org/uct2ufe/) runs uploaded scans through a microFE pipeline. It requires a Sheffield Google account and states no licence, so it is not a data source.

### 3. ETH frequency series (the Frontiers 2020 paper you named)

- Paper: [Scheuren et al., Front Bioeng Biotechnol 2020, DOI 10.3389/fbioe.2020.566346](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2020.566346/full). Female C57BL/6J, CV6, 5 minutes per bout, 3 times per week for 4 weeks. Groups: sham 0 N, static 8 N, and 8 N cyclic at 2, 5 and 10 Hz (600, 1,500 and 3,000 cycles). n = 5 to 8 per group, weekly scans, 10.5 µm.
- Data statement (verbatim): "All datasets presented in this study are included in the article/Supplementary Material." No repository link. The [ETH Research Collection record](https://www.research-collection.ethz.ch/entities/publication/34fea96b-e900-44d8-bb04-63603927196e) (DOI 10.3929/ethz-b-000449350, CC BY 4.0) lists no linked datasets and no downloadable data files.
- The same animals are reused in [Marques et al., Front Bioeng Biotechnol 2023, DOI 10.3389/fbioe.2023.1140673](https://pmc.ncbi.nlm.nih.gov/articles/PMC10126906/): sham 8, static 8, 2 Hz 7, 5 Hz 5, 10 Hz 8 mice. Its data statement says inquiries go to the corresponding author. The [ETH record](https://www.research-collection.ethz.ch/entities/publication/cfb8ef1c-cb74-43fd-b5e2-a57768c6078d) has the paper PDF only.
- FE: homogeneous or heterogeneous hexahedral models (E = 14.8 GPa bone), 1% displacement rescaled to 8 N (4 N for sham). Outputs are SED, effective strain and the SED gradient ([Marques 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10126906/)). Its supplementary file is a 3.5 MB PDF, so it holds no strain fields.
- Contact: ram@ethz.ch is the correspondence address on the [Frontiers paper](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2020.566346/full).
- Static vs cyclic: the 17256047 record does not contain this contrast (it has 0 N vs 8 N at 10 Hz only), so this request is needed for the static vs cyclic question.

### 4. Columbia (Guo lab)

- Abstract: [ORS 2026 #695](https://www.ors.org/wp-content/uploads/AM26/695.pdf), "Longitudinal multiscale finite element modeling for characterizing 4D bone modeling and remodeling in a mouse tibia" (Shyu, Fagre-Guerriero, Guo). 9 female 16-week-old C57BL/6J mice, 9 N peak, 5 weeks, 5 consecutive days per week, weekly in vivo proximal-tibia scans beginning 2 weeks before loading (5 µm cortical VOI), whole tibiae at 20 µm, ex vivo whole-tibia scan at week 5. Loading cycles and frequency are not stated. No data availability statement.
- Related paper: [Frontiers in Medical Engineering 2025, DOI 10.3389/fmede.2025.1547895](https://www.frontiersin.org/journals/medical-engineering/articles/10.3389/fmede.2025.1547895/full). 4 mice per group (WT and SOST KO), 1 to 9 N haversine at 2 Hz, 100 cycles, 5 µm, weekly scans. Its data statement points to [github.com/pts2116/BBL-ModRemod](https://github.com/pts2116/BBL-ModRemod). I listed that repo: it has MATLAB scripts, Scanco `.COM` batch files and two result spreadsheets, about 2.8 MB, no images, no licence ([GitHub API](https://api.github.com/repos/pts2116/BBL-ModRemod/contents)). The 4-per-group study and the 9-mouse abstract may or may not be the same animals. UNVERIFIED.
- FE: abstract states 15 GPa, ν = 0.3, 9 N, submodels of a 5 µm cortical VOI driven by 20 µm whole-tibia models. The Frontiers paper reports no FE strain values for the experiment.
- Contacts: X. Edward Guo, ed.guo@columbia.edu ([JBMR author page](https://onlinelibrary.wiley.com/doi/abs/10.1002/jbmr.4338)); presenting author gbf2113@columbia.edu ([ORS abstract](https://www.ors.org/wp-content/uploads/AM26/695.pdf)). Both are UNVERIFIED as current.

### 5. Others (request only, lower priority)

- [Schulte et al., PLoS ONE 2013, DOI 10.1371/journal.pone.0062172](https://pmc.ncbi.nlm.nih.gov/articles/PMC3634859/): CV6, 9 N (8 N + 1 N preload), 10 Hz, 3,000 cycles, 3 times per week for 4 weeks, weekly scans, 10.5 µm. n = 9 loaded, 8 control. No repository link in the page I read.
- [Sci Adv 2020, DOI 10.1126/sciadv.aax8301](https://pmc.ncbi.nlm.nih.gov/articles/PMC7060058/): tibia, 12 N, 40 cycles per day, 2 Hz, alternate days for 2 weeks, scans out to 12 weeks, 9 µm, 18 loaded and 6 control mice. "Additional data ... may be requested from the authors."
- [Mendeley Data h2cnrvhg76](https://data.mendeley.com/datasets/h2cnrvhg76/1): single ex vivo mouse tibia scan, no loading, so not relevant.

## Fit for a Registered Report on existing data

**What PCI RR says.** PCI RR accepts "secondary RRs" on existing data and requires you to state a bias-control level. Level 3 means the authors have not observed the data. Level 2 means they have not observed the key variables. Level 1 means they have not yet run the proposed analyses on the key variables. Levels 1 and 2 "will usually be required to include stringent countermeasures against overfitting, such as the adoption of conservative inferential statistical thresholds, recruitment of a blinded analyst, or multiverse/specification analysis." A key variable counts as observed once "the raw data that will go into the analysis that answers the research question has been seen." PCI RR "will not consider studies where the authors already know the outcomes of the prospective (planned) analyses" ([PCI RR guide for authors, section 2.6](https://rr.peercommunityin.org/PCIRegisteredReports/help/guide_for_authors)).

**The problem.** The remodeling outcome (formation/resorption vs SED, mechanostat curves) is already published for ETH and Sheffield, and the ETH layer-1 CSVs contain it. Honest claims are therefore limited.

**How I would handle it** (my reading of the policy, not a PCI RR ruling; confirm through a pre-submission enquiry to the PCI RR managing board):

1. **Claim Level 1, not 2 or 3, unless you avoid the data entirely.** You have read abstracts and figures, so Level 3 is out. Level 2 is defensible only if you never open the layer-0 masks or layer-1 CSVs before Stage 1 and state that you have read the papers' aggregate results. Level 1 is the safe claim, with the countermeasures below.
2. **Define the key analysis as something the data owners have not done.** They tested SED vs observed remodeling. Your planned analysis is a forward prediction: run your frozen law from the 8 N-loaded (and 0 N) baseline masks, then score predicted vs observed voxel-level formation, resorption and quiescence. That is a different outcome statistic, but the aggregate trend is known, so state that openly.
3. **Freeze before looking.** Fix parameters from other sources (the repo's literature-calibrated values) or tune on one dataset only. Tag the repo and deposit it on Zenodo before touching layer-0 data. Record the Zenodo MD5 checksums (listed in the [record](https://zenodo.org/records/17256047)) and your download timestamps as an audit trail.
4. **Split the data.** Calibrate (if at all) on the 0 N group or on the Sheffield mouse. Confirm on held-out ETH vehicle animals. Keep the drug groups for exploratory work only.
5. **Add the countermeasures PCI RR names:** conservative thresholds (for example alpha = 0.005), a blinded analyst who codes group labels as A/B for scoring, and a multiverse analysis across threshold, registration and voxel-to-element mapping choices.
6. **Pre-specify outcome-neutral checks:** positive control (a model that uses the true measured SED should reproduce the published conditional-probability trend), negative control (shuffled labels), and a minimum registration quality.
7. **Declare disclosures.** State which papers and figures you have read, that you have not opened the voxel data, and who has seen what.

A fallback if PCI RR declines: post the same plan on OSF with a timestamp and report as a preregistered secondary analysis. UNVERIFIED that every journal accepts that substitute.

## Needs Wes

- Approve each email below before anything is sent. All are NOT SENT.
- Confirm or fix the contact addresses (Guo and Fagre-Guerriero are UNVERIFIED as current).
- Choose the bias-control level you can truthfully claim, and decide whether to open the layer-0 and layer-1 data before Stage 1.
- Decide whether craniofacial relevance of a vertebral/tibial test is enough for the preprint. It tests the stimulus-to-remodeling mapping, not maxillary geometry.

## Draft emails (all NOT SENT)

### Email 1 (NOT SENT): Dall'Ara, University of Sheffield

To: e.dallara@sheffield.ac.uk
Subject: Request for the full mouse tibia loading dataset (Cheong et al. 2020, Acta Biomaterialia)

Dear Professor Dall'Ara,

I am an independent researcher (MorphoGenix) building an open finite element model of strain-driven bone remodeling. I want to test its remodeling law against published in vivo data, and your OVX and tibial loading study (doi:10.1016/j.actbio.2020.09.011) is the best match I have found.

The ORDA record (doi:10.15131/shef.data.12927365) contains one mouse and says the full database is available from you. Could you share the registered micro-CT images for all loaded and unloaded mice at weeks 14 to 22? If the micro-FE models, strain or SED fields and body weights exist, those would help most. The applied load frequency in Hz would also help.

I plan to run my model forward from the baseline scans and compare predicted and observed formation and resorption. I would pre-register the analysis, offer you co-authorship or acknowledgement as you prefer, credit the ORDA records, and release all code openly. I would use the data only for this analysis. Please let me know under what licence you are happy for me to cite and redistribute derived results.

Best regards,
Wes Gilbert
Independent researcher, MorphoGenix
wslgilbert50@gmail.com

### Email 2 (NOT SENT): Müller, ETH Zürich

To: ram@ethz.ch
Subject: Request for static vs cyclic CV6 loading micro-CT data (Scheuren 2020, Marques 2023)

Dear Professor Müller,

I am an independent researcher (MorphoGenix) building an open finite element model of strain-driven bone remodeling and testing it on published in vivo data. Thank you for depositing the 2025 anabolic therapy dataset on Zenodo (doi:10.5281/zenodo.17256047); I plan to use its vehicle groups.

I am writing about a second dataset. The caudal vertebra loading study (Scheuren et al., doi:10.3389/fbioe.2020.566346; reused in Marques et al., doi:10.3389/fbioe.2023.1140673) compares sham, static 8 N, and 2, 5 and 10 Hz loading, and the data statement points only to the paper and its supplement. Could you share the segmented time-lapse images and the micro-FE results (SED fields) per animal and week? Segmented masks and SED are enough. I also have one question on the Zenodo record: could you confirm that folders m2, m3 and m4 correspond to weeks 20, 22 and 24?

I would pre-register the analysis, offer you and the authors co-authorship or acknowledgement, cite both papers, and release my code openly.

Best regards,
Wes Gilbert
Independent researcher, MorphoGenix
wslgilbert50@gmail.com

### Email 3 (NOT SENT): Guo lab, Columbia

To: ed.guo@columbia.edu; cc gbf2113@columbia.edu (addresses UNVERIFIED)
Subject: Request for time-lapse tibial loading micro-CT data (ORS 2026 abstract 695)

Dear Professor Guo, Dr Shyu and Dr Fagre-Guerriero,

I am an independent researcher (MorphoGenix) building an open finite element model of strain-driven bone remodeling, and I am looking for in vivo time-lapse micro-CT data to test its remodeling law. Your ORS 2026 abstract on longitudinal multiscale FE modeling in the mouse tibia (9 mice, 9 N, 5 weeks, weekly scans) matches my needs closely.

Could you share any of the following? (1) The weekly in vivo proximal tibia scans, or the segmented and registered versions. (2) The week-5 ex vivo whole-tibia scans. (3) The FE models, strain fields or the load cycle count and frequency. I understand that unpublished data may not be shareable yet, and I am happy to wait or to work with a subset.

I would offer co-authorship or acknowledgement as you prefer, cite your work, pre-register my analysis and release all code openly. I am also happy to run a call to explain what I would do.

Best regards,
Wes Gilbert
Independent researcher, MorphoGenix
wslgilbert50@gmail.com

## Search coverage

- **Zenodo:** searched with keywords on mouse tibia/caudal vertebra loading, time-lapse micro-CT and bone remodeling, and by author for Müller, Schulte, Walle, Dall'Ara and Guo. The only relevant record is 17256047 ([Zenodo search API](https://zenodo.org/api/records)). A 2026 record titled "Four-dimensional Time-resolved Analysis of Skeletal Regenerative Material Performance In Vivo" came up and I did not open it. UNVERIFIED.
- **Figshare / ORDA:** the four Sheffield records above plus supplementary datasheets ([Figshare API](https://api.figshare.com/v2/articles/search)) and the [NLM Dataset Catalog listing](https://datasetcatalog.nlm.nih.gov/searchResults?filters=agent:Cheong%252C%2520Vee%2520San&sort=rel&page=1&size=10). The 2019 and 2020 Sheffield records on non-invasive failure-load prediction use ex vivo scans, so I excluded them ([GtR](https://gtr.ukri.org/project/CA6AC550-785C-4833-B91A-72FB0C54878A?pn=0&fetchSize=25&selectedSortableField=date&selectedSortOrder=ASC)).
- **Dryad:** keyword searches returned nothing relevant to rodent loading and time-lapse micro-CT ([Dryad search API](https://datadryad.org/api/v2/search)).
- **ETH Research Collection:** the paper records show no linked datasets. Its search API did not respond to my script, so I used the web pages only. A deposited dataset I could not find by web search might exist. UNVERIFIED.
- **ORMIR:** the community list has one mouse tibia dataset (22 baseline scans, no loading), so not relevant ([ORMIR datasets](https://www.ormir.org/datasets.html)).
- **Rat:** I found no open rat in vivo loading with time-lapse micro-CT data. Rat results were bench or FE studies without a data repository.
- **Not checked:** Willie/Birkhold (Berlin, Montréal) tibia and Bristol cortical-loading datasets. Data availability UNVERIFIED.
