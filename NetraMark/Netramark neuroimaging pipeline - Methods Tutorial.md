Netramark fMRI & MEG Neuroimaging Analysis Pipeline Documentation


# Pre-installation checklist:


## Technical Requirements:

Disk space:

fMRI: ~1 GB per scan

~260 MB / scan for FreeSurfer outputs

~670 MB / scan for main pipeline outputs

MEG: ~16 GB per scan

~260 MB / scan for FreeSurfer outputs

400 MB (coregistration)

167 MB (forward_models)

169 MB (inverse_models)

12.6 GB (morphs)

2.0 GB (source_estimates)

271 MB (source_spaces)

10 MB (timeseries)

FreeSurfer must be fully installed prior to any pipeline script runs

Must be installed by user/administrator (i.e. cannot be automatically installed via CLI package installers, unlike most other required pipeline libraries); available here

Valid FreeSurfer license file also required (free, requires online registration here)

Use of FreeSurfer version 7.4.1 is strongly recommended to avoid compatibility issues

Correct pointers / paths must be set in the “FreeSurfer settings” section of the main ‘config.yaml’ file (explained more fully below)

Other package / library dependencies must be installed

Can be done in CLI via pip / conda / etc.

See pipeline package’s ‘requirements.txt’ or ‘requirements-dev.txt’

Setting up a dedicated python environment is recommended, but not strictly mandatory

Pipeline was built & tested using Python version 3.8.10


## Other requirements:

All input data (whether MEG or fMRI) must be BIDS-compliant (including filenaming and metadata presence)

BIDS filenames must, at minimum, include a ‘sub-*’, ‘ses-*’, and datatype-prefix component (other BIDS elements optional)

Any MEG data must be cleaned (e.g. via ICA or some other artifact-removal procedure) prior to pipeline ingestion.

All input data must be placed in a single parent folder, either all together or in subdirectories, including paired metadata sidecar files.

e.g. setting the main data folder (via the ‘MRI_data_directory’ field) to ‘.../data/MRI’ will tell the pipeline to scan all files and subdirectories in this parent folder for any ‘.nii’ or ‘.nii.gz’ files, and all discovered files will be considered part of the input dataset.

If using Craddock maps for parcellation, make sure to set a full file-path to the directory containing the Craddock ‘.nii’ files  in the ‘[atlases][Craddock][craddock_dir]’ field of the ‘config.yaml’ file.


## FreeSurfer configuration (required):

Many steps of the pipeline reference FreeSurfer library functions, and thus require that FreeSurfer is properly configured on the root system.

Most scripts in the pipeline automatically retrieve the following FreeSurfer environmental variables, all of which are mandatory and stored in the following section of the ‘config.yaml’ file:

# ==== freesurfer settings ====

freesurfer:

home:          /opt/freesurfer-7.4.1

license:       /opt/freesurfer-7.4.1/license.txt

subjects_dir:  /opt/freesurfer-7.4.1/subjects

setup_script:   SetUpFreeSurfer.sh

* Note: the ‘setup_script’ in the above refers to setting up environmental variables; not package installation

The default FreeSurfer paths (already included in the CONFIG file) correspond to the standard default installation locations (on a linux machine), and should not need to be changed unless: (a) a different operating system is used, or (b) the default FreeSurfer installation paths were changed by the administrator during installation.

Any script that requires FreeSurfer will have a dedicated code cell — typically the third cell in the notebook — that loads the relevant FreeSurfer parameters, which can be identified via the ‘### LOAD FREESURFER:’ header comment.

When running the cell, you should see the following two printed outputs:

freesurfer-<OS type>-XXXXXXXX_64-7.4.1-YYYYYYY-ZZZZZZZ		←(‘X’ / ‘Y’ / ‘Z’ contain installation package version info and may vary)

CompletedProcess(args='recon-all --version', returncode=0)

If you see this output — particularly the final ‘CompletedProcess’ line with a return code of ‘0’ — then you can confirm that the script will be able to locate and execute the FreeSurfer functions correctly. If you receive an error or see other outputs, contact your system administrator to verify the FreeSurfer package locations and double-check the config.yaml settings to ensure they’re pointing to the correct locations.


# Tutorial:

OVERVIEW

The full analysis pipeline consists of sequential scripts, which can be executed in numerical order, end-to-end. The intended workflow divides the overall pipeline into discrete sub-stages (listed below), after each of which the user either needs to (a) check & verify QC outputs and/or (b) make dataset- and run-specific analysis / modeling decisions before continuing. Note that so-called “optional” user actions are strongly recommended, but not strictly necessary:

(STAGE 0) INITIAL INSTALLATION & SETUP [multiple ‘SETUP’ scripts (dataset- and analysis-dependent)]: Install pipeline package & dependencies; unpack datasets (MRI + fMRI/MEG) to known locations; pre-harmonize datasets to BIDS format (if not already BIDS-compliant).

>>> USER ACTION (REQUIRED): Update & verify core ‘CONFIG.yaml’ settings & path variables.

(STAGE 1) Dataset ingestion [Script #01]: Scans input datasets (MRI + fMRI or MEG) and builds central “run manifest” tables defining the analysis plan.

>>> USER ACTION (OPTIONAL): Review final manifest (‘subject_manifest.csv’ in main project folder) for data completeness and alignment with analytical goals, etc.

(STAGE 2) Pre-processing — MRI [Script #02]: Performs volumetric reconstruction of subject MRIs, masking of primary brain volume, and segmentation of whole-volume brain into anatomical regions. (Note: this is by far the most time- and computationally-intensive stage of the full pipeline.)

>>> USER ACTION (REQUIRED): Review script outputs (text summaries and runtime logs, etc.) to confirm no major errors during reconstruction.

(STAGE 3) Pre-processing — fMRI/MEG [Scripts #03-06]: Exact functions vary depending on whether the fMRI or MEG pipeline branches are being used, but these stages generally perform anatomical coregistration, brain volume masking, de-warping, de-noising, and alignment of final preprocessed data to standardized brain atlas templates.

>>> USER ACTION (OPTIONAL): Visual inspection of QC outputs (-BBR; -denoising; -motion; -SDC) for any gross errors.

(STAGE 4) Parcellation [Script #07]: Assigns all within-brain voxels to ROIs, based on the parcellation scheme chosen during pipeline configuration.

(STAGE 5) ROI labeling [AUX script: “create_ROI_labels”; can technically be run at any time after pipeline setup/configuration]: Automatically creates anatomical and functional labels for chosen ROI parcellation scheme, using standardized brain atlases selected in pipeline configuration settings.

>>> USER ACTION (OPTIONAL): Inspection of QC (-alignment) for gross mismatches / misaligned overlays, & optionally review ROI labels.

(STAGE 6) Extract per-subject anatomical measures [AUX script: “extract_morphometry”, any time after script #02 and prior to final export in script #14]: Extracts whole-brain and regional anatomical volumes for each subject, to serve as anatomical covariates in the final output dataset.

(STAGE 7) Network analysis [Script #08]: Generates per-ROI neural activity time-series data; computes correlations across ROIs; computes centrality metrics; generates epoched network graph of brain dynamics.

>>> USER ACTION (REQUIRED): Adjust epoching parameters for network-graph computation; review final time-series & graph metrics outputs; and (if continuing on to ML stage) set up file-pathing CONFIG fields and set hyperparameter regimes for stage 8.

(STAGE 8) Dynamic state-space modeling [Scripts #09-11]: Freezes a split train/test dataset for ML; trains multiple HMM models on frozen dataset according to user-provided hyperparameter space; automatically selects the best model within each hyperparameter set; evaluates all winning models from each hyperparameter regime and presents evaluation metrics to the user in a table for final model selection.

>>> USER ACTION (REQUIRED): Review cross-model evaluation metrics, and manually select a final model to use for state-space analyses.

(STAGE 9) Extract model-based features [Scripts #12-14]: Employs user-selected final HMM model to label time-resolved state trajectories for each subject/session; computes higher-level feature summaries per subject/session (state occupancy rates, etc.); extracts state definitions from final model; collects and exports all selected feature sets as ML-ready tables. (Note: Make sure the AUX scripts ‘extract_morphometry’ and ‘create_ROI_labels’ scripts have also been run, otherwise certain data outputs won’t be available during final dataset assembly.)

>>> END OF PIPELINE


## Walkthrough:

Note — the tutorial below assumes a simplified workflow:

no per-file QC metadata available / used

no within-dataset subsetting / filtering — i.e. we will be processing an entire dataset in a single large batch

uses base/default subdirectory names & structure

uses default processing parameters wherever possible

In this tutorial, we walk through the complete end-to-end functional neuroimaging analysis pipeline, starting from raw MRI and fMRI data and ending with a single, analysis-ready dataset containing multi-modal brain features per subject and session. Along the way, we perform anatomical reconstruction, fMRI preprocessing and quality control, atlas-based parcellation, time-resolved network-graph construction, and dynamic state-space modeling using Hidden Markov Models (HMMs).

The ultimate goal of the pipeline is to convert high-dimensional, noisy neuroimaging time-series data into a structured set of interpretable features that summarize how whole-brain connectivity patterns evolve over time and differ across individuals. These features can be used for a wide range of scientific and clinical analyses, including group comparisons, correlational studies, and predictive modeling (e.g. identifying neural signatures associated with treatment response or symptom trajectories).

We’ll be using the MRI and fMRI data from the CANBIND Integrated Biological Markers for the Prediction of Treatment Response in Depression dataset — specifically, the first three study time-points (baseline, week 2, and week 8). To keep the tutorial simple, we’ll be performing a relatively conventional analysis, which means we will be mostly setting only the most crucial required parameters, and otherwise relying on default pipeline settings wherever possible.


### Stage 1 — Pipeline initialization & setup

Setting up Data Ingestion:

We begin by assuming the pipeline (and all of its dependencies) have already been installed; see this checklist for the basic requirements.

With everything installed, we first download and extract each set of MRI and fMRI data to separate folders on our local system. Then, we update the following mandatory CONFIG.yaml fields:

(1) ‘root_output_directory’: the full system filepath to our main project output directory (NB: this directory must already exist, so the user may need to manually create this first). All pipeline outputs will be stored under this parent directory, either directly or within user-specified, stage-specific subfolders.

(2) ‘MRI_data_directory’: the full system filepath to our source MRI data folder

(3) ‘fMRI_data_directory’: the full system filepath to our source fMRI data folder

Note: Filepaths shown above are in Linux format, and will look different if using Windows / MacOS.

Next, we prepare the files for ingestion into the pipeline. This dataset is already in BIDS-compliant format, so our ingestion scripts will automatically parse its filenames and metadata to extract various critical subject- and file-level identifiers, with a little bit of additional guidance from us to ensure a clean, complete, and fully-auditable final dataset.

For instance, we can see from inspecting the dataset’s base BIDS-compliant filenames that they come in the following standardized format:

‘sub-CBN01CAM0002_ses-01SE01MR_task-rest_run-1_echo-1_bold‘

The initial ‘sub-<SUBJECT>’ field is mandatory in BIDS format, and always forms the first filename element — so our ingestion script will automatically extract this ‘subject_ID’ as the string ‘CBN01CAM0002’.

The final filename suffix — ‘..._bold’ — always forms the last filename element, and specifies the BIDS ‘datatype’, which we will return to shortly.

The second section — ‘..._ses-<SESSION>’ — contains the BIDS session identifier, which is an arbitrary ID string. This field is technically non-mandatory in BIDs format, but in practice will almost always be included as the second filename element in most datasets (if this field happens to be missing, then it will have to be added by the user prior to ingestion).

To make these session tags more user-legible, we can use the ‘session_ID_mappings’ field in our CONFIG file, which consists of a dictionary with a set of {key:value} mappings. These mappings will convert each arbitrary BIDS-session identifier into a custom, human-readable ‘session_ID’ that will be used to identify time-points throughout the rest of the pipeline. Note that, to ensure that no data is inadvertently lost, the ingestion scripts require that all BIDS session identifiers present in the dataset are fully mapped; if an un-mapped BIDS session identifier is encountered, the script will raise an error to notify the user.

In this dataset, manual inspection reveals that all the baseline files (for both MRI and fMRI) use a BIDS-session identifier of ‘01SE01MR’, and all week-8 sessions use the label ‘03SE01MR’, whereas the week-2 sessions mostly use the tag ‘02SE01MR’, but with a few stray files using ‘02SE01aMR’ — so we provide all four of these mappings into the CONFIG file accordingly:

With this field complete, this will ensure that all input files are now consistently grouped together despite any upstream metadata variations (e.g. such as we observed in the case of the week-2 data, which will now be properly consolidated into a single label/category).

Note that the same ‘session_ID_mappings’ dictionary is re-used downstream to perform chronology-based dataset filtering — for this reason, it is important to provide the re-mapping labels in chronological order (as we just did above, by defining ‘baseline’ → ‘week2’ → ‘week8’ in that order).

Next, we provide the ‘datatype_mappings’, which re-map BIDS-datatype codes to custom labels:

The above settings will give files with ‘*_T1w’ and ‘*_T2w’ BIDS suffixes a common label of ‘MRI’ in our final data catalogue, and files with ‘*_bold’ will be labeled as ‘fMRI’.

Note that both of the ‘session_ID_mappings’ and ‘datatype_mappings’ CONFIG fields are strictly mandatory, in that they must contain {string:string} pairs that fully cover all possible BIDS identifiers extracted from the original dataset. The actual re-mapping itself, however, is technically optional, as the user could in principle decide to retain the BIDS-derived identifiers by simply mapping the same strings together (for example by replacing the string ‘baseline’ with ‘01SE01MR’ in the first example, or ‘fMRI’ with ‘bold’ in the second). Nevertheless, replacing these with more human-readable labels is generally recommended for downstream transparency and auditability.

Next, there are some additional — though optional — steps we can take to further clean up the dataset identifiers.

First, in our example CANBIND dataset, the control and experimental group data happen to be separated by being contained in different subdirectories, with the folder names containing the substrings ‘control’ or ‘MDD’:

This means that we can extract a label for whether a given subject is in the control or experimental group by looking for these different substrings in the filepaths of our ingested files (see next image).

Additionally, from our initial manual dataset inspection we can also observe that, in the present dataset, all BIDS subject identifiers begin with a common prefix string ‘CBN01*’:

sub-CBN01CAM0002_ses-01SE01MR_task-rest_run-1_echo-1_bold

...

sub-CBN01MCU0001_ses-01SE01MR_task-rest_run-1_echo-1_bold

...

sub-CBN01QNS0003_ses-01SE01MR_task-rest_run-1_echo-1_bold

Since this prefix is uninformative — i.e. it is not useful for discriminating between different subjects — we can optionally tell our script to automatically strip this prefix from all our incoming subject_IDs, so that we retain only the unique part of each BIDS identifier.

We can implement both of the above by setting the ‘group_identifiers’ and ‘strip_subjectID_substrings’ fields in the CONFIG file:

The ‘group_identifiers’ field is another dictionary variable that provides {string:string} re-mappings which will detect the different substrings (‘control’ vs. ‘MDD’) in the filepaths, and will automatically provide these (here unchanged) as a third, optional ‘group_ID’ identifier labeling the experimental groups (see the dedicated documentation section for the data catalogue builder script for more detailed info). The ‘strip_subjectID_substrings’ field contains a single target substring we want to eliminate from all BIDS-derived subject identifiers, which will help clean up our final downstream subject_IDs:

subject_ID: ‘CBN01CAM0002’ → ‘CAM0002’

subject_ID: ‘CBN01MCU0001’ → ‘MCU0001’

subject_ID: ‘CBN01QNS0003’ → ‘QNS0003’

With these steps, we’ve ensured that our dataset will be labeled with the core subject- and file-level identifiers necessary to track them throughout the rest of the pipeline:

subject_ID (mandatory)

session_ID (mandatory)

group_ID    (optional)

Throughout our pipeline, specific target MRI, fMRI, and/or MEG files — i.e. the core objects of analysis that get passed through our different processing stages — are always identified by a unique combination of [<subject_ID>_<session_ID>]; whereas the ‘group_ID’ identifier we created here is an extra, non-mandatory data decoration occasionally used for optional dataset subsetting and data labeling further downstream.

Setting the Analysis Plan:

Finally, we need to tell the pipeline which sets of MRI and fMRI (or MEG) files to use.

In any dataset, we need each subject to have at least one MRI scan so that we can perform a full 3D reconstruction of the brain volume. This is a strictly necessary analysis prerequisite for our pipeline, as it provides an anchoring, subject-specific alignment template for coregistering and parcellating our fMRI (or MEG) data later on.

In our current dataset, each subject actually received multiple MRI scans over the course of the study, but we only need one MRI image in order to perform an anatomical reconstruction. Therefore, we just need to tell the pipeline which of the multiple individual MRI files to pull from, which we can do using the ‘MRI_selection’ CONFIG field:

This field must be a string corresponding to one of the “preset” names, or a specific ‘session_ID’ label — for full description of what these different presets are, see the full documentation for the “build_run_manifest” script.

In contrast to our MRI requirements, in any given dataset we may have access to any number of different fMRI and/or MEG scans for a given subject (including zero, e.g. if a subject was enrolled in a study but dropped out before any neuroimaging sessions were scheduled). Depending on our intended analysis goals, we may want to analyze all available sessions, or only a subset of scans from specific time-points (e.g. only ‘baseline’ sessions if we are looking to detect prospective predictive factors; or only ‘endpoint’ sessions if we are looking for neural correlates of specific post-treatment outcomes, etc.).

Therefore, unlike MRI, we may need to specify a more complex set of fMRI/MEG sessions to pass through the pipeline for analysis, which we can do using the ‘fMRI_selection’ (or ‘MEG_selection’) CONFIG field:

This field can be a string (if using an analysis preset), or a list of strings (if passing a custom list of ‘session_ID’ labels). For now we will keep our example simple and tell our pipeline that we only want to analyze fMRI scans from the ‘baseline’ study time-point. For more information on this parameter, as well as the rest of the (here un-used/un-modified) CONFIG parameters for this pipeline stage, see the full documentation for the “build_run_manifest” script.

With these core parameters set, we can go ahead and run the following scripts:

[SETUP-01] — Base Data Catalogue Builder

This script will scan the provided data directories and build a base (un-decorated) catalogue of all available data.

[SETUP-03] — Collect fMRI Parameters

This script harvests fMRI acquisition parameters from BIDS metadata, and stores all parameters in a ‘fMRI_parameters.csv’ file.

[01] Build Run Manifest

This script compiles a final manifest of subjects/files to pass through the pipeline for full processing, representing the final intended sample pool.


### Stage 2 — Data ingestion & manifest review

After running the pre-flight scripts and the first main pipeline script, we should now see the following output files in our main project directory:

At this stage, the main table we want to inspect is ‘subject_manifest.csv’ (for detailed breakdown of all output artifacts at this stage, see the detailed documentation for each of the individual scripts listed above).

Our ‘subject_manifest.csv’ table should look like this:

At this stage, we should verify that our analysis has been set up correctly. Depending on the analysis settings, this table will contain a variable number of columns for us to check.

MRI columns: In all cases, we should see one pair of MRI-related columns, designating the ‘MRI_filename’ and the corresponding ‘MRI_session_ID’ that was selected for each subject. In this case, we used the ‘first_any’ preset, and since all our subjects’ first MRIs were taken at the baseline study visit, all the values are ‘baseline’, with no other NaNs/nulls/missing values — so far so good.

fMRI (or MEG) columns: There are two possible scenarios here. If only a single time-point was selected for analysis — e.g. if ‘fMRI_selection’ used a single-time-point preset such as ‘first_any’, or a manual session_ID selection string list containing a single entry like ‘[‘baseline’]’ — then we should see another pair of columns listing the filename and session_ID for each single scan file, as we do here. If multiple scan files per subject were selected during pipeline configuration, we would instead see a variable number of columns for each selected time point, named in the format ‘<datatype>_<session_ID>_filename’, listing the base filename of each target file. (For example, if we had provided the list ‘[‘baseline’, ‘week2’, ‘week8’]’ in the ‘fMRI_selection’ CONFIG field, we would have no dedicated ‘fMRI_session_ID’ column, but instead the three columns: ‘fMRI_baseline_filename’, ‘fMRI_week2_filename’, and ‘fMRI_week8_filename’.)

Once we’ve confirmed that the subject manifest has been generated as intended, without encountering errors along the way, we can continue on to run the following scripts:

[02] MRI reconstruction

This script will gather up all the MRI files we included in the subject manifest we just built, and submit these to FreeSurfer’s ‘recon-all’ pipeline to generate one full 3d volumetric reconstructions for each unique subject_ID in our manifest.


### Stage 3 — Validate MRI reconstructions

The volumetric reconstructions are by far the longest single stage of the pipeline — 2-10 hours per job, depending on local system specs — so be prepared for long (possibly multi-day) processing times for this stage if you are passing in a full dataset’s worth of subjects at once. (See the dedicated documentation for MRI reconstruction for some usage tips you can perform ahead of time to verify that everything is set up correctly before running large batches of subjects, which will maximize the chances of success for the first full run.)

As the reconstructions are completed, there are a few outputs we can check. The simplest — though not 100% fool-proof — method is to make sure that we see one subdirectory per unique subject_ID in our FreeSurfer ‘.../subjects/’ folder:

However, subjects simply having subdirectories within this folder is not a surefire guarantee, as ‘recon-all’ jobs which fail halfway through can sometimes still leave behind partial output data in subject-specific folders.

Thus to complement the folder-checks, we can also check the runtime logs within the MRI_reconstruction script notebook itself. FreeSurfer reconstructions create a very large and detailed log of intermediary outputs, but any successfully-completed jobs should always end with the following lines printed in the console:

...

...

...

Started at Mon Nov 10 23:16:51 UTC 2025

Ended   at Tue Nov 11 01:33:15 UTC 2025

#@#%# recon-all-run-time-hours 2.273

recon-all -s <subject_ID> finished without error at Tue Nov 11 01:33:15 UTC 2025

FreeSurfer output will be displayed as console / cell-level print-outs within the script notebooks themselves; a full copy of all FreeSurfer outputs are also automatically captured for each job and saved in the ‘.../subjects/_launcher_logs’ folder (see above image, top-left) as ‘<subject_ID>.recon-all.stdout.log’ files.

In most cases, however, detailed checks of FreeSurfer output logs should not be necessary except for deep debugging. In general, the vast majority of errors will cause the ‘recon-all’ pipeline to exit and will throw an explicit error within the script notebook itself, making it rather obvious to the user if an error has occurred (albeit unfortunately stopping the entire processing batch in the process). If the script completes without raising a hard error, it can usually be safely assumed that the reconstructions were successful — and any seemingly-successful runs with underlying issues will likely become evident during the next stage when we check various visual QC pipeline outputs.

Once we’ve verified — one way or another — that all the FreeSurfer reconstructions completed without error, we can continue on to run the following scripts, which collectively will pre-process our fMRI data:

[03] Motion correction

this script uses rigid-body motion estimation to re-sample the original imaging scan to a motion-corrected 4D series.

[04] De-warping

This script fixes scanner-physics distortions that arise from directionality-dependent, tissue-density-induced signal-timing differences inherent in fMRI acquisition.

[05] De-noising

This script removes non-neural noise in the fMRI time series by regressing out motion- and physiology-related nuisance signals and (if necessary) censoring frames which exceeded a certain threshold of motion.


### Stage 4 — Inspect preprocessing QC artifacts

We’ve just run the bulk of the fMRI preprocessing section of the pipeline, during which we performed rigid-body motion-correction (script #03), fMRI signal de-warping (script #04), removed noisy signal components, and performed a final re-coregistration of the fully-preprocessed fMRI volume back onto each subject’s reconstructed anatomical volume (script #05).

While a lack of errors during these scripts is generally a healthy sign, processing issues do not always necessarily result in hard code errors — thus the scripts have been designed to output various QC plots that can be visually checked to verify that preprocessing was successful / free of subtle issues that may not typically trigger the raising of hard code failures.

These QC outputs can be found in the following folders (assuming default names provided in the upstream pipeline CONFIG file):

Motion-correction brain masks in ‘.../{main}/QC-motion/’

FD and DVARS plots in ‘.../{main}/QC-denoising/’

Grey-matter, white-matter, and CSF/ventricular masking overlays in ‘.../{main}/QC-BBR/’

For each set of outputs, there are key signs to look for to verify successful processing. For more detailed explanations of the processes these plots represent, see the detailed documentation for each of the respective scripts in question — for now we’ll just be summarizing the main visual checks we can make, heuristically.

Whole-brain motion masks:

What to check for:

For all subjects, mask coverage (blue) should clearly follow the outer brain periphery, without gross excursions into peripheral non-anatomical space.

Note: noticeable gaps in mask coverage — typically around temporal and orbitofrontal regions — is completely normal (due to intrinsic fMRI signal attenuation), so these masks should not necessarily cover the full brain comprehensively. Rather, we’re checking for correct (even if non-complete) isolation of within-brain voxels here — not complete anatomical coverage.

Figure: Example of a correctly-generated fMRI motion mask. These masks are based on input fMRI signal intensity, and so some drop-out of signal / mask under-coverage — particularly within orbitofrontal and temporal regions — is both expected and normal. Here the masking procedure has correctly isolated brain tissue without extending into non-brain-tissue regions. All voxels within this mask will have rigid motion-correction applied to them (while potential signal distortions within the mask-excluded brain regions will be corrected for in the next “de-warping” step).

Error mitigation / correction:

If certain subjects (or groups of subjects) exhibit incorrect masking, it is likely that the primary- and/or “fallback” masking parameters will have to be adjusted: see “Key parameters & troubleshooting” section of the script-specific documentation for more detailed information about these parameters and what they control.

FD & DVARS plots:

What to check for:

Total number / proportion of censored frames should not fall below ~80%.

Censored frames are marked as dots (middle two subplots) or as grey bands (bottom subplot).

Different subjects will naturally exhibit a greater or lesser amount of detectable head movements, but in general, the percentage of “kept frames” should be >80%.

i.e. see “kept X/X frames (X%)” readout in the top left of the main plot space.

Figure: Example of a motion-correction & denoising summary plot. Top: Rigid motion plot showing framewise displacements of the head, with planar movements in 3d coordinates (dx; dy; dz), and with rotational movements shown in radians (rx; ry; rz). NB: FD = Framewise Displacement; DVARS = Delta Variation in Signal, also sometimes known as “spatial standard deviation of successive difference images”; these are two common threshold-based methods for detecting spatial displacements in fMRI. Interpretation: In this particular example we can see evidence of a large head movement around frame #120 (see top sub-plot). The clustering of dots across the FD / DVARS / GS subplots show that this subject’s head movement was correctly identified and that the affected frames were censored out of the final signal (dots & threshold-crossings within subplots), as well as a number of smaller head displacements. The high percentage of kept frames (here 92.5%) indicates that the source data was cleaned/corrected without too much data loss.

Error mitigation / correction:

In general, subjects / scans with a high frame drop-out rate are often difficult to rescue directly without resorting to potentially distortion-prone approaches, such as multi-frame interpolation across runs of censored frames. Thus, the “safest” option for dealing with a scan with excessive head motion is often simply to exclude it from analysis.

BBR plots:

The pipeline uses boundary-based registration (BBR) to align the final preprocessed fMRI volume back onto the subject’s T1 anatomy. The accuracy of this process is fairly easy to assess with simple visual overlays.

Note that coordinate-space (i.e. affine-based) transformations can often fail along just one of the 3d axes, so multiple views of the brain volume are typically  required to detect misalignments — e.g. see the failure example below, in which the coronal view appears well-aligned but hides an erroneous displacement along the sagittal plane.

Figure: Example of a misalignment between MRI (greyscale) and functional imaging (in color; in this case, a MEG source estimate rather than fMRI). Note that this example image is taken from an older version of the pipeline, so it is not formatted like the current QC outputs; nevertheless, the gross misalignment of the outer tissue boundaries across the two overlays is obvious evidence of a failed alignment, and resembles what would be seen in the current pipeline version (see next figure).

Figure: Example of an ideal MRI <-> functional imaging overlay (in this case, fMRI data) using boundary-based registration (BBR), where outer tissue boundaries are properly aligned. Note that the drop-out of fMRI signal strength (“EPI intensity”) — most often within temporal and orbitofrontal regions — is both a normal feature of fMRI signal acquisition and is compensated for elsewhere in the pipeline.

Overlay of FS-masks (GM, WM, CSF/V) on final fMRI image:

To further fine-tune the alignment across volumes, as well as to furnish our voxels with additional anatomical labels, we grab the original grey- and white-matter masks, as well as the CSF/ventricular masks, from each of our FreeSurfer analyses, and then re-project these back into our final coregistration coordinate space (i.e. from anatomical T1 space → native EPI space).

Once again, these are fairly easy to validate using the QC overlays that are automatically generated at this stage of the pipeline. The grey-matter masks (in green) should hug the general outline of the brain; the white-matter masks (blue) should cover only the denser regions underneath the cortical mantle, and the CSF/ventricular masks should cover a small region along the brain’s central midline.

See figures below for a representative failure- and success case, respectively:

Figure: Example of a gross error in brain-masking. In this example, the threshold for mask inclusion was set too low, and the mask erroneously picked up a number of background voxels with very small, near-zero signals, leading to an over-inclusive mask which extended into surrounding non-tissue space. An error like this can be corrected by adjusting the lower mask threshold upwards (see next fig., bottom, for the same example corrected).

Figure: Two examples of QC output from a successful mask-projection, including a corrected version of the same example brain (bottom) from the previous example figure (masking errors). Green = the grey-matter mask; blue = the white-matter mask; and yellow = the ventricular/CSF mask. Note that, in this plot, the greyscale image is the fMRI (EPI space) signal, so we once again see some drop-off in temporal and frontal regions — this is normal.

With these QC outputs checked and the preprocessing steps validated, we can continue on to run the following scripts:

[06] Brainmap alignment

This script pre-computes a projection of the user-selected parcellation atlas, from MNI coordinate space, through anatomical T1 space, and ultimately into each subjects’ native EPI (fMRI scanner) coordinate space, aligning the reference map to each subjects’ individual brain anatomy.

[AUX] Create ROI labels

This script cross-references each of the ROIs from the user-selected parcellation scheme against standard brain atlases, and automatically generates anatomical- and functional labels for each ROI.


### Stage 5 — Confirm brain-map projections / parcellation alignments

NOTE: If you have not already done so during initial pipeline installation / setup, make sure you have set the full system filepath to the folder containing the Craddock map ‘.nii’ files in the ‘[atlases][Craddock][craddock_dir]’ field of the CONFIG file before continuing.

In order to perform ROI-based parcellation, we need to place our fMRI volumes in the same coordinate space as our target brain-map (in this example we’re using the default 50-ROI Craddock brain map).

Importantly, any time we re-sample the original fMRI data into a new coordinate space we necessarily introduce distortions (e.g. due to interpolations across neighboring voxels between spaces with different native voxel resolutions). Therefore, in this pipeline we take a distortion-minimizing approach, by re-sampling the target brain map from MNI space into native fMRI EPI scanner space, rather than the other way around (we use T1 as an intermediary space, so the full transformation chain takes the brain maps from MNI → T1 → EPI space).

Once again, a successful chain of transformations is easy to validate via the use of visual overlays. There are two sets of QC plots we can check:

Initial MNI → T1 transformations

Default location: ‘.../{main}/QC-alignment/atlas_in_T1/’

Technically an optional QC output, which comes already-enabled under default pipeline configurations.

See first figure below.

Final MNI → T1 → EPI coordinate-space projections:

Default QC location: ‘.../{main}/QC-alignment/’ (always enabled)

See second figure below.

Parcellation: Partial brain atlas projection, from MNI → T1

Figure: Brain map projection from MNI → T1. Four examples of the same common ROI parcellation scheme (Craddock map w/ 50 ROIs) projected into the T1 anatomical (e.g. original MRI coordinate) space of four different subjects. Each view is automatically centered on the center-of-mass of the whole brain along each 3D axis, resulting in slightly different x/y/z slice coordinates across subjects. Despite obvious differences in brain anatomy as well as size and orientation, we can clearly observe that individual ROIs have been mapped on to consistent anatomical regions across subjects, indicating that the first half of the projection of the parcellation atlas from MNI → T1 → EPI space has been successful.

Parcellation: Full brain atlas projection, from MNI → T1 → EPI (downsampled)

Figure: Final results of brain map projection from T1 → EPI space. Here the same four example subjects from the previous figure are shown, this time overlaying the final Craddock-50 ROIs over the MRI images in the final, subject EPI (i.e. native fMRI) coordinate space. Note: these final figures are intrinsically lower-resolution because projecting into EPI space entails down-sampling the (considerably higher-res) MRIs and brainmaps into native fMRI resolution. Nevertheless, consistent ROI projection is still evident, verifying that the full MNI → T1 → EPI space transformation chain was executed successfully.

Error mitigation / correction:

Errors in brainmap projection may arise due to miscalculated affines (from upstream), or incorrect sequencing of transformations. The pipeline is automatically set up to try different permutations of the full transformation chain, and to automatically select the best procedure (judged via total voxel-coverage percentage of the end result) — however subtle errors are still possible; see this script’s full documentation for more detail about affines are used and how to intervene manually in the transformation process.

Parcellation output — neural time-series:

If our parcellation map was successfully projected, then we can see that our parcellation output folder (‘.../{main}/parcellation/’ by default) should contain a subdirectory for each file analyzed, each containing two files: one provenance .json file (which stores all the parameters used to generate each time-series), and a large table containing our actual time-series data.

Given that we aren’t shown every possible ROI in the QC plots we inspected previously, one additional quick sanity-check we can perform here is to inspect the second ‘num_voxels’ column of this output table and make sure it contains no null / NaN / ‘0’ values, which would likely indicate that the parcellation was unsuccessful (although this is not always 100% necessarily the case — see the “usage notes” in the detailed script documentation for an example edge-case). In general, inspection should reveal that all ROIs contain at least some voxels — albeit with some considerable variance in total size across ROIs (see below) — which is a strong indication that the brainmap’s voxels were successfully intersected with the fMRI voxels during map projection.

Time-series data can be outputted in one of two ways: either as raw means of total neural activity within ROI, or with an additional step in which Z-score normalization is applied. Here we have used the default — and generally recommended — normalization option, as raw means can be prone to certain distortions if not normalized (for example, smaller ROIs contain fewer voxels, which can increase the variance of the mean and thus produce unintentional differences in scale across ROIs).

This time-series data will be consumed during the next step in the pipeline, in which we will compute the correlation structure across ROIs, and use this to build a time-resolved network graph of the brain’s state during each of our sampling windows, from which we can derive our final graph-based connectivity metrics.

Review automated ROI labeling output (optional):

Finally, we can also review the anatomical- and functional labels generated by our auxiliary script, which creates one dedicated report figure per ROI in the user-selected parcellation scheme (see example below). For each ROI, both sets of labels are determined via a voxel-by-voxel “voting” process, where the label confidence is measured by the degree of each ROI’s overlap with anatomical and functional networks from the Harvard-Oxford brain atlas (for anatomical labels) and the YEO-7 and YEO-17 atlases (for functional labels).

ROI labeling: Detailed functional- and anatomical reports (per ROI)

Figure: ROI summary figures generated by the auxiliary script. The top three “hits” for each ROI are reported for each category. For anatomical labels, the ‘ambiguous’ metadata field is set to ‘True’ if the top anatomical region identified does not report >50% overlap with a single anatomical region, as per the Harvard-Oxford brain atlas. (Note that region-assignment ambiguity is not itself abnormal, as it can be a common consequence of overlapping two sets of brain maps with very different numbers and sizes of ROIs.) “Coarse” functional labels are derived from the YEO-7 functional brain atlas, whereas “fine” functional labels are derived from the YEO-17 atlas.

With these stages complete, we can continue on to the final steps in the “core” pipeline:

[08] Compute correlations

This script will compute neural time-series data from each of the ROIs in our selected parcellation scheme, compute inter-ROI correlational structure, and use this correlation structure to build a time-resolved network graph, which can be used to further compute our final target set of graph-based measures, such as per-region centrality and network-wide modularity metrics.

[AUX] Extract morphometry

This (technically optional) script uses each subjects’ FreeSurfer brain segmentation data to generate a table containing morphological measures, such as total brain size and the size of many individual structures in the brain; these are useful as additional correlates for many analyses.


### Stage 6 — Compute neural time-series & network graphs

At this stage, the pipeline transitions from “signal preprocessing” to “signal interpretation.” Instead of operating directly on voxel-level fMRI data, we now work with region-of-interest-level time-series and the network relationships between those regions. The goal here is to characterize how different brain regions interact across time, and how these interactions organize into distinct whole-brain connectivity patterns.

In the following scripts, we use our now-fully-preprocessed fMRI data to generate neural-activity time-series data within each ROI, and ultimately compute a complete, time-resolved network graph of overall brain activity. Conceptually, this stage converts raw ROI time-series into a sequence of brain “snapshots,” where each snapshot is represented as a network graph: ROIs are nodes; correlations are edges; and graph-theoretic metrics summarize the role of each node and the overall network structure at that moment in time. These graph representations form the core inputs for all subsequent modeling stages (see stages 7-8 below).

There are a few important CONFIG parameters which affect output processing at this stage:

‘correlation_types’: [string] toggles between correlation options

‘centrality_types’: [list of strings] sets the list of (ROI-level) centrality-based graph metrics yielded as output

‘graph_metrics’: [list of strings] sets the list of (brain-wide) connectivity metrics yielded as output

For ‘correlation_type’, the pipeline currently supports two options: Pearson correlation (simple) and partial correlation (complex). The latter is strongly recommended for most contexts, as it better isolates correlations between specific ROI pairs, although it necessarily requires a larger window size (and thus reduces the total number of samples in the output; see full script documentation for more details).

To keep things relatively simple we’ll also request just two types of centrality, and one whole-brain Q-modularity measure, to receive as our graph_metrics output for now.

The above two parameters can be set via the following CONFIG fields:

When we run the script, we’ll see an interim print-out, where we can confirm a few important details:

All our subjects/files ended up with the same number of raw, un-windowed / un-epoched samples (295), which is important to confirm for any analyses that require uniform data length across subjects (e.g. most resting-state analyses). This also simplifies the selection of epoching parameters (see below).

Our current epoching parameters use a sliding window of 30 frames, with a 15-frame (50%) overlap between successive windows, yielding 18 epochs in total per scan file.

Our epoching regime will cause the 10 final trailing samples to be un-used (because our minimum step size is 15 samples).

If we want, we can change the epoching parameters using the ‘epoch_length’ and ‘epoch_overlap’ CONFIG fields below:

Under the current settings, we’ll be getting 18 full, uniformly-sized windows per scan, which is fairly typical for resting-state fMRI datasets. (See the detailed script documentation for notes about how to choose epoching parameters, the various trade-offs arising from increasing the overlap percentage and other epoching parameters, and other important analytical considerations).

(Important caveat: overlapping windows increase the number of samples, but they do not create independent observations—so downstream statistical tests should not treat overlapping windows as independent datapoints.)

When this script finishes running, we should see two folders within our target output subdirectory (‘.../{main}/graph_metrics/’ by default):

The ‘un-epoched’ folder contains both whole-brain and within-ROI measures, computed across the entire span of each scan file. (These are relatively trivial to compute and so are always generated regardless of epoching parameters.) This folder contains three files per file scanned:

A ‘<subject_ID>_<session_ID>_un-epoched_centrality.csv’ table

A ‘<subject_ID>_<session_ID>_un-epoched_global_metrics.csv’ table

A ‘<subject_ID>_<session_ID>_un-epoched_provenance.json’ file

The provenance files contain a complete record of all relevant pipeline parameters used to acquire the graph_metrics dataset, and are largely for auditing purposes only. The centrality tables contain single measures representing the overall centrality of each ROI, measured across the entire span of the fMRI session. The global_metrics tables contain summary statistics (mean, variance, min, max, and skew) for each centrality type computed.

The ‘epoched’ folder is generally the more important class of outputs, as this is the output that is consumed by subsequent stages of the pipeline. This folder contains a subfolder named after the epoching parameters used (e.g. ‘*_window-<epoch_length>_overlap-<epoch_overlap>_*’; e.g. see local directory address bar in image below), and this subfolder contains three files per subject:

A ‘<subject_ID>_<session_ID>_*_timecourses.csv’ table

A ‘<subject_ID>_<session_ID>_*_summaries.csv’ table

A ‘<subject_ID>_<session_ID>_*_provenance.json’ file (diagnostic / parameter logging only)

These ‘epoched’ outputs are the primary inputs for dynamic analyses. Each row in the ‘*_timecourses.csv’ tables corresponds to a single time window and contains network-derived features such as ROI centrality values and whole-brain modularity. The accompanying ‘*_summaries.csv’ tables aggregate these time-resolved features across windows, providing simple descriptive statistics that can be useful for exploratory analyses or quality checks.

Together, these outputs describe how brain network structure fluctuates over time within each scan, forming the basis for identifying recurring connectivity patterns using state-space models in the next stage.

Review Extracted Morphology Measures (optional)

If we like, we can also review the output tables containing our morphology measures from the auxiliary script. Generally, these tables are trivial to fill if the upstream FreeSurfer MRI reconstructions were successful.

Note: Technically, this script can be run any time after script #02 (volumetric reconstruction) and before script #14 (the final data wrangling).

The morphology output tables are stored in the subfolder ‘.../{main}/morphology/’ by default, and should consist of a single .csv table potentially containing multiple sets of columns. The pipeline gives us three options for expressing anatomical volumes:

Raw volume units (mm^3)

The most “direct” measure of volume, but otherwise not usually recommended as this is confounded by subject morphology (e.g. total body-/head size) if used as a statistical correlate across subjects.

Total areal volumes divided by estimated total brain volume (ETIV), which is an estimate of total cranial volume originally calculated during FreeSurfer reconstruction.

ETIV corrects for straightforward body-/head-size confounding and is often the best choice for across-subject comparisons.

Total areal volumes divided by the Brain Segmentation Volume (BSV), which is an estimate of total current brain volume also calculated during reconstruction.

BSV normalizes by CURRENT brain size ('BrainSegVol' in default FreeSurfer parlance); i.e. partially controls for global atrophy; useful for detecting regions that are spared/vulnerable under atrophic conditions, etc.

‘ETIV’ is enabled by default as it is the most generally appropriate, but the pipeline can be configured to export one or more of the above volumetric units using the ‘[morphometry][export_representations]’ CONFIG field, and for each option will export a complete set of anatomical measures whose column names include the prefix substrings [‘__RAW__’, ‘__ETIV__’, or ‘__BSV__’], respectively, for easy auditing and selection.

With our graph metrics computed and anatomical data extracted, next we can feed the time-resolved output files into ML models to help identify recurring dynamical brain states. To prepare for the ML stage, we can now go ahead and run the following scripts:

[09] — Create ML Features

This script turns per-window brain-network features into a clean ML dataset, and writes it to disk in a consistent, reproducible format.

[10] HMM Training & Evaluation

This script trains multiple candidate HMMs across a range of hyperparameter settings (e.g. number of states, covariance structure, optional dimensionality reduction), and automatically selects the best-performing model within each predefined model “family.”

[11] Cross-Model Evaluation

This script evaluates the remaining candidate models across stability and generalizability criteria, and guides the user in selecting a single final model to serve as the canonical representation of brain dynamics for the dataset.

REQUIRED ACTION ITEM: User selection of final “production” model (manual input required)


### Stage 7 — Dynamic State-Space Modeling

In this stage, we model the brain as a system that transitions between a small number of recurring connectivity “states” over time. Each state represents a characteristic pattern of whole-brain connectivity, and the Hidden Markov Model (HMM) learns both what these states look like, and how the brain moves between them.

A useful mental model is that an HMM learns (1) a small library of recurring network “templates,” and (2) the switching dynamics between templates across time windows.

Because each “state” picked up by the model is defined in terms of interpretable network-level features (e.g., ROI-specific centrality, modularity, and connectivity structure), the learned templates can be “decoded” back into anatomically and functionally meaningful patterns (e.g. using the ROI labels generated earlier in the pipeline, etc.). In this way, HMM states serve as a compact, noise-reducing bridge between high-dimensional neural time-series data and scientifically-interpretable descriptions of large-scale brain organization and dynamics — which can also be systematically compared across subjects, time points, or clinical groups.

Script #09 — Freezing datasets for ML training

The first script in this stage collects our time-resolved graph-based measures (the ‘epoched’ data from the previous stage) and creates a stable, frozen dataset used to train our ML models, including uniform training- and test data splits.

As usual, we will use the default pipeline configuration, which is set up to use 1/5th of our input data as the holdout ‘test’ set (see ‘holdout_fraction’ below), and the remaining 80% for model-training. We enforce ‘split_by_subject’ to ensure that all data for a given subject is assigned to either the test- or training set together, i.e. we prevent data leakage by having data from the same subject appear in both datasets. We also enable ‘enforce_class_balance’ to use the ‘group_ID’ identifiers to make sure we keep a consistent balance of control- vs. experimental group subjects across both datasets. We also standardize our data (‘standardize’ == True) to normalize the input data across subjects.

Note: this script automatically attempts to retrieve the target epoched graph-metrics data from the ‘[ML_prep][input_data_directory]’ CONFIG path set here; because this needs to specify a full system path, this field will need to be updated to match the main project folder location on each user’s machine:

…and it specifically searches this directory path for all .csv files containing the CURRENT ‘epoch_length’ and ‘epoch_overlap’ parameters, set in the previous section of the CONFIG file:

i.e. under the above settings, the dataset-freezing script will look for any & all .csv files in ‘.../{main}/graph_metrics/epoched/’ that contain the substring ‘window-30_overlap-15’. If you encounter errors during the “data auditing” section of this script, double-check these CONFIG fields to make sure that they have not been inadvertently changed since you initially generated the epoched data.

When the data are successfully collected, split, and frozen, you should see the following output in the final cell of this script:

We now have a fully-frozen, fully-consistent set of training data to test different models on, and we can continue on to the next script in this stage.

Script #10 — Model training & initial (within-class) model selection:

In this script we’ll begin actually training some candidate models. We begin by loading up the frozen dataset we created during the last script. (Note: under default configurations, the pipeline will automatically reference a pointer file to automatically retrieve the correct target dataset — for more info on this behavior and how to change it, see the script-specific documentation for script #09.) When the frozen dataset is successfully loaded, we should see the following output:

Here the number of individual input sequences matches the total number of scan files in our dataset (193) — so far so good.

Now the script is set up to begin training models, according to the user-specified hyperparameter regimes set within the ‘HMM_protocols’ field of the CONFIG file. Here we can use a variety of “preset” hyperparameter regimes, or we can create our own:

We can select one or more presets to run via the ‘protocols_to_run’ CONFIG field, by providing a list of strings of the target protocol names (exact matches):

For each “protocol”, the script will now train multiple models (specified by the ‘num_initializations’ hyperparameter) for each ‘K’ hyperparameter.

Our total model space will undergo two rounds of selection. First, within each ‘K’, we automatically keep the “best” model, and then summarize each best-K model in a protocol-specific table. Ultimately, for each protocol executed, we’ll see an output block like this:

Here we can see all the output models for the ‘full_pca80’ protocol, i.e. the single best model (out of 10 initializations) for each ‘K’ hyperparameter value we tested. This process will be repeated for each protocol until all are complete.

Once all the protocols have been iterated over (producing one output block like the above for each), we move onto the second round of selection. In this stage, a single model K is chosen from within each protocol, so that we end up with one candidate model from each protocol we used during initial training. These are then presented in a final table at the end of the script:

Here we can see one model (i.e. the best ‘K’ hyperparameter value) per protocol type that we originally asked for in the ‘protocols_to_run’ CONFIG field.

A copy of each winning model in our batch has been saved to a training-dataset-specific subdirectory in our main project folder (‘.../{main}/models_HMM/’ by default), along with full provenance details for every single model trained and each model-selection step.

So far we’ve been able to automate model-selection within individual hyperparameter regimes; this is possible because models from the same “family” can be directly and quantitatively compared (via the AIC / BIC / val-LogL metrics above), and so “within-family” model-selection is a mathematically straightforward process of picking the greatest- or least criterion value (depending on the evaluation metric used).

However, in order to perform the final selection between the different model “families” in the final table above, we can no longer rely on simple arithmetic comparisons across evaluation metrics. Instead, we have to perform a semi-subjective judgment involving various trade-offs involving the total number of model parameters, model stability, generalizability, etc. Therefore, in the next step we’ll perform cross-model evaluation, which will require manual intervention from the user in order to officially select the final model.

Script #11 — Cross-model evaluation & final (user-driven) model selection:

First the script loads the “winning” models from each “family” (protocol preset) from the previous step:

From here, the script will go on to perform multiple splits-tests of the data on each of the models in the batch, including dedicated tests for model stability (consistency of predictions across different subsets of data) and generalizability (how well a model predicts out-of-training-sample data).

Ultimately, we get the following read-out, including several metrics per evaluation test type:

Stability-testing metrics:

Occupancy correlation: Measures whether the fraction of time spent in each state is consistent across random half-splits (higher = better).

Transition Frobenius distance: Measures how similar the state-to-state transition matrices are between splits (lower = better).

Dwell time Jensen–Shannon divergence: Measures whether state duration distributions match across splits (lower = better).

Generalization-testing metrics:

test_logL_per_timepoint_mean: Higher is better, though scores can't be objectively compared across models.

test_logL_per_timepoint_std: Lower is better; indicates LESS variability across training folds.

At this point, the model-evaluation metrics we see in this table are no longer directly commensurate (as each depends on the number of parameters used by each model and other “protocol-specific” variables), and therefore can no longer be compared using single numerical criteria. Instead, we evaluate models holistically across multiple dimensions, balancing stability, generalizability, and model complexity.

Stability metrics assess whether a model learns consistent state definitions when trained on different subsets of the data. High occupancy correlations and low transition/dwell divergences indicate that the inferred states are robust and not overly sensitive to random sampling variation.

Generalizability metrics assess how well a model explains held-out data. Higher average log-likelihood and lower variability across folds suggest that a model captures meaningful structure rather than overfitting noise.

When multiple models perform similarly on these criteria, we generally prefer simpler models (fewer states and fewer effective parameters), as they are easier to interpret and more likely to generalize beyond the current dataset.

Once we select the final model:

It becomes the canonical representation of brain dynamics for this dataset

All downstream features (occupancies, transitions, state metrics, ML predictors) depend on it

In this tutorial, we select the ‘full_pca80’ model because it exhibits excellent stability, strong generalization performance, and relatively low complexity compared to more heavily parameterized alternatives. We can perform this selection in the designated user-input cell immediately below the main evaluation results table, using the ‘FINAL_MODEL_SELECTION’ variable (we only need to edit this single line; everything else will be taken care of for us, provided we entered a valid string value for this selection variable):

With our final model “recipe” selected, the script will finish off by training one final, authoritative version of the model, this time using all of the available training data (rather than splits). After a few minutes, we should see the following output displayed under the final cell in the script:

This final saved model (‘.joblib’ object) will become our final “production” model that we use to assign state labels to the fMRI time-points in our original dataset.


### Stage 8 — Extracting Model-Based Data & Other Feature-Engineering

In the final stages of the pipeline, we transform the abstract HMM state definitions into concrete, subject- and session-level features that can be used directly in statistical analyses or predictive models. These scripts are where we convert “the model’s internal outputs” into interpretable variables (occupancies, transitions, stability/fidelity metrics) that can be merged with anatomy and graph features in the final master table.

Script #12 — Extracting & Engineering Time-Resolved (i.e. State-Level) Features

Summary: This script applies the final selected HMM to the original epoched graph-feature sequences and produces a time-resolved state annotation for each scan (one state label per time window), along with summary measures describing each scan’s state dynamics.

Our chosen model was a 4-state model, so it classifies whole-brain fMRI activity for each of the 18 time-points (epochs) we extracted for each scan file — in this case, using state labels ‘0’-’3’ representing each of our 4 HMM model states. We also collect a variety of other useful summary measures including state-to-state transition probabilities, dwell-time measures, “hard” state-occupancy rates, and more (see script-specific documentation for full feature breakdown):

“Hard” state labels (per window) tell us which connectivity regime was most likely active (highest posterior probability; see below) in each epoch.

Occupancy features tell us how much of the scan was spent in each state (and can be compared across subjects/groups).

Transition features tell us how often the brain switches between states and which state-to-state transitions are the most- and least- common.

Dwell-time features tell us whether states are “sticky” (long contiguous runs) versus brief/transient.

This script also outputs a supplemental table called ‘decoded_time_resolved.csv’, which contains the specific posterior probability (‘postProb’, or in other words, the overall state-likelihood) values the model assigned to each possible state, at each time point per file, as well as the maximum postProb value (i.e. the postProb of whichever state label “won” for a given time-point), and entropy. Entropy, in this context, quantifies how “confident” the model is about its state assignment at a given time point; low entropy indicates that one state strongly dominates the posterior probability distribution, whereas higher entropy indicates ambiguity or overlap between multiple possible states. (Practically, entropy is a compact uncertainty measure: it is low when one state clearly dominates, and higher when the model is “torn” between states. This can be useful for flagging borderline windows or scans with unstable state expression.)

In the ‘decoded_time_resolved’ table, instead of one row per subject / scan file like we had in the previous table, we now have one row per time-point per file, showing the full state-probability landscape within each epoch:

One thing we can notice with this table is that the ‘posterior_probability_max’ column is always extremely high (~98% or higher), indicating that our model is always extremely confident in its state assignments. Uniformly extreme confidence can be a healthy sign (e.g. of clear state separation), but it can also occur when model assumptions strongly constrain the solution — so we will treat this as a diagnostic signal rather than a guarantee of biological correctness per se.

Beyond this, for now, this table is not really needed unless we want to perform an in-depth inspection of how different states’ probabilities fluctuate over time — otherwise, this table is mostly used downstream, by other scripts which compute “soft” state-occupancy rates and other subject-/session-level features (see below).

Script #13 — Extracting & Engineering Additional Subject-/Session-Level Features

This script focuses on how individual subjects or sessions express each learned connectivity state. It computes measures such as “soft” state occupancy (how much time is spent in each state) and fidelity metrics (how closely an individual’s state patterns match the canonical, model-defined state signatures).

In addition to “what states occur,” this script is designed to quantify “how canonical” each subject/session’s state expressions are relative to the dataset-level model. This is useful when we want to treat the global HMM as a reference template and ask how closely each subject conforms to (or deviates from) that template.

The key feature families in this table include:

“Soft” occupancy: Unlike “hard” occupancy (counting only the winning state), “soft” occupancy uses posterior probabilities — so it captures partial membership and produces smoother, often more stable estimates of time spent in each state.

In other words: “hard” occupancy reflects how often the model assigns a time window to a given state after forcing a single-state decision; “soft” occupancy reflects the average posterior probability mass assigned to each state and captures uncertainty in state assignments.

Side note: Soft occupancy is generally preferred for cross-subject comparisons and state interpretation, while hard occupancy is required for transition- and duration-based analyses.

State-profile fidelity (correlation / cosine similarity): These compare a subject/session’s state-specific feature profile to the model’s global state signature; higher values indicate the subject expresses that state in a way that closely matches the canonical pattern learned from the full dataset.

‘MeanAbsZdiff-Global’ (per state): Summarizes how far a subject/session’s state profile deviates from the canonical state definition, averaged across features; larger values indicate more idiosyncratic state expression.

Practically speaking, the tables from this script produce features that can be used as subject-level phenotypes:

“How much does subject X express each state?”

“How typical is subject X’s expression of each state?”

These complement Script #12’s time-resolved labels and transition/dwell metrics.

Script #14 — Final data-wrangling and export

The final script consolidates all outputs from across the pipeline into a single, wide, analysis-ready table. Each row corresponds to a unique [subject_ID × session_ID] combination, and columns span multiple feature families, including anatomical measures, network-graph metrics, and HMM-derived state features, such that we end up with a single row of multi-modal data per scan file..

The final dataset is saved to a subdirectory defined by the ‘final_output_dir’ CONFIG field (‘.../{main}/_final_output/’ by default), written to a file called ‘master_output.csv’. This table is typically very large (e.g. containing several thousands of columns), depending on how many output types were toggled/enabled in our main configuration settings.

Note that the different “families” of output columns (e.g. groups of columns containing anatomical measures, graph metrics, HMM-derived features, etc.) are each given unique prefixes, to facilitate feature-selection or filtering downstream.

This final dataset represents a compact but information-rich summary of brain structure, function, and dynamics, suitable for a wide range of downstream analyses — from classical statistical modeling to machine-learning–based prediction, as well as exploratory or hypothesis-generating studies.

At this point, most users will either (1) run group-level statistics (e.g., group contrasts controlling for covariates), (2) perform correlational analyses against clinical scales or outcomes, (3) build predictive models (classification/regression), or (4) explore data-driven structure (clustering, dimensionality reduction, factor models). The key advantage is that all of these downstream analyses now operate on a single, reproducible, audit-friendly table rather than on scattered intermediate files.



| This tutorial is written for scientists and clinician-scientists who want to run a complete end-to-end fMRI connectivity analysis without needing to deeply modify code. You will get the most value from this pipeline if you are comfortable editing a small number of configuration fields (e.g. filepaths, dataset identifiers, and a few core analysis toggles), and you want a reproducible way to produce connectivity- and state-based features for group comparison, correlational analysis, or predictive modeling. |



![image35.png](images/image35.png)


![image16.png](images/image16.png)


![image30.png](images/image30.png)


![image10.png](images/image10.png)


![image7.png](images/image7.png)


![image39.png](images/image39.png)


![image17.png](images/image17.png)


![image24.png](images/image24.png)


![image12.png](images/image12.png)


![image37.png](images/image37.png)


![image36.png](images/image36.png)


![image14.png](images/image14.png)


![image6.png](images/image6.png)


![image21.png](images/image21.png)


![image22.png](images/image22.png)


![image15.png](images/image15.png)


![image27.png](images/image27.png)


![image40.png](images/image40.png)


![image3.png](images/image3.png)


![image33.png](images/image33.png)


![image2.png](images/image2.png)


![image4.png](images/image4.png)


![image11.png](images/image11.png)


![image9.png](images/image9.png)


![image32.png](images/image32.png)


![image5.png](images/image5.png)


![image25.png](images/image25.png)


![image38.png](images/image38.png)


![image34.png](images/image34.png)


![image18.png](images/image18.png)


![image29.png](images/image29.png)


![image28.png](images/image28.png)


![image31.png](images/image31.png)


![image26.png](images/image26.png)


![image23.png](images/image23.png)


![image13.png](images/image13.png)


![image8.png](images/image8.png)


![image41.png](images/image41.png)


![image19.png](images/image19.png)


![image1.png](images/image1.png)


![image20.png](images/image20.png)
