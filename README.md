# Conformer-Q

Conformer-Q is a workbench for low-energy conformer exploration and quantum-chemical property analysis. The first example is the aphid alarm pheromone `(E)-beta-farnesene` (`EBF`, PubChem CID `5281517`).

The current release targets macOS Apple Silicon (`arm64`). The source tree does not bundle Ketcher, CREST, xTB, or ORCA. Install these third-party programs separately and follow their respective licenses.

## Features

- Includes an isomeric SMILES example for EBF and accepts other SMILES inputs.
- Reuses a local Ketcher editor for molecular drawing, then normalizes and validates the structure with RDKit.
- Calculates molecular formula, molecular weight, cLogP, TPSA, hydrogen-bond counts, rotatable bonds, and related properties with RDKit.
- Diagnoses flexible chains, rings, macrocycles, spiro/bridged systems, possible intramolecular hydrogen bonds, formal charges, and unusual-element risks.
- Uses `CREST iMTD-GC + GFN2-xTB` to search for low-energy conformer ensembles.
- Supports gas-phase and `ALPB(water)` implicit-water conformer searches. The selected environment is passed to CREST and included in task deduplication.
- Uses a reduced CREST quick-search preset by default; users can explicitly select a full `iMTD-GC` search.
- Sends long-running searches to a task center so users can monitor progress, stop tasks, or reopen completed results.
- Generates a task fingerprint from the normalized structure, computational protocol, and formal charge. Duplicate submissions reuse running or completed tasks.
- Displays estimated and remaining time. CREST estimates are initially based on structural complexity and can later use historical runtimes for similar tasks. ORCA estimates start from the number of selected representative conformers and update dynamically from observed runtimes.
- Provides `ETKDGv3 + MMFF94` quick previews for immediate inspection and coarse screening.
- Uses local `3Dmol.js` for interactive three-dimensional conformer viewing, energy-ranked switching, and display-style control.
- Analyzes conformer ensembles by relative-energy windows, recommending `TFD` or heavy-atom `RMSD` clustering based on structural type, and shows the lowest-energy representative of each family.
- Generates 3D ESP, empirical hydrophobic-potential, HOMO, and LUMO maps for selected ORCA-refined conformers, with adjustable color scales/isosurfaces and PNG export.
- Sends the current conformer and selected low-energy ensemble to the optional `Molecular_Derivative_Designer` integration as a 3D design template.
- Exports 3D conformers as `SDF` and energy-ranked results as `CSV`.

`CREST/GFN2-xTB` improves exploration of low-energy regions, but it is not a mathematical enumeration of the continuous potential-energy surface. A low-energy conformer ensemble is not equivalent to a protein-bound conformation and does not prove biological activity. Important candidates should be further refined and re-ranked with ORCA DFT or a higher-level method.

## Start the application

```bash
cd Conformer-Q
python3 -m pip install -r requirements-dev.txt
python3 app.py
```

Open <http://127.0.0.1:5062/> in a browser. The modeling and submission page is available at `/build`; the background task list is available at <http://127.0.0.1:5062/tasks>.

You can also double-click the Conformer-Q launcher script. The launcher checks only Python, Flask, and RDKit. If Ketcher, CREST, xTB, or ORCA is missing, structure diagnostics and quick previews remain available and the corresponding advanced features show installation guidance in the interface.

## Typical workflow

1. `/build`: draw or enter a structure, confirm the 2D structure, review diagnostics, choose the sampling environment and computational protocol, and inspect the time estimate.
2. `/tasks/<run_id>`: inspect the stages, log summary, runtime, and stop controls for one background task. `/tasks` lists and can delete completed or stopped task records.
3. `/results/<run_id>`: analyze completed conformers in 3D, review 2D confirmation, properties, energy ranking, and exported files.

## Install third-party programs

### Ketcher molecular editor

1. Download a standalone package from [Ketcher Releases](https://github.com/epam/ketcher/releases).
2. Extract it and make sure `index.html` is located at:

   ```text
   Conformer-Q/engines/ketcher-standalone-3.7.0/standalone/index.html
   ```

3. If it is installed elsewhere, set the path before starting the app:

   ```bash
   export CONFORMER_Q_KETCHER_DIR="/path/to/ketcher/standalone"
   ```

Ketcher is optional; users can still enter isomeric SMILES directly in the application.

### CREST and xTB conformer search

Obtain the programs from the official [CREST](https://github.com/crest-lab/crest) and [xTB](https://github.com/grimme-lab/xtb) projects. With conda, they can be installed into an existing environment with:

```bash
conda install -c conda-forge crest xtb
```

If `crest` and `xtb` are on `PATH`, Conformer-Q discovers them automatically. Otherwise, set explicit paths:

```bash
export CONFORMER_Q_CREST_BIN="/path/to/crest"
export CONFORMER_Q_XTB_BIN="/path/to/xtb"
```

Conformer-Q invokes CREST using `--legacy --gfn2 -xnam <xtb>` and supports both gas-phase and `ALPB(water)` implicit-solvent searches.

### ORCA refinement

Obtain an appropriate macOS version from the [official ORCA channel](https://www.faccts.de/orca/) and follow its usage and licensing requirements. The current workflow has been validated with ORCA `6.0.1`. The extracted directory can be placed at:

```text
Conformer-Q/engines/orca-6.0.1/orca
```

You can also specify the installation with environment variables:

```bash
export CONFORMER_Q_ORCA_DIR="/path/to/orca_6_0_1"
# Or specify the executable directly:
export CONFORMER_Q_ORCA_BIN="/path/to/orca_6_0_1/orca"
```

ORCA is not distributed with this repository. Read and follow the ORCA license. Without ORCA, CREST searches and quick previews can still run independently.

### Executable discovery order

Conformer-Q checks `CONFORMER_Q_*` environment variables first, then installation directories under `engines/`, and finally the system `PATH`. Local third-party installation directories are excluded by `.gitignore` and are not committed to Git.

## Protocol logic

The application diagnoses the structure before running the selected conformer protocol:

| Structure type | Recommended strategy |
| --- | --- |
| Flexible chain, such as EBF | `CREST/GFN2-xTB` primary search; optionally cross-check with systematic torsion enumeration |
| Relatively rigid ring system | Generate an initial geometry quickly, then confirm with xTB/ORCA |
| Spiro or bridged system | Preserve topology and stereochemistry, then use CREST search and ORCA refinement |
| Macrocycle | Run multiple independent CREST searches, consider solvent models, and refine representative structures |
| Possible intramolecular hydrogen bonds or charged molecules | Study relevant solvent, pH, and microstate definitions separately |

CREST receives the formal charge from the submitted structure. For compounds with multiple protonation states or tautomers, define the target microstate first and calculate each relevant state separately.

The conformer-search environment is part of the computational protocol:

| Environment | CREST parameter | Interpretation |
| --- | --- | --- |
| Gas phase | No solvent parameter by default | Low-energy conformer preferences for an isolated molecule |
| Implicit water | `-alpb water` | Low-energy conformer preferences in a continuum-water approximation |

Gas-phase and implicit-water searches produce different task fingerprints and are never reused across environments. Implicit water is not an explicit-water simulation and does not represent a protein-binding pocket. Higher-accuracy refinement should use the same environment definition.

## Result analysis

The results page preserves the original conformer ensemble and energy ranking while providing family analysis for interpretation and refinement selection:

| Analysis | Purpose |
| --- | --- |
| Energy window | Examine the size of the low-energy ensemble within `0.5 / 1.0 / 2.0 / 5.0 kcal/mol` windows |
| Heavy-atom RMSD clustering | General grouping by overall geometric similarity |
| TFD clustering | Compare conformer families for flexible molecules dominated by torsional freedom |
| Family representative | The lowest-energy conformer in each cluster, available for 3D inspection and refinement selection |

The interface recommends an analysis mode from the structure diagnosis: flexible chains default to `TFD`; rigid, bridged, spiro, and macrocyclic systems initially use heavy-atom `RMSD`. Macrocycles, intramolecular hydrogen-bond systems, and charged molecules still require specialized features and independent-search validation; clustering alone is not a substitute for those checks.

Results are organized into two stages. Stage one displays CREST screening results and lets users select low-energy family representatives. Stage two preselects the first `10` representatives from the low-energy families, while allowing the selection to be changed, and can submit an `ORCA 6.0.1 r2SCAN-3c Opt + Freq` task. Gas-phase candidates are refined in the gas phase; implicit-water candidates use `CPCM(Water)` to maintain a continuum-water definition. The thermodynamic analysis temperature is configurable and defaults to `298.15 K`.

After ORCA refinement, the page prioritizes the final low-energy conformer summary. The 3D refinement view includes a fixed-height, scrollable ranking of `Delta G` and Boltzmann populations, allowing direct conformer switching and overlay selection.

The selected 3D conformer has a decision-oriented interpretation panel. It distinguishes evidence levels and answers why the conformer matters, how it differs from the dominant conformer `#1`, and whether it should be retained:

| Stage | Displayed information | Interpretation boundary |
| --- | --- | --- |
| CREST/xTB or quick preview | Significant dihedral flips and folding changes relative to the current lowest-energy conformer, plus family-level guidance on whether refinement is worthwhile | Useful for screening and interpretation, not a final free-energy conclusion |
| After ORCA refinement | `ΔG`, Boltzmann population at the selected temperature, and distinctive shape relative to the dominant conformer, with a suggestion about retaining it for property analysis or docking | Applies only to the sampled conformers that completed refinement |

By default, significant changes are reported for key dihedrals differing from `#1` by at least `30 deg`. `Folded`/`extended` state, principal-axis dimensions, and radius of gyration remain available as expandable measurement evidence. Intramolecular hydrogen bonds are reported only as distance-and-angle candidates. The displayed `TPSA` is the topological polar surface area of the connected structure, not the exposed 3D polar surface area of one conformer.

After refinement, the application removes unstable minima with significant imaginary frequencies (current threshold: `< -20 cm-1`), deduplicates optimized structures using structure-appropriate `TFD` or heavy-atom `RMSD`, then ranks them by relative Gibbs free energy `ΔG` and calculates Boltzmann populations at the selected temperature. The final low-energy set is defined by default as the independent conformers needed to reach a cumulative Boltzmann population of `95%`. The results table shows both the originating CREST energy and the ORCA `ΔG`.

If an ORCA task is stopped after at least one representative has completed, the task page can open a partial-results analysis. Its `ΔG` ranking and Boltzmann populations apply only within the completed subset and are explicitly not presented as a final low-energy conformer-set conclusion.

The ORCA results page supports multicolor overlays of `2-5` independent refined conformers, using the lowest-`ΔG` conformer as the default reference. For flexible molecules such as EBF, the application identifies a locally rigid fragment defined by double bonds or rings and includes directly adjacent heavy atoms as alignment anchors. Users can switch to whole-molecule heavy-atom alignment or select a specific fragment. The page reports anchor RMSD and full-molecule RMSD after local alignment: the former describes the common-fragment fit, while the latter shows folding differences in the remaining chain. This tool explains conformational differences; it does not replace torsional analysis or free-energy ranking.

## 3D property maps

The ORCA results page provides a 3D property-mapping module for the selected refined conformer:

| Map | Data source | Interpretation boundary |
| --- | --- | --- |
| ESP | `orca_vpot` evaluates the electrostatic potential from the `r2SCAN-3c` wavefunction and maps it onto a VDW visualization surface | Blue indicates negative potential and red indicates positive potential; the default symmetric scale is `+/- 0.05 a.u.` |
| HOMO / LUMO | `orca_plot` exports orbital cube files from the conformer's `.gbw` wavefunction | Orange/blue show orbital phase, not electrostatic charge; keep the same isovalue when comparing images |
| Empirical hydrophobic potential | A 3D Gaussian projection of RDKit Wildman-Crippen atomic cLogP contributions | The blue-to-brown scale shows relative distribution; brown indicates hydrophobic character. This is not an ORCA quantum-chemical observable |

The first map generation runs ORCA post-processing and caches cube data. ESP requires an additional calculation of the grid potential. Reopening the same conformer and map reuses the cache. During export, the view is temporarily redrawn at high resolution; PNG files include the map type, conformer number, and current scale or isovalue so the plotting conditions are preserved.

For flexible molecules such as EBF, low-frequency vibrations can strongly affect entropy and conformer free-energy ranking. The current `Opt + Freq` workflow provides an executable first-pass thermodynamic screen; further validation should evaluate low-frequency treatment and higher-level single-point corrections.

CREST coordinate parsing preserves the atom-order template used when the task was submitted. Canonical SMILES is used only for structure identification and task deduplication, preventing atom reordering during canonicalization from mismatching 3D coordinates and bond connectivity.

## Task management

Long-running tasks write `task.json` files in their output directories containing the structure fingerprint, protocol, status, start time, estimated duration, and process identifier. Tasks can be followed after a page refresh or reopening the application, and completed results can be reloaded from the task page. Time estimates are scheduling guidance, not guarantees of computational completion time; unfamiliar structure classes may have larger errors before historical samples are available.

## EBF data source

- PubChem CID 5281517: <https://pubchem.ncbi.nlm.nih.gov/compound/5281517>
- Structure input: `CC(=CCC/C(=C/CCC(=C)C=C)/C)C`
- Molecular formula: `C15H24`

## Optional integration

The derivative-design entry point on the results page connects to `Molecular_Derivative_Designer` at `http://127.0.0.1:5063/`. This integration is optional and can be ignored when using Conformer-Q by itself.

## Roadmap

1. Add a CREST search-log viewer and convergence reports for multiple independent searches to improve runtime estimates.
2. Add enumeration of reasonable protonation states and tautomers.
3. Add convergence reports for repeated independent searches and higher-level single-point corrections.
4. Add controlled analog design, project saving, and property-comparison tables.

## Third-party frontend components

- `web/vendor/3Dmol-min.js`: `3Dmol.js 2.4.2`, used for offline browser-based visualization of 3D conformers.
- The license text is stored in `web/vendor/3Dmol-LICENSE.txt`.
