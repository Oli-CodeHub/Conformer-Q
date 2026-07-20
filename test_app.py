import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rdkit import RDLogger

import app as workbench
import orca_refinement_worker
from app import EBF, app


class MolecularWorkbenchTest(unittest.TestCase):
    def setUp(self):
        self.temporary_output = tempfile.TemporaryDirectory()
        self.temporary_tools = tempfile.TemporaryDirectory()
        self.temporary_ketcher = tempfile.TemporaryDirectory()
        self.previous_output = workbench.OUTPUT_DIR
        self.previous_tool_paths = {
            "CREST_BIN": workbench.CREST_BIN,
            "XTB_BIN": workbench.XTB_BIN,
            "COMPUTE_BIN_DIRS": workbench.COMPUTE_BIN_DIRS,
            "ORCA_BIN": workbench.ORCA_BIN,
            "ORCA_DIR": workbench.ORCA_DIR,
            "ORCA_PLOT_BIN": workbench.ORCA_PLOT_BIN,
            "ORCA_VPOT_BIN": workbench.ORCA_VPOT_BIN,
            "KETCHER_DIR": workbench.KETCHER_DIR,
        }
        tool_dir = Path(self.temporary_tools.name) / "bin"
        tool_dir.mkdir()
        for name in ("crest", "xtb", "orca", "orca_plot", "orca_vpot"):
            executable = tool_dir / name
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
        workbench.CREST_BIN = tool_dir / "crest"
        workbench.XTB_BIN = tool_dir / "xtb"
        workbench.COMPUTE_BIN_DIRS = [tool_dir]
        workbench.ORCA_BIN = tool_dir / "orca"
        workbench.ORCA_DIR = tool_dir
        workbench.ORCA_PLOT_BIN = tool_dir / "orca_plot"
        workbench.ORCA_VPOT_BIN = tool_dir / "orca_vpot"
        ketcher_dir = Path(self.temporary_ketcher.name)
        (ketcher_dir / "index.html").write_text("<!doctype html>", encoding="utf-8")
        workbench.KETCHER_DIR = ketcher_dir
        workbench.OUTPUT_DIR = Path(self.temporary_output.name)
        workbench.JOBS.clear()
        self.client = app.test_client()

    def tearDown(self):
        for job in workbench.JOBS.values():
            if not job.get("closed"):
                job["log_handle"].close()
        workbench.JOBS.clear()
        workbench.OUTPUT_DIR = self.previous_output
        for name, value in self.previous_tool_paths.items():
            setattr(workbench, name, value)
        self.temporary_output.cleanup()
        self.temporary_tools.cleanup()
        self.temporary_ketcher.cleanup()

    def test_ebf_conformer_generation_and_downloads(self):
        response = self.client.post(
            "/api/analyze",
            json={"name": EBF["name"], "smiles": EBF["smiles"], "num_conformers": 10, "seed": 20260525},
        )
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["properties"]["formula"], "C15H24")
        self.assertEqual(result["profile"]["topology"], "柔性链状分子")
        self.assertGreater(result["retained_conformers"], 0)
        self.assertEqual(result["conformers"][0]["relative_energy_kcal_mol"], 0.0)
        persisted = self.client.get(f"/api/results/{result['run_id']}").get_json()
        self.assertEqual(persisted["run_id"], result["run_id"])
        conformer_response = self.client.get(f"/api/conformer/{result['run_id']}/1")
        self.assertEqual(conformer_response.status_code, 200)
        self.assertIn(b"V2000", conformer_response.data)
        for path in result["files"].values():
            download_response = self.client.get(path)
            self.assertEqual(download_response.status_code, 200)
            download_response.close()

    def test_ebf_ensemble_analysis_recommends_torsion_families(self):
        result = self.client.post(
            "/api/analyze",
            json={"name": EBF["name"], "smiles": EBF["smiles"], "num_conformers": 10, "seed": 20260525},
        ).get_json()
        response = self.client.get(f"/api/results/{result['run_id']}/ensemble?energy_window=5")
        self.assertEqual(response.status_code, 200)
        analysis = response.get_json()
        self.assertEqual(analysis["topology"], "柔性链状分子")
        self.assertEqual(analysis["recommendation"]["method"], "tfd")
        self.assertEqual(analysis["method"], "tfd")
        self.assertGreater(analysis["family_count"], 0)
        self.assertLessEqual(analysis["family_count"], analysis["selected_conformers"])
        self.assertIn("representative_rank", analysis["families"][0])

    def test_conformer_interpretation_reports_ebf_geometry_features(self):
        result = self.client.post(
            "/api/analyze",
            json={"name": EBF["name"], "smiles": EBF["smiles"], "num_conformers": 4, "seed": 20260525},
        ).get_json()
        response = self.client.get(f"/api/results/{result['run_id']}/interpretation/1")
        self.assertEqual(response.status_code, 200)
        interpretation = response.get_json()
        self.assertEqual(interpretation["evidence_level"], "预览证据")
        self.assertEqual(interpretation["decision"]["label"], "最低能参考候选")
        self.assertIn("高级精修", interpretation["decision"]["recommendation"])
        self.assertGreater(len(interpretation["key_torsions"]), 0)
        self.assertGreater(interpretation["shape"]["length_angstrom"], interpretation["shape"]["depth_angstrom"])
        self.assertGreater(interpretation["shape"]["end_to_end_distance_angstrom"], 0)
        self.assertIn("折叠", interpretation["shape"]["label"])
        self.assertEqual(interpretation["intramolecular_hydrogen_bonds"], [])
        self.assertEqual(interpretation["tpsa_angstrom_squared"], 0.0)

    def test_refinement_plan_saves_selected_family_representatives(self):
        result = self.client.post(
            "/api/analyze",
            json={"name": EBF["name"], "smiles": EBF["smiles"], "num_conformers": 10, "seed": 20260525},
        ).get_json()
        analysis = self.client.get(f"/api/results/{result['run_id']}/ensemble?energy_window=5").get_json()
        ranks = [family["representative_rank"] for family in analysis["families"][:2]]
        response = self.client.post(
            f"/api/results/{result['run_id']}/refinement-plan",
            json={
                "method": analysis["method"],
                "threshold": analysis["threshold"],
                "energy_window": 5,
                "representative_ranks": ranks,
            },
        )
        self.assertEqual(response.status_code, 200)
        plan = response.get_json()
        self.assertEqual(plan["status"], "prepared")
        self.assertEqual(plan["selection_count"], len(ranks))
        self.assertEqual(plan["next_stage"], "ORCA 优化、频率计算与自由能分析")
        stored = self.client.get(f"/api/results/{result['run_id']}/refinement-plan").get_json()
        self.assertEqual(stored["selected_conformers"], plan["selected_conformers"])

    def test_orca_refinement_task_is_created_and_returns_refined_result(self):
        class RunningProcess:
            pid = 141421

            def poll(self):
                return None

        source = self.client.post(
            "/api/analyze",
            json={"name": EBF["name"], "smiles": EBF["smiles"], "num_conformers": 4, "seed": 20260525},
        ).get_json()
        analysis = self.client.get(f"/api/results/{source['run_id']}/ensemble?energy_window=5").get_json()
        representative = analysis["families"][0]["representative_rank"]
        self.client.post(
            f"/api/results/{source['run_id']}/refinement-plan",
            json={
                "method": analysis["method"],
                "threshold": analysis["threshold"],
                "energy_window": 5,
                "representative_ranks": [representative],
            },
        )
        with patch.object(workbench.subprocess, "Popen", return_value=RunningProcess()):
            response = self.client.post(
                f"/api/results/{source['run_id']}/refinement/start",
                json={"temperature_kelvin": 310.15},
            )
        self.assertEqual(response.status_code, 200)
        task_view = response.get_json()
        self.assertEqual(task_view["task_type"], "orca")
        refinement_dir = workbench.OUTPUT_DIR / task_view["run_id"]
        self.assertIn("r2SCAN-3c", (refinement_dir / "conf_001.inp").read_text())
        self.assertIn("Freq", (refinement_dir / "conf_001.inp").read_text())
        self.assertIn("%freq Temp 310.15 end", (refinement_dir / "conf_001.inp").read_text())
        self.assertNotIn("CPCM(Water)", (refinement_dir / "conf_001.inp").read_text())

        task = workbench.load_task(task_view["run_id"])
        (refinement_dir / "conf_001.xyz").write_text((refinement_dir / "conf_001_input.xyz").read_text())
        task["status"] = "completed"
        task["completed_conformers"] = [
            {
                "source_rank": representative,
                "family_id": 1,
                "energy_hartree": -100.0,
                "gibbs_free_energy_hartree": -99.99,
                "xyz_file": "conf_001.xyz",
            }
        ]
        task["progress"] = "ORCA 精修完成。"
        task["completed_at"] = task["created_at"] + 5
        workbench.save_task(task)
        refined = self.client.get(f"/api/results/{task_view['run_id']}").get_json()
        self.assertEqual(refined["method"], workbench.ORCA_METHOD)
        self.assertEqual(refined["conformers"][0]["relative_energy_kcal_mol"], 0.0)
        self.assertEqual(refined["temperature_kelvin"], 310.15)
        self.assertEqual(refined["conformers"][0]["boltzmann_population_percent"], 100.0)
        self.assertEqual(refined["stationary_point_validation"]["excluded_count"], 0)
        self.assertIn("相对吉布斯自由能", refined["method_scope"])
        interpretation = self.client.get(f"/api/results/{task_view['run_id']}/interpretation/1").get_json()
        self.assertEqual(interpretation["evidence_level"], "精修证据")
        self.assertEqual(interpretation["energy"]["label"], "ΔG")
        self.assertEqual(interpretation["energy"]["boltzmann_population_percent"], 100.0)
        self.assertEqual(interpretation["decision"]["label"], "主导参考构象")

    def test_orca_water_refinement_input_uses_cpcm(self):
        text = workbench.orca_input_text(0, "water", "candidate.xyz", 298.15)
        self.assertIn("CPCM(Water)", text)
        self.assertIn("Freq", text)
        self.assertIn("%freq Temp 298.15 end", text)
        self.assertIn("* xyzfile 0 1 candidate.xyz", text)

    def test_orca_worker_preserves_symlink_path_for_orca_binary(self):
        link_path = Path("/tmp/conformer-q-tools/runtime/orca-6.0.1/orca")
        resolved_path = orca_refinement_worker.orca_executable_path(str(link_path))
        self.assertEqual(resolved_path, link_path)

    def test_stopped_orca_task_exposes_completed_subset_as_partial_result(self):
        source = self.client.post(
            "/api/analyze",
            json={"name": EBF["name"], "smiles": EBF["smiles"], "num_conformers": 3, "seed": 20260525},
        ).get_json()
        run_id = "orca-stopped-partial"
        run_dir = workbench.OUTPUT_DIR / run_id
        run_dir.mkdir()
        supplier = workbench.Chem.SDMolSupplier(str(workbench.OUTPUT_DIR / source["run_id"] / "conformers.sdf"), removeHs=False)
        workbench.Chem.MolToXYZFile(supplier[0], str(run_dir / "conf_001.xyz"), confId=0)
        task = {
            "task_type": "orca",
            "run_id": run_id,
            "fingerprint": "stopped-partial",
            "source_run_id": source["run_id"],
            "name": "stopped partial refinement",
            "canonical_smiles": source["canonical_smiles"],
            "sampling_environment": "gas",
            "temperature_kelvin": 298.15,
            "selected_conformers": [{"source_rank": 1}, {"source_rank": 2}],
            "completed_conformers": [
                {
                    "source_rank": 1,
                    "family_id": 1,
                    "energy_hartree": -100.0,
                    "gibbs_free_energy_hartree": -99.99,
                    "xyz_file": "conf_001.xyz",
                },
            ],
            "status": "stopped",
            "progress": "ORCA 精修任务已停止。",
            "created_at": 1,
            "completed_at": 2,
            "estimated_seconds": 2400,
            "estimate_basis": "test",
        }
        workbench.save_task(task)
        response = self.client.get(f"/api/results/{run_id}")
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertTrue(result["is_partial_result"])
        self.assertEqual(result["completed_refinement_count"], 1)
        self.assertEqual(result["requested_conformers"], 2)
        self.assertIn("不能视为最终低能构象集合", result["method_scope"])
        interpretation = self.client.get(f"/api/results/{run_id}/interpretation/1").get_json()
        self.assertEqual(interpretation["decision"]["label"], "部分结果参考构象")
        self.assertIn("待其余精修完成", interpretation["decision"]["recommendation"])

    def test_orca_refined_conformers_can_be_aligned_for_overlay(self):
        source = self.client.post(
            "/api/analyze",
            json={"name": EBF["name"], "smiles": EBF["smiles"], "num_conformers": 3, "seed": 20260525},
        ).get_json()
        run_id = "orca-overlay-check"
        run_dir = workbench.OUTPUT_DIR / run_id
        run_dir.mkdir()
        source_sdf = workbench.OUTPUT_DIR / source["run_id"] / "conformers.sdf"
        (run_dir / "conformers.sdf").write_bytes(source_sdf.read_bytes())
        overlay_result = {
            **source,
            "run_id": run_id,
            "task_type": "orca",
            "is_partial_result": True,
            "conformers": [
                {
                    **source["conformers"][0],
                    "rank": 1,
                    "relative_gibbs_free_energy_kcal_mol": 0.0,
                    "boltzmann_population_percent": 60.0,
                },
                {
                    **source["conformers"][1],
                    "rank": 2,
                    "relative_gibbs_free_energy_kcal_mol": 0.4,
                    "boltzmann_population_percent": 40.0,
                },
            ],
        }
        workbench.save_result_payload(run_id, overlay_result)
        response = self.client.post(
            f"/api/results/{run_id}/overlay",
            json={"ranks": [1, 2], "reference_rank": 1},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["reference_rank"], 1)
        self.assertTrue(data["is_partial_result"])
        self.assertEqual(len(data["conformers"]), 2)
        self.assertEqual(data["conformers"][0]["rmsd_angstrom"], 0.0)
        self.assertGreaterEqual(data["conformers"][1]["rmsd_angstrom"], 0.0)
        self.assertEqual(data["selected_anchor"]["kind"], "local")
        self.assertGreater(len(data["anchor_options"]), 2)
        self.assertIn("局部锚点", data["alignment_method"])
        self.assertIn("whole_rmsd_after_alignment_angstrom", data["conformers"][1])
        self.assertIn("V2000", data["conformers"][1]["mol_block"])
        whole_response = self.client.post(
            f"/api/results/{run_id}/overlay",
            json={"ranks": [1, 2], "reference_rank": 1, "anchor_id": "whole"},
        )
        self.assertEqual(whole_response.status_code, 200)
        whole_data = whole_response.get_json()
        self.assertEqual(whole_data["selected_anchor"]["kind"], "whole")
        self.assertIn("整体重原子", whole_data["alignment_method"])

    def test_orca_property_map_generates_empirical_hydrophobic_cube_for_selected_rank(self):
        source = self.client.post(
            "/api/analyze",
            json={"name": EBF["name"], "smiles": EBF["smiles"], "num_conformers": 2, "seed": 20260525},
        ).get_json()
        run_id = "orca-property-check"
        run_dir = workbench.OUTPUT_DIR / run_id
        run_dir.mkdir()
        (run_dir / "conformers.sdf").write_bytes(
            (workbench.OUTPUT_DIR / source["run_id"] / "conformers.sdf").read_bytes()
        )
        (run_dir / "conf_001.gbw").write_text("wavefunction placeholder")
        workbench.save_task(
            {
                "task_type": "orca",
                "run_id": run_id,
                "fingerprint": "property-result",
                "source_run_id": source["run_id"],
                "name": "property map test",
                "canonical_smiles": source["canonical_smiles"],
                "sampling_environment": "gas",
                "temperature_kelvin": 298.15,
                "selected_conformers": [{"source_rank": 1}],
                "completed_conformers": [{"source_rank": 1, "xyz_file": "conf_001.xyz"}],
                "status": "completed",
                "progress": "complete",
                "created_at": 1,
                "completed_at": 2,
                "estimated_seconds": 1,
                "estimate_basis": "test",
            }
        )
        result = {
            **source,
            "run_id": run_id,
            "task_type": "orca",
            "conformers": [
                {
                    **source["conformers"][0],
                    "rank": 1,
                    "source_rank": 1,
                    "relative_gibbs_free_energy_kcal_mol": 0.0,
                    "boltzmann_population_percent": 100.0,
                }
            ],
        }
        workbench.save_result_payload(run_id, result)
        response = self.client.post(
            f"/api/results/{run_id}/property-map",
            json={"rank": 1, "map_type": "hydrophobic"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["render_mode"], "surface")
        self.assertIn("经验", data["label"])
        cube = self.client.get(data["cube_url"])
        self.assertEqual(cube.status_code, 200)
        self.assertIn(b"Wildman-Crippen", cube.data)
        cube.close()

    def test_frontier_orbital_parser_uses_final_contiguous_orbital_table(self):
        output = Path(self.temporary_output.name) / "frontier.out"
        output.write_text(
            "ORBITAL ENERGIES\n  NO   OCC E(Eh) E(eV)\n 0 2.0000 -0.4 -10.0\n 1 0.0000 0.1 2.0\n\n"
            "ORBITAL ENERGIES\n  NO   OCC E(Eh) E(eV)\n 56 2.0000 -0.2 -5.5\n 57 0.0000 -0.05 -1.5\n\n"
            " 114 2.0000 0.0001 0.002\n",
            encoding="utf-8",
        )
        orbitals = workbench.parse_frontier_orbitals(output)
        self.assertEqual(orbitals["homo"]["index"], 56)
        self.assertEqual(orbitals["lumo"]["index"], 57)
        self.assertEqual(orbitals["gap_ev"], 4.0)

    def test_orca_duplicate_optimized_geometries_are_merged(self):
        source = self.client.post(
            "/api/analyze",
            json={"name": EBF["name"], "smiles": EBF["smiles"], "num_conformers": 4, "seed": 20260525},
        ).get_json()
        run_id = "orca-duplicate-check"
        run_dir = workbench.OUTPUT_DIR / run_id
        run_dir.mkdir()
        first_xyz = run_dir / "conf_001.xyz"
        supplier = workbench.Chem.SDMolSupplier(str(workbench.OUTPUT_DIR / source["run_id"] / "conformers.sdf"), removeHs=False)
        workbench.Chem.MolToXYZFile(supplier[0], str(first_xyz), confId=0)
        (run_dir / "conf_002.xyz").write_text(first_xyz.read_text())
        (run_dir / "conf_003.xyz").write_text(first_xyz.read_text())
        task = {
            "task_type": "orca",
            "run_id": run_id,
            "fingerprint": "duplicate-result",
            "source_run_id": source["run_id"],
            "name": "duplicate optimized geometries",
            "canonical_smiles": source["canonical_smiles"],
            "sampling_environment": "gas",
            "temperature_kelvin": 298.15,
            "selected_conformers": [{"source_rank": 1}, {"source_rank": 2}, {"source_rank": 3}],
            "completed_conformers": [
                {"source_rank": 1, "family_id": 1, "energy_hartree": -100.0, "gibbs_free_energy_hartree": -99.99, "xyz_file": "conf_001.xyz"},
                {"source_rank": 2, "family_id": 2, "energy_hartree": -99.9, "gibbs_free_energy_hartree": -99.98, "xyz_file": "conf_002.xyz"},
                {
                    "source_rank": 3,
                    "family_id": 3,
                    "energy_hartree": -100.1,
                    "gibbs_free_energy_hartree": -100.09,
                    "significant_imaginary_frequencies_cm_1": [-45.0],
                    "xyz_file": "conf_003.xyz",
                },
            ],
            "status": "completed",
            "progress": "complete",
            "created_at": 1,
            "completed_at": 2,
            "estimated_seconds": 1,
            "estimate_basis": "test",
        }
        workbench.save_task(task)
        result = self.client.get(f"/api/results/{run_id}").get_json()
        self.assertEqual(result["stationary_point_validation"]["excluded_count"], 1)
        self.assertEqual(result["deduplication"]["input_count"], 2)
        self.assertEqual(result["deduplication"]["independent_count"], 1)
        self.assertEqual(result["final_ensemble_count"], 1)
        self.assertEqual(result["conformers"][0]["merged_count"], 2)

    def test_macrocycle_ensemble_recommendation_uses_rmsd_initial_screen(self):
        profile = workbench.profile_molecule(workbench.Chem.MolFromSmiles("C1CCCCCCCCCC1"))
        recommendation = workbench.ensemble_analysis_recommendation(profile)
        self.assertEqual(recommendation["method"], "rmsd")
        self.assertIn("宏环", recommendation["label"])

    def test_invalid_smiles_reports_user_error(self):
        RDLogger.DisableLog("rdApp.error")
        try:
            response = self.client.post("/api/analyze", json={"smiles": "not_a_structure"})
        finally:
            RDLogger.EnableLog("rdApp.error")
        self.assertEqual(response.status_code, 400)
        self.assertIn("SMILES", response.get_json()["error"])

    def test_missing_conformer_is_rejected(self):
        response = self.client.get("/api/conformer/missing-run/1")
        self.assertEqual(response.status_code, 404)

    def test_local_3dmol_viewer_asset_is_available(self):
        response = self.client.get("/vendor/3Dmol-min.js")
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.data), 100_000)
        response.close()
        style_response = self.client.get("/styles/workspace.css")
        self.assertEqual(style_response.status_code, 200)
        self.assertIn(b"--accent: #007aff", style_response.data)
        style_response.close()

    def test_profiler_recognizes_macrocycle_and_spiro_topologies(self):
        macrocycle = self.client.post("/api/profile", json={"smiles": "C1CCCCCCCCCC1"}).get_json()["profile"]
        spiro = self.client.post("/api/profile", json={"smiles": "C1CCC2(CC1)CCCC2"}).get_json()["profile"]
        self.assertEqual(macrocycle["topology"], "宏环分子")
        self.assertGreater(macrocycle["macrocycles"], 0)
        self.assertEqual(spiro["topology"], "螺环或桥环骨架")
        self.assertGreater(spiro["spiro_atoms"], 0)

    def test_compute_tools_are_detected(self):
        tools = self.client.get("/api/tools").get_json()
        self.assertTrue(tools["ketcher"]["available"])
        self.assertTrue(tools["crest"]["available"])
        self.assertTrue(tools["xtb"]["available"])
        self.assertTrue(tools["orca"]["available"])
        self.assertEqual(tools["orca"]["validated_version"], "6.0.1")
        self.assertEqual(tools["orca"]["install_relative_path"], "engines/orca-6.0.1")

    def test_task_page_is_available(self):
        for path, text in [("/build", "新建计算"), ("/tasks", "任务中心")]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn(text.encode(), response.data)
            response.close()
        self.assertEqual(self.client.get("/").status_code, 302)
        response = self.client.get("/ketcher/index.html")
        self.assertEqual(response.status_code, 200)
        response.close()
        results_page = self.client.get("/results/example")
        self.assertIn("阶段二 · 高级精修准备".encode(), results_page.data)
        self.assertIn("当前构象判断".encode(), results_page.data)
        self.assertIn(b"loadInterpretation", results_page.data)
        self.assertIn("相对主导构象 #1 的变化".encode(), results_page.data)
        self.assertIn("是否保留".encode(), results_page.data)
        self.assertIn("保存候选方案".encode(), results_page.data)
        self.assertIn("提交 ORCA 高级精修任务".encode(), results_page.data)
        self.assertIn("正在检查 ORCA 精修引擎".encode(), results_page.data)
        self.assertIn(b"loadOrcaStatus", results_page.data)
        self.assertIn("叠合已选构象".encode(), results_page.data)
        self.assertIn("自动局部锚点".encode(), results_page.data)
        self.assertIn("显示氢原子".encode(), results_page.data)
        self.assertIn("三维性质作图".encode(), results_page.data)
        self.assertIn(b"generatePropertyMap", results_page.data)
        self.assertIn(b'VolumeData(propertyCube,"cube")', results_page.data)
        self.assertIn(b"Gradient.CustomLinear", results_page.data)
        self.assertIn("构象能量排序".encode(), results_page.data)
        self.assertIn(b"orcaRows", results_page.data)
        self.assertIn(b"fitPropertyView", results_page.data)
        self.assertIn("导出 PNG 图片".encode(), results_page.data)
        self.assertIn(b"formulaMarkup", results_page.data)
        self.assertLess(results_page.data.index(b'id="finalAnalysis"'), results_page.data.index(b'id="samplingStage"'))
        results_page.close()

    def test_missing_orca_returns_installation_guidance(self):
        source = self.client.post(
            "/api/analyze",
            json={"name": EBF["name"], "smiles": EBF["smiles"], "num_conformers": 3, "seed": 20260525},
        ).get_json()
        analysis = self.client.get(f"/api/results/{source['run_id']}/ensemble?energy_window=5").get_json()
        representative = analysis["families"][0]["representative_rank"]
        self.client.post(
            f"/api/results/{source['run_id']}/refinement-plan",
            json={
                "method": analysis["method"],
                "threshold": analysis["threshold"],
                "energy_window": 5,
                "representative_ranks": [representative],
            },
        )
        with patch.object(workbench, "ORCA_BIN", Path("/missing/orca")):
            response = self.client.post(f"/api/results/{source['run_id']}/refinement/start", json={})
        self.assertEqual(response.status_code, 503)
        data = response.get_json()
        self.assertIn("ORCA 6.0.1", data["error"])
        self.assertEqual(data["install_relative_path"], "engines/orca-6.0.1")
        self.assertIn("faccts.de/orca", data["download_url"])

    def test_duplicate_crest_submission_reuses_running_task(self):
        class RunningProcess:
            pid = 314159

            def poll(self):
                return None

        payload = {"name": EBF["name"], "smiles": EBF["smiles"], "search_quality": "quick"}
        with patch.object(workbench.subprocess, "Popen", return_value=RunningProcess()) as launch:
            first = self.client.post("/api/crest/start", json=payload).get_json()
            second = self.client.post("/api/crest/start", json=payload).get_json()

        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(launch.call_count, 1)
        tasks = self.client.get("/api/tasks").get_json()["tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertGreater(tasks[0]["estimated_seconds"], 0)
        detail = self.client.get(f"/api/tasks/{first['run_id']}").get_json()
        self.assertGreater(len(detail["pipeline"]), 0)
        response = self.client.get(f"/tasks/{first['run_id']}")
        self.assertEqual(response.status_code, 200)
        response.close()

    def test_crest_plan_identifies_existing_task_before_submit(self):
        class RunningProcess:
            pid = 271828

            def poll(self):
                return None

        payload = {"name": EBF["name"], "smiles": EBF["smiles"], "search_quality": "quick"}
        with patch.object(workbench.subprocess, "Popen", return_value=RunningProcess()):
            task = self.client.post("/api/crest/start", json=payload).get_json()
        plan = self.client.post("/api/crest/plan", json=payload).get_json()
        self.assertEqual(plan["duplicate_task"]["run_id"], task["run_id"])

    def test_water_environment_changes_protocol_and_crest_command(self):
        class RunningProcess:
            pid = 161803

            def poll(self):
                return None

        gas_payload = {"name": EBF["name"], "smiles": EBF["smiles"], "search_quality": "quick"}
        water_payload = {**gas_payload, "sampling_environment": "water"}
        with patch.object(workbench.subprocess, "Popen", return_value=RunningProcess()) as launch:
            gas = self.client.post("/api/crest/start", json=gas_payload).get_json()
            water = self.client.post("/api/crest/start", json=water_payload).get_json()

        self.assertNotEqual(gas["run_id"], water["run_id"])
        self.assertEqual(gas["sampling_environment_label"], "气相")
        self.assertEqual(water["sampling_environment_label"], "隐式水环境 · ALPB(water)")
        self.assertEqual(launch.call_count, 2)
        self.assertNotIn("-alpb", launch.call_args_list[0].args[0])
        self.assertIn("-alpb", launch.call_args_list[1].args[0])
        self.assertIn("water", launch.call_args_list[1].args[0])
        plan = self.client.post("/api/crest/plan", json=water_payload).get_json()
        self.assertEqual(plan["duplicate_task"]["run_id"], water["run_id"])

    def test_crest_single_conformer_gc_stop_is_treated_as_converged_result(self):
        run_id = "crest-single-conformer"
        run_dir = workbench.OUTPUT_DIR / run_id
        run_dir.mkdir()
        mol = workbench.Chem.MolFromSmiles("CC")
        conformer_mol, _, _ = workbench.generate_conformers(mol, 1, 20260621)
        xyz_lines = [str(conformer_mol.GetNumAtoms()), "-10.0 single conformer"]
        conformer = conformer_mol.GetConformer(0)
        for atom in conformer_mol.GetAtoms():
            position = conformer.GetAtomPosition(atom.GetIdx())
            xyz_lines.append(f"{atom.GetSymbol()} {position.x:.6f} {position.y:.6f} {position.z:.6f}")
        (run_dir / "crest_conformers.xyz").write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")
        (run_dir / "crest.log").write_text(
            "CREGEN> E lowest :   -10.00000\n"
            " 1 structures remain within     5.00 kcal/mol window\n"
            "ERROR STOP\n"
            " Warning, file confcross_0.xyz does not exist!\n"
            "\n"
            " Not enough structures to perform GC!\n",
            encoding="utf-8",
        )
        workbench.save_task(
            {
                "task_type": "crest",
                "run_id": run_id,
                "fingerprint": "single-conformer",
                "name": "single conformer",
                "canonical_smiles": "CC",
                "coordinate_smiles": "CC",
                "formal_charge": 0,
                "search_quality": "quick",
                "sampling_environment": "gas",
                "status": "failed",
                "progress": "任务未正常完成，请查看运行日志。",
                "error": "任务未正常完成，请查看运行日志。",
                "created_at": 1,
                "completed_at": 2,
                "estimated_seconds": 120,
                "estimate_basis": "test",
                "pid": 0,
            }
        )

        task = self.client.get(f"/api/tasks/{run_id}").get_json()
        self.assertEqual(task["status"], "completed")
        self.assertTrue(task["single_conformer_converged"])
        self.assertIn("单一主导构象", task["progress"])

        result = self.client.get(f"/api/results/{run_id}").get_json()
        self.assertEqual(result["retained_conformers"], 1)
        self.assertTrue(result["single_conformer_converged"])
        self.assertIn("单一主导构象", result["method_scope"])

    def test_build_page_exposes_sampling_environment_choices(self):
        response = self.client.get("/build")
        self.assertEqual(response.status_code, 200)
        self.assertIn("低能构象采样环境".encode(), response.data)
        self.assertIn(b'option value="gas"', response.data)
        self.assertIn(b'option value="water"', response.data)
        response.close()

    def test_only_finished_task_can_be_deleted_with_its_output_directory(self):
        run_id = "crest-delete-check"
        run_dir = workbench.OUTPUT_DIR / run_id
        run_dir.mkdir()
        task = {
            "run_id": run_id,
            "fingerprint": "delete-check",
            "name": "historical calculation",
            "canonical_smiles": "CC",
            "search_quality": "quick",
            "status": "running",
            "created_at": 1,
            "estimated_seconds": 120,
            "estimate_basis": "test",
            "pid": 0,
        }
        workbench.save_task(task)
        with patch.object(workbench, "process_is_running", return_value=True):
            refused = self.client.delete(f"/api/tasks/{run_id}")
        self.assertEqual(refused.status_code, 409)
        self.assertTrue(run_dir.exists())

        task["status"] = "stopped"
        task["progress"] = "任务已停止。"
        workbench.save_task(task)
        deleted = self.client.delete(f"/api/tasks/{run_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse(run_dir.exists())


if __name__ == "__main__":
    unittest.main()
