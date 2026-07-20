#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, redirect, request, send_file
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, Crippen, Descriptors, Draw, Lipinski, rdMolAlign, rdMolDescriptors, rdMolTransforms, TorsionFingerprints
from rdkit.Geometry import Point3D
from rdkit.ML.Cluster import Butina


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
VENDOR_DIR = ROOT / "web" / "vendor"
STYLE_DIR = ROOT / "web" / "styles"
ENGINE_DIR = ROOT / "engines"
DEFAULT_KETCHER_DIR = ENGINE_DIR / "ketcher-standalone-3.7.0" / "standalone"
LEGACY_KETCHER_DIR = ROOT.parent / "HTE Experiment Designer" / "ketcher_local" / "ketcher-standalone-3.7.0" / "standalone"
if os.environ.get("CONFORMER_Q_KETCHER_DIR"):
    KETCHER_DIR = Path(os.environ["CONFORMER_Q_KETCHER_DIR"]).expanduser()
elif (DEFAULT_KETCHER_DIR / "index.html").exists():
    KETCHER_DIR = DEFAULT_KETCHER_DIR
else:
    KETCHER_DIR = LEGACY_KETCHER_DIR
PACKAGED_CREST_ENV = Path(
    os.environ.get("CONFORMER_Q_CREST_DIR", str(ENGINE_DIR / "crest-3.0.2"))
).expanduser()
LEGACY_CREST_ENV = Path.home() / ".local" / "share" / "conformer-q-tools" / "envs" / "crest2"
ORCA_VALIDATED_VERSION = "6.0.1"
ORCA_INSTALL_RELATIVE_PATH = "engines/orca-6.0.1"
ORCA_DOWNLOAD_URL = "https://www.faccts.de/orca/"
PACKAGED_ORCA_DIR = Path(
    os.environ.get("CONFORMER_Q_ORCA_DIR", str(ENGINE_DIR / "orca-6.0.1"))
).expanduser()
LEGACY_ORCA_DIR = Path.home() / "Library" / "orca_6_0_1"


def resolve_executable(env_name: str, command_name: str, directories: list[Path]) -> Path:
    candidates: list[Path] = []
    configured = os.environ.get(env_name)
    if configured:
        configured_path = Path(configured).expanduser()
        candidates.extend(
            [
                configured_path,
                configured_path / command_name,
                configured_path / "bin" / command_name,
            ]
        )
    for directory in directories:
        candidates.extend([directory / command_name, directory / "bin" / command_name])
    discovered = shutil.which(command_name)
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return Path("/__conformer_q_missing__") / command_name


CREST_BIN = resolve_executable("CONFORMER_Q_CREST_BIN", "crest", [PACKAGED_CREST_ENV, LEGACY_CREST_ENV])
XTB_BIN = resolve_executable("CONFORMER_Q_XTB_BIN", "xtb", [PACKAGED_CREST_ENV, LEGACY_CREST_ENV])
COMPUTE_BIN_DIRS = sorted({CREST_BIN.parent, XTB_BIN.parent}, key=str)
ORCA_BIN = resolve_executable("CONFORMER_Q_ORCA_BIN", "orca", [PACKAGED_ORCA_DIR, LEGACY_ORCA_DIR])
ORCA_DIR = ORCA_BIN.parent
ORCA_PLOT_BIN = ORCA_DIR / "orca_plot"
ORCA_VPOT_BIN = ORCA_DIR / "orca_vpot"
ORCA_WORKER = ROOT / "orca_refinement_worker.py"
OUTPUT_DIR.mkdir(exist_ok=True)
JOBS: dict[str, dict] = {}
CREST_METHOD = "CREST 3.0.2 iMTD-GC + GFN2-xTB (legacy external-xTB backend)"
ORCA_METHOD = f"ORCA {ORCA_VALIDATED_VERSION} r2SCAN-3c Opt + Freq"
GAS_CONSTANT_KCAL = 0.00198720425864083
SAMPLING_ENVIRONMENTS = {
    "gas": {
        "label": "气相",
        "description": "孤立分子的低能构象搜索",
        "crest_args": [],
    },
    "water": {
        "label": "隐式水环境 · ALPB(water)",
        "description": "连续介质水溶液近似下的低能构象搜索",
        "crest_args": ["-alpb", "water"],
    },
}

EBF = {
    "name": "(E)-beta-farnesene (EBF)",
    "common_name": "Aphid alarm pheromone",
    "cid": 5281517,
    "smiles": "CC(=CCC/C(=C/CCC(=C)C=C)/C)C",
    "source": "https://pubchem.ncbi.nlm.nih.gov/compound/5281517",
}

app = Flask(__name__)
FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))


def draw_svg(mol: Chem.Mol) -> str:
    display_mol = Chem.Mol(mol)
    AllChem.Compute2DCoords(display_mol)
    drawer = Draw.MolDraw2DSVG(500, 280)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    drawer.DrawMolecule(display_mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def molecular_properties(mol: Chem.Mol) -> dict:
    properties = {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "clogp": round(Crippen.MolLogP(mol), 2),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
    }
    properties["lipinski_violations"] = sum(
        [
            properties["molecular_weight"] > 500,
            properties["clogp"] > 5,
            properties["hbd"] > 5,
            properties["hba"] > 10,
        ]
    )
    return properties


def conformer_torsions(mol: Chem.Mol, reference: Chem.Mol | None = None) -> list[dict]:
    conformer = mol.GetConformer()
    reference_conformer = reference.GetConformer() if reference is not None else None
    torsions = []
    seen_bonds = set()
    for begin, end in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts):
        bond_key = tuple(sorted((begin, end)))
        if bond_key in seen_bonds:
            continue
        seen_bonds.add(bond_key)
        begin_neighbors = [
            atom.GetIdx()
            for atom in mol.GetAtomWithIdx(begin).GetNeighbors()
            if atom.GetIdx() != end and atom.GetAtomicNum() > 1
        ]
        end_neighbors = [
            atom.GetIdx()
            for atom in mol.GetAtomWithIdx(end).GetNeighbors()
            if atom.GetIdx() != begin and atom.GetAtomicNum() > 1
        ]
        if not begin_neighbors or not end_neighbors:
            continue
        atoms = (begin_neighbors[0], begin, end, end_neighbors[0])
        angle = float(rdMolTransforms.GetDihedralDeg(conformer, *atoms))
        reference_angle = (
            float(rdMolTransforms.GetDihedralDeg(reference_conformer, *atoms))
            if reference_conformer is not None
            else angle
        )
        change = (angle - reference_angle + 180.0) % 360.0 - 180.0
        torsions.append(
            {
                "atoms": [atom_idx + 1 for atom_idx in atoms],
                "atom_label": "-".join(f"{mol.GetAtomWithIdx(atom_idx).GetSymbol()}{atom_idx + 1}" for atom_idx in atoms),
                "central_bond": [begin + 1, end + 1],
                "angle_degrees": round(angle, 1),
                "reference_angle_degrees": round(reference_angle, 1),
                "change_from_lowest_degrees": round(change, 1),
            }
        )
    torsions.sort(key=lambda row: (-abs(row["change_from_lowest_degrees"]), row["central_bond"]))
    return torsions


def intramolecular_hbond_candidates(mol: Chem.Mol) -> list[dict]:
    conformer = mol.GetConformer()
    features = FEATURE_FACTORY.GetFeaturesForMol(mol)
    donors = [feature for feature in features if feature.GetFamily() == "Donor"]
    acceptors = [feature for feature in features if feature.GetFamily() == "Acceptor"]
    candidates = []
    for donor in donors:
        donor_idx = donor.GetAtomIds()[0]
        hydrogens = [
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(donor_idx).GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        ]
        for acceptor in acceptors:
            acceptor_idx = acceptor.GetAtomIds()[0]
            if acceptor_idx == donor_idx:
                continue
            donor_position = np.array(conformer.GetAtomPosition(donor_idx))
            acceptor_position = np.array(conformer.GetAtomPosition(acceptor_idx))
            donor_acceptor = float(np.linalg.norm(donor_position - acceptor_position))
            if donor_acceptor > 3.5:
                continue
            for hydrogen_idx in hydrogens:
                hydrogen_position = np.array(conformer.GetAtomPosition(hydrogen_idx))
                hydrogen_acceptor = float(np.linalg.norm(hydrogen_position - acceptor_position))
                left = donor_position - hydrogen_position
                right = acceptor_position - hydrogen_position
                denominator = np.linalg.norm(left) * np.linalg.norm(right)
                if denominator == 0:
                    continue
                angle = math.degrees(math.acos(float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))))
                if hydrogen_acceptor <= 2.6 and angle >= 120.0:
                    candidates.append(
                        {
                            "donor_atom": f"{mol.GetAtomWithIdx(donor_idx).GetSymbol()}{donor_idx + 1}",
                            "acceptor_atom": f"{mol.GetAtomWithIdx(acceptor_idx).GetSymbol()}{acceptor_idx + 1}",
                            "distance_angstrom": round(donor_acceptor, 2),
                            "angle_degrees": round(angle, 1),
                        }
                    )
    return candidates


def conformer_shape_description(mol: Chem.Mol, profile: dict) -> dict:
    heavy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    coordinates = np.array([list(mol.GetConformer().GetAtomPosition(atom_idx)) for atom_idx in heavy_atoms])
    centered = coordinates - coordinates.mean(axis=0)
    if len(heavy_atoms) > 1:
        _, axes = np.linalg.eigh(np.cov(centered.T))
        extents = sorted(np.ptp(centered @ axes, axis=0), reverse=True)
        radius_gyration = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    else:
        extents = [0.0, 0.0, 0.0]
        radius_gyration = 0.0
    graph_distances = Chem.GetDistanceMatrix(mol)
    terminal_pair = max(
        ((first, second) for first in heavy_atoms for second in heavy_atoms if second > first),
        key=lambda pair: graph_distances[pair[0], pair[1]],
        default=(heavy_atoms[0], heavy_atoms[0]),
    )
    path = Chem.GetShortestPath(mol, *terminal_pair)
    conformer = mol.GetConformer()
    endpoint_distance = float(
        (conformer.GetAtomPosition(terminal_pair[0]) - conformer.GetAtomPosition(terminal_pair[1])).Length()
    )
    contour_length = sum(
        float((conformer.GetAtomPosition(path[index]) - conformer.GetAtomPosition(path[index + 1])).Length())
        for index in range(len(path) - 1)
    )
    folding_ratio = endpoint_distance / contour_length if contour_length else 1.0
    if profile.get("topology") == "柔性链状分子":
        if folding_ratio >= 0.78:
            label = "较伸展 · extended"
        elif folding_ratio <= 0.52:
            label = "较折叠 · folded"
        else:
            label = "中间折叠态"
    else:
        label = "整体形状描述"
    return {
        "label": label,
        "length_angstrom": round(float(extents[0]), 2),
        "width_angstrom": round(float(extents[1]), 2),
        "depth_angstrom": round(float(extents[2]), 2),
        "radius_of_gyration_angstrom": round(radius_gyration, 2),
        "end_to_end_distance_angstrom": round(endpoint_distance, 2),
        "end_to_end_atoms": [atom_idx + 1 for atom_idx in terminal_pair],
        "folding_ratio": round(folding_ratio, 3),
        "definition": "length / width / depth 为重原子坐标沿主轴的范围；折叠分类基于拓扑最远重原子的空间距离与链路径长度之比。",
    }


def conformer_decision_explanation(
    rank: int,
    is_orca: bool,
    is_partial_result: bool,
    row: dict,
    shape: dict,
    reference_shape: dict,
    torsions: list[dict],
) -> dict:
    endpoint_change = round(shape["end_to_end_distance_angstrom"] - reference_shape["end_to_end_distance_angstrom"], 2)
    gyration_change = round(shape["radius_of_gyration_angstrom"] - reference_shape["radius_of_gyration_angstrom"], 2)
    significant_torsions = [torsion for torsion in torsions if abs(torsion["change_from_lowest_degrees"]) >= 30.0][:3]
    geometry_distinct = bool(significant_torsions) or abs(endpoint_change) >= 0.75 or abs(gyration_change) >= 0.35
    if endpoint_change >= 0.60:
        shape_change = f"端到端距离比 #1 增加 {endpoint_change:.2f} Å，链段整体更展开。"
    elif endpoint_change <= -0.60:
        shape_change = f"端到端距离比 #1 减少 {abs(endpoint_change):.2f} Å，链段整体更折回。"
    else:
        shape_change = f"端到端距离与 #1 接近（变化 {endpoint_change:+.2f} Å），整体折叠尺度相似。"
    if significant_torsions:
        torsion_text = "；".join(
            f"{torsion['atom_label']} 翻转 {torsion['change_from_lowest_degrees']:+.1f}°"
            for torsion in significant_torsions
        )
        structural_difference = f"{torsion_text}。{shape_change}"
    else:
        structural_difference = f"未检测到超过 30° 的主导二面角变化。{shape_change}"

    if rank == 1:
        if is_orca:
            label = "部分结果参考构象" if is_partial_result else "主导参考构象"
            importance = (
                f"这是{'已完成子集中' if is_partial_result else '精修后'}自由能最低的参考形态，"
                f"在{'该子集' if is_partial_result else '当前系综'}中的占比为 "
                f"{row.get('boltzmann_population_percent', 0):.3f}%。"
            )
            recommendation = (
                "作为部分结果的参照保留；待其余精修完成后再确认最终主导形态。"
                if is_partial_result
                else "作为主导参考构象保留，用于判断其他低能形态是否提供额外空间覆盖。"
            )
        else:
            label = "最低能参考候选"
            importance = "这是当前采样结果的最低能参考形态，后续候选均与它比较几何差异。"
            recommendation = "优先纳入高级精修；当前阶段尚不能用它代表最终主导构象。"
    elif is_orca:
        population = float(row.get("boltzmann_population_percent") or 0.0)
        delta_g = float(row.get("relative_gibbs_free_energy_kcal_mol") or 0.0)
        if population >= 5.0 and geometry_distinct:
            label = "部分结果中的竞争形态" if is_partial_result else "重要竞争低能形态"
            importance = (
                f"它仅比 #1 高 {delta_g:.4f} kcal/mol，且在"
                f"{'已完成子集' if is_partial_result else '当前系综'}中仍占 {population:.3f}%；同时提供不同三维形态。"
            )
            recommendation = (
                "建议在已完成结果中保留并继续关注；待全部精修完成后再决定最终性质分析或 docking 集合。"
                if is_partial_result
                else "建议保留，进入后续性质场分析或 docking 构象集合。"
            )
        elif population >= 5.0:
            label = "部分结果中的近似形态" if is_partial_result else "热力学相关但形态近似"
            importance = f"它在{'已完成子集' if is_partial_result else '当前系综'}中占 {population:.3f}%，但与 #1 的主要几何差异有限。"
            recommendation = (
                "暂时保留统计信息；等待完整精修结果再判断是否需要作为独立结构使用。"
                if is_partial_result
                else "可保留其权重用于系综统计；作为独立 docking 结构的优先级较低。"
            )
        else:
            label = "低占比候选"
            importance = f"它在当前系综中的占比为 {population:.3f}%，对主要分布贡献较小。"
            recommendation = "仅在需要覆盖稀有形态时保留，不作为首轮下游分析重点。"
    else:
        delta_e = float(row.get("relative_energy_kcal_mol") or 0.0)
        if geometry_distinct:
            label = "几何差异候选"
            importance = f"它与当前最低能候选相差 {delta_e:.4f} kcal/mol，但表现出不同的链段构型。"
            recommendation = "若其代表独立构象家族，建议纳入 ORCA 精修以确认能量排序。"
        else:
            label = "近似重复候选"
            importance = f"它与最低能候选相差 {delta_e:.4f} kcal/mol，且未体现主要形状变化。"
            recommendation = "优先查看其是否与 #1 归入同一家族；若是，不必重复送入精修。"
    return {
        "reference_rank": 1,
        "label": label,
        "importance": importance,
        "structural_difference": structural_difference,
        "recommendation": recommendation,
        "geometry_distinct": geometry_distinct,
        "endpoint_change_angstrom": endpoint_change,
        "radius_of_gyration_change_angstrom": gyration_change,
        "significant_torsions": significant_torsions,
    }


def available_tools() -> dict:
    packaged_orca = (PACKAGED_ORCA_DIR / "orca").exists()
    orca_available = ORCA_BIN.exists()
    return {
        "rdkit": {"available": True, "role": "快速构象预览与结构诊断"},
        "ketcher": {
            "available": (KETCHER_DIR / "index.html").exists(),
            "path": str(KETCHER_DIR),
            "role": "二维结构绘制",
            "installation_message": (
                "请下载 Ketcher standalone 发行包，并解压到 "
                "engines/ketcher-standalone-3.7.0/standalone/，或设置 CONFORMER_Q_KETCHER_DIR。"
            ),
        },
        "xtb": {"available": XTB_BIN.exists(), "path": str(XTB_BIN), "role": "GFN2-xTB 半经验量子优化"},
        "crest": {"available": CREST_BIN.exists(), "path": str(CREST_BIN), "role": "增强采样构象搜索"},
        "orca": {
            "available": orca_available,
            "path": str(ORCA_BIN) if orca_available else None,
            "role": "后续 DFT 精修与重排序",
            "validated_version": ORCA_VALIDATED_VERSION,
            "project_installed": packaged_orca,
            "install_relative_path": ORCA_INSTALL_RELATIVE_PATH,
            "download_url": ORCA_DOWNLOAD_URL,
            "installation_message": (
                f"请从 ORCA 官方渠道取得 macOS 版 ORCA {ORCA_VALIDATED_VERSION}，"
                f"并将解压后的安装目录放入项目的 {ORCA_INSTALL_RELATIVE_PATH}/，"
                "或设置 CONFORMER_Q_ORCA_DIR / CONFORMER_Q_ORCA_BIN。"
            ),
        },
    }


def profile_molecule(mol: Chem.Mol) -> dict:
    rings = list(mol.GetRingInfo().AtomRings())
    ring_sizes = sorted(len(ring) for ring in rings)
    rotatable = Lipinski.NumRotatableBonds(mol)
    macrocycles = [size for size in ring_sizes if size >= 9]
    spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    fragments = len(Chem.GetMolFrags(mol))
    special_elements = sorted(
        {atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 20 and atom.GetSymbol() not in {"Br", "I"}}
    )
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    potential_imbh = hbd > 0 and hba > 0 and (rotatable > 1 or bool(rings))

    risks = []
    if macrocycles:
        risks.append("宏环闭环约束使扭转角强耦合，需要增强采样和多次验证。")
    if spiro_atoms or bridgeheads:
        risks.append("螺环/桥环拓扑存在，不能按独立单键简单枚举。")
    if potential_imbh:
        risks.append("存在潜在分子内氢键，搜索结果可能对溶剂和质子化状态敏感。")
    if formal_charge:
        risks.append("分子带形式电荷，应在指定 pH/溶剂与微观状态下解释构象。")
    if fragments > 1:
        risks.append("检测到多片段体系，需要考虑非共价复合物协议。")
    if special_elements:
        risks.append("包含特殊元素，需确认 xTB/DFT 参数与化学状态。")

    if macrocycles:
        topology = "宏环分子"
        recommended = "CREST/GFN2-xTB 多次独立搜索，并在溶剂模型下进行 ORCA 精修"
    elif spiro_atoms or bridgeheads:
        topology = "螺环或桥环骨架"
        recommended = "ETKDG 合理起始几何 + CREST/GFN2-xTB 搜索 + ORCA 精修"
    elif rings and rotatable >= 4:
        topology = "含环的柔性分子"
        recommended = "CREST/GFN2-xTB 主搜索，并保留不同环/侧链构象家族"
    elif rings:
        topology = "较刚性环状分子"
        recommended = "ETKDG 快速检查后，以 GFN2-xTB 优化和必要的 ORCA 精修确认"
    elif rotatable >= 4:
        topology = "柔性链状分子"
        recommended = "CREST/GFN2-xTB 主搜索；旋转键有限时可用系统扭转枚举交叉验证"
    else:
        topology = "刚性或低柔性链状分子"
        recommended = "少量初始构象经 GFN2-xTB 优化，关键结果再以 ORCA 精修"
    return {
        "topology": topology,
        "rotatable_bonds": rotatable,
        "ring_count": len(rings),
        "ring_sizes": ring_sizes,
        "macrocycles": len(macrocycles),
        "spiro_atoms": spiro_atoms,
        "bridgehead_atoms": bridgeheads,
        "formal_charge": formal_charge,
        "fragments": fragments,
        "potential_intramolecular_hbond": potential_imbh,
        "special_elements": special_elements,
        "risks": risks or ["未识别到需要额外处理的主要拓扑或电荷风险。"],
        "recommended_protocol": recommended,
        "scientific_scope": "构象搜索寻找给定模型下的低能构象集合，不证明完整连续势能面或生物结合构象。",
    }


def sampling_environment(value: str | None) -> dict | None:
    return SAMPLING_ENVIRONMENTS.get(value or "gas")


def task_fingerprint(canonical_smiles: str, search_quality: str, formal_charge: int, environment: str = "gas") -> str:
    specification = (
        f"crest3.0.2|legacy-external-gfn2-xtb|coordinate-order-v2|{search_quality}|charge={formal_charge}"
        f"|environment={environment}|{canonical_smiles}"
    )
    return hashlib.sha256(specification.encode("utf-8")).hexdigest()


def task_path(run_id: str) -> Path:
    return OUTPUT_DIR / run_id / "task.json"


def load_task(run_id: str) -> dict | None:
    path = task_path(run_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_task(task: dict) -> None:
    path = task_path(task["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def estimate_task_seconds(mol: Chem.Mol, search_quality: str, fingerprint: str) -> tuple[int, str]:
    completed_durations = []
    for path in OUTPUT_DIR.glob("crest-*/task.json"):
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "completed":
            duration = previous.get("duration_seconds")
            if isinstance(duration, (int, float)) and duration > 0:
                completed_durations.append(float(duration))
    if completed_durations:
        estimate = int(sum(completed_durations[-3:]) / len(completed_durations[-3:]))
        return max(30, estimate), "相同结构和协议的历史完成耗时"

    profile = profile_molecule(mol)
    atoms = Chem.AddHs(mol).GetNumAtoms()
    seconds = 120 + atoms * 8 + profile["rotatable_bonds"] * 180 + profile["ring_count"] * 180
    if profile["macrocycles"]:
        seconds += 1800
    if search_quality == "full":
        seconds *= 3
    elif search_quality == "smoke":
        seconds = min(seconds, 180)
    return max(120, int(seconds)), "按原子数、可旋转键、环系和采样等级的初始估算"


def process_is_running(task: dict) -> bool:
    job = JOBS.get(task["run_id"])
    if job is not None:
        return job["process"].poll() is None
    pid = task.get("pid")
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def crest_progress(run_id: str) -> tuple[str, str]:
    text = ""
    log_path = OUTPUT_DIR / run_id / "crest.log"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    if "Optimizing all" in text:
        return "正在优化增强采样得到的候选结构...", "candidate_optimization"
    if "Meta-MD" in text:
        return "正在进行 Meta-MD 增强采样...", "enhanced_sampling"
    if "Initial Geometry Optimization" in text:
        return "正在进行 GFN2-xTB 初始几何优化...", "initial_optimization"
    return "正在初始化 CREST 搜索...", "initializing"


def task_view(task: dict) -> dict:
    elapsed = max(0, int((task.get("completed_at") or time.time()) - task["created_at"]))
    estimate = int(task["estimated_seconds"])
    remaining = max(0, estimate - elapsed) if task["status"] == "running" else 0
    task_type = task.get("task_type", "crest")
    if task_type == "orca":
        progress = task.get("progress", "正在准备 ORCA 高级精修...")
        stage = task["status"] if task["status"] != "running" else "refinement"
    else:
        progress, stage = crest_progress(task["run_id"]) if task["status"] == "running" else (task.get("progress", ""), task["status"])
    environment_key = task.get("sampling_environment", "gas")
    environment = sampling_environment(environment_key) or SAMPLING_ENVIRONMENTS["gas"]
    response = {
        "run_id": task["run_id"],
        "name": task["name"],
        "canonical_smiles": task["canonical_smiles"],
        "status": task["status"],
        "task_type": task_type,
        "method": ORCA_METHOD if task_type == "orca" else CREST_METHOD,
        "search_quality": task.get("search_quality", "refinement"),
        "sampling_environment": environment_key,
        "sampling_environment_label": environment["label"],
        "created_at": task["created_at"],
        "estimated_seconds": estimate,
        "estimate_basis": task["estimate_basis"],
        "elapsed_seconds": elapsed,
        "remaining_seconds": remaining,
        "estimate_exceeded": task["status"] == "running" and elapsed > estimate,
        "progress": progress,
        "stage": stage,
        "fingerprint": task["fingerprint"],
    }
    if task.get("single_conformer_converged"):
        response["single_conformer_converged"] = True
    if task_type == "orca":
        response["source_run_id"] = task["source_run_id"]
        response["selection_count"] = len(task["selected_conformers"])
        response["completed_count"] = len(task.get("completed_conformers", []))
        response["temperature_kelvin"] = task.get("temperature_kelvin", 298.15)
        if task["status"] == "running" and response["completed_count"]:
            durations = [
                float(item["duration_seconds"])
                for item in task.get("completed_conformers", [])
                if isinstance(item.get("duration_seconds"), (int, float)) and item["duration_seconds"] > 0
            ]
            if durations:
                seconds_per_conformer = statistics.median(durations)
                eta_basis = "按已完成 ORCA 构象的实际耗时中位数动态估算"
                eta_confidence = "observed"
            else:
                seconds_per_conformer = elapsed / response["completed_count"]
                eta_basis = "当前任务缺少逐构象时间记录，按已完成数量与累计耗时保守粗估"
                eta_confidence = "legacy_rough"
            pending_count = max(0, response["selection_count"] - response["completed_count"])
            dynamic_remaining = max(0, int(round(seconds_per_conformer * pending_count)))
            response.update(
                {
                    "dynamic_eta_available": True,
                    "dynamic_remaining_seconds": dynamic_remaining,
                    "dynamic_total_seconds": elapsed + dynamic_remaining,
                    "observed_seconds_per_conformer": int(round(seconds_per_conformer)),
                    "dynamic_eta_basis": eta_basis,
                    "dynamic_eta_confidence": eta_confidence,
                }
            )
    if "error" in task:
        response["error"] = task["error"]
    return response


def find_reusable_task(
    fingerprint: str,
    canonical_smiles: str | None = None,
    search_quality: str | None = None,
    formal_charge: int | None = None,
    environment: str = "gas",
) -> dict | None:
    for path in sorted(OUTPUT_DIR.glob("crest-*/task.json"), reverse=True):
        previous = load_task(path.parent.name)
        if previous is None:
            continue
        same_fingerprint = previous.get("fingerprint") == fingerprint
        legacy_gas_task = (
            environment == "gas"
            and "sampling_environment" not in previous
            and previous.get("canonical_smiles") == canonical_smiles
            and previous.get("search_quality") == search_quality
            and previous.get("formal_charge") == formal_charge
        )
        if not same_fingerprint and not legacy_gas_task:
            continue
        previous = refresh_crest_task(previous)
        if previous["status"] in {"running", "completed"}:
            return previous
    return None


def task_pipeline(task: dict) -> list[dict]:
    if task.get("task_type") == "orca":
        count = len(task["selected_conformers"])
        completed = len(task.get("completed_conformers", []))
        steps = [
            {"key": "prepared", "label": "代表构象与协议确认", "status": "completed"},
            {
                "key": "refinement",
                "label": f"ORCA 优化与频率计算（{completed} / {count}）",
                "status": task["status"] if task["status"] in {"failed", "stopped"} else ("completed" if task["status"] == "completed" else "running"),
            },
            {
                "key": "completed",
                "label": "自由能排序与玻尔兹曼分布",
                "status": "completed" if task["status"] == "completed" else "pending",
            },
        ]
        return steps
    stage_order = [
        ("submitted", "任务提交与结构确认"),
        ("initializing", "生成初始三维几何"),
        ("initial_optimization", "GFN2-xTB 初始优化"),
        ("enhanced_sampling", "CREST Meta-MD 增强采样"),
        ("candidate_optimization", "候选结构优化与筛选"),
        ("completed", "构象集合与结果文件"),
    ]
    view = task_view(task)
    current_stage = view["stage"]
    if task["status"] in {"failed", "stopped"}:
        _, current_stage = crest_progress(task["run_id"])
    current_index = next((index for index, (key, _) in enumerate(stage_order) if key == current_stage), 0)
    if task["status"] == "completed":
        current_index = len(stage_order) - 1
    stages = []
    for index, (key, label) in enumerate(stage_order):
        if index < current_index or task["status"] == "completed":
            status = "completed"
        elif index == current_index and task["status"] == "running":
            status = "running"
        elif index == current_index and task["status"] in {"failed", "stopped"}:
            status = task["status"]
        else:
            status = "pending"
        stages.append({"key": key, "label": label, "status": status})
    return stages


def log_excerpt(run_id: str, limit: int = 12) -> list[str]:
    task = load_task(run_id)
    filename = "worker.log" if task and task.get("task_type") == "orca" else "crest.log"
    path = OUTPUT_DIR / run_id / filename
    if not path.exists():
        return []
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    return lines[-limit:]


def save_result_payload(run_id: str, payload: dict) -> None:
    path = OUTPUT_DIR / run_id / "result.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_result_payload(run_id: str) -> dict | None:
    path = OUTPUT_DIR / run_id / "result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def enrich_result_context(run_id: str, result: dict) -> dict:
    enriched = dict(result)
    task = load_task(run_id)
    if task is None:
        return enriched
    environment_key = task.get("sampling_environment", "gas")
    environment = sampling_environment(environment_key) or SAMPLING_ENVIRONMENTS["gas"]
    enriched["sampling_environment"] = environment_key
    enriched["sampling_environment_label"] = environment["label"]
    if enriched.get("method", "").startswith("CREST"):
        if task.get("single_conformer_converged"):
            enriched["single_conformer_converged"] = True
            enriched["method_scope"] = (
                f"单一主导构象结果：在{environment['label']}下，CREST 采样与去重后仅保留 1 个低能构象；"
                "后续 GC 交叉步骤因构象数不足停止。该结果适合作为刚性分子的主导构象参考，"
                "重点构象仍建议在相同环境定义下使用 ORCA 精修。"
            )
        else:
            enriched.setdefault(
                "method_scope",
                f"研究搜索：在{environment['label']}下由增强采样与半经验量子能量得到构象集合；"
                "当前用于构象家族初筛，最终排序仍需在相同环境定义下进行高级精修。",
            )
    return enriched


def overlay_alignment_anchors(reference: Chem.Mol) -> list[dict]:
    heavy_atoms = [atom.GetIdx() for atom in reference.GetAtoms() if atom.GetAtomicNum() > 1]
    whole = {
        "id": "whole",
        "label": f"整体重原子（{len(heavy_atoms)} 个）",
        "atom_indices": heavy_atoms,
        "kind": "whole",
    }
    if len(heavy_atoms) < 3:
        return [{"id": "auto", "label": "自动推荐 · 整体重原子", **{key: whole[key] for key in ("atom_indices", "kind")}}, whole]

    rotatable_edges = {
        tuple(sorted(match[:2]))
        for match in reference.GetSubstructMatches(Lipinski.RotatableBondSmarts)
        if len(match) >= 2
    }
    remaining = set(heavy_atoms)
    rigid_components: list[set[int]] = []
    while remaining:
        root = remaining.pop()
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            atom = reference.GetAtomWithIdx(current)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor.GetAtomicNum() == 1 or neighbor_idx not in remaining:
                    continue
                if tuple(sorted((current, neighbor_idx))) in rotatable_edges:
                    continue
                remaining.remove(neighbor_idx)
                component.add(neighbor_idx)
                stack.append(neighbor_idx)
        rigid_components.append(component)

    local_anchors = []
    for component in rigid_components:
        has_rigid_feature = any(
            bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing()
            for bond in reference.GetBonds()
            if bond.GetBeginAtomIdx() in component and bond.GetEndAtomIdx() in component
        )
        if not has_rigid_feature:
            continue
        expanded = set(component)
        for atom_idx in component:
            for neighbor in reference.GetAtomWithIdx(atom_idx).GetNeighbors():
                if neighbor.GetAtomicNum() > 1:
                    expanded.add(neighbor.GetIdx())
        if len(expanded) < 3 or len(expanded) == len(heavy_atoms):
            continue
        local_anchors.append(tuple(sorted(expanded)))

    unique_anchors = sorted(set(local_anchors), key=lambda atoms: (-len(atoms), atoms))
    options = []
    for index, atoms in enumerate(unique_anchors, start=1):
        display_atoms = ", ".join(str(atom_idx + 1) for atom_idx in atoms)
        options.append(
            {
                "id": f"fragment-{index}",
                "label": f"局部刚性片段 {index}（{len(atoms)} 个重原子）",
                "description": f"原子编号 {display_atoms}",
                "atom_indices": list(atoms),
                "kind": "local",
            }
        )
    recommended = options[0] if options else whole
    auto_label = (
        f"自动局部锚点（{len(recommended['atom_indices'])} 个重原子）"
        if options
        else "自动推荐 · 整体重原子"
    )
    return [
        {
            "id": "auto",
            "label": auto_label,
            "description": "按结构拓扑选择最大的局部刚性锚点。" if options else "未识别到足够的局部刚性片段。",
            "atom_indices": recommended["atom_indices"],
            "kind": recommended["kind"],
        },
        whole,
        *options,
    ]


def overlay_atom_rmsd(probe: Chem.Mol, reference: Chem.Mol, atom_indices: list[int]) -> float:
    probe_conformer = probe.GetConformer()
    reference_conformer = reference.GetConformer()
    squared_distance = 0.0
    for atom_idx in atom_indices:
        probe_position = probe_conformer.GetAtomPosition(atom_idx)
        reference_position = reference_conformer.GetAtomPosition(atom_idx)
        squared_distance += (probe_position - reference_position).LengthSq()
    return math.sqrt(squared_distance / len(atom_indices))


def refinement_plan_path(run_id: str) -> Path:
    return OUTPUT_DIR / run_id / "refinement_plan.json"


def load_refinement_plan(run_id: str) -> dict | None:
    path = refinement_plan_path(run_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_refinement_plan(run_id: str, plan: dict) -> None:
    path = refinement_plan_path(run_id)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def orca_input_text(charge: int, environment_key: str, xyz_filename: str = "input.xyz", temperature_kelvin: float = 298.15) -> str:
    solvent = " CPCM(Water)" if environment_key == "water" else ""
    return "\n".join(
        [
            f"! r2SCAN-3c TightSCF Opt Freq{solvent}",
            "%pal nprocs 1 end",
            "%maxcore 2000",
            f"%freq Temp {temperature_kelvin:.2f} end",
            f"* xyzfile {charge} 1 {xyz_filename}",
            "",
        ]
    )


def orca_refinement_fingerprint(source_run_id: str, plan: dict, temperature_kelvin: float) -> str:
    ranks = ",".join(str(row["rank"]) for row in plan["selected_conformers"])
    specification = (
        f"orca6|r2scan-3c-opt-freq|temperature={temperature_kelvin:.2f}|environment={plan['sampling_environment']}"
        f"|source={source_run_id}|ranks={ranks}"
    )
    return hashlib.sha256(specification.encode("utf-8")).hexdigest()


def orca_property_artifacts(run_id: str, rank: int, result: dict) -> tuple[Path, str, dict]:
    if result.get("task_type") != "orca":
        raise ValueError("性质作图仅支持 ORCA 精修后的构象。")
    row = next((item for item in result["conformers"] if item["rank"] == rank), None)
    if row is None:
        raise ValueError("当前精修结果中不存在所选构象。")
    task = load_task(run_id)
    if task is None:
        raise ValueError("缺少该 ORCA 结果的任务信息，无法定位波函数文件。")
    source_rank = row["source_rank"]
    completed = next(
        (item for item in task.get("completed_conformers", []) if item["source_rank"] == source_rank),
        None,
    )
    if completed is None:
        raise ValueError("所选构象没有对应的 ORCA 完成记录。")
    prefix = Path(completed["xyz_file"]).stem
    run_dir = OUTPUT_DIR / run_id
    if not (run_dir / f"{prefix}.gbw").exists():
        raise ValueError("所选构象缺少 ORCA .gbw 波函数文件，无法生成量子性质图。")
    return run_dir, prefix, row


def parse_frontier_orbitals(output_path: Path) -> dict:
    text = output_path.read_text(encoding="utf-8", errors="ignore")
    blocks = text.split("ORBITAL ENERGIES")
    if len(blocks) < 2:
        raise ValueError("ORCA 输出中未找到轨道能量表。")
    orbital_pattern = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", re.MULTILINE)
    orbitals = []
    collecting = False
    for line in blocks[-1].splitlines():
        match = orbital_pattern.match(line)
        if match:
            collecting = True
            orbitals.append(
                {
                    "index": int(match.group(1)),
                    "occupation": float(match.group(2)),
                    "energy_hartree": float(match.group(3)),
                    "energy_ev": float(match.group(4)),
                }
            )
        elif collecting:
            break
    occupied = [row for row in orbitals if row["occupation"] > 0.0]
    virtual = [row for row in orbitals if row["occupation"] == 0.0]
    if not occupied or not virtual:
        raise ValueError("ORCA 输出中无法识别 HOMO/LUMO。")
    homo = occupied[-1]
    lumo = virtual[0]
    return {
        "homo": homo,
        "lumo": lumo,
        "gap_ev": round(lumo["energy_ev"] - homo["energy_ev"], 4),
    }


def run_orca_plot_until_file(run_dir: Path, prefix: str, commands: str, output_name: str) -> Path:
    output_path = run_dir / output_name
    if output_path.exists() and output_path.stat().st_size > 1000:
        return output_path
    if not ORCA_PLOT_BIN.exists():
        raise ValueError("未检测到 ORCA orca_plot 工具，无法生成体数据。")
    environment = os.environ.copy()
    environment["PATH"] = f"{ORCA_DIR}:{environment.get('PATH', '')}"
    process = subprocess.Popen(
        [str(ORCA_PLOT_BIN), f"{prefix}.gbw", "-i"],
        cwd=run_dir,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        process.stdin.write(commands)
        process.stdin.flush()
        deadline = time.time() + 90
        while time.time() < deadline:
            if output_path.exists() and output_path.stat().st_size > 1000:
                return output_path
            if process.poll() is not None:
                break
            time.sleep(0.1)
        raise ValueError("ORCA 体数据生成超时或未生成输出文件。")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def ensure_orca_orbital_cube(run_dir: Path, prefix: str, orbital_index: int, target_path: Path) -> Path:
    generated = run_orca_plot_until_file(
        run_dir,
        prefix,
        f"2\n{orbital_index}\n11\n",
        f"{prefix}.mo{orbital_index}a.cube",
    )
    if not target_path.exists():
        shutil.move(str(generated), str(target_path))
    return target_path


def ensure_orca_density_cube(run_dir: Path, prefix: str) -> Path:
    return run_orca_plot_until_file(run_dir, prefix, "1\n2\ny\n11\n", f"{prefix}.eldens.cube")


def cube_header_and_grid(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = abs(int(lines[2].split()[0]))
    origin = np.array([float(value) for value in lines[2].split()[1:4]])
    dimensions = [abs(int(lines[index].split()[0])) for index in range(3, 6)]
    vectors = [np.array([float(value) for value in lines[index].split()[1:4]]) for index in range(3, 6)]
    points = []
    bohr_to_angstrom = 0.529177210903
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                point = (origin + x * vectors[0] + y * vectors[1] + z * vectors[2]) * bohr_to_angstrom
                points.append(tuple(float(value) for value in point))
    return lines[: atom_count + 6], points


def write_cube_values(path: Path, header: list[str], values: list[float]) -> None:
    lines = list(header)
    for offset in range(0, len(values), 6):
        lines.append(" ".join(f"{value: .6e}" for value in values[offset : offset + 6]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_orca_esp_cube(run_dir: Path, prefix: str, target_path: Path) -> Path:
    if target_path.exists() and target_path.stat().st_size > 1000:
        return target_path
    if not ORCA_VPOT_BIN.exists():
        raise ValueError("未检测到 ORCA orca_vpot 工具，无法生成静电势。")
    density_cube = ensure_orca_density_cube(run_dir, prefix)
    header, points = cube_header_and_grid(density_cube)
    property_dir = target_path.parent
    grid_path = property_dir / f"{prefix}_esp_grid.xyz"
    potential_path = property_dir / f"{prefix}_esp_values.out"
    grid_path.write_text(
        str(len(points)) + "\n" + "\n".join(f"{x:.10f} {y:.10f} {z:.10f}" for x, y, z in points) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(ORCA_VPOT_BIN), f"{prefix}.gbw", f"{prefix}.scfp", str(grid_path), str(potential_path)],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0 or not potential_path.exists():
        raise ValueError("ORCA 静电势后处理失败，无法生成 ESP 图层。")
    potential_values = [
        float(line.split()[-1])
        for line in potential_path.read_text(encoding="utf-8").splitlines()[1:]
        if line.split()
    ]
    if len(potential_values) != len(points):
        raise ValueError("静电势网格数量与电子密度表面不一致。")
    write_cube_values(target_path, header, potential_values)
    grid_path.unlink(missing_ok=True)
    potential_path.unlink(missing_ok=True)
    return target_path


def write_empirical_hydrophobic_cube(molecule: Chem.Mol, target_path: Path) -> Path:
    if target_path.exists() and target_path.stat().st_size > 1000:
        return target_path
    conformer = molecule.GetConformer()
    atom_positions = np.array(
        [[conformer.GetAtomPosition(index).x, conformer.GetAtomPosition(index).y, conformer.GetAtomPosition(index).z]
         for index in range(molecule.GetNumAtoms())]
    )
    contributions = [float(value[0]) for value in Crippen._GetAtomContribs(molecule)]
    margin = 3.0
    minimum = atom_positions.min(axis=0) - margin
    maximum = atom_positions.max(axis=0) + margin
    dimensions = (32, 32, 32)
    spacing = (maximum - minimum) / np.array([dimension - 1 for dimension in dimensions])
    sigma = 1.25
    values = []
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                point = minimum + np.array([x, y, z]) * spacing
                squared = ((atom_positions - point) ** 2).sum(axis=1)
                values.append(float(np.dot(contributions, np.exp(-squared / (2 * sigma * sigma)))))
    angstrom_to_bohr = 1.88972612546
    header = [
        "Conformer-Q empirical hydrophobic field from RDKit Wildman-Crippen atom contributions",
        "Spatial Gaussian projection; not an ORCA quantum observable",
        f"{molecule.GetNumAtoms():5d} {minimum[0] * angstrom_to_bohr: .6f} {minimum[1] * angstrom_to_bohr: .6f} {minimum[2] * angstrom_to_bohr: .6f}",
        f"{dimensions[0]:5d} {spacing[0] * angstrom_to_bohr: .6f}  0.000000  0.000000",
        f"{dimensions[1]:5d}  0.000000 {spacing[1] * angstrom_to_bohr: .6f}  0.000000",
        f"{dimensions[2]:5d}  0.000000  0.000000 {spacing[2] * angstrom_to_bohr: .6f}",
    ]
    for index, atom in enumerate(molecule.GetAtoms()):
        position = atom_positions[index] * angstrom_to_bohr
        header.append(
            f"{atom.GetAtomicNum():5d} {0.0: .6f} {position[0]: .6f} {position[1]: .6f} {position[2]: .6f}"
        )
    write_cube_values(target_path, header, values)
    return target_path


def cube_value_extent(path: Path) -> float:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = abs(int(lines[2].split()[0]))
    values = [
        abs(float(value))
        for line in lines[atom_count + 6 :]
        for value in line.split()
    ]
    return max(values) if values else 1.0


def find_reusable_orca_task(fingerprint: str) -> dict | None:
    for path in sorted(OUTPUT_DIR.glob("orca-*/task.json"), reverse=True):
        task = load_task(path.parent.name)
        if task is None or task.get("fingerprint") != fingerprint:
            continue
        task = refresh_task(task)
        if task["status"] in {"running", "completed"}:
            return task
    return None


def ensemble_analysis_recommendation(profile: dict) -> dict:
    topology = profile["topology"]
    if topology == "柔性链状分子":
        return {
            "method": "tfd",
            "threshold": 0.10,
            "energy_window": 2.0,
            "label": "扭转构象家族分析",
            "reason": "该分子的主要自由度来自可旋转单键；TFD 更适合比较链段扭转模式。",
            "caution": "低能家族仍应通过独立 CREST 搜索和高级能量精修确认。",
        }
    if topology == "含环的柔性分子":
        return {
            "method": "tfd",
            "threshold": 0.10,
            "energy_window": 2.0,
            "label": "环系与侧链扭转初筛",
            "reason": "先以扭转差异识别柔性侧链及环构象差异，再结合 RMSD 检查整体形状。",
            "caution": "环骨架差异不应仅由单一聚类指标解释。",
        }
    if topology == "宏环分子":
        return {
            "method": "rmsd",
            "threshold": 1.00,
            "energy_window": 2.0,
            "label": "宏环折叠初筛",
            "reason": "宏环扭转角高度耦合，先按重原子 RMSD 提取候选折叠家族。",
            "caution": "宏环需要进一步联合环扭转、跨环距离、氢键模式和多次独立搜索验证。",
        }
    if topology == "螺环或桥环骨架":
        return {
            "method": "rmsd",
            "threshold": 0.75,
            "energy_window": 2.0,
            "label": "刚性骨架取向分析",
            "reason": "刚性骨架构象变化更适合先比较整体重原子形状。",
            "caution": "应额外检查取代基方向和关键二面角。",
        }
    return {
        "method": "rmsd",
        "threshold": 0.50,
        "energy_window": 2.0,
        "label": "低能候选去重分析",
        "reason": "该体系预期构象自由度有限，重原子 RMSD 足以识别主要几何差异。",
        "caution": "若低能候选数量仍多，应升级采样或精修方案。",
    }


def cluster_conformer_ensemble(run_id: str, result: dict, method: str, energy_window: float, threshold: float) -> dict:
    sdf_path = OUTPUT_DIR / run_id / "conformers.sdf"
    if not sdf_path.exists():
        raise ValueError("缺少可用于构象家族分析的 SDF 结果文件。")

    rows = sorted(result["conformers"], key=lambda row: row["rank"])
    selected_rows = [row for row in rows if row["relative_energy_kcal_mol"] <= energy_window + 1e-8]
    if not selected_rows:
        raise ValueError("当前能量窗口内没有可分析构象。")

    molecules = [mol for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False) if mol is not None]
    if len(molecules) < len(rows):
        raise ValueError("SDF 构象数量与结果索引不一致。")

    ensemble = Chem.RemoveHs(Chem.Mol(molecules[selected_rows[0]["rank"] - 1]))
    ensemble.RemoveAllConformers()
    for row in selected_rows:
        heavy = Chem.RemoveHs(Chem.Mol(molecules[row["rank"] - 1]))
        ensemble.AddConformer(heavy.GetConformer(), assignId=True)

    requested_method = method
    method_note = ""
    if method == "tfd":
        non_ring, ring = TorsionFingerprints.CalculateTorsionLists(ensemble)
        if not non_ring and not ring:
            method = "rmsd"
            method_note = "分子没有可用于 TFD 的有效扭转，自动改用重原子 RMSD 聚类。"

    if len(selected_rows) == 1:
        clusters = ((0,),)
    else:
        if method == "tfd":
            distances = TorsionFingerprints.GetTFDMatrix(ensemble)
        else:
            distances = rdMolAlign.GetAllConformerBestRMS(ensemble, numThreads=0)
        clusters = Butina.ClusterData(distances, len(selected_rows), threshold, isDistData=True, reordering=True)

    families = []
    for members in clusters:
        member_rows = sorted((selected_rows[index] for index in members), key=lambda row: row["relative_energy_kcal_mol"])
        representative = member_rows[0]
        families.append(
            {
                "representative_rank": representative["rank"],
                "minimum_relative_energy_kcal_mol": representative["relative_energy_kcal_mol"],
                "member_count": len(member_rows),
                "member_ranks": [row["rank"] for row in member_rows],
            }
        )
    families.sort(key=lambda family: (family["minimum_relative_energy_kcal_mol"], -family["member_count"]))
    for index, family in enumerate(families, start=1):
        family["family_id"] = index

    cutoffs = [0.5, 1.0, 2.0, 5.0]
    energy_counts = [
        {"energy_window": cutoff, "count": sum(row["relative_energy_kcal_mol"] <= cutoff + 1e-8 for row in rows)}
        for cutoff in cutoffs
    ]
    return {
        "total_conformers": len(rows),
        "energy_window": energy_window,
        "selected_conformers": len(selected_rows),
        "energy_counts": energy_counts,
        "requested_method": requested_method,
        "method": method,
        "method_label": "扭转指纹 TFD" if method == "tfd" else "重原子 RMSD",
        "threshold": threshold,
        "method_note": method_note,
        "family_count": len(families),
        "families": families,
    }


def generate_conformers(mol: Chem.Mol, num_conformers: int, seed: int) -> tuple[Chem.Mol, list[dict], str]:
    three_d_mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.35
    params.numThreads = 0
    conf_ids = list(AllChem.EmbedMultipleConfs(three_d_mol, numConfs=num_conformers, params=params))
    if not conf_ids:
        raise ValueError("RDKit 未能为该分子生成三维构象。")

    if AllChem.MMFFHasAllMoleculeParams(three_d_mol):
        optimisation = AllChem.MMFFOptimizeMoleculeConfs(three_d_mol, numThreads=0, maxIters=500)
        force_field = "MMFF94"
    else:
        optimisation = AllChem.UFFOptimizeMoleculeConfs(three_d_mol, numThreads=0, maxIters=500)
        force_field = "UFF"

    rows = []
    for conf_id, (status, energy) in zip(conf_ids, optimisation):
        rows.append(
            {
                "conf_id": conf_id,
                "energy_kcal_mol": float(energy),
                "converged": status == 0,
            }
        )
    rows.sort(key=lambda row: row["energy_kcal_mol"])
    minimum = rows[0]["energy_kcal_mol"]
    for rank, row in enumerate(rows, start=1):
        relative_energy = max(0.0, row["energy_kcal_mol"] - minimum)
        row["rank"] = rank
        row["energy_kcal_mol"] = round(row["energy_kcal_mol"], 4)
        row["relative_energy_kcal_mol"] = round(relative_energy, 4)
    return three_d_mol, rows, force_field


def parse_crest_ensemble(path: Path, template: Chem.Mol) -> tuple[Chem.Mol, list[dict]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    molecule = Chem.AddHs(template)
    atoms = molecule.GetNumAtoms()
    offset = 0
    energies: list[float] = []
    conformers = []
    while offset < len(lines):
        count = int(lines[offset].strip())
        if count != atoms:
            raise ValueError("CREST 构象原子数与输入结构不一致。")
        comment = lines[offset + 1].strip().split()
        energies.append(float(comment[0]))
        conformer = Chem.Conformer(atoms)
        for atom_index, line in enumerate(lines[offset + 2 : offset + 2 + atoms]):
            fields = line.split()
            conformer.SetAtomPosition(atom_index, Point3D(float(fields[1]), float(fields[2]), float(fields[3])))
        conformers.append(conformer)
        offset += atoms + 2
    molecule.RemoveAllConformers()
    minimum = min(energies)
    rows = []
    for index, (energy, conformer) in enumerate(zip(energies, conformers), start=1):
        conf_id = molecule.AddConformer(conformer, assignId=True)
        rows.append(
            {
                "conf_id": conf_id,
                "rank": index,
                "energy_hartree": round(energy, 10),
                "relative_energy_kcal_mol": round((energy - minimum) * 627.509474, 4),
                "energy_kcal_mol": round((energy - minimum) * 627.509474, 4),
                "converged": True,
            }
        )
    rows.sort(key=lambda row: row["relative_energy_kcal_mol"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return molecule, rows


def save_results(
    molecule_name: str,
    canonical_smiles: str,
    molecule: Chem.Mol,
    conformers: list[dict],
    properties: dict,
    method: str,
    energy_label: str = "Approximate_Energy_kcal_mol",
    run_id: str | None = None,
) -> tuple[str, dict]:
    run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = run_dir / "conformers.sdf"
    csv_path = run_dir / "conformer_energies.csv"

    writer = Chem.SDWriter(str(sdf_path))
    molecule.SetProp("_Name", molecule_name)
    molecule.SetProp("Canonical_SMILES", canonical_smiles)
    molecule.SetProp("Conformer_Method", method)
    for row in conformers:
        molecule.SetProp("Conformer_Rank", str(row["rank"]))
        molecule.SetProp(energy_label, str(row["energy_kcal_mol"]))
        molecule.SetProp("Relative_Energy_kcal_mol", str(row["relative_energy_kcal_mol"]))
        for property_name in (
            "source_rank",
            "family_id",
            "source_relative_energy_kcal_mol",
            "gibbs_free_energy_hartree",
            "relative_gibbs_free_energy_kcal_mol",
            "boltzmann_population_percent",
            "cumulative_population_percent",
            "in_final_ensemble",
            "merged_source_ranks",
            "merged_family_ids",
            "merged_count",
            "significant_imaginary_frequencies_cm_1",
        ):
            if property_name in row:
                molecule.SetProp(property_name, str(row[property_name]))
        writer.write(molecule, confId=row["conf_id"])
    writer.close()

    fieldnames = [
        "rank",
        "conf_id",
        "source_rank",
        "family_id",
        "source_relative_energy_kcal_mol",
        "merged_source_ranks",
        "merged_family_ids",
        "merged_count",
        "significant_imaginary_frequencies_cm_1",
        "energy_hartree",
        "energy_kcal_mol",
        "gibbs_free_energy_hartree",
        "relative_energy_kcal_mol",
        "relative_gibbs_free_energy_kcal_mol",
        "boltzmann_population_percent",
        "cumulative_population_percent",
        "in_final_ensemble",
        "converged",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(conformers)

    summary = run_dir / "summary.txt"
    summary.write_text(
        "\n".join(
            [
                molecule_name,
                f"Canonical SMILES: {canonical_smiles}",
                f"Formula: {properties['formula']}",
                f"Conformers retained: {len(conformers)}",
                f"Method: {method}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    files = {
        "sdf": f"/api/download/{run_id}/conformers.sdf",
        "csv": f"/api/download/{run_id}/conformer_energies.csv",
    }
    return run_id, files


@app.get("/")
def index():
    return redirect("/build")


@app.get("/build")
def build_page():
    return send_file(ROOT / "web" / "build.html")


@app.get("/tasks")
def tasks_page():
    return send_file(ROOT / "web" / "tasks.html")


@app.get("/tasks/<run_id>")
def task_detail_page(run_id: str):
    return send_file(ROOT / "web" / "task-detail.html")


@app.get("/results/<run_id>")
def results_page(run_id: str):
    return send_file(ROOT / "web" / "results.html")


@app.get("/vendor/<path:filename>")
def vendor_asset(filename: str):
    target = (VENDOR_DIR / filename).resolve()
    if VENDOR_DIR.resolve() not in target.parents or not target.exists():
        return jsonify({"error": "Vendor asset not found."}), 404
    return send_file(target)


@app.get("/styles/<path:filename>")
def style_asset(filename: str):
    target = (STYLE_DIR / filename).resolve()
    if STYLE_DIR.resolve() not in target.parents or not target.exists():
        return jsonify({"error": "Stylesheet not found."}), 404
    return send_file(target)


@app.get("/ketcher/<path:filename>")
def ketcher_asset(filename: str):
    target = (KETCHER_DIR / filename).resolve()
    if KETCHER_DIR.resolve() not in target.parents or not target.exists():
        return jsonify({"error": "Ketcher asset not found."}), 404
    return send_file(target)


@app.get("/api/example")
def example():
    return jsonify(EBF)


@app.get("/api/tools")
def tools():
    return jsonify(available_tools())


@app.post("/api/profile")
def profile():
    payload = request.get_json(force=True) or {}
    mol = Chem.MolFromSmiles((payload.get("smiles") or "").strip())
    if mol is None:
        return jsonify({"error": "无法解析该 SMILES，请检查结构字符串。"}), 400
    return jsonify({"profile": profile_molecule(mol), "properties": molecular_properties(mol), "svg": draw_svg(mol)})


@app.post("/api/crest/plan")
def crest_plan():
    payload = request.get_json(force=True) or {}
    mol = Chem.MolFromSmiles((payload.get("smiles") or "").strip())
    quality = payload.get("search_quality") or "quick"
    environment_key = payload.get("sampling_environment") or "gas"
    environment = sampling_environment(environment_key)
    if mol is None:
        return jsonify({"error": "无法解析该 SMILES，请检查结构字符串。"}), 400
    if quality not in {"quick", "full", "smoke"}:
        return jsonify({"error": "未知的 CREST 搜索等级。"}), 400
    if environment is None:
        return jsonify({"error": "未知的构象采样环境。"}), 400
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    fingerprint = task_fingerprint(canonical_smiles, quality, charge, environment_key)
    estimate, basis = estimate_task_seconds(mol, quality, fingerprint)
    duplicate = find_reusable_task(fingerprint, canonical_smiles, quality, charge, environment_key)
    return jsonify(
        {
            "canonical_smiles": canonical_smiles,
            "formal_charge": charge,
            "estimated_seconds": estimate,
            "estimate_basis": basis,
            "method": f"{CREST_METHOD} · {environment['label']}",
            "sampling_environment": environment_key,
            "sampling_environment_label": environment["label"],
            "sampling_environment_description": environment["description"],
            "duplicate_task": task_view(duplicate) if duplicate else None,
        }
    )


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(force=True) or {}
    smiles = (payload.get("smiles") or "").strip()
    name = (payload.get("name") or "Untitled molecule").strip()
    try:
        num_conformers = max(1, min(int(payload.get("num_conformers") or 30), 200))
        seed = int(payload.get("seed") or 20260525)
    except (TypeError, ValueError):
        return jsonify({"error": "构象数量和随机种子必须为整数。"}), 400
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "无法解析该 SMILES，请检查结构字符串。"}), 400

    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    properties = molecular_properties(mol)
    try:
        three_d_mol, conformers, force_field = generate_conformers(mol, num_conformers, seed)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    method = f"ETKDGv3 + {force_field}"
    run_id, files = save_results(name, canonical_smiles, three_d_mol, conformers, properties, method)
    result = {
            "run_id": run_id,
            "name": name,
            "canonical_smiles": canonical_smiles,
            "properties": properties,
            "profile": profile_molecule(mol),
            "svg": draw_svg(mol),
            "conformers": conformers,
            "requested_conformers": num_conformers,
            "retained_conformers": len(conformers),
            "method": method,
            "method_scope": "快速预览：力场局部优化结果，不代表系统构象搜索或量子化学能量排序。",
            "files": files,
        }
    save_result_payload(run_id, result)
    return jsonify(result)


@app.post("/api/crest/start")
def start_crest():
    payload = request.get_json(force=True) or {}
    smiles = (payload.get("smiles") or "").strip()
    name = (payload.get("name") or "Untitled molecule").strip()
    search_quality = payload.get("search_quality") or "full"
    environment_key = payload.get("sampling_environment") or "gas"
    environment = sampling_environment(environment_key)
    if search_quality not in {"quick", "full", "smoke"}:
        return jsonify({"error": "未知的 CREST 搜索等级。"}), 400
    if environment is None:
        return jsonify({"error": "未知的构象采样环境。"}), 400
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "无法解析该 SMILES，请检查结构字符串。"}), 400
    if not CREST_BIN.exists() or not XTB_BIN.exists():
        return jsonify({"error": "CREST/GFN2-xTB 计算环境尚未安装。"}), 503
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    coordinate_smiles = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    fingerprint = task_fingerprint(canonical_smiles, search_quality, formal_charge, environment_key)
    if not payload.get("force"):
        previous = find_reusable_task(fingerprint, canonical_smiles, search_quality, formal_charge, environment_key)
        if previous is not None:
            response = task_view(previous)
            response["duplicate"] = True
            response["duplicate_message"] = (
                "相同结构和协议的任务正在运行，已返回现有任务。"
                if previous["status"] == "running"
                else "相同结构和协议已有完成结果，已复用现有任务。"
            )
            return jsonify(response)

    initial, _, _ = generate_conformers(mol, 1, 20260525)
    estimate, estimate_basis = estimate_task_seconds(mol, search_quality, fingerprint)
    run_id = f"crest-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True)
    input_xyz = run_dir / "input.xyz"
    Chem.MolToXYZFile(initial, str(input_xyz), confId=0)
    log_handle = (run_dir / "crest.log").open("w", encoding="utf-8")
    env = os.environ.copy()
    compute_paths = ":".join(str(path) for path in COMPUTE_BIN_DIRS)
    env["PATH"] = f"{compute_paths}:{env.get('PATH', '')}"
    command = [str(CREST_BIN), "input.xyz", "--legacy", "--gfn2", "-xnam", str(XTB_BIN), "-T", "4", "-chrg", str(formal_charge)]
    command.extend(environment["crest_args"])
    if search_quality == "quick":
        command.append("-quick")
    elif search_quality == "smoke":
        command.append("-mquick")
    process = subprocess.Popen(
        command,
        cwd=run_dir,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    task = {
        "task_type": "crest",
        "run_id": run_id,
        "fingerprint": fingerprint,
        "name": name,
        "canonical_smiles": canonical_smiles,
        "coordinate_smiles": coordinate_smiles,
        "formal_charge": formal_charge,
        "search_quality": search_quality,
        "sampling_environment": environment_key,
        "method": CREST_METHOD,
        "status": "running",
        "created_at": time.time(),
        "estimated_seconds": estimate,
        "estimate_basis": estimate_basis,
        "pid": process.pid,
    }
    save_task(task)
    JOBS[run_id] = {
        "process": process,
        "log_handle": log_handle,
        "mol": mol,
        "name": name,
        "canonical_smiles": canonical_smiles,
        "properties": molecular_properties(mol),
        "search_quality": search_quality,
        "sampling_environment": environment_key,
    }
    response = task_view(task)
    response["profile"] = profile_molecule(mol)
    response["duplicate"] = False
    return jsonify(response)


def build_crest_response(task: dict) -> dict:
    cached = load_result_payload(task["run_id"])
    if cached is not None:
        return cached
    mol = Chem.MolFromSmiles(task["canonical_smiles"])
    coordinate_template = Chem.MolFromSmiles(task.get("coordinate_smiles", task["canonical_smiles"]))
    properties = molecular_properties(mol)
    molecule, conformers = parse_crest_ensemble(OUTPUT_DIR / task["run_id"] / "crest_conformers.xyz", coordinate_template)
    _, files = save_results(
        task["name"],
        task["canonical_smiles"],
        molecule,
        conformers,
        properties,
        CREST_METHOD,
        energy_label="Relative_Energy_kcal_mol",
        run_id=task["run_id"],
    )
    response = task_view(task)
    environment_label = response["sampling_environment_label"]
    single_conformer_converged = bool(task.get("single_conformer_converged"))
    response.update(
        {
            "properties": properties,
            "profile": profile_molecule(mol),
            "svg": draw_svg(mol),
            "conformers": conformers,
            "requested_conformers": None,
            "retained_conformers": len(conformers),
            "single_conformer_converged": single_conformer_converged,
            "method_scope": (
                f"单一主导构象结果：在{environment_label}下，CREST 采样与去重后仅保留 1 个低能构象；"
                "后续 GC 交叉步骤因构象数不足停止。该结果适合作为刚性分子的主导构象参考，"
                "重点构象仍建议在相同环境定义下使用 ORCA 精修。"
                if single_conformer_converged
                else (
                    f"研究搜索：在{environment_label}下由增强采样与半经验量子能量得到构象集合；"
                    "重点构象仍建议在相同环境定义下使用 ORCA 精修。"
                )
            ),
            "files": files,
        }
    )
    save_result_payload(task["run_id"], response)
    return response


def conformer_from_xyz(path: Path, atom_count: int) -> Chem.Conformer:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or int(lines[0].strip()) != atom_count:
        raise ValueError("ORCA 优化坐标与输入结构原子数不一致。")
    conformer = Chem.Conformer(atom_count)
    for atom_index, line in enumerate(lines[2 : 2 + atom_count]):
        fields = line.split()
        conformer.SetAtomPosition(atom_index, Point3D(float(fields[1]), float(fields[2]), float(fields[3])))
    return conformer


def deduplicate_refined_candidates(molecule: Chem.Mol, candidates: list[dict], profile: dict) -> tuple[list[dict], dict]:
    recommendation = ensemble_analysis_recommendation(profile)
    method = recommendation["method"]
    threshold = recommendation["threshold"]
    ensemble = Chem.RemoveHs(Chem.Mol(molecule))
    if method == "tfd":
        non_ring, ring = TorsionFingerprints.CalculateTorsionLists(ensemble)
        if not non_ring and not ring:
            method = "rmsd"
            threshold = 0.50
    if len(candidates) == 1:
        clusters = ((0,),)
    else:
        distances = (
            TorsionFingerprints.GetTFDMatrix(ensemble)
            if method == "tfd"
            else rdMolAlign.GetAllConformerBestRMS(ensemble, numThreads=0)
        )
        clusters = Butina.ClusterData(distances, len(candidates), threshold, isDistData=True, reordering=True)
    independent = []
    for members in clusters:
        rows = sorted((candidates[index] for index in members), key=lambda row: row["gibbs_free_energy_hartree"])
        representative = dict(rows[0])
        representative["merged_source_ranks"] = [row["source_rank"] for row in rows]
        representative["merged_family_ids"] = [row["family_id"] for row in rows]
        representative["merged_count"] = len(rows)
        independent.append(representative)
    independent.sort(key=lambda row: row["gibbs_free_energy_hartree"])
    return independent, {
        "method": method,
        "method_label": "扭转指纹 TFD" if method == "tfd" else "重原子 RMSD",
        "threshold": threshold,
        "input_count": len(candidates),
        "independent_count": len(independent),
    }


def build_orca_response(task: dict) -> dict:
    cached = load_result_payload(task["run_id"])
    if cached is not None:
        return cached
    source_sdf = OUTPUT_DIR / task["source_run_id"] / "conformers.sdf"
    source_molecules = [mol for mol in Chem.SDMolSupplier(str(source_sdf), removeHs=False) if mol is not None]
    if not source_molecules:
        raise ValueError("无法读取用于 ORCA 精修的来源构象。")
    all_completed = task.get("completed_conformers", [])
    if not all_completed:
        raise ValueError("ORCA 精修任务尚未生成可用能量结果。")
    is_partial_result = task["status"] != "completed" or len(all_completed) < len(task["selected_conformers"])
    excluded_imaginary = [
        item for item in all_completed if item.get("significant_imaginary_frequencies_cm_1", [])
    ]
    completed = [
        item for item in all_completed if not item.get("significant_imaginary_frequencies_cm_1", [])
    ]
    if not completed:
        raise ValueError("所有 ORCA 精修结构均存在显著虚频，不能据此计算玻尔兹曼分布。")
    candidate_molecule = Chem.Mol(source_molecules[0])
    candidate_molecule.RemoveAllConformers()
    candidates = []
    source_rows = {row["rank"]: row for row in load_result_payload(task["source_run_id"])["conformers"]}
    for item in completed:
        conformer = conformer_from_xyz(OUTPUT_DIR / task["run_id"] / item["xyz_file"], candidate_molecule.GetNumAtoms())
        conf_id = candidate_molecule.AddConformer(conformer, assignId=True)
        candidates.append(
            {
                "conf_id": conf_id,
                "source_rank": item["source_rank"],
                "family_id": item["family_id"],
                "energy_hartree": round(item["energy_hartree"], 10),
                "energy_kcal_mol": round(item["energy_hartree"] * 627.509474, 4),
                "gibbs_free_energy_hartree": round(item["gibbs_free_energy_hartree"], 10),
                "source_relative_energy_kcal_mol": source_rows[item["source_rank"]]["relative_energy_kcal_mol"],
                "significant_imaginary_frequencies_cm_1": item.get("significant_imaginary_frequencies_cm_1", []),
                "converged": True,
            }
        )
    canonical_mol = Chem.MolFromSmiles(task["canonical_smiles"])
    independent, deduplication = deduplicate_refined_candidates(candidate_molecule, candidates, profile_molecule(canonical_mol))
    minimum_g = independent[0]["gibbs_free_energy_hartree"]
    temperature = float(task["temperature_kelvin"])
    weights = [
        math.exp(-((row["gibbs_free_energy_hartree"] - minimum_g) * 627.509474) / (GAS_CONSTANT_KCAL * temperature))
        for row in independent
    ]
    normalization = sum(weights)
    molecule = Chem.Mol(source_molecules[0])
    molecule.RemoveAllConformers()
    conformers = []
    cumulative = 0.0
    reached_coverage = False
    for rank, (row, weight) in enumerate(zip(independent, weights), start=1):
        conf_id = molecule.AddConformer(candidate_molecule.GetConformer(row["conf_id"]), assignId=True)
        relative_g = (row["gibbs_free_energy_hartree"] - minimum_g) * 627.509474
        population = weight / normalization * 100
        cumulative += population
        include = not reached_coverage
        if cumulative >= 95.0:
            reached_coverage = True
        conformers.append(
            {
                **row,
                "rank": rank,
                "conf_id": conf_id,
                "relative_energy_kcal_mol": round(relative_g, 4),
                "relative_gibbs_free_energy_kcal_mol": round(relative_g, 4),
                "boltzmann_population_percent": round(population, 3),
                "cumulative_population_percent": round(cumulative, 3),
                "in_final_ensemble": include,
            }
        )
    properties = molecular_properties(canonical_mol)
    _, files = save_results(
        task["name"],
        task["canonical_smiles"],
        molecule,
        conformers,
        properties,
        ORCA_METHOD,
        energy_label="ORCA_Electronic_Energy_kcal_mol",
        run_id=task["run_id"],
    )
    response = task_view(task)
    response.update(
        {
            "properties": properties,
            "profile": profile_molecule(canonical_mol),
            "svg": draw_svg(canonical_mol),
            "conformers": conformers,
            "requested_conformers": len(task["selected_conformers"]),
            "retained_conformers": len(conformers),
            "completed_refinement_count": len(all_completed),
            "is_partial_result": is_partial_result,
            "source_run_id": task["source_run_id"],
            "temperature_kelvin": temperature,
            "deduplication": deduplication,
            "stationary_point_validation": {
                "imaginary_frequency_threshold_cm_1": -20.0,
                "excluded_count": len(excluded_imaginary),
                "excluded_source_ranks": [row["source_rank"] for row in excluded_imaginary],
            },
            "final_ensemble_count": sum(row["in_final_ensemble"] for row in conformers),
            "population_coverage_target_percent": 95.0,
            "method_scope": (
                (
                    f"部分分析：ORCA 任务已停止，目前仅有 {len(all_completed)} / {len(task['selected_conformers'])} "
                    f"个代表构象完成优化与频率计算；在{response['sampling_environment_label']}定义下排除显著虚频后，"
                    f"按 {temperature:.2f} K 计算的排序与玻尔兹曼比例仅适用于已完成子集，不能视为最终低能构象集合。"
                    if is_partial_result
                    else f"最终分析：在{response['sampling_environment_label']}定义下进行 ORCA r2SCAN-3c 优化与频率计算；"
                    f"排除存在显著虚频的结构后，按 {temperature:.2f} K 的相对吉布斯自由能排序并计算玻尔兹曼分布。"
                )
            ),
            "files": files,
        }
    )
    save_result_payload(task["run_id"], response)
    return response


def refresh_crest_task(task: dict) -> dict:
    if task["status"] != "running":
        if task["status"] == "failed" and crest_single_conformer_gc_stop(task["run_id"]):
            return mark_crest_single_conformer_converged(task)
        return task
    if process_is_running(task):
        return task
    job = JOBS.get(task["run_id"])
    if job is not None and not job.get("closed"):
        job["log_handle"].close()
        job["closed"] = True
    log_path = OUTPUT_DIR / task["run_id"] / "crest.log"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    conformers_path = OUTPUT_DIR / task["run_id"] / "crest_conformers.xyz"
    has_conformers = conformers_path.exists()
    single_conformer_gc_stop = has_conformers and "Not enough structures to perform GC!" in log_text
    if has_conformers and "CREST terminated normally." in log_text:
        task["status"] = "completed"
        task["progress"] = "搜索完成，可查看构象集合。"
    elif single_conformer_gc_stop:
        return mark_crest_single_conformer_converged(task)
    else:
        task["status"] = "failed"
        task["progress"] = "任务未正常完成，请查看运行日志。"
        task["error"] = task["progress"]
    task["completed_at"] = time.time()
    task["duration_seconds"] = max(0, int(task["completed_at"] - task["created_at"]))
    save_task(task)
    return task


def refresh_orca_task(task: dict) -> dict:
    if task["status"] != "running" or process_is_running(task):
        return task
    job = JOBS.get(task["run_id"])
    if job is not None and not job.get("closed"):
        job["log_handle"].close()
        job["closed"] = True
    updated = load_task(task["run_id"]) or task
    if updated["status"] == "running":
        updated["status"] = "failed"
        updated["progress"] = "ORCA 精修进程已结束，但未生成完整结果。"
        updated["error"] = updated["progress"]
        updated["completed_at"] = time.time()
        updated["duration_seconds"] = max(0, int(updated["completed_at"] - updated["created_at"]))
        save_task(updated)
    return updated


def crest_single_conformer_gc_stop(run_id: str) -> bool:
    run_dir = OUTPUT_DIR / run_id
    conformers_path = run_dir / "crest_conformers.xyz"
    log_path = run_dir / "crest.log"
    if not conformers_path.exists() or not log_path.exists():
        return False
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    return "Not enough structures to perform GC!" in log_text


def mark_crest_single_conformer_converged(task: dict) -> dict:
    task["status"] = "completed"
    task["progress"] = "CREST 收敛到单一主导构象；GC 交叉步骤因构象数不足而停止，已保留可用结果。"
    task["single_conformer_converged"] = True
    task.pop("error", None)
    if "completed_at" not in task:
        task["completed_at"] = time.time()
    task["duration_seconds"] = max(0, int(task["completed_at"] - task["created_at"]))
    save_task(task)
    return task


def refresh_task(task: dict) -> dict:
    return refresh_orca_task(task) if task.get("task_type") == "orca" else refresh_crest_task(task)


@app.get("/api/tasks")
def list_tasks():
    tasks = []
    for path in OUTPUT_DIR.glob("*/task.json"):
        task = load_task(path.parent.name)
        if task is not None:
            tasks.append(task_view(refresh_task(task)))
    tasks.sort(key=lambda task: task["created_at"], reverse=True)
    return jsonify({"tasks": tasks})


@app.get("/api/tasks/<run_id>")
def task_detail(run_id: str):
    task = load_task(run_id)
    if task is None:
        return jsonify({"error": "任务中心未找到该计算任务。"}), 404
    task = refresh_task(task)
    response = task_view(task)
    response["pipeline"] = task_pipeline(task)
    response["log_excerpt"] = log_excerpt(run_id)
    return jsonify(response)


@app.delete("/api/tasks/<run_id>")
def delete_task(run_id: str):
    task = load_task(run_id)
    if task is None:
        return jsonify({"error": "任务中心未找到该历史任务。"}), 404
    task = refresh_task(task)
    if task["status"] == "running":
        return jsonify({"error": "运行中的任务不能删除，请先停止任务。"}), 409
    run_dir = (OUTPUT_DIR / run_id).resolve()
    if OUTPUT_DIR.resolve() not in run_dir.parents:
        return jsonify({"error": "任务目录不合法。"}), 400
    JOBS.pop(run_id, None)
    shutil.rmtree(run_dir)
    return jsonify({"run_id": run_id, "status": "deleted"})


@app.get("/api/crest/status/<run_id>")
def crest_status(run_id: str):
    task = load_task(run_id)
    if task is None:
        return jsonify({"error": "任务中心未找到该 CREST 任务。"}), 404
    task = refresh_crest_task(task)
    if task["status"] == "completed":
        return jsonify(build_crest_response(task))
    return jsonify(task_view(task))


@app.get("/api/results/<run_id>")
def result_data(run_id: str):
    cached = load_result_payload(run_id)
    if cached is not None:
        return jsonify(enrich_result_context(run_id, cached))
    task = load_task(run_id)
    if task is None:
        return jsonify({"error": "未找到该计算结果。"}), 404
    task = refresh_task(task)
    has_stopped_orca_results = (
        task.get("task_type") == "orca"
        and task["status"] == "stopped"
        and bool(task.get("completed_conformers"))
    )
    if task["status"] != "completed" and not has_stopped_orca_results:
        return jsonify({"error": "任务尚未完成，结果分析暂不可用。", "status": task["status"]}), 409
    try:
        result = build_orca_response(task) if task.get("task_type") == "orca" else build_crest_response(task)
    except ValueError as exc:
        return jsonify({"error": str(exc), "status": "analysis_failed"}), 422
    return jsonify(enrich_result_context(run_id, result))


@app.get("/api/results/<run_id>/interpretation/<int:rank>")
def conformer_interpretation(run_id: str, rank: int):
    result = load_result_payload(run_id)
    if result is None:
        task = load_task(run_id)
        if task is None:
            return jsonify({"error": "未找到该计算结果。"}), 404
        task = refresh_task(task)
        has_stopped_orca_results = (
            task.get("task_type") == "orca"
            and task["status"] == "stopped"
            and bool(task.get("completed_conformers"))
        )
        if task["status"] != "completed" and not has_stopped_orca_results:
            return jsonify({"error": "任务尚未完成，构象解释暂不可用。"}), 409
        try:
            result = build_orca_response(task) if task.get("task_type") == "orca" else build_crest_response(task)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422
    result = enrich_result_context(run_id, result)
    rows = {row["rank"]: row for row in result["conformers"]}
    if rank not in rows:
        return jsonify({"error": "当前结果中不存在该构象。"}), 404
    sdf_path = OUTPUT_DIR / run_id / "conformers.sdf"
    molecules = [molecule for molecule in Chem.SDMolSupplier(str(sdf_path), removeHs=False) if molecule is not None]
    if len(molecules) < max(rank, 1):
        return jsonify({"error": "无法读取用于解释的三维构象。"}), 422
    molecule = Chem.Mol(molecules[rank - 1])
    reference = Chem.Mol(molecules[0])
    row = rows[rank]
    is_orca = result.get("task_type") == "orca"
    is_crest = str(result.get("method", "")).startswith("CREST")
    evidence_level = "精修证据" if is_orca else ("初筛证据" if is_crest else "预览证据")
    energy_label = "ΔG" if is_orca else "ΔE"
    energy_value = (
        row.get("relative_gibbs_free_energy_kcal_mol")
        if is_orca
        else row.get("relative_energy_kcal_mol")
    )
    profile = result.get("profile") or profile_molecule(Chem.RemoveHs(molecule))
    shape = conformer_shape_description(molecule, profile)
    reference_shape = conformer_shape_description(reference, profile)
    torsions = conformer_torsions(molecule, reference)
    decision = conformer_decision_explanation(
        rank,
        is_orca,
        bool(result.get("is_partial_result")),
        row,
        shape,
        reference_shape,
        torsions,
    )
    hydrogen_bonds = intramolecular_hbond_candidates(molecule)
    scope = (
        "基于 ORCA 优化与频率分析后的几何和自由能排序；玻尔兹曼占比仅针对已采样并成功精修的构象集合。"
        if is_orca
        else (
            "基于 CREST/xTB 采样几何与初筛相对能量，用于选择后续精修候选；不代表最终自由能排序。"
            if is_crest
            else "基于快速力场预览几何，仅用于查看形状与扭转特征，不代表量子化学能量排序。"
        )
    )
    return jsonify(
        {
            "rank": rank,
            "evidence_level": evidence_level,
            "energy": {
                "label": energy_label,
                "relative_kcal_mol": energy_value,
                "rank": rank,
                "boltzmann_population_percent": row.get("boltzmann_population_percent") if is_orca else None,
                "temperature_kelvin": result.get("temperature_kelvin") if is_orca else None,
            },
            "family_id": row.get("family_id"),
            "source_rank": row.get("source_rank"),
            "decision": decision,
            "shape": shape,
            "reference_shape": reference_shape,
            "key_torsions": torsions[:8],
            "intramolecular_hydrogen_bonds": hydrogen_bonds,
            "tpsa_angstrom_squared": result.get("properties", {}).get("tpsa"),
            "tpsa_scope": "TPSA 是连接结构的拓扑极性指标，并非该构象的三维暴露极性面积。",
            "summary": decision["importance"],
            "scope": scope,
        }
    )


@app.post("/api/results/<run_id>/overlay")
def result_conformer_overlay(run_id: str):
    result = load_result_payload(run_id)
    if result is None:
        task = load_task(run_id)
        if task is None:
            return jsonify({"error": "未找到该精修结果。"}), 404
        task = refresh_task(task)
        has_partial_results = (
            task.get("task_type") == "orca"
            and task["status"] == "stopped"
            and bool(task.get("completed_conformers"))
        )
        if task["status"] != "completed" and not has_partial_results:
            return jsonify({"error": "ORCA 精修结果尚不可用于叠合。"}), 409
        try:
            result = build_orca_response(task) if task.get("task_type") == "orca" else None
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422
    if not result or result.get("task_type") != "orca":
        return jsonify({"error": "构象叠合当前仅支持 ORCA 高级精修结果。"}), 400
    payload = request.get_json(silent=True) or {}
    try:
        ranks = list(dict.fromkeys(int(rank) for rank in payload.get("ranks", [])))
        reference_rank = int(payload.get("reference_rank", ranks[0] if ranks else 0))
    except (TypeError, ValueError):
        return jsonify({"error": "叠合构象编号无效。"}), 400
    if len(ranks) < 2:
        return jsonify({"error": "请选择至少两个精修构象进行叠合。"}), 400
    if len(ranks) > 5:
        return jsonify({"error": "为保持视图清晰，一次最多叠合 5 个构象。"}), 400
    if reference_rank not in ranks:
        return jsonify({"error": "参考构象必须包含在已勾选的构象中。"}), 400
    rows = {row["rank"]: row for row in result["conformers"]}
    if any(rank not in rows for rank in ranks):
        return jsonify({"error": "所选构象不属于当前精修结果。"}), 400
    sdf_path = OUTPUT_DIR / run_id / "conformers.sdf"
    molecules = [mol for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False) if mol is not None]
    if len(molecules) < max(ranks):
        return jsonify({"error": "无法读取用于叠合的精修坐标。"}), 422
    reference = Chem.Mol(molecules[reference_rank - 1])
    anchor_options = overlay_alignment_anchors(reference)
    anchor_id = str(payload.get("anchor_id", "auto"))
    anchor = next((option for option in anchor_options if option["id"] == anchor_id), None)
    if anchor is None:
        return jsonify({"error": "选择的叠合锚点不可用于当前结构。"}), 400
    heavy_atom_indices = [atom.GetIdx() for atom in reference.GetAtoms() if atom.GetAtomicNum() > 1]
    atom_map = [(atom_idx, atom_idx) for atom_idx in anchor["atom_indices"]]
    if not atom_map:
        return jsonify({"error": "当前结构没有可用于重原子叠合的原子。"}), 422
    overlay_rows = []
    for rank in ranks:
        molecule = Chem.Mol(molecules[rank - 1])
        anchor_rmsd = 0.0 if rank == reference_rank else rdMolAlign.AlignMol(molecule, reference, atomMap=atom_map)
        whole_rmsd = overlay_atom_rmsd(molecule, reference, heavy_atom_indices)
        row = rows[rank]
        overlay_rows.append(
            {
                "rank": rank,
                "is_reference": rank == reference_rank,
                "rmsd_angstrom": round(float(anchor_rmsd), 4),
                "anchor_rmsd_angstrom": round(float(anchor_rmsd), 4),
                "whole_rmsd_after_alignment_angstrom": round(float(whole_rmsd), 4),
                "relative_gibbs_free_energy_kcal_mol": row["relative_gibbs_free_energy_kcal_mol"],
                "boltzmann_population_percent": row["boltzmann_population_percent"],
                "mol_block": Chem.MolToMolBlock(molecule),
            }
        )
    alignment_method = (
        f"局部锚点刚体叠合（RDKit AlignMol，{len(anchor['atom_indices'])} 个重原子）"
        if anchor["kind"] == "local"
        else "整体重原子一一对应的最小二乘刚体叠合（RDKit AlignMol）"
    )
    return jsonify(
        {
            "reference_rank": reference_rank,
            "alignment_method": alignment_method,
            "selected_anchor": anchor,
            "anchor_options": anchor_options,
            "interpretation_scope": (
                "锚点 RMSD 描述所选同源化学片段的拟合程度，全分子 RMSD 描述在该对齐下其余链段的展开差异；"
                "该比较用于解释构象形状，不替代扭转指纹、自由能排序或独立采样验证。"
            ),
            "is_partial_result": bool(result.get("is_partial_result")),
            "conformers": overlay_rows,
        }
    )


@app.post("/api/results/<run_id>/property-map")
def result_property_map(run_id: str):
    result = load_result_payload(run_id)
    if result is None:
        task = load_task(run_id)
        if task is None:
            return jsonify({"error": "未找到该精修结果。"}), 404
        task = refresh_task(task)
        if task.get("task_type") != "orca" or (
            task["status"] != "completed" and not (task["status"] == "stopped" and task.get("completed_conformers"))
        ):
            return jsonify({"error": "性质作图仅支持已有 ORCA 精修结果。"}), 409
        try:
            result = build_orca_response(task)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422
    payload = request.get_json(silent=True) or {}
    try:
        rank = int(payload.get("rank", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "所选构象编号无效。"}), 400
    map_type = str(payload.get("map_type", "esp")).lower()
    if map_type not in {"esp", "hydrophobic", "homo", "lumo"}:
        return jsonify({"error": "所选性质图类型无效。"}), 400
    try:
        run_dir, prefix, row = orca_property_artifacts(run_id, rank, result)
        property_dir = run_dir / "property_maps"
        property_dir.mkdir(exist_ok=True)
        molecule_supplier = Chem.SDMolSupplier(str(run_dir / "conformers.sdf"), removeHs=False)
        molecule = molecule_supplier[rank - 1] if rank <= len(molecule_supplier) else None
        if molecule is None:
            raise ValueError("无法读取用于性质图的精修构象坐标。")
        response = {
            "run_id": run_id,
            "rank": rank,
            "source_rank": row["source_rank"],
            "map_type": map_type,
            "relative_gibbs_free_energy_kcal_mol": row["relative_gibbs_free_energy_kcal_mol"],
            "boltzmann_population_percent": row["boltzmann_population_percent"],
        }
        if map_type in {"homo", "lumo"}:
            frontier = parse_frontier_orbitals(run_dir / f"{prefix}.out")
            orbital = frontier[map_type]
            cube_path = ensure_orca_orbital_cube(
                run_dir,
                prefix,
                orbital["index"],
                property_dir / f"rank_{rank:03d}_{map_type}_mo{orbital['index']}.cube",
            )
            response.update(
                {
                    "label": map_type.upper(),
                    "description": (
                        f"ORCA {ORCA_VALIDATED_VERSION} r2SCAN-3c 分子轨道；"
                        "红/蓝等值面表示相反轨道相位，不代表电荷正负。"
                    ),
                    "cube_url": f"/api/property/{run_id}/{cube_path.name}",
                    "render_mode": "orbital",
                    "orbital_index": orbital["index"],
                    "orbital_energy_ev": orbital["energy_ev"],
                    "homo_lumo_gap_ev": frontier["gap_ev"],
                    "suggested_isovalue": 0.03,
                }
            )
        elif map_type == "esp":
            cube_path = ensure_orca_esp_cube(run_dir, prefix, property_dir / f"rank_{rank:03d}_esp.cube")
            response.update(
                {
                    "label": "ESP",
                    "description": (
                        f"ORCA {ORCA_VALIDATED_VERSION} r2SCAN-3c 波函数静电势投影于 VDW 可视化表面；"
                        "负值区域偏富电子，正值区域偏缺电子；默认使用对称 ±0.05 a.u. 色标便于比较。"
                    ),
                    "cube_url": f"/api/property/{run_id}/{cube_path.name}",
                    "render_mode": "surface",
                    "unit": "a.u.",
                    # Nuclear-adjacent grid values are not a useful color range for a molecular surface.
                    "suggested_scale": 0.05,
                }
            )
        else:
            cube_path = write_empirical_hydrophobic_cube(molecule, property_dir / f"rank_{rank:03d}_hydrophobic.cube")
            response.update(
                {
                    "label": "经验疏水势",
                    "description": (
                        "RDKit Wildman-Crippen 原子 cLogP 贡献的三维高斯投影；"
                        "用于解释疏水区域分布，不是 ORCA 量子化学可观测量。"
                    ),
                    "cube_url": f"/api/property/{run_id}/{cube_path.name}",
                    "render_mode": "surface",
                    "unit": "relative cLogP field",
                    "suggested_scale": round(cube_value_extent(cube_path), 5),
                }
            )
        return jsonify(response)
    except (ValueError, subprocess.TimeoutExpired) as exc:
        return jsonify({"error": str(exc)}), 422


@app.get("/api/property/<run_id>/<filename>")
def property_cube(run_id: str, filename: str):
    property_dir = (OUTPUT_DIR / run_id / "property_maps").resolve()
    target = (property_dir / filename).resolve()
    if property_dir not in target.parents or not target.exists() or target.suffix != ".cube":
        return jsonify({"error": "性质体数据不存在。"}), 404
    return send_file(target, mimetype="text/plain")


@app.get("/api/results/<run_id>/ensemble")
def result_ensemble_analysis(run_id: str):
    result = load_result_payload(run_id)
    if result is None:
        task = load_task(run_id)
        if task is None:
            return jsonify({"error": "未找到该计算结果。"}), 404
        task = refresh_crest_task(task)
        if task["status"] != "completed":
            return jsonify({"error": "任务尚未完成，结果分析暂不可用。", "status": task["status"]}), 409
        result = build_crest_response(task)
    result = enrich_result_context(run_id, result)

    recommendation = ensemble_analysis_recommendation(result["profile"])
    method = request.args.get("method", "recommended")
    if method == "recommended":
        method = recommendation["method"]
    if method not in {"tfd", "rmsd"}:
        return jsonify({"error": "未知的构象聚类方法。"}), 400
    try:
        energy_window = float(request.args.get("energy_window", recommendation["energy_window"]))
        default_threshold = recommendation["threshold"] if method == recommendation["method"] else (0.10 if method == "tfd" else 0.75)
        threshold = float(request.args.get("threshold", default_threshold))
    except ValueError:
        return jsonify({"error": "能量窗口和聚类阈值必须为数字。"}), 400
    if energy_window <= 0 or energy_window > 20 or threshold <= 0:
        return jsonify({"error": "能量窗口或聚类阈值超出有效范围。"}), 400

    try:
        analysis = cluster_conformer_ensemble(run_id, result, method, energy_window, threshold)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    analysis["recommendation"] = recommendation
    analysis["topology"] = result["profile"]["topology"]
    return jsonify(analysis)


@app.get("/api/results/<run_id>/refinement-plan")
def get_refinement_plan(run_id: str):
    result = load_result_payload(run_id)
    if result is None and load_task(run_id) is None:
        return jsonify({"error": "未找到该计算结果。"}), 404
    plan = load_refinement_plan(run_id)
    return jsonify({"status": "not_prepared"} if plan is None else plan)


@app.post("/api/results/<run_id>/refinement-plan")
def prepare_refinement_plan(run_id: str):
    result = load_result_payload(run_id)
    if result is None:
        task = load_task(run_id)
        if task is None:
            return jsonify({"error": "未找到该计算结果。"}), 404
        task = refresh_crest_task(task)
        if task["status"] != "completed":
            return jsonify({"error": "任务尚未完成，不能准备高级精修。"}), 409
        result = build_crest_response(task)
    result = enrich_result_context(run_id, result)
    payload = request.get_json(force=True) or {}
    method = payload.get("method")
    if method not in {"tfd", "rmsd"}:
        return jsonify({"error": "请先完成有效的构象家族分析。"}), 400
    try:
        energy_window = float(payload.get("energy_window"))
        threshold = float(payload.get("threshold"))
        ranks = sorted({int(rank) for rank in payload.get("representative_ranks", [])})
    except (TypeError, ValueError):
        return jsonify({"error": "精修候选参数无效。"}), 400
    if not ranks:
        return jsonify({"error": "至少选择一个构象家族代表。"}), 400
    try:
        analysis = cluster_conformer_ensemble(run_id, result, method, energy_window, threshold)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    representatives = {family["representative_rank"]: family for family in analysis["families"]}
    if any(rank not in representatives for rank in ranks):
        return jsonify({"error": "所选构象不属于当前分析条件下的家族代表，请刷新后重试。"}), 400
    rows = {row["rank"]: row for row in result["conformers"]}
    selected = [
        {
            "rank": rank,
            "relative_energy_kcal_mol": rows[rank]["relative_energy_kcal_mol"],
            "family_id": representatives[rank]["family_id"],
            "family_member_count": representatives[rank]["member_count"],
        }
        for rank in ranks
    ]
    plan = {
        "status": "prepared",
        "source_run_id": run_id,
        "created_at": time.time(),
        "sampling_environment": result.get("sampling_environment", "gas"),
        "sampling_environment_label": result.get("sampling_environment_label", "气相"),
        "selection_count": len(selected),
        "selected_conformers": selected,
        "analysis_conditions": {
            "method": method,
            "method_label": analysis["method_label"],
            "energy_window_kcal_mol": energy_window,
            "threshold": threshold,
        },
        "next_stage": "ORCA 优化、频率计算与自由能分析",
        "proposed_output": "精修后相对自由能重排序、几何去重与指定温度玻尔兹曼分布",
        "message": "已保存高级精修候选方案，可提交 ORCA 优化与频率计算任务。",
    }
    save_refinement_plan(run_id, plan)
    return jsonify(plan)


@app.post("/api/results/<source_run_id>/refinement/start")
def start_orca_refinement(source_run_id: str):
    plan = load_refinement_plan(source_run_id)
    source_result = load_result_payload(source_run_id)
    if plan is None or plan.get("status") != "prepared":
        return jsonify({"error": "请先保存高级精修候选方案。"}), 400
    if source_result is None:
        return jsonify({"error": "未找到来源构象结果。"}), 404
    if not ORCA_BIN.exists():
        return jsonify(
            {
                "error": f"未检测到已验证的 ORCA {ORCA_VALIDATED_VERSION}，无法提交高级精修。",
                "installation_message": (
                    f"请从 ORCA 官方渠道下载 macOS 版 ORCA {ORCA_VALIDATED_VERSION}，"
                    f"并解压到项目目录 {ORCA_INSTALL_RELATIVE_PATH}/ 后重新打开页面。"
                ),
                "download_url": ORCA_DOWNLOAD_URL,
                "install_relative_path": ORCA_INSTALL_RELATIVE_PATH,
            }
        ), 503
    payload = request.get_json(silent=True) or {}
    try:
        temperature_kelvin = float(payload.get("temperature_kelvin", 298.15))
    except (TypeError, ValueError):
        return jsonify({"error": "温度必须为有效数值。"}), 400
    if temperature_kelvin < 50 or temperature_kelvin > 1000:
        return jsonify({"error": "温度需要位于 50 至 1000 K 之间。"}), 400
    fingerprint = orca_refinement_fingerprint(source_run_id, plan, temperature_kelvin)
    if not payload.get("force"):
        reusable = find_reusable_orca_task(fingerprint)
        if reusable is not None:
            response = task_view(reusable)
            response["duplicate"] = True
            return jsonify(response)
    source_task = load_task(source_run_id)
    molecule = Chem.MolFromSmiles(source_result["canonical_smiles"])
    formal_charge = source_task.get("formal_charge", 0) if source_task else sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    environment_key = plan["sampling_environment"]
    supplier = Chem.SDMolSupplier(str(OUTPUT_DIR / source_run_id / "conformers.sdf"), removeHs=False)
    selected = []
    run_id = f"orca-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True)
    for index, row in enumerate(plan["selected_conformers"], start=1):
        source_rank = row["rank"]
        source_molecule = supplier[source_rank - 1] if source_rank <= len(supplier) else None
        if source_molecule is None:
            shutil.rmtree(run_dir)
            return jsonify({"error": f"无法读取代表构象 #{source_rank}。"}), 422
        prefix = f"conf_{index:03d}"
        xyz_name = f"{prefix}_input.xyz"
        Chem.MolToXYZFile(source_molecule, str(run_dir / xyz_name), confId=0)
        (run_dir / f"{prefix}.inp").write_text(
            orca_input_text(formal_charge, environment_key, xyz_name, temperature_kelvin),
            encoding="utf-8",
        )
        selected.append(
            {
                "source_rank": source_rank,
                "family_id": row["family_id"],
                "source_relative_energy_kcal_mol": row["relative_energy_kcal_mol"],
            }
        )
    task = {
        "task_type": "orca",
        "run_id": run_id,
        "fingerprint": fingerprint,
        "source_run_id": source_run_id,
        "name": f"{source_result['name']} · 高级精修",
        "canonical_smiles": source_result["canonical_smiles"],
        "formal_charge": formal_charge,
        "sampling_environment": environment_key,
        "temperature_kelvin": temperature_kelvin,
        "selected_conformers": selected,
        "completed_conformers": [],
        "status": "running",
        "progress": f"正在准备 ORCA 优化与频率计算，共 {len(selected)} 个代表构象...",
        "created_at": time.time(),
        "estimated_seconds": max(120, len(selected) * 1200),
        "estimate_basis": "按代表构象数量及优化/频率计算估算；后续将由 ORCA 完成历史校准",
        "timing_schema": "per_conformer_v1",
    }
    save_task(task)
    log_handle = (run_dir / "worker.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(ORCA_WORKER), str(run_dir), str(ORCA_BIN)],
        cwd=run_dir,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    task["pid"] = process.pid
    save_task(task)
    JOBS[run_id] = {"process": process, "log_handle": log_handle}
    response = task_view(task)
    response["duplicate"] = False
    return jsonify(response)


@app.post("/api/crest/stop/<run_id>")
def stop_crest(run_id: str):
    task = load_task(run_id)
    if task is None:
        return jsonify({"error": "任务中心未找到该 CREST 任务。"}), 404
    task = refresh_crest_task(task)
    if task["status"] != "running":
        return jsonify(task_view(task))
    job = JOBS.get(run_id)
    try:
        os.killpg(int(task["pid"]), signal.SIGTERM)
    except ProcessLookupError:
        pass
    if job is not None and not job.get("closed"):
        job["log_handle"].close()
        job["closed"] = True
    task["status"] = "stopped"
    task["progress"] = "任务已停止。"
    task["completed_at"] = time.time()
    task["duration_seconds"] = max(0, int(task["completed_at"] - task["created_at"]))
    save_task(task)
    return jsonify(task_view(task))


@app.post("/api/tasks/<run_id>/stop")
def stop_task(run_id: str):
    task = load_task(run_id)
    if task is None:
        return jsonify({"error": "任务中心未找到该任务。"}), 404
    if task.get("task_type") != "orca":
        return stop_crest(run_id)
    task = refresh_task(task)
    if task["status"] != "running":
        return jsonify(task_view(task))
    try:
        os.killpg(int(task["pid"]), signal.SIGTERM)
    except ProcessLookupError:
        pass
    job = JOBS.get(run_id)
    if job is not None and not job.get("closed"):
        job["log_handle"].close()
        job["closed"] = True
    task["status"] = "stopped"
    task["progress"] = "ORCA 精修任务已停止。"
    task["completed_at"] = time.time()
    task["duration_seconds"] = max(0, int(task["completed_at"] - task["created_at"]))
    save_task(task)
    return jsonify(task_view(task))


@app.get("/api/download/<run_id>/<filename>")
def download(run_id: str, filename: str):
    if filename not in {"conformers.sdf", "conformer_energies.csv"}:
        return jsonify({"error": "Unknown result file."}), 404
    target = (OUTPUT_DIR / run_id / filename).resolve()
    if OUTPUT_DIR.resolve() not in target.parents or not target.exists():
        return jsonify({"error": "Result file not found."}), 404
    return send_file(target, as_attachment=True)


@app.get("/api/conformer/<run_id>/<int:rank>")
def conformer_model(run_id: str, rank: int):
    sdf_path = (OUTPUT_DIR / run_id / "conformers.sdf").resolve()
    if OUTPUT_DIR.resolve() not in sdf_path.parents or not sdf_path.exists() or rank < 1:
        return jsonify({"error": "Conformer not found."}), 404
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    if rank > len(supplier) or supplier[rank - 1] is None:
        return jsonify({"error": "Conformer not found."}), 404
    mol_block = Chem.MolToMolBlock(supplier[rank - 1])
    return Response(mol_block, mimetype="chemical/x-mdl-molfile")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5062, debug=False)
