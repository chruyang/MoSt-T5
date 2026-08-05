#!/usr/bin/env python3
"""Bounded, no-LMDB E3FP preflight for the frozen PCQM4Mv2 train-3D SDF.

This is an R1 diagnostic gate, not an adapter and not a dataset builder.  In
normal mode it streams at most 1,000 SDF records (128 by default) from the
compressed OGB archive, creates a tagged in-memory copy of each RDKit Mol,
and applies the frozen minimal explicit-H projection:

    RemoveHsParameters.removeDefiningBondStereo = True

The post-projection Mol is the *only* atom universe used for coordinates and
E3FP.  Before projection every source atom receives the integer property
``_r1_source_atom_index``.  Every surviving atom must retain a valid, unique,
ordered source tag; compacted post-RemoveHs indices are never used as a proxy.

The E3FP invocation preserves the historical P1 semantic from
``process/process_qc_step1_e3fp.py``:

    bits=4096, level=3, rdkit_invariants=True,
    all_iters=True, exclude_floating=False

Unlike the historical script, feature slots are filled by each E3FP shell's
explicit ``center_atom`` and ``radius``.  The script never infers an iteration
from ``all_shells`` list position.  It opens no LMDB, writes no molecule data,
does not extract the SDF, and emits no raw SMILES.  Its sole write is a new,
small JSON sidecar report supplied through ``--output``.

Remote synthetic smoke (no archive needed)::

  /root/miniconda3/envs/3dmolt5/bin/python -B pcqm_e3fp_preflight.py \\
    --self-test \\
    --e3fp-source /root/autodl-tmp/MoSt-T5/tokenization/3d_tokenization \\
    --output /root/autodl-fs/most-t5-r1/reports/<run>/e3fp_preflight_selftest.json

Remote bounded archive gate::

  /root/miniconda3/envs/3dmolt5/bin/python -B pcqm_e3fp_preflight.py \\
    --archive /root/autodl-fs/most-t5-p0/sources/pcqm4mv2/ogb-pcqm4mv2-train-3d-v1/archive/pcqm4m-v2-train.sdf.tar.gz \\
    --e3fp-source /root/autodl-tmp/MoSt-T5/tokenization/3d_tokenization \\
    --max-records 128 \\
    --output /root/autodl-fs/most-t5-r1/reports/<run>/e3fp_preflight_128.json

The caller must replace ``<run>`` with a new remote report directory.  This
tool intentionally has no full-corpus mode.
"""

from __future__ import print_function

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path


MAX_PREFLIGHT_RECORDS = 1000
DEFAULT_PREFLIGHT_RECORDS = 128
SOURCE_ATOM_TAG = "_r1_source_atom_index"
FP_BITS = 4096
FP_LEVEL = 3

# These five fields are deliberately the same explicit parameter semantic as
# process/process_qc_step1_e3fp.py.  Other E3FP options remain the frozen
# defaults of the supplied historical E3FP source and are recorded after the
# first Fingerprinter is constructed.
HISTORICAL_E3FP_INVOCATION = {
    "bits": FP_BITS,
    "level": FP_LEVEL,
    "rdkit_invariants": True,
    "all_iters": True,
    "exclude_floating": False,
}

HYDROGEN_PROJECTION_PROFILE = {
    "profile_id": "project_explicit_hydrogens_before_e3fp_v1",
    "source_atom_tag": SOURCE_ATOM_TAG,
    "sdf_parser": {"sanitize": True, "remove_hs": False},
    "projection": {
        "operation": "Chem.RemoveHs(Chem.Mol(tagged_source_mol), parameters, sanitize=True)",
        "only_nondefault_RemoveHsParameters_override": {
            "removeDefiningBondStereo": True,
        },
    },
    "forbidden": [
        "manual_atom_deletion",
        "compacted_post_projection_atom_index_as_source_index",
        "separate_smiles_rebuild_or_conformer_generation",
        "e3fp_from_a_different_molecule_than_coordinates",
    ],
}


class RecordRejected(RuntimeError):
    """Expected record-level rejection with a closed, non-sensitive reason."""

    def __init__(self, reason_code, stage):
        super(RecordRejected, self).__init__(reason_code)
        self.reason_code = reason_code
        self.stage = stage


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_regular_file(path, label):
    result = Path(path).expanduser()
    if not result.is_file():
        raise FileNotFoundError("{} is not a regular file: {}".format(label, result))
    return result.resolve()


def require_new_output(path):
    output = Path(path).expanduser()
    if output.exists():
        raise FileExistsError("--output must be a new sidecar path, found existing: {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.parent.is_dir():
        raise NotADirectoryError("output parent is not a directory: {}".format(output.parent))
    return output.resolve()


def write_json_new(path, payload):
    """Atomically write a *new* JSON sidecar without accepting overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to replace an existing sidecar: {}".format(path))
    temporary = path.with_name("." + path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError("temporary sidecar path already exists: {}".format(temporary))
    try:
        with open(str(temporary), "x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        # The parent process owns this fresh report path.  os.replace is used
        # only after the target's new-path invariant was checked above.
        os.replace(str(temporary), str(path))
    except Exception:
        # No delete is attempted: an interrupted temporary report is evidence
        # that should be inspected rather than silently removed.
        raise


def find_sdf_member(archive):
    for member in archive:
        if member.isfile() and member.name.lower().endswith(".sdf"):
            return member
    raise RuntimeError("no regular .sdf member found in archive")


def resolve_e3fp_source(path):
    """Accept either the historical 3d_tokenization root or its e3fp child."""
    supplied = Path(path).expanduser()
    if not supplied.is_dir():
        raise NotADirectoryError("--e3fp-source is not a directory: {}".format(supplied))
    supplied = supplied.resolve()
    if (supplied / "e3fp" / "pipeline.py").is_file():
        import_root = supplied
        package_root = supplied / "e3fp"
    elif supplied.name == "e3fp" and (supplied / "pipeline.py").is_file():
        import_root = supplied.parent
        package_root = supplied
    else:
        raise FileNotFoundError(
            "--e3fp-source must be a historical 3d_tokenization root containing e3fp/pipeline.py "
            "or the e3fp package itself: {}".format(supplied)
        )
    required = {
        "pipeline": package_root / "pipeline.py",
        "fingerprinter": package_root / "fingerprint" / "fprinter.py",
    }
    for label, file_path in required.items():
        if not file_path.is_file():
            raise FileNotFoundError("historical E3FP {} source missing: {}".format(label, file_path))
    return import_root, package_root, required


def import_locked_e3fp(import_root, package_root):
    """Import only the caller-pinned E3FP source, never an ambient package."""
    root_text = str(import_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        import e3fp
        from e3fp.fingerprint.fprinter import signed_to_unsigned_int
        from e3fp.pipeline import fprints_from_mol_verbose
    except ImportError as exc:
        raise RuntimeError("unable to import E3FP from --e3fp-source: {}".format(type(exc).__name__)) from exc

    module_file = Path(getattr(e3fp, "__file__", "")).resolve()
    try:
        module_file.relative_to(package_root)
    except ValueError as exc:
        raise RuntimeError(
            "ambient E3FP import escaped the supplied source root: imported {}, expected under {}".format(
                module_file, package_root
            )
        ) from exc
    return {
        "fprints_from_mol_verbose": fprints_from_mol_verbose,
        "signed_to_unsigned_int": signed_to_unsigned_int,
        "module_file": module_file,
        "module_version": getattr(e3fp, "__version__", None),
    }


def finite_single_conformer(mol, stage):
    conformer_count = int(mol.GetNumConformers())
    if conformer_count != 1:
        raise RecordRejected("SDF_CONFORMER_COUNT_NOT_ONE", stage)
    try:
        positions = mol.GetConformer(0).GetPositions()
    except Exception as exc:
        raise RecordRejected("SDF_CONFORMER_ACCESS_FAILED", stage) from exc
    if len(positions) != mol.GetNumAtoms():
        raise RecordRejected("ATOM_COORDINATE_COUNT_MISMATCH", stage)
    for row in positions:
        for value in row:
            if not math.isfinite(float(value)):
                raise RecordRejected("NONFINITE_COORDINATES", stage)


def tag_source_atoms(Chem, mol):
    """Return a tagged source copy and validate the pre-projection tag domain."""
    tagged = Chem.Mol(mol)
    source_atom_count = int(tagged.GetNumAtoms())
    if source_atom_count <= 0:
        raise RecordRejected("ZERO_SOURCE_ATOMS", "source_atom_tags")
    for source_index, atom in enumerate(tagged.GetAtoms()):
        atom.SetIntProp(SOURCE_ATOM_TAG, int(source_index))
    source_mapping = validate_source_tags(tagged, source_atom_count, require_full_domain=True)
    return tagged, source_atom_count, source_mapping


def validate_source_tags(mol, source_atom_count, require_full_domain):
    tags = []
    for atom in mol.GetAtoms():
        if not atom.HasProp(SOURCE_ATOM_TAG):
            raise RecordRejected("SOURCE_ATOM_TAG_MISSING", "source_atom_tags")
        try:
            tag = int(atom.GetIntProp(SOURCE_ATOM_TAG))
        except Exception as exc:
            raise RecordRejected("SOURCE_ATOM_TAG_NOT_INTEGER", "source_atom_tags") from exc
        if tag < 0 or tag >= source_atom_count:
            raise RecordRejected("SOURCE_ATOM_TAG_OUT_OF_RANGE", "source_atom_tags")
        tags.append(tag)

    if len(tags) != len(set(tags)):
        raise RecordRejected("SOURCE_ATOM_TAG_NOT_UNIQUE", "source_atom_tags")
    if require_full_domain and sorted(tags) != list(range(source_atom_count)):
        raise RecordRejected("SOURCE_ATOM_TAG_DOMAIN_INVALID", "source_atom_tags")
    if not require_full_domain and tags != sorted(tags):
        raise RecordRejected("SOURCE_ATOM_TAG_ORDER_NOT_PRESERVED", "source_atom_tags")
    return tags


def project_hydrogens(Chem, tagged_source_mol, source_atom_count):
    parameters = Chem.RemoveHsParameters()
    if not hasattr(parameters, "removeDefiningBondStereo"):
        raise RuntimeError("installed RDKit lacks RemoveHsParameters.removeDefiningBondStereo")
    parameters.removeDefiningBondStereo = True
    try:
        geometry_mol = Chem.RemoveHs(Chem.Mol(tagged_source_mol), parameters, sanitize=True)
        Chem.SanitizeMol(geometry_mol)
        Chem.AssignStereochemistry(geometry_mol, cleanIt=True, force=True)
    except Exception as exc:
        raise RecordRejected("HYDROGEN_PROJECTION_FAILED", "hydrogen_projection") from exc

    model_atom_count = int(geometry_mol.GetNumAtoms())
    if model_atom_count <= 0:
        raise RecordRejected("ZERO_MODEL_ATOMS", "hydrogen_projection")
    retained_tags = validate_source_tags(geometry_mol, source_atom_count, require_full_domain=False)
    residual_hydrogens = sum(atom.GetAtomicNum() == 1 for atom in geometry_mol.GetAtoms())
    if residual_hydrogens:
        raise RecordRejected("HYDROGEN_PROJECTION_RESIDUAL_H", "hydrogen_projection")
    if any(atom.GetAtomicNum() <= 1 for atom in geometry_mol.GetAtoms()):
        # E3FP fingerprints only atoms with atomic number > 1.  Keeping a
        # dummy/non-heavy model row would create a padding target, so reject
        # rather than fabricate a row.
        raise RecordRejected("GEOMETRY_NON_E3FP_ATOM", "hydrogen_projection")
    finite_single_conformer(geometry_mol, "geometry_coordinates")
    return geometry_mol, retained_tags


def shell_level_from_radius(shell, radius_multiplier):
    try:
        radius = float(shell.radius)
    except Exception as exc:
        raise RecordRejected("E3FP_SHELL_RADIUS_INVALID", "e3fp_matrix") from exc
    if not math.isfinite(radius) or radius < 0.0:
        raise RecordRejected("E3FP_SHELL_RADIUS_INVALID", "e3fp_matrix")
    if radius == 0.0:
        return 0
    multiplier = float(radius_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise RecordRejected("E3FP_RADIUS_MULTIPLIER_INVALID", "e3fp_matrix")
    level = int(round(radius / multiplier))
    # Radius is assigned by ShellsGenerator as level * radius_multiplier.  A
    # tight tolerance rejects a future E3FP implementation that changes the
    # interpretation rather than silently deriving a list-order column.
    tolerance = max(1e-10, abs(multiplier) * 1e-9)
    if level < 0 or not math.isclose(radius, level * multiplier, rel_tol=0.0, abs_tol=tolerance):
        raise RecordRejected("E3FP_SHELL_RADIUS_UNMAPPABLE", "e3fp_matrix")
    return level


def build_explicit_shell_matrix(np, fingerprinter, signed_to_unsigned_int, model_atom_count):
    """Build [atom, level] slots solely from each shell's center and radius."""
    matrix = np.full((model_atom_count, FP_LEVEL + 1), -1, dtype=np.int32)
    slots_seen = set()
    shells_seen = 0
    for shell in fingerprinter.all_shells:
        shells_seen += 1
        try:
            center_atom = int(shell.center_atom)
        except Exception as exc:
            raise RecordRejected("E3FP_SHELL_CENTER_INVALID", "e3fp_matrix") from exc
        if center_atom < 0 or center_atom >= model_atom_count:
            raise RecordRejected("E3FP_SHELL_CENTER_OUT_OF_RANGE", "e3fp_matrix")
        level = shell_level_from_radius(shell, fingerprinter.radius_multiplier)
        if level > FP_LEVEL:
            # An implementation should not emit a shell above the explicitly
            # requested level.  It is never silently clipped.
            raise RecordRejected("E3FP_SHELL_LEVEL_ABOVE_REQUESTED", "e3fp_matrix")
        slot = (center_atom, level)
        if slot in slots_seen:
            raise RecordRejected("E3FP_DUPLICATE_CENTER_RADIUS_SLOT", "e3fp_matrix")
        slots_seen.add(slot)
        if getattr(shell, "identifier", None) is None:
            raise RecordRejected("E3FP_SHELL_IDENTIFIER_MISSING", "e3fp_matrix")
        try:
            folded = int(signed_to_unsigned_int(int(shell.identifier)) % FP_BITS)
        except Exception as exc:
            raise RecordRejected("E3FP_IDENTIFIER_FOLD_FAILED", "e3fp_matrix") from exc
        if folded < 0 or folded >= FP_BITS:
            raise RecordRejected("E3FP_IDENTIFIER_OUT_OF_RANGE", "e3fp_matrix")
        matrix[center_atom, level] = folded

    if shells_seen == 0:
        raise RecordRejected("E3FP_NO_SHELLS", "e3fp_matrix")
    if matrix.shape != (model_atom_count, FP_LEVEL + 1):
        raise RecordRejected("E3FP_SHAPE_INVALID", "e3fp_matrix")
    if int(matrix.min()) < -1 or int(matrix.max()) >= FP_BITS:
        raise RecordRejected("E3FP_VALUE_RANGE_INVALID", "e3fp_matrix")
    if bool(np.any(matrix[:, 0] == -1)):
        raise RecordRejected("E3FP_LEVEL0_MISSING", "e3fp_matrix")
    if bool(np.any(np.all(matrix == -1, axis=1))):
        raise RecordRejected("E3FP_ALL_PADDING_MODEL_ROW", "e3fp_matrix")
    return matrix, {"shells_seen": int(shells_seen), "slots_populated": int(len(slots_seen))}


def generate_e3fp(np, e3fp_api, geometry_mol, ordinal):
    """Run the historical E3FP semantic on the already-projected Mol only."""
    # The legacy E3FP helper requires _Name.  This synthetic name is derived
    # solely from a bounded ordinal and is not an SDF title or a raw SMILES.
    geometry_mol.SetProp("_Name", "r1_pcqm_e3fp_preflight_{:06d}".format(ordinal))
    # This historical E3FP copy logs routine generation progress through the
    # root logger rather than a package logger.  Mute only INFO-level chatter
    # while it runs, then restore the caller's logging state.  Warnings and
    # errors remain visible and the E3FP parameters/results are untouched.
    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    try:
        if previous_root_level < logging.WARNING:
            root_logger.setLevel(logging.WARNING)
        fprints, fingerprinter = e3fp_api["fprints_from_mol_verbose"](
            geometry_mol,
            fprint_params=dict(HISTORICAL_E3FP_INVOCATION),
        )
    except Exception as exc:
        raise RecordRejected("E3FP_GENERATION_FAILED", "e3fp_generation") from exc
    finally:
        root_logger.setLevel(previous_root_level)
    if not fprints:
        raise RecordRejected("E3FP_EMPTY_FINGERPRINT_RESULT", "e3fp_generation")
    matrix, shell_summary = build_explicit_shell_matrix(
        np,
        fingerprinter,
        e3fp_api["signed_to_unsigned_int"],
        int(geometry_mol.GetNumAtoms()),
    )
    resolved = {
        "bits": int(fingerprinter.bits),
        "level": int(fingerprinter.level),
        "radius_multiplier": float(fingerprinter.radius_multiplier),
        "stereo": bool(fingerprinter.stereo),
        "include_disconnected": bool(fingerprinter.include_disconnected),
        "rdkit_invariants": bool(fingerprinter.rdkit_invariants),
        "exclude_floating": bool(fingerprinter.exclude_floating),
        "remove_duplicate_substructs": bool(fingerprinter.remove_duplicate_substructs),
        "fingerprint_type": getattr(fingerprinter.fp_type, "__name__", str(fingerprinter.fp_type)),
        "all_iters": True,
    }
    if resolved["bits"] != FP_BITS or resolved["level"] != FP_LEVEL:
        raise RecordRejected("E3FP_RESOLVED_CONFIG_MISMATCH", "e3fp_generation")
    if resolved["rdkit_invariants"] is not True or resolved["exclude_floating"] is not False:
        raise RecordRejected("E3FP_RESOLVED_CONFIG_MISMATCH", "e3fp_generation")
    return matrix, shell_summary, resolved


def summarize_matrix(matrix, shell_summary):
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "dtype": str(matrix.dtype),
        "value_min": int(matrix.min()),
        "value_max": int(matrix.max()),
        "padding_cells": int((matrix == -1).sum()),
        "valid_cells": int((matrix != -1).sum()),
        "all_padding_rows": int((matrix == -1).all(axis=1).sum()),
        "sha256": sha256_bytes(matrix.tobytes(order="C")),
        **shell_summary,
    }


def process_one_record(Chem, np, e3fp_api, ordinal, mol):
    """Return one raw-free status record; all expected defects are explicit."""
    summary = {"sdf_record_index": int(ordinal), "status": None}
    if mol is None:
        summary.update({"status": "reject", "stage": "sdf_parse", "reason_code": "SDF_PARSE_FAILED"})
        return summary
    try:
        finite_single_conformer(mol, "source_coordinates")
        tagged_source, source_atom_count, source_tags = tag_source_atoms(Chem, mol)
        geometry_mol, model_to_source = project_hydrogens(Chem, tagged_source, source_atom_count)
        matrix, shell_summary, resolved_config = generate_e3fp(np, e3fp_api, geometry_mol, ordinal)
        matrix_summary = summarize_matrix(matrix, shell_summary)
        summary.update(
            {
                "status": "ok",
                "source_atom_count": int(source_atom_count),
                "model_atom_count": int(geometry_mol.GetNumAtoms()),
                "source_atom_tag_validation": {
                    "pre_projection_tag_count": int(len(source_tags)),
                    "post_projection_tag_count": int(len(model_to_source)),
                    "pre_projection_full_domain": True,
                    "post_projection_unique": True,
                    "post_projection_in_range": True,
                    "post_projection_order_preserved": True,
                    "model_to_source_atom_index_sha256": sha256_bytes(
                        ",".join(str(value) for value in model_to_source).encode("utf-8")
                    ),
                },
                "e3fp": matrix_summary,
                "resolved_e3fp_config": resolved_config,
            }
        )
    except RecordRejected as exc:
        summary.update({"status": "reject", "stage": exc.stage, "reason_code": exc.reason_code})
    except Exception as exc:
        # Do not serialize exception text: an RDKit/E3FP exception can include
        # an SDF title or molecule representation.  The class is enough for a
        # reproducible triage rerun without leaking raw molecular content.
        summary.update(
            {
                "status": "reject",
                "stage": "unexpected",
                "reason_code": "UNEXPECTED_{}_ERROR".format(type(exc).__name__.upper()),
            }
        )
    return summary


def synthetic_molecule(Chem):
    """Create one deterministic explicit-H conformer entirely in memory."""
    from rdkit.Chem import AllChem

    base = Chem.MolFromSmiles("CCO")
    if base is None:
        raise RuntimeError("synthetic molecule construction failed")
    mol = Chem.AddHs(base)
    if hasattr(AllChem, "ETKDGv3"):
        parameters = AllChem.ETKDGv3()
    else:
        parameters = AllChem.ETKDGv2()
    parameters.randomSeed = 12648430
    status = int(AllChem.EmbedMolecule(mol, parameters))
    if status != 0:
        raise RuntimeError("deterministic synthetic conformer embedding failed")
    return mol


def aggregate_results(record_summaries):
    reason_counts = Counter()
    stage_counts = Counter()
    source_atom_histogram = Counter()
    model_atom_histogram = Counter()
    matrix_shape_histogram = Counter()
    total_e3fp_rows = 0
    total_padding_cells = 0
    total_valid_cells = 0
    all_padding_rows = 0
    global_min = None
    global_max = None
    for summary in record_summaries:
        if summary["status"] == "ok":
            reason_counts["E3FP_OK"] += 1
            source_atom_histogram[str(summary["source_atom_count"])] += 1
            model_atom_histogram[str(summary["model_atom_count"])] += 1
            e3fp = summary["e3fp"]
            matrix_shape_histogram["{}x{}".format(*e3fp["shape"])] += 1
            total_e3fp_rows += e3fp["shape"][0]
            total_padding_cells += e3fp["padding_cells"]
            total_valid_cells += e3fp["valid_cells"]
            all_padding_rows += e3fp["all_padding_rows"]
            global_min = e3fp["value_min"] if global_min is None else min(global_min, e3fp["value_min"])
            global_max = e3fp["value_max"] if global_max is None else max(global_max, e3fp["value_max"])
        else:
            reason_counts[summary["reason_code"]] += 1
            stage_counts[summary["stage"]] += 1
    return {
        "reason_counts": dict(sorted(reason_counts.items())),
        "failure_stage_counts": dict(sorted(stage_counts.items())),
        "source_atom_count_histogram": dict(sorted(source_atom_histogram.items(), key=lambda item: int(item[0]))),
        "model_atom_count_histogram": dict(sorted(model_atom_histogram.items(), key=lambda item: int(item[0]))),
        "e3fp_matrix_shape_histogram": dict(sorted(matrix_shape_histogram.items())),
        "e3fp_rows_total": int(total_e3fp_rows),
        "e3fp_padding_cells_total": int(total_padding_cells),
        "e3fp_valid_cells_total": int(total_valid_cells),
        "e3fp_all_padding_rows_total": int(all_padding_rows),
        "e3fp_value_global_min": global_min,
        "e3fp_value_global_max": global_max,
    }


def stream_archive_records(Chem, archive_path, max_records, np, e3fp_api):
    records = []
    member_name = None
    member_size = None
    with tarfile.open(str(archive_path), mode="r|gz") as archive:
        member = find_sdf_member(archive)
        member_name = member.name
        member_size = int(member.size)
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("cannot open SDF tar member: {}".format(member.name))
        try:
            supplier = Chem.ForwardSDMolSupplier(stream, sanitize=True, removeHs=False)
            for ordinal, mol in enumerate(supplier):
                records.append(process_one_record(Chem, np, e3fp_api, ordinal, mol))
                if ordinal + 1 >= max_records:
                    break
        finally:
            stream.close()
    return records, {
        "sdf_member": member_name,
        "sdf_member_uncompressed_bytes": member_size,
        "records_seen": int(len(records)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="", help="frozen OGB train-3D .sdf.tar.gz; required unless --self-test")
    parser.add_argument(
        "--e3fp-source",
        required=True,
        help="historical remote 3d_tokenization root (or its e3fp child); no ambient E3FP package is accepted",
    )
    parser.add_argument("--output", required=True, help="new JSON sidecar report path")
    parser.add_argument("--max-records", type=int, default=DEFAULT_PREFLIGHT_RECORDS)
    parser.add_argument("--sample-limit", type=int, default=8, help="bounded raw-free successful record summaries")
    parser.add_argument("--self-test", action="store_true", help="run one in-memory explicit-H synthetic molecule instead of reading an archive")
    args = parser.parse_args()

    if args.max_records < 1 or args.max_records > MAX_PREFLIGHT_RECORDS:
        parser.error("--max-records must be within [1, {}]".format(MAX_PREFLIGHT_RECORDS))
    if args.sample_limit < 0:
        parser.error("--sample-limit must be non-negative")
    if args.self_test and args.archive:
        parser.error("--self-test and --archive are mutually exclusive")
    if not args.self_test and not args.archive:
        parser.error("--archive is required unless --self-test is selected")

    output_path = require_new_output(args.output)
    archive_path = None if args.self_test else require_regular_file(args.archive, "archive")
    import_root, package_root, e3fp_files = resolve_e3fp_source(args.e3fp_source)

    try:
        import numpy as np
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise RuntimeError("NumPy and RDKit are required for this preflight") from exc
    e3fp_api = import_locked_e3fp(import_root, package_root)

    if args.self_test:
        record_summaries = [process_one_record(Chem, np, e3fp_api, 0, synthetic_molecule(Chem))]
        archive_observed = {
            "synthetic_input_only": True,
            "sdf_member": None,
            "sdf_member_uncompressed_bytes": None,
            "records_seen": 1,
        }
        records_requested = 1
    else:
        record_summaries, archive_observed = stream_archive_records(
            Chem, archive_path, args.max_records, np, e3fp_api
        )
        records_requested = int(args.max_records)

    aggregate = aggregate_results(record_summaries)
    successful = [item for item in record_summaries if item["status"] == "ok"]
    rejected = [item for item in record_summaries if item["status"] != "ok"]
    errors = []
    if archive_observed["records_seen"] != records_requested:
        errors.append("input ended before the requested bounded record count")
    if rejected:
        errors.append("one or more tested records failed the frozen projection/E3FP contract")
    if aggregate["e3fp_all_padding_rows_total"] != 0:
        errors.append("a successful result reported an all-padding E3FP model row")
    if aggregate["e3fp_value_global_min"] is not None and aggregate["e3fp_value_global_min"] < -1:
        errors.append("a successful E3FP matrix was below the padding floor")
    if aggregate["e3fp_value_global_max"] is not None and aggregate["e3fp_value_global_max"] >= FP_BITS:
        errors.append("a successful E3FP matrix exceeded the locked bit range")

    report = {
        "schema_version": "most-t5-r1/pcqm-e3fp-preflight/v1",
        "created_utc": utc_now(),
        "scope": {
            "mode": "synthetic_hermetic" if args.self_test else "bounded_archive_stream",
            "archive_streamed_not_extracted": not args.self_test,
            "synthetic_input_only": bool(args.self_test),
            "records_requested": records_requested,
            "records_seen": archive_observed["records_seen"],
            "complete_archive_scan": False,
            "lmdb_opened": False,
            "lmdb_records_written": 0,
            "source_artifacts_modified": False,
            "data_download_performed": False,
            "local_data_transfer": False,
            "raw_smiles_emitted": False,
            "raw_molecule_records_emitted": False,
        },
        "inputs": {
            "archive": str(archive_path) if archive_path is not None else None,
            "archive_bytes": int(archive_path.stat().st_size) if archive_path is not None else None,
            "e3fp_source_supplied": str(Path(args.e3fp_source).expanduser()),
            "e3fp_import_root": str(import_root),
            "e3fp_package_root": str(package_root),
            "e3fp_module_file": str(e3fp_api["module_file"]),
            "e3fp_module_version": e3fp_api["module_version"],
            "e3fp_source_file_sha256": {
                label: sha256_file(file_path) for label, file_path in sorted(e3fp_files.items())
            },
            "rdkit_version": rdBase.rdkitVersion,
        },
        "hydrogen_projection": HYDROGEN_PROJECTION_PROFILE,
        "e3fp_invocation": HISTORICAL_E3FP_INVOCATION,
        "archive_observed": archive_observed,
        "results": {
            "records_ok": int(len(successful)),
            "records_rejected": int(len(rejected)),
            **aggregate,
            "successful_record_summaries": successful[: args.sample_limit],
            "rejected_record_summaries": rejected[: args.sample_limit],
            "summaries_truncated": len(successful) > args.sample_limit or len(rejected) > args.sample_limit,
        },
        "pass": not errors,
        "errors": errors,
        "interpretation": {
            "pass_meaning": "The selected bounded records passed this same-Mol H-projection, source-tag, coordinate, and E3FP matrix contract. It does not admit PCQM4Mv2 to P1 or prove a full release.",
            "failure_meaning": "Any rejection is a reject-ledger design input, not a signal to pad geometry with zeros or infer source indices from compacted atom positions.",
            "next_gate": "Only after an accepted bounded preflight should the team design a single remote full-pass adapter with membership and reject ledgers; this tool has no full-corpus mode.",
        },
    }
    write_json_new(output_path, report)
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "records_seen": report["scope"]["records_seen"],
                "records_ok": report["results"]["records_ok"],
                "records_rejected": report["results"]["records_rejected"],
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
