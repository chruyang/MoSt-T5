from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest
import uuid

from rdkit import Chem

from most_t5_next.p2.build_qm9_standard_target_overlay_v1 import (
    QM9StandardTargetOverlayError,
    build_overlay,
)


class QM9StandardTargetOverlayTest(unittest.TestCase):
    def _fixture(self, *, duplicate: bool = False) -> tuple[Path, Path, Path]:
        root = Path("tmp")
        root.mkdir(exist_ok=True)
        token = uuid.uuid4().hex
        records = root / f"qm9_overlay_{token}.records.jsonl"
        rows = []
        for index, smiles in enumerate(("CCO", "C")):
            rows.append({
                "smiles": smiles,
                "split": "train" if index == 0 else "dev",
                "record": {
                    "record_id": f"r{index}",
                    "storage_key": f"k{index}",
                },
            })
        records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        sdf = root / f"qm9_overlay_{token}.sdf"
        writer = Chem.SDWriter(str(sdf))
        source = (("gdb_1", "CCO"), ("gdb_2", "C"))
        if duplicate:
            source = source + (("gdb_3", "OCC"),)
        for name, smiles in source:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            mol.SetProp("_Name", name)
            writer.write(mol)
        writer.close()

        csv_path = root / f"qm9_overlay_{token}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["mol_id", "mu", "alpha", "r2", "u0", "u0_atom"]
            writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
            writer_csv.writeheader()
            for index, (name, _) in enumerate(source):
                writer_csv.writerow({
                    "mol_id": name,
                    "mu": 1 + index,
                    "alpha": 2 + index,
                    "r2": 3 + index,
                    "u0": -4 - index,
                    "u0_atom": -5 - index,
                })
        for path in (records, sdf, csv_path):
            self.addCleanup(path.unlink, missing_ok=True)
        return records, sdf, csv_path

    def test_kekule_sdf_is_normalized_to_aromatic_identity(self) -> None:
        records, sdf, csv_path = self._fixture()
        row = json.loads(records.read_text(encoding="utf-8").splitlines()[0])
        row["smiles"] = "c1ccccc1"
        records.write_text(json.dumps(row) + "\n", encoding="utf-8")
        writer = Chem.SDWriter(str(sdf))
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol.SetProp("_Name", "gdb_1")
        # RDKit's SDF writer kekulizes aromatic bonds by default.  Avoid the
        # version-specific ``kekulize=`` Python keyword used by older wheels.
        writer.write(mol)
        writer.close()
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        csv_path.write_text("\n".join((lines[0], lines[1])) + "\n", encoding="utf-8")
        overlays, rejects, _ = build_overlay(
            probe_records_path=records,
            qm9_sdf_path=sdf,
            qm9_csv_path=csv_path,
        )
        self.assertEqual(len(overlays), 1)
        self.assertEqual(rejects, [])

    def test_exact_join_preserves_probe_order(self) -> None:
        records, sdf, csv_path = self._fixture()
        overlays, rejects, report = build_overlay(
            probe_records_path=records,
            qm9_sdf_path=sdf,
            qm9_csv_path=csv_path,
        )
        self.assertEqual([row["record_id"] for row in overlays], ["r0", "r1"])
        self.assertEqual(overlays[0]["targets"]["mu"], 1.0)
        self.assertEqual(rejects, [])
        self.assertEqual(report["joined_by_split"], {"dev": 1, "train": 1})

    def test_ambiguous_identity_is_rejected(self) -> None:
        records, sdf, csv_path = self._fixture(duplicate=True)
        overlays, rejects, report = build_overlay(
            probe_records_path=records,
            qm9_sdf_path=sdf,
            qm9_csv_path=csv_path,
        )
        self.assertEqual([row["record_id"] for row in overlays], ["r1"])
        self.assertEqual(rejects[0]["reason"], "AMBIGUOUS_STANDARD_IDENTITY")
        self.assertEqual(report["ambiguous_canonical_identities"], 1)

    def test_sdf_csv_order_mismatch_is_rejected(self) -> None:
        records, sdf, csv_path = self._fixture()
        text = csv_path.read_text(encoding="utf-8").replace("gdb_1", "gdb_9", 1)
        csv_path.write_text(text, encoding="utf-8")
        with self.assertRaises(QM9StandardTargetOverlayError):
            build_overlay(
                probe_records_path=records,
                qm9_sdf_path=sdf,
                qm9_csv_path=csv_path,
            )


if __name__ == "__main__":
    unittest.main()
