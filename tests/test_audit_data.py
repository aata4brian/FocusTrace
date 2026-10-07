"""Artificial schema fixtures only; never participant evidence.

Run: python -m unittest discover -s tests -p test_audit_data.py -v
"""
import csv
import importlib.util
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_data", ROOT / "scripts/audit_data.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fixture(root, pid="T01", sid="FIXTURE-T01", sequence_id="CUSTOM"):
    data, sessions = root / "data", root / "outputs/sessions"
    for name in ("features", "events", "responses", "profiles", "raw_video", "database"):
        (data / name).mkdir(parents=True, exist_ok=True)
    sessions.mkdir(parents=True, exist_ok=True)
    sequence = list(audit.CODES)
    intervals = [dict(state="CALIBRATION", block_code="CALIBRATION", conceptual_condition="CALIBRATION", start=0., end=.2, duration=.2)]
    for i, code in enumerate(sequence):
        a = round(.2 + .4 * i, 5)
        intervals.append(dict(state="BLOCK", block_code=code, conceptual_condition=audit.labels(code)[0], start=a, end=round(a+.4, 5), duration=.4))
    rows = []
    for n in range(26):
        t = round(.05 + n*.1, 5)
        item = next(i for i in intervals if i["start"] <= t < i["end"])
        code = item["block_code"]
        row = {k: float(n+1)/100 for k in audit.FEATURE_FIELDS}
        row.update(participant_id=pid, session_id=sid, sequence_id=sequence_id,
            timestamp_sec=t, frame_index=n, system_state=item["state"], block_number=sequence.index(code)+1 if code in sequence else 0,
            block_code=code, conceptual_condition=item["conceptual_condition"], binary_label=audit.labels(code)[1], gaze_category="CENTER")
        row.update({key: 1 for key in audit.FLAGS})
        for source, delta in audit.BASE_FIELDS.items():
            row[delta] = row[source] - .015 if item["state"] == "BLOCK" else None
        rows.append(row)
    write_csv(data / "features" / f"{pid}_features.csv", rows, audit.FEATURE_FIELDS)
    frames = [dict(frame_index=i, timestamp_sec=r["timestamp_sec"], encoded_time_sec=i/10) for i,r in enumerate(rows)]
    write_csv(data / "raw_video" / f"{pid}_frame_timestamps.csv", frames, frames[0])
    (data / "raw_video" / f"{pid}_raw.mp4").write_bytes(b"ARTIFICIAL TEST PLACEHOLDER: auditor does not decode video")
    event_rows = [dict(participant_id=pid, session_id=sid, event_type="INTERVAL", system_state=i["state"], block_code=i["block_code"], end_time_sec=i["end"]) for i in intervals]
    write_csv(data / "events" / f"{pid}_events.csv", event_rows, event_rows[0])
    write_csv(data / "responses" / f"{pid}_responses.csv", [], ["participant_id", "session_id"])
    profile = {k:dict(n=2, median=.015, mad=.005, valid=True) for k in audit.BASE_FIELDS}
    audit.write_json(data / "profiles" / f"{pid}_baseline.json", dict(participant_id=pid, session_id=sid, minimum_valid_samples=2, valid=True, features=profile))
    meta = dict(participant_id=pid, session_id=sid, sequence_id=sequence_id, sequence=sequence, mode="RESEARCH", state="FINISHED", block_sec=.4, calibration_sec=.2, intervals=intervals)
    audit.write_json(sessions / f"{pid}_metadata.json", meta)
    audit.write_json(sessions / f"{pid}_integrity_report.json", dict(participant_id=pid, session_id=sid, status="VALID", issues=[]))
    con = sqlite3.connect(data / "database/tracefokus.sqlite3")
    con.executescript("CREATE TABLE sessions(participant_id TEXT,session_id TEXT,sequence_id TEXT,sequence_json TEXT,state TEXT,mode TEXT); CREATE TABLE reviews(id INTEGER,session_id TEXT,start_sec REAL,end_sec REAL,final_verified_label TEXT);")
    con.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?)", (pid,sid,sequence_id,json.dumps(sequence),"FINISHED","RESEARCH"))
    for i, block in enumerate(intervals[1:]):
        con.execute("INSERT INTO reviews VALUES(?,?,?,?,?)", (i,sid,block["start"],block["end"],block["conceptual_condition"]))
    con.commit(); con.close()
    return data, sessions, rows


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_audit(self, data, sessions, expected=1):
        obj = audit.Audit(self.root, data, sessions, self.root / "result", expected=expected)
        summary = obj.run()
        return obj, summary

    def test_source_only_is_blocked_and_inventory_has_no_fake_participants(self):
        obj, summary = self.run_audit(self.root/"data", self.root/"outputs/sessions", expected=18)
        self.assertEqual(summary["participants_found"], 0)
        self.assertEqual(summary["data_health"], "BLOCKED")
        self.assertFalse(summary["training_authorized"])
        self.assertEqual(obj.inventory, [])
        self.assertIn("NO_PARTICIPANTS", {i["code"] for i in obj.issues})

    def test_consistent_fixture_keeps_every_raw_byte_unchanged(self):
        data, sessions, _ = fixture(self.root)
        before = {str(p): audit.digest(p) for folder in (data, sessions) for p in folder.rglob("*") if p.is_file()}
        obj, summary = self.run_audit(data, sessions)
        self.assertEqual(summary["critical_issues"], 0, obj.issues)
        self.assertEqual(summary["participant_status"], {"T01":"PASS"})
        after = {str(p): audit.digest(p) for folder in (data, sessions) for p in folder.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(summary["training_authorized"])

    def test_nonstandard_participant_id_is_discovered_from_content(self):
        data, sessions, _ = fixture(self.root, pid="SUBJECT-27", sid="CUSTOM-SESSION")
        _, summary = self.run_audit(data, sessions)
        self.assertEqual(summary["participants"], ["SUBJECT-27"])

    def test_reset_and_duplicate_timestamp_are_errors_before_any_sort(self):
        data, sessions, rows = fixture(self.root)
        rows[10]["timestamp_sec"] = rows[0]["timestamp_sec"]
        write_csv(data/"features/T01_features.csv", rows, audit.FEATURE_FIELDS)
        obj, summary = self.run_audit(data, sessions)
        self.assertIn("TIMESTAMP_INVALID", {i["code"] for i in obj.issues})
        self.assertEqual(summary["participant_status"]["T01"], "REVIEW")

    def test_wrong_label_is_not_silently_corrected(self):
        data, sessions, rows = fixture(self.root)
        rows[2]["binary_label"] = 1
        write_csv(data/"features/T01_features.csv", rows, audit.FEATURE_FIELDS)
        before = audit.digest(data/"features/T01_features.csv")
        obj, _ = self.run_audit(data, sessions)
        self.assertIn("LABEL_MAPPING", {i["code"] for i in obj.issues})
        self.assertEqual(before, audit.digest(data/"features/T01_features.csv"))

    def test_wrong_preset_gets_explicit_warning_not_automatic_exclusion(self):
        data, sessions, _ = fixture(self.root, pid="P02", sequence_id="P01")
        obj, summary = self.run_audit(data, sessions)
        issue = next(i for i in obj.issues if i["code"] == "OTHER_PARTICIPANT_PRESET")
        self.assertEqual(issue["severity"], "WARNING")
        self.assertEqual(summary["excluded"], 0)

    def test_malformed_csv_is_not_accepted_as_partial_data(self):
        data, sessions, _ = fixture(self.root)
        with (data/"features/T01_features.csv").open("a") as f:
            f.write("only,two\n")
        obj, _ = self.run_audit(data, sessions)
        self.assertIn("MALFORMED_CSV", {i["code"] for i in obj.issues})
        self.assertEqual(obj.inventory[0]["rows"], 0)

    def test_infinity_and_invalid_numbers_block_preprocessing(self):
        data, sessions, rows = fixture(self.root)
        rows[3]["wrist_motion"] = "inf"
        rows[4]["head_angular_speed"] = "broken"
        rows[5]["gaze_valid"] = 2
        write_csv(data/"features/T01_features.csv", rows, audit.FEATURE_FIELDS)
        obj, _ = self.run_audit(data, sessions)
        self.assertEqual(sum(i["code"] == "INVALID_NUMERIC" for i in obj.issues), 2)
        self.assertIn("INVALID_FLAG", {i["code"] for i in obj.issues})

    def test_swapped_baseline_identifiers_are_detected(self):
        data, sessions, _ = fixture(self.root)
        path = data/"profiles/T01_baseline.json"
        value = json.loads(path.read_text()); value["participant_id"] = "SOMEONE_ELSE"
        audit.write_json(path, value)
        obj, _ = self.run_audit(data, sessions)
        self.assertIn("ID_MISMATCH", {i["code"] for i in obj.issues})

    def test_existing_output_is_never_overwritten(self):
        data, sessions, _ = fixture(self.root)
        self.run_audit(data, sessions)
        with self.assertRaises(FileExistsError):
            self.run_audit(data, sessions)

    def test_wal_database_is_read_from_copy_and_originals_stay_unchanged(self):
        data, sessions, _ = fixture(self.root)
        path = data/"database/tracefokus.sqlite3"
        con = sqlite3.connect(path)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("UPDATE sessions SET state='INTERRUPTED'")
            con.commit()
            before = {str(p): audit.digest(p) for p in path.parent.iterdir() if p.is_file()}
            obj, summary = self.run_audit(data, sessions)
            self.assertIn("SESSION_NOT_FINISHED", {i["code"] for i in obj.issues})
            after = {str(p): audit.digest(p) for p in path.parent.iterdir() if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(summary["data_health"], "BLOCKED")
        finally:
            con.close()

    def test_output_inside_raw_data_is_rejected(self):
        data, sessions, _ = fixture(self.root)
        with self.assertRaises(ValueError):
            audit.Audit(self.root, data, sessions, data/"new_audit").run()

    def test_unknown_metadata_shape_fails_closed(self):
        data, sessions, _ = fixture(self.root)
        path = sessions/"T01_metadata.json"
        meta = json.loads(path.read_text()); meta["intervals"] = "broken"
        audit.write_json(path, meta)
        obj, summary = self.run_audit(data, sessions)
        self.assertEqual(summary["participant_status"]["T01"], "REVIEW")
        self.assertIn("INTERVALS_INVALID", {i["code"] for i in obj.issues})


if __name__ == "__main__":
    unittest.main()
