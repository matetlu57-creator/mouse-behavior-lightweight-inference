from __future__ import annotations

from mouse_behavior.identity.labels import annotate_identity_labels


def test_behavior_records_keep_slots_and_add_animal_labels():
    records = [
        {
            "actor_id": 1,
            "target_id": 0,
            "members": "[0, 1]",
        }
    ]

    annotate_identity_labels(records, ("RFID-A", "RFID-B"))

    assert records[0]["actor_id"] == 1
    assert records[0]["actor_id_label"] == "RFID-B"
    assert records[0]["target_id_label"] == "RFID-A"
    assert records[0]["members_labels"] == ["RFID-A", "RFID-B"]
