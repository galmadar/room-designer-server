"""/interpret with the model stubbed out: no network, no key, no cost.

The scan below is the one from the endpoint's contract, so the numbers the
tests assert on are the numbers the app was promised.
"""

import json

import pytest
from fastapi.testclient import TestClient

import app as server

ROOM = {
    "kind": "kitchen",
    "widthMetres": 3.4,
    "lengthMetres": 2.8,
    "objects": [
        {"id": "A1", "category": "refrigerator", "widthMetres": 0.6, "depthMetres": 0.6, "heightMetres": 2.0},
        {"id": "B2", "category": "sofa", "widthMetres": 1.9, "depthMetres": 0.9, "heightMetres": 0.8},
    ],
}

client = TestClient(server.app)


def stub(monkeypatch, reply, calls=None):
    """Make the model say exactly `reply` — a dict to be encoded, or raw text."""
    text = reply if isinstance(reply, str) else json.dumps(reply)

    def fake(instruction, *, max_tokens, what):
        if calls is not None:
            calls.append(instruction)
        return text

    monkeypatch.setattr(server, "_ask_model", fake)


def interpret(sentence, room=None):
    response = client.post("/interpret", json={"sentence": sentence, "room": room if room is not None else ROOM})
    assert response.status_code == 200, response.text
    return response.json()


def test_all_three_corrections_at_once(monkeypatch):
    stub(monkeypatch, {
        "roomKind": "guest room",
        "objectEdits": [{"id": "A1", "category": "wardrobe", "widthMetres": None, "depthMetres": None, "heightMetres": None}],
        "vagueSizes": [{"id": "A1", "field": "widthMetres", "direction": "bigger", "ask": "Wider than what, though?"}],
        "unchanged": [],
    })
    body = interpret("It's a guest room. That fridge is a wardrobe, and it's wider than that.")

    assert body == {
        "roomKind": "guest room",
        "objectEdits": [
            {"id": "A1", "category": "wardrobe", "widthMetres": None, "depthMetres": None, "heightMetres": None},
        ],
        "questions": [
            {
                "id": "A1",
                "field": "widthMetres",
                "ask": "Wider than what, though?",
                "options": [
                    {"label": "Twice that — 1.2 m", "value": 1.2},
                    {"label": "Half the room — 1.7 m", "value": 1.7},
                ],
            },
        ],
        "unchanged": [],
    }


def test_only_the_room_is_renamed(monkeypatch):
    stub(monkeypatch, {"roomKind": "playroom", "objectEdits": [], "vagueSizes": [], "unchanged": []})
    body = interpret("Actually this is the playroom.")

    assert body == {"roomKind": "playroom", "objectEdits": [], "questions": [], "unchanged": []}


def test_only_an_object_is_renamed(monkeypatch):
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [{"id": "B2", "category": "daybed"}],
        "vagueSizes": [],
        "unchanged": [],
    })
    body = interpret("The sofa is really a daybed.")

    assert body["roomKind"] is None
    assert body["objectEdits"] == [
        {"id": "B2", "category": "daybed", "widthMetres": None, "depthMetres": None, "heightMetres": None},
    ]
    assert body["questions"] == []


def test_a_vague_size_asks_instead_of_guessing(monkeypatch):
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [],
        "vagueSizes": [{"id": "A1", "field": "widthMetres", "direction": "bigger", "ask": "How wide is it really?"}],
        "unchanged": [],
    })
    body = interpret("That fridge is much bigger than that.")

    assert body["objectEdits"] == []
    question = body["questions"][0]
    assert question["ask"] == "How wide is it really?"
    # Every number is measured off the scan: 2 x 0.6 m, and half of a 3.4 m room.
    assert [option["value"] for option in question["options"]] == [1.2, 1.7]
    assert [option["label"] for option in question["options"]] == ["Twice that — 1.2 m", "Half the room — 1.7 m"]


def test_a_precise_size_edits_instead_of_asking(monkeypatch):
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [{"id": "A1", "widthMetres": 1.2}],
        "vagueSizes": [],
        "unchanged": [],
    })
    body = interpret("Make the fridge 1.2 metres wide.")

    assert body["objectEdits"] == [
        {"id": "A1", "category": None, "widthMetres": 1.2, "depthMetres": None, "heightMetres": None},
    ]
    assert body["questions"] == []


def test_a_number_wins_over_a_question_about_the_same_field(monkeypatch):
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [{"id": "A1", "widthMetres": 1.5}],
        "vagueSizes": [{"id": "A1", "field": "widthMetres", "direction": "bigger", "ask": "How wide?"}],
        "unchanged": [],
    })
    body = interpret("The fridge is about a metre and a half wide.")

    assert body["objectEdits"][0]["widthMetres"] == 1.5
    assert body["questions"] == []


def test_an_empty_sentence_never_reaches_the_model(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("an empty sentence must not cost a model call")

    monkeypatch.setattr(server, "_ask_model", boom)
    assert interpret("   ") == {"roomKind": None, "objectEdits": [], "questions": [], "unchanged": []}


def test_a_malformed_reply_is_a_shrug_not_a_500(monkeypatch):
    stub(monkeypatch, "I'm afraid I can't help with that — {{{ not json at all")
    body = interpret("It's a guest room.")

    assert body == {"roomKind": None, "objectEdits": [], "questions": [], "unchanged": [server.NOT_UNDERSTOOD]}


def test_a_reply_that_is_a_json_list_is_a_shrug_too(monkeypatch):
    stub(monkeypatch, "[1, 2, 3]")
    assert interpret("It's a guest room.")["unchanged"] == [server.NOT_UNDERSTOOD]


def test_a_fenced_reply_is_still_read(monkeypatch):
    stub(monkeypatch, '```json\n{"roomKind": "study", "objectEdits": [], "vagueSizes": [], "unchanged": []}\n```')
    assert interpret("It's a study.")["roomKind"] == "study"


def test_an_id_that_does_not_exist_is_dropped(monkeypatch):
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [{"id": "ZZ", "category": "wardrobe"}],
        "vagueSizes": [{"id": "ZZ", "field": "widthMetres", "direction": "bigger", "ask": "How wide?"}],
        "unchanged": [],
    })
    body = interpret("The tall one by the window is a wardrobe and it's wider.")

    assert body["objectEdits"] == []
    assert body["questions"] == []
    assert body["unchanged"] == [server.NO_SUCH_THING]


def test_a_room_with_no_objects_still_answers(monkeypatch):
    stub(monkeypatch, {
        "roomKind": "study",
        "objectEdits": [{"id": "A1", "category": "wardrobe"}],
        "vagueSizes": [],
        "unchanged": [],
    })
    body = interpret("It's a study.", room={"kind": None, "widthMetres": 3.4, "lengthMetres": 2.8, "objects": []})

    assert body["roomKind"] == "study"
    assert body["objectEdits"] == []


def test_a_sentence_about_nothing_relevant_changes_nothing(monkeypatch):
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [],
        "vagueSizes": [],
        "unchanged": ["I could not act on \"what's the weather like\"."],
    })
    body = interpret("What's the weather like?")

    assert body == {
        "roomKind": None,
        "objectEdits": [],
        "questions": [],
        "unchanged": ['I could not act on "what\'s the weather like".'],
    }


def test_an_edit_that_settles_nothing_is_dropped(monkeypatch):
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [{"id": "A1", "category": None, "widthMetres": None}],
        "vagueSizes": [],
        "unchanged": [],
    })
    assert interpret("Hmm.")["objectEdits"] == []


def test_nonsense_numbers_are_refused(monkeypatch):
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [{"id": "A1", "widthMetres": -3, "depthMetres": 900, "heightMetres": "wide"}],
        "vagueSizes": [],
        "unchanged": [],
    })
    assert interpret("Make it minus three metres wide.")["objectEdits"] == []


def test_a_smaller_question_offers_smaller_options(monkeypatch):
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [],
        "vagueSizes": [{"id": "B2", "field": "widthMetres", "direction": "smaller", "ask": "How narrow?"}],
        "unchanged": [],
    })
    options = interpret("The sofa is narrower than that.")["questions"][0]["options"]

    # Half of 1.9 m, and a quarter off it — both rounded to the nearest 5 cm.
    assert [option["value"] for option in options] == [0.95, 1.4]
    assert all(option["value"] < 1.9 for option in options)


def test_a_height_question_falls_back_to_multiples(monkeypatch):
    """The scan measures the floor but not the ceiling, so a height has no room span."""
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [],
        "vagueSizes": [{"id": "A1", "field": "heightMetres", "direction": "bigger", "ask": "How tall?"}],
        "unchanged": [],
    })
    options = interpret("The fridge is taller than that.")["questions"][0]["options"]

    assert [option["value"] for option in options] == [2.3, 3.0]


def test_a_missing_ask_gets_a_default(monkeypatch):
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [],
        "vagueSizes": [{"id": "A1", "field": "depthMetres", "direction": "bigger"}],
        "unchanged": [],
    })
    assert interpret("It sticks out further.")["questions"][0]["ask"] == "How deep should it be?"


def test_the_instruction_carries_the_scan_and_the_sentence(monkeypatch):
    calls = []
    stub(monkeypatch, {"roomKind": None, "objectEdits": [], "vagueSizes": [], "unchanged": []}, calls)
    interpret("It's a guest room.")

    assert "refrigerator" in calls[0] and "3.4" in calls[0] and "It's a guest room." in calls[0]


def test_a_missing_room_is_not_a_422(monkeypatch):
    stub(monkeypatch, {"roomKind": "study", "objectEdits": [], "vagueSizes": [], "unchanged": []})
    response = client.post("/interpret", json={"sentence": "It's a study."})

    assert response.status_code == 200, response.text
    assert response.json()["roomKind"] == "study"


def test_a_tall_thing_can_still_grow_to_the_ceiling(monkeypatch):
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [],
        "vagueSizes": [{"id": "C1", "field": "heightMetres", "direction": "bigger", "ask": "How tall?"}],
        "unchanged": [],
    })
    body = interpret("The wardrobe is taller than that.", room={
        "kind": "bedroom", "widthMetres": 4.0, "lengthMetres": 3.2,
        "objects": [{"id": "C1", "category": "storage", "widthMetres": 0.9, "depthMetres": 0.6, "heightMetres": 2.1}],
    })

    assert [option["value"] for option in body["questions"][0]["options"]] == [2.4, 3.0]


def test_one_option_is_no_choice_at_all(monkeypatch):
    """Two or three, or none: a single option is the guess this endpoint exists to avoid."""
    stub(monkeypatch, {
        "roomKind": None,
        "objectEdits": [],
        "vagueSizes": [{"id": "D1", "field": "heightMetres", "direction": "bigger", "ask": "How tall?"}],
        "unchanged": [],
    })
    body = interpret("That wardrobe is taller than that.", room={
        "kind": "bedroom", "widthMetres": 4.0, "lengthMetres": 3.2,
        "objects": [{"id": "D1", "category": "wardrobe", "widthMetres": 0.9, "depthMetres": 0.6, "heightMetres": 2.9}],
    })

    assert body["questions"] == []
    assert body["unchanged"] == [server.NO_NEW_SIZE]


@pytest.mark.parametrize("sentence", ["", "   ", "\n"])
def test_every_empty_sentence_is_the_same_shape(sentence):
    assert interpret(sentence) == {"roomKind": None, "objectEdits": [], "questions": [], "unchanged": []}
