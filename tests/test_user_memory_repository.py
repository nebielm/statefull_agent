import json

from app.repositories import user_memory


def test_load_user_data_returns_empty_dict_for_missing_file(tmp_path):
    missing_file = tmp_path / "missing.json"

    assert user_memory.load_user_data(str(missing_file)) == {}


def test_lookup_user_value_reads_nested_category_value():
    user_data = {
        "profile": {"city": "Berlin"},
        "preferences": {"diet": "vegetarian"},
    }

    assert user_memory.lookup_user_value(user_data, "city") == "Berlin"
    assert user_memory.lookup_user_value(user_data, "diet") == "vegetarian"
    assert user_memory.lookup_user_value(user_data, "unknown") is None


def test_is_valid_key_blocks_immutable_and_unknown_keys():
    assert user_memory.is_valid_key("name", "profile") is False
    assert user_memory.is_valid_key("city", "profile") is True
    assert user_memory.is_valid_key("unknown", "profile") is False


def test_controlled_structured_data_storage_writes_expected_json(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))

    result = user_memory.controlled_structured_data_storage(
        user_id="user-1",
        key="city",
        value="Berlin",
        category="profile",
    )

    assert result == "Stored city: Berlin"
    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"city": "Berlin"}}}


def test_controlled_structured_data_storage_reports_no_change(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(json.dumps({"user-1": {"profile": {"city": "Berlin"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))

    result = user_memory.controlled_structured_data_storage(
        user_id="user-1",
        key="city",
        value="Berlin",
        category="profile",
    )

    assert result == "No change for city"


def test_controlled_structured_data_storage_rejects_invalid_schema_key(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))

    result = user_memory.controlled_structured_data_storage(
        user_id="user-1",
        key="city",
        value="Berlin",
        category="preferences",
    )

    assert result == "city is not allowed in preferences."
    assert not data_file.exists()


def test_retrieve_structured_memory_filters_by_category_and_key(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(
        json.dumps(
            {
                "user-1": {
                    "profile": {"city": "Berlin", "job": "Engineer"},
                    "preferences": {"diet": "vegetarian"},
                }
            }
        )
    )
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))

    results = user_memory.retrieve_structured_memory(
        "user-1",
        relevant_categories=["profile"],
        relevant_keys=["city"],
    )

    assert results == [
        {
            "key": "city",
            "value": "Berlin",
            "category": "profile",
            "score": 1.0,
        }
    ]
