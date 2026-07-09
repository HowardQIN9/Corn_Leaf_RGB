from pathlib import Path

from scripts.extract_leaf2_crops import parse_leaf_filename


def test_parse_green_profile_filename() -> None:
    info = parse_leaf_filename(Path("LeafDoc_109_tall_1_Leaf2_1780576946303_green_profiles.csv"))

    assert info is not None
    assert info.prefix == "LeafDoc"
    assert info.plot_number == "109"
    assert info.geno == "tall"
    assert info.plant_number == "1"
    assert info.leaf == "Leaf2"
    assert info.timestamp == "1780576946303"
    assert info.output_suffix == "_green_profiles"
