"""The unified conversion interface.

The four converter packages are exercised by their own suites; what is tested
here is only the layer on top -- dispatch, option validation, and the specific
differences each adapter has to reconcile. Every underlying converter is
replaced, so no suite here needs a dataset, lerobot, or torch.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from xense_dataset_convert import (
    ConversionError,
    ConvertResult,
    Format,
    UnsupportedConversion,
    available,
    convert,
)


def test_importing_the_facade_does_not_import_the_converters():
    """The registry must stay lazy.

    ``import lerobot2mcap`` executes ``from lerobot.datasets.lerobot_dataset
    import LeRobotDataset`` at module level, which drags in torch. An engine
    asking what conversions exist, or running an HDF5 one, must not pay that.
    Checked in a fresh interpreter because this suite's own imports would
    otherwise mask it.
    """
    probe = (
        "import sys; import xense_dataset_convert as f; f.available();"
        "leaked = [n for n in ('lerobot2mcap', 'lerobot2zarr', 'mcap2lerobot',"
        " 'hdf52lerobot', 'lerobot', 'torch') if n in sys.modules];"
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"facade import pulled in {result.stdout.strip()}"


def test_available_lists_every_pair():
    assert set(available()) == {
        (Format.LEROBOT, Format.MCAP),
        (Format.LEROBOT, Format.ZARR),
        (Format.MCAP, Format.LEROBOT),
        (Format.UMI_HDF5, Format.LEROBOT),
    }


def test_format_is_built_from_a_config_value():
    assert Format("umi_hdf5") is Format.UMI_HDF5
    assert str(Format.LEROBOT) == "lerobot"


def test_unsupported_pair_names_the_alternatives(tmp_path):
    with pytest.raises(UnsupportedConversion, match="Cannot convert zarr to mcap"):
        convert(Format.ZARR, Format.MCAP, tmp_path, tmp_path / "out")


def test_a_bare_string_is_refused_with_the_fix(tmp_path):
    with pytest.raises(TypeError, match=r"Format\('lerobot'\)"):
        convert("lerobot", Format.MCAP, tmp_path, tmp_path / "out")


def test_unknown_option_is_rejected_and_lists_what_is_accepted(tmp_path):
    with pytest.raises(TypeError) as caught:
        convert(
            Format.UMI_HDF5, Format.LEROBOT, tmp_path, tmp_path / "out", quat_ordr="xyzw"
        )
    message = str(caught.value)
    assert "'quat_ordr'" in message
    assert "quat_order" in message  # the accepted spelling is offered


# -- per-adapter reconciliations ------------------------------------------


def _install(monkeypatch, name: str, **attributes) -> types.ModuleType:
    """Put a stand-in module in sys.modules for an adapter to import."""
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_lerobot_to_mcap_turns_false_into_an_error(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        "lerobot2mcap",
        DEFAULT_CONVERTER_FUNCTIONS="/somewhere/converter_functions.yaml",
        convert_dataset=lambda **kwargs: False,
    )
    with pytest.raises(ConversionError, match="failed to convert"):
        convert(Format.LEROBOT, Format.MCAP, tmp_path, tmp_path / "out")


def test_lerobot_to_mcap_defaults_the_converter_functions_path(monkeypatch, tmp_path):
    seen = {}

    def fake_convert_dataset(**kwargs):
        seen.update(kwargs)
        return True

    _install(
        monkeypatch,
        "lerobot2mcap",
        DEFAULT_CONVERTER_FUNCTIONS="/shipped/converter_functions.yaml",
        convert_dataset=fake_convert_dataset,
    )
    result = convert(Format.LEROBOT, Format.MCAP, tmp_path, tmp_path / "out")

    assert seen["converter_functions_path"] == Path("/shipped/converter_functions.yaml")
    assert seen["dataset_root"] == tmp_path
    assert result.output_path == tmp_path / "out"


def test_lerobot_to_zarr_opens_the_dataset_from_a_path(monkeypatch, tmp_path):
    """The caller passes a path; the adapter does the opening."""
    opened = {}

    class FakeConverter:
        def __init__(self, dataset, output_path, **kwargs):
            opened["dataset"] = dataset
            opened["output_path"] = output_path

        def convert(self, limit=None):
            opened["limit"] = limit
            return opened["output_path"]

    def fake_open(root, repo_id=None, episodes=None):
        opened["root"] = root
        opened["repo_id"] = repo_id
        return "opened-dataset"

    module = _install(monkeypatch, "lerobot2zarr", **{})
    _install(
        monkeypatch,
        "lerobot2zarr.zarr_converter",
        LeRobotZarrConverter=FakeConverter,
        open_lerobot_dataset=fake_open,
    )
    module.zarr_converter = sys.modules["lerobot2zarr.zarr_converter"]

    result = convert(
        Format.LEROBOT, Format.ZARR, tmp_path, tmp_path / "out.zarr", limit=5
    )

    assert opened["root"] == tmp_path
    assert opened["dataset"] == "opened-dataset"
    assert opened["limit"] == 5
    assert result.output_path == tmp_path / "out.zarr"


def test_mcap_to_lerobot_defaults_repo_id_to_the_output_name(monkeypatch, tmp_path):
    seen = {}

    class FakeConverter:
        def __init__(self, inputs, output_root, repo_id, config):
            seen["repo_id"] = repo_id
            seen["inputs"] = inputs
            seen["config"] = config
            self.output_root = output_root

        def convert(self):
            return self.output_root

    class FakeConfig:
        def __init__(self, **kwargs):
            seen["config_kwargs"] = kwargs

    _install(monkeypatch, "mcap2lerobot", **{})
    _install(monkeypatch, "mcap2lerobot.config", ConverterConfig=FakeConfig)
    _install(monkeypatch, "mcap2lerobot.converter", McapToLeRobotConverter=FakeConverter)

    output = tmp_path / "converted_dataset"
    convert(Format.MCAP, Format.LEROBOT, tmp_path / "episodes", output, fps=30)

    assert seen["repo_id"] == "converted_dataset"
    # Only what the caller set, so the dataclass defaults survive.
    assert seen["config_kwargs"] == {"fps": 30}


def test_lerobot_targets_report_totals_from_the_written_metadata(monkeypatch, tmp_path):
    output = tmp_path / "dataset"
    (output / "meta").mkdir(parents=True)
    (output / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 1472, "fps": 20})
    )

    class FakeConverter:
        def __init__(self, *args, **kwargs):
            pass

        def convert(self):
            return output

    class FakeConfig:
        def __init__(self, **kwargs):
            pass

    _install(monkeypatch, "mcap2lerobot", **{})
    _install(monkeypatch, "mcap2lerobot.config", ConverterConfig=FakeConfig)
    _install(monkeypatch, "mcap2lerobot.converter", McapToLeRobotConverter=FakeConverter)

    result = convert(Format.MCAP, Format.LEROBOT, tmp_path / "episodes", output)

    assert result.details == {"total_episodes": 2, "total_frames": 1472}


def test_missing_metadata_leaves_details_empty_rather_than_failing(monkeypatch, tmp_path):
    """The totals are a convenience on an already-successful conversion."""
    output = tmp_path / "dataset"

    class FakeConverter:
        def __init__(self, *args, **kwargs):
            pass

        def convert(self):
            return output

    class FakeConfig:
        def __init__(self, **kwargs):
            pass

    _install(monkeypatch, "mcap2lerobot", **{})
    _install(monkeypatch, "mcap2lerobot.config", ConverterConfig=FakeConfig)
    _install(monkeypatch, "mcap2lerobot.converter", McapToLeRobotConverter=FakeConverter)

    result = convert(Format.MCAP, Format.LEROBOT, tmp_path / "episodes", output)

    assert isinstance(result, ConvertResult)
    assert result.details == {}


def test_umi_hdf5_passes_overrides_and_overwrite_through(monkeypatch, tmp_path):
    seen = {}

    class FakeConfig:
        @classmethod
        def from_yaml(cls, path):
            seen["config_path"] = path
            return cls()

        def with_overrides(self, **overrides):
            seen["overrides"] = overrides
            return self

    class FakeConverter:
        def __init__(self, inputs, output_root, config, overwrite):
            seen["inputs"] = inputs
            seen["overwrite"] = overwrite
            self.output_root = output_root

        def convert(self):
            return self.output_root

    _install(monkeypatch, "hdf52lerobot", **{})
    _install(monkeypatch, "hdf52lerobot.cli", DEFAULT_CONFIG=Path("/shipped/hdf52lerobot.yaml"))
    _install(monkeypatch, "hdf52lerobot.config", ConverterConfig=FakeConfig)
    _install(monkeypatch, "hdf52lerobot.converter", Hdf5ToLeRobotConverter=FakeConverter)

    convert(
        Format.UMI_HDF5,
        Format.LEROBOT,
        ["/data/ep0", "/data/ep1"],
        tmp_path / "out",
        task="pick up the cup",
        robot_type="bi_rdt_gripper",
        overwrite=True,
    )

    assert seen["overrides"] == {"task": "pick up the cup", "robot_type": "bi_rdt_gripper"}
    assert seen["overwrite"] is True
    assert seen["inputs"] == [Path("/data/ep0"), Path("/data/ep1")]
    # The shipped config does not exist in the test environment, so it is skipped.
    assert "config_path" not in seen


def test_several_inputs_are_refused_where_only_one_makes_sense(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        "lerobot2mcap",
        DEFAULT_CONVERTER_FUNCTIONS="/shipped/converter_functions.yaml",
        convert_dataset=lambda **kwargs: True,
    )
    with pytest.raises(ValueError, match="exactly one input"):
        convert(Format.LEROBOT, Format.MCAP, [tmp_path, tmp_path], tmp_path / "out")


def test_empty_inputs_are_refused(tmp_path):
    with pytest.raises(ValueError, match="inputs is empty"):
        convert(Format.LEROBOT, Format.MCAP, [], tmp_path / "out")


# -- the mocks above would hide a misspelled attribute; these do not ----------


def test_adapters_name_attributes_that_really_exist():
    """Guard against the stand-in modules masking a typo.

    Three of the four packages import cleanly here (their heavy dependencies are
    all lazy), so the names the adapters reach for are checked against the real
    thing.
    """
    import hdf52lerobot.cli
    import hdf52lerobot.config
    import hdf52lerobot.converter
    import lerobot2zarr.zarr_converter
    import mcap2lerobot.config
    import mcap2lerobot.converter

    assert isinstance(hdf52lerobot.cli.DEFAULT_CONFIG, Path)
    assert hasattr(hdf52lerobot.config.ConverterConfig, "from_yaml")
    assert hasattr(hdf52lerobot.config.ConverterConfig, "with_overrides")
    assert hasattr(hdf52lerobot.converter, "Hdf5ToLeRobotConverter")
    assert hasattr(lerobot2zarr.zarr_converter, "LeRobotZarrConverter")
    assert hasattr(lerobot2zarr.zarr_converter, "open_lerobot_dataset")
    assert hasattr(mcap2lerobot.config, "ConverterConfig")
    assert hasattr(mcap2lerobot.converter, "McapToLeRobotConverter")


def test_lerobot2mcap_still_provides_what_its_adapter_uses():
    """Same check for lerobot2mcap, which cannot be imported without torch.

    Read as source instead, so the guard costs nothing and works in CI.
    """
    import ast

    source = (Path(__file__).resolve().parent.parent / "lerobot2mcap" / "__init__.py").read_text()
    tree = ast.parse(source)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "convert_dataset" in functions
    assert "DEFAULT_CONVERTER_FUNCTIONS" in assigned


def test_umi_hdf5_option_names_match_the_config_it_overrides():
    """Every override the adapter accepts must be a real ConverterConfig field."""
    from hdf52lerobot.config import ConverterConfig

    from xense_dataset_convert.adapters.umi_hdf5_to_lerobot import OVERRIDE_OPTIONS

    fields = set(ConverterConfig.__dataclass_fields__)
    assert OVERRIDE_OPTIONS <= fields, OVERRIDE_OPTIONS - fields


def test_mcap_option_names_match_the_config_it_builds():
    from mcap2lerobot.config import ConverterConfig

    from xense_dataset_convert.adapters.mcap_to_lerobot import CONFIG_OPTIONS

    fields = set(ConverterConfig.__dataclass_fields__)
    assert CONFIG_OPTIONS == fields, CONFIG_OPTIONS ^ fields


def test_lerobot2mcap_looks_up_the_distribution_by_its_real_name():
    """Renaming the distribution must not break `import lerobot2mcap`.

    Its __version__ comes from importlib.metadata at module level, so a name
    that no longer matches pyproject raises PackageNotFoundError on import --
    which nothing else here would catch, since CI never installs the project
    and lerobot2mcap has no suite of its own.
    """
    import re

    root = Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text()
    distribution = re.search(r'^name = "([^"]+)"', pyproject, re.M).group(1)

    source = (root / "lerobot2mcap" / "__init__.py").read_text()
    looked_up = re.findall(r'version\("([^"]+)"\)', source)

    assert looked_up, "lerobot2mcap no longer looks up a distribution version"
    assert set(looked_up) == {distribution}, (
        f"lerobot2mcap asks importlib.metadata for {looked_up}, "
        f"but the distribution is named {distribution!r}"
    )
