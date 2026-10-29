from __future__ import annotations

import os
import pickle
import random
import shutil
import subprocess
from math import exp
from random import randint

import pytest
import torch
from numpy.random import poisson

from dxtbx import flumpy
from dxtbx.model.experiment_list import ExperimentListFactory
from scitbx.array_family import flex as scitbx_flex

import dials_gpu.command_line.find_spots
from dials.algorithms.image.threshold import (
    DispersionExtendedThreshold,
    DispersionThreshold,
)
from dials.array_family import flex
from dials.model.data import PixelList, PixelListLabeller
from dials_gpu.spot_finding.connected_components import PixelListLabellerGPU
from dials_gpu.spot_finding.threshold import (
    DispersionExtendedSpotFinderThresholdGPU,
    DispersionSpotFinderThresholdGPU,
    dispersion_extended_threshold_gpu,
    dispersion_threshold_gpu,
    flex_to_tensor,
    tensor_to_flex_bool,
)


# =============================================================================
# 1. Equivalent of dials/tests/command_line/test_find_spots.py
# =============================================================================


def _check_expected_results(reflections):
    """Check expected results for standard processing."""
    assert len(reflections) in range(653, 655)
    refl = reflections[0]
    assert refl["intensity.sum.value"] == pytest.approx(42)
    assert refl["bbox"] == pytest.approx((1398, 1400, 513, 515, 0, 1))
    assert refl["xyzobs.px.value"] == pytest.approx(
        (1399.1190476190477, 514.2142857142857, 0.5)
    )
    assert "shoebox" in reflections


def test_find_spots_from_images(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            "output.reflections=spotfinder.refl",
            "output.shoeboxes=True",
            "algorithm=dispersion",
        ]
        + list(dials_data("centroid_test_data").glob("centroid*.cbf")),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    _check_expected_results(reflections)
    assert not reflections.experiment_identifiers().values(), list(
        reflections.experiment_identifiers().values()
    )


def test_find_spots_from_images_override_maximum(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            "maximum_trusted_value=100",
            "output.reflections=spotfinder.refl",
            "output.shoeboxes=True",
            "algorithm=dispersion",
        ]
        + list(dials_data("centroid_test_data").glob("centroid*.cbf")),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    sbox = reflections["shoebox"]
    for s in sbox:
        assert flex.max(s.data) <= 100


def test_find_spots_from_zero_indexed_cbf(dials_data, tmp_path):
    one_indexed_cbf = dials_data("centroid_test_data") / "centroid_0001.cbf"
    zero_indexed_cbf = tmp_path / "centroid_0000.cbf"
    shutil.copy(one_indexed_cbf, zero_indexed_cbf)

    result = subprocess.run(
        [shutil.which("dials_gpu.find_spots"), "nproc=1", zero_indexed_cbf],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "strong.refl").is_file()
    assert b"Saved 0 reflections to" not in result.stdout, "No spots found on 0000.cbf"


def test_find_spots_from_images_output_experiments(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            "output.reflections=spotfinder.refl",
            "output.shoeboxes=True",
            "algorithm=dispersion",
            "output.experiments=spotfinder.expt",
        ]
        + list(dials_data("centroid_test_data").glob("centroid*.cbf")),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    _check_expected_results(reflections)
    assert reflections.experiment_identifiers().values(), list(
        reflections.experiment_identifiers().values()
    )


def _import_images(filenames, tmp_path, extra_args=None):
    filenames = list(filenames)
    args = [shutil.which("dials.import")] + (extra_args or []) + filenames
    res = subprocess.run(args, cwd=tmp_path, capture_output=True)
    if res.returncode == 0 and (tmp_path / "imported.expt").is_file():
        return
    from dxtbx.model import Experiment, ExperimentList
    from dxtbx.model.experiment_list import ExperimentListFactory
    from dials.util.multi_dataset_handling import generate_experiment_identifiers

    expts = ExperimentListFactory.from_filenames([os.fspath(f) for f in sorted(filenames)])
    if extra_args and "oscillation=0,0" in extra_args:
        grid_expts = ExperimentList()
        for expt in expts:
            imageset = expt.imageset
            scan = imageset.get_scan()
            scan.set_oscillation((0.0, 0.0))
            imageset.set_scan(scan)
            start, end = scan.get_array_range()
            offset = start
            for j in range(start, end):
                subset = imageset[j - offset : j - offset + 1]
                grid_expts.append(
                    Experiment(
                        imageset=imageset,
                        beam=imageset.get_beam(),
                        detector=imageset.get_detector(),
                        goniometer=imageset.get_goniometer(),
                        scan=subset.get_scan(),
                        crystal=None,
                    )
                )
        expts = grid_expts
    generate_experiment_identifiers(expts)
    expts.as_file(tmp_path / "imported.expt")


def test_find_spots_from_imported_experiments(dials_data, tmp_path):
    _import_images(dials_data("centroid_test_data").glob("centroid*.cbf"), tmp_path)

    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            tmp_path / "imported.expt",
            "output.reflections=spotfinder.refl",
            "output.shoeboxes=True",
            "algorithm=dispersion",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    _check_expected_results(reflections)
    assert len(reflections.experiment_identifiers().values()) == 1, list(
        reflections.experiment_identifiers().values()
    )


def test_find_spots_from_imported_as_grid(dials_data, tmp_path):
    _import_images(
        dials_data("centroid_test_data").glob("centroid*.cbf"),
        tmp_path,
        extra_args=["oscillation=0,0"],
    )

    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            tmp_path / "imported.expt",
            "output.reflections=spotfinder.refl",
            "output.shoeboxes=True",
            "algorithm=dispersion",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    assert len(set(reflections["id"])) == 9, len(set(reflections["id"]))


def test_find_spots_with_resolution_filter(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            "output.reflections=spotfinder.refl",
            "output.shoeboxes=False",
            "algorithm=dispersion",
            "filter.d_min=2",
            "filter.d_max=15",
        ]
        + list(dials_data("centroid_test_data").glob("centroid*.cbf")),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    assert len(reflections) in range(467, 469)
    assert "shoebox" not in reflections


def test_find_spots_with_hot_mask(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            "write_hot_mask=True",
            "output.reflections=spotfinder.refl",
            "algorithm=dispersion",
            "output.shoeboxes=False",
        ]
        + list(dials_data("centroid_test_data").glob("centroid*.cbf")),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()
    assert (tmp_path / "hot_mask_0.pickle").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    assert len(reflections) in range(653, 655)
    assert "shoebox" not in reflections

    with (tmp_path / "hot_mask_0.pickle").open("rb") as f:
        mask = pickle.load(f)
    assert len(mask) == 1
    assert mask[0].count(False) == 12


def test_find_spots_with_hot_mask_with_prefix(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            "write_hot_mask=True",
            "hot_mask_prefix=my_hot_mask",
            "output.reflections=spotfinder.refl",
            "output.shoeboxes=False",
            "algorithm=dispersion",
        ]
        + list(dials_data("centroid_test_data").glob("centroid*.cbf")),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()
    assert (tmp_path / "my_hot_mask_0.pickle").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    assert len(reflections) in range(653, 655)
    assert "shoebox" not in reflections
    with (tmp_path / "my_hot_mask_0.pickle").open("rb") as f:
        mask = pickle.load(f)
    assert len(mask) == 1
    assert mask[0].count(False) == 12


def test_find_spots_with_generous_parameters(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            "min_spot_size=3",
            "max_separation=3",
            "output.reflections=spotfinder.refl",
            "algorithm=dispersion",
        ]
        + list(dials_data("centroid_test_data").glob("centroid*.cbf")),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    assert len(reflections) in range(678, 680)


def test_find_spots_with_user_defined_mask(dials_data, tmp_path):
    mask_pickle = dials_data("centroid_test_data") / "mask.pickle"
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            "output.reflections=spotfinder.refl",
            "output.shoeboxes=True",
            "algorithm=dispersion",
            f"lookup.mask={mask_pickle}",
        ]
        + list(dials_data("centroid_test_data").glob("centroid*.cbf")),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    experiments = ExperimentListFactory.from_json_file(
        dials_data("centroid_test_data") / "experiments.json"
    )
    assert len(experiments) == 1
    imageset = experiments.imagesets()[0]
    detector = imageset.get_detector()
    beam = imageset.get_beam()
    for x, y, z in reflections["xyzobs.px.value"]:
        d = detector[0].get_resolution_at_pixel(beam.get_s0(), (x, y))
        assert d >= 3


def test_find_spots_with_user_defined_region(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            "output.reflections=spotfinder.refl",
            "output.shoeboxes=True",
            "region_of_interest=800,1200,800,1200",
        ]
        + list(dials_data("centroid_test_data").glob("centroid*.cbf")),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    x, y, z = reflections["xyzobs.px.value"].parts()
    assert x.all_ge(800)
    assert y.all_ge(800)
    assert x.all_lt(1200)
    assert y.all_lt(1200)


def test_find_spots_with_image_exclusions(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            "output.reflections=spotfinder.refl",
            "output.shoeboxes=True",
            "exclude_images=4:6",
        ]
        + list(dials_data("centroid_test_data").glob("centroid*.cbf")),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    _, _, z = reflections["xyzobs.px.value"].parts()
    assert len(z.select((z > 3) & (z < 6.5))) == 0


def test_find_spots_with_xfel_stills(dials_data, tmp_path):
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            str(
                dials_data("image_examples")
                / "LCLS_cspad_nexus-idx-20130301060858801.cbf",
            ),
            "output.reflections=spotfinder.refl",
            "algorithm=dispersion",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "spotfinder.refl").is_file()

    reflections = flex.reflection_table.from_file(tmp_path / "spotfinder.refl")
    assert len(reflections) == 1148


def test_find_spots_with_per_image_statistics(dials_data, tmp_path):
    result = subprocess.run(
        [shutil.which("dials_gpu.find_spots"), "nproc=1", "per_image_statistics=True"]
        + list(dials_data("centroid_test_data").glob("centroid*.cbf")),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "strong.refl").is_file()
    assert b"Number of centroids per image for imageset 0:" in result.stdout
    assert (
        b"|   image |   #spots |   #spots_no_ice |   total_intensity |" in result.stdout
    )


@pytest.mark.parametrize(
    "blur,expected_nref", [("None", 559), ("narrow", 721), ("wide", 739)]
)
def test_find_spots_radial_profile(dials_data, blur, expected_nref, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reflections = dials_gpu.command_line.find_spots.run(
        [
            "nproc=1",
            "threshold.algorithm=radial_profile",
            f"blur={blur}",
        ]
        + [
            os.fspath(f) for f in dials_data("centroid_test_data").glob("centroid*.cbf")
        ],
        return_results=True,
    )
    assert len(reflections) == expected_nref


# =============================================================================
# 2. Equivalent of dials/tests/algorithms/image/threshold/test_local.py
# =============================================================================


class TestLocalThresholdGPU:
    def setup_class(self):
        spot = scitbx_flex.double(scitbx_flex.grid(11, 11))
        for j in range(11):
            for i in range(11):
                spot[j, i] = exp(-((j - 5) ** 2 + (i - 5) ** 2) / 2**2)

        self.image = scitbx_flex.double(scitbx_flex.grid(500, 500))
        for _ in range(50):
            x = randint(0, 500)
            y = randint(0, 500)
            for j in range(0, 11):
                for i in range(0, 11):
                    xx = x + i - 5
                    yy = y + j - 5
                    if xx >= 0 and yy >= 0 and xx < 500 and yy < 500:
                        self.image[yy, xx] = poisson(100 * spot[j, i])

        background = scitbx_flex.double(list(poisson(5, 500 * 500)))
        background.reshape(scitbx_flex.grid(500, 500))
        self.image += background

        self.mask = scitbx_flex.random_bool(500 * 500, 0.99)
        self.mask.reshape(scitbx_flex.grid(500, 500))
        self.gain = scitbx_flex.random_double(500 * 500) + 1.0
        self.gain.reshape(scitbx_flex.grid(500, 500))
        self.size = (3, 3)
        self.min_count = 2

    def test_dispersion_threshold_gpu(self):
        nsig_b = 3
        nsig_s = 3
        algorithm_cpu = DispersionThreshold(
            self.image.all(), self.size, nsig_b, nsig_s, 0, self.min_count
        )
        res_cpu = flex.bool(flex.grid(self.image.all()))
        res_cpu_gain = flex.bool(flex.grid(self.image.all()))
        algorithm_cpu(self.image, self.mask, res_cpu)
        algorithm_cpu(self.image, self.mask, self.gain, res_cpu_gain)

        gpu_thresh = DispersionSpotFinderThresholdGPU(
            kernel_size=self.size,
            n_sigma_b=nsig_b,
            n_sigma_s=nsig_s,
            min_count=self.min_count,
            global_threshold=0,
        )
        res_gpu = gpu_thresh.compute_threshold(self.image, self.mask)
        res_gpu_gain = gpu_thresh.compute_threshold(self.image, self.mask, gain=self.gain)

        assert res_cpu == res_gpu
        assert res_cpu_gain == res_gpu_gain

    def test_dispersion_extended_threshold_gpu(self):
        nsig_b = 3
        nsig_s = 3
        algorithm_cpu = DispersionExtendedThreshold(
            self.image.all(), self.size, nsig_b, nsig_s, 0, self.min_count
        )
        res_cpu = flex.bool(flex.grid(self.image.all()))
        res_cpu_gain = flex.bool(flex.grid(self.image.all()))
        algorithm_cpu(self.image, self.mask, res_cpu)
        algorithm_cpu(self.image, self.mask, self.gain, res_cpu_gain)

        gpu_thresh = DispersionExtendedSpotFinderThresholdGPU(
            kernel_size=self.size,
            n_sigma_b=nsig_b,
            n_sigma_s=nsig_s,
            min_count=self.min_count,
            global_threshold=0,
        )
        res_gpu = gpu_thresh.compute_threshold(self.image, self.mask)
        res_gpu_gain = gpu_thresh.compute_threshold(self.image, self.mask, gain=self.gain)

        assert res_cpu == res_gpu
        assert res_cpu_gain == res_gpu_gain

    @pytest.mark.parametrize(
        "gpu_fn", [dispersion_threshold_gpu, dispersion_extended_threshold_gpu]
    )
    def test_dispersion_algorithm_symmetry_gpu(self, gpu_fn):
        img_t = flex_to_tensor(self.image, dtype=torch.float64)
        mask_t = flex_to_tensor(self.mask, dtype=torch.bool)
        res1 = gpu_fn(img_t, mask_t, kernel_size=(3, 3), nsig_b=3, nsig_s=3, global_threshold=0, min_local=self.min_count)
        res2 = gpu_fn(img_t.T, mask_t.T, kernel_size=(3, 3), nsig_b=3, nsig_s=3, global_threshold=0, min_local=self.min_count)
        assert torch.equal(res1, res2.T)


# =============================================================================
# 3. Equivalent of dials/tests/model/data/test_pixel_list.py
# =============================================================================


def test_pixel_list_labeller_gpu_3d_and_2d():
    size = (300, 300)
    sf = 0
    labeller_cpu = PixelListLabeller()
    labeller_gpu = PixelListLabellerGPU()

    for i in range(3):
        image = scitbx_flex.random_int_gaussian_distribution(size[0] * size[1], 100, 5)
        mask = scitbx_flex.random_bool(size[0] * size[1], 0.2)
        image.reshape(scitbx_flex.grid(size))
        mask.reshape(scitbx_flex.grid(size))
        pl = PixelList(sf + i, image, mask)
        labeller_cpu.add(pl)
        labeller_gpu.add(pl)

    assert len(labeller_cpu.values()) == len(labeller_gpu.values())
    assert labeller_cpu.values().all_eq(labeller_gpu.values())
    assert labeller_cpu.labels_3d().all_eq(labeller_gpu.labels_3d())
    assert labeller_cpu.labels_2d().all_eq(labeller_gpu.labels_2d())


def test_pixel_list_labeller_gpu_with_no_points():
    size = (200, 200)
    labeller_gpu = PixelListLabellerGPU()
    for i in range(3):
        image = scitbx_flex.random_int_gaussian_distribution(size[0] * size[1], 100, 5)
        mask = scitbx_flex.bool(size[0] * size[1], False)
        image.reshape(scitbx_flex.grid(size))
        mask.reshape(scitbx_flex.grid(size))
        pl = PixelList(i, image, mask)
        labeller_gpu.add(pl)

    assert len(labeller_gpu.coords()) == 0
    assert len(labeller_gpu.labels_2d()) == 0
    assert len(labeller_gpu.labels_3d()) == 0


# =============================================================================
# 4. Equivalent of test_spot_counts_per_image.py & test_find_hot_pixels.py
# =============================================================================


def test_spot_counts_per_image_with_gpu(dials_data, tmp_path):
    path = dials_data("centroid_test_data")
    _import_images(path.glob("*.cbf*"), tmp_path, extra_args=["output.experiments=imported.expt"])

    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "imported.expt",
            "nproc=1",
            "min_spot_size=3",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert tmp_path.joinpath("strong.refl").is_file()

    result = subprocess.run(
        [
            shutil.which("dials.spot_counts_per_image"),
            "imported.expt",
            "strong.refl",
            "plot=spot_counts.png",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert tmp_path.joinpath("spot_counts.png").is_file()


def test_find_hot_pixels_with_gpu(dials_data, tmp_path):
    images = dials_data("centroid_test_data").glob("centroid*.cbf")
    result = subprocess.run(
        [
            shutil.which("dials_gpu.find_spots"),
            "nproc=1",
            "output.experiments=spotfinder.expt",
            "output.reflections=spotfinder.refl",
            "output.shoeboxes=True",
        ]
        + sorted(images),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    result = subprocess.run(
        [
            shutil.which("dials.find_hot_pixels"),
            "input.experiments=spotfinder.expt",
            "input.reflections=spotfinder.refl",
            "output.mask=hot_pixels.mask",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert tmp_path.joinpath("hot_pixels.mask").is_file()
    assert (
        b"Found 8 hot pixels" in result.stdout or b"Found 9 hot pixels" in result.stdout
    )
