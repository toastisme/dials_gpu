from __future__ import annotations

import copy
import math
import pathlib
import random
import shutil
import subprocess
import sys
from copy import deepcopy
from math import pi

import numpy as np
import pytest
from annlib_ext import AnnAdaptor
from cctbx.sgtbx import space_group, space_group_symbols
from cctbx.uctbx import unit_cell
from dxtbx.model import Beam, Crystal, Detector, Experiment, Goniometer, ScanFactory
from dxtbx.model.experiment_list import ExperimentList, ExperimentListFactory
from libtbx import phil
from libtbx.phil import parse
from libtbx.test_utils import approx_equal
from rstbx.symmetry.constraints.parameter_reduction import symmetrize_reduce_enlarge
from scitbx import matrix
from dials.array_family import flex
from scitbx.math import five_number_summary

from dials.algorithms.refinement import DialsRefineConfigError
from dials.algorithms.refinement.outlier_detection.outlier_base import (
    phil_scope as outlier_phil_scope,
)
from dials.algorithms.refinement.parameterisation.autoreduce import (
    AutoReduce,
    phil_scope as ar_phil_scope,
)
from dials.algorithms.refinement.parameterisation.beam_parameters import (
    BeamParameterisation,
)
from dials.algorithms.refinement.parameterisation.crystal_parameters import (
    CrystalOrientationParameterisation,
    CrystalUnitCellParameterisation,
)
from dials.algorithms.refinement.parameterisation.detector_parameters import (
    DetectorParameterisationHierarchical,
    DetectorParameterisationMultiPanel,
    DetectorParameterisationSinglePanel,
)
from dials.algorithms.refinement.parameterisation.goniometer_parameters import (
    GoniometerParameterisation,
)
from dials.algorithms.refinement.parameterisation.parameter_report import (
    ParameterReporter,
)
from dials.algorithms.refinement.parameterisation.scan_varying_beam_parameters import (
    ScanVaryingBeamParameterisation,
)
from dials.algorithms.refinement.parameterisation.scan_varying_crystal_parameters import (
    ScanVaryingCrystalOrientationParameterisation,
    ScanVaryingCrystalUnitCellParameterisation,
)
from dials.algorithms.refinement.parameterisation.scan_varying_detector_parameters import (
    ScanVaryingDetectorParameterisationSinglePanel,
)
from dials.algorithms.refinement.parameterisation.scan_varying_goniometer_parameters import (
    ScanVaryingGoniometerParameterisation,
)
from dials.algorithms.refinement.prediction.managed_predictors import ScansRayPredictor
from dials.algorithms.refinement.refiner import (
    _trim_scans_to_observations,
    phil_scope as refiner_phil_scope,
)
from dials.algorithms.refinement.reflection_manager import (
    BlockCalculator,
    ReflectionManager,
    phil_scope as refman_phil_scope,
)
from dials.algorithms.spot_prediction import IndexGenerator, ray_intersection
from dials.util.slice import slice_reflections
from dials_gpu.command_line.refine import _find_disjoint_sets
from dials_gpu.refinement import (
    CentroidOutlierFactoryGPU,
    ExperimentsPredictorFactoryGPU,
    GaussNewtonIterationsGPU,
    Journal,
    LBFGScurvsGPU,
    LeastSquaresPositionalResidualWithRmsdCutoffGPU,
    LevenbergMarquardtIterationsGPU,
    RefinerFactoryGPU,
    RefinerGPU,
    ReflectionManagerFactoryGPU,
    ScansExperimentsPredictorGPU,
    ScanVaryingPredictionParameterisationGPU,
    SphericalRelpStillsPredictionParameterisationGPU,
    StillsExperimentsPredictorGPU,
    StillsPredictionParameterisationGPU,
    XYPhiPredictionParameterisationGPU,
    _patch_crystal_uc_parameterisations,
)

DIALS_REPO_ROOT = pathlib.Path("/home/davidm/diffraction/dials/modules/dials")
if str(DIALS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DIALS_REPO_ROOT))

from tests.algorithms.refinement import geometry_phil, minimiser_phil, setup_geometry
from tests.algorithms.refinement.test_multi_panel_detector_parameterisation import (
    make_panel_in_array,
)

BIN_DIR = pathlib.Path(sys.executable).parent
DIALS_GPU_REFINE = str(BIN_DIR / "dials_gpu.refine")


# =============================================================================
# 1. Port of tests/algorithms/refinement/test_centroid_outlier.py
# =============================================================================


@pytest.mark.parametrize(
    "method,colnames,expected_nout",
    [
        ("tukey", ("x_resid", "y_resid", "phi_resid"), 34),
        ("mcd", ("x_resid", "y_resid", "phi_resid"), 35),
        ("sauter_poon", ("miller_index", "xyzobs.px.value", "xyzcal.px"), 34),
    ],
)
def test_centroid_outlier_gpu(dials_data, method, colnames, expected_nout):
    flex.set_random_seed(42)
    data_dir = dials_data("refinement_test_data")
    residuals = flex.reflection_table.from_file(
        data_dir / "centroid_outlier_residuals.refl"
    )
    params = outlier_phil_scope.extract()
    params.outlier.algorithm = method
    params.outlier.sauter_poon.px_sz = (0.1, 0.1)
    outlier_detector = CentroidOutlierFactoryGPU.from_parameters_and_colnames(
        params, colnames
    )
    outlier_detector(residuals)
    outliers = residuals.get_flags(residuals.flags.centroid_outlier)
    assert outliers.count(True) == expected_nout


# =============================================================================
# 2. Port of tests/algorithms/refinement/test_reflection_manager.py
# =============================================================================


def test_scan_margin_gpu(dials_data):
    data_dir = dials_data("l_cysteine_dials_output")
    experiments = ExperimentListFactory.from_json_file(
        data_dir / "indexed.expt", check_format=False
    )
    reflections = flex.reflection_table.from_file(data_dir / "indexed.refl")
    orig_phi = reflections["xyzobs.mm.value"].parts()[2]

    reflections.set_flags(
        flex.bool(len(reflections), True), reflections.flags.predicted
    )

    refman = ReflectionManager(reflections, experiments)
    refman.finalise()
    refs1 = refman.get_matches()
    phi1 = refs1["xyzobs.mm.value"].parts()[2]

    margin = 1.0
    refman2 = ReflectionManager(reflections, experiments, scan_margin=margin)
    refman2.finalise()
    refs2 = refman2.get_matches()
    phi2 = refs2["xyzobs.mm.value"].parts()[2]

    assert min(orig_phi) == min(phi1)
    assert max(orig_phi) == max(phi1)
    assert min(phi2) == pytest.approx(min(phi1) + math.radians(margin), abs=1e-3)
    assert max(phi2) == pytest.approx(max(phi1) - math.radians(margin), abs=1e-3)


# =============================================================================
# 3. Port of tests/algorithms/refinement/test_refiner_config.py
# =============================================================================


@pytest.mark.parametrize(
    "detector_parameterisation_choice",
    ["automatic", "single", "multiple", "hierarchical"],
)
def test_multi_panel_parameterisations_gpu(
    dials_data, detector_parameterisation_choice
):
    data_dir = dials_data("iterative_cspad_refinement")
    exp_file = data_dir / "cspad_refined_experiments_step6_level2_300.json"
    ref_file = data_dir / "cspad_reflections_step7_300.pickle"

    reflections = flex.reflection_table.from_file(ref_file)
    experiments = ExperimentListFactory.from_json_file(exp_file, check_format=False)

    params = refiner_phil_scope.fetch(source=phil.parse("")).extract()
    params.refinement.parameterisation.detector.panels = (
        detector_parameterisation_choice
    )

    if detector_parameterisation_choice == "single":
        with pytest.raises(DialsRefineConfigError):
            RefinerFactoryGPU.from_parameters_data_experiments(
                params, reflections, experiments
            )
    else:
        refiner = RefinerFactoryGPU.from_parameters_data_experiments(
            params, reflections, experiments
        )
        assert refiner.experiment_type == "stills"


def test_trim_scans_to_observations_gpu(dials_data):
    data_dir = dials_data("l_cysteine_dials_output")
    experiments = ExperimentListFactory.from_json_file(
        data_dir / "indexed.expt", check_format=False
    )
    reflections = flex.reflection_table.from_file(data_dir / "indexed.refl")

    image_ranges = [e.scan.get_image_range() for e in experiments]
    osc_ranges = [e.scan.get_oscillation_range() for e in experiments]
    for a, b in zip(image_ranges, [(1, 1700), (1, 1700), (1, 1700), (1, 1800)]):
        assert a == b
    for a, b in zip(
        osc_ranges, [(-145.0, 25.0), (-145.0, 25.0), (-145.0, 25.0), (0.0, 180.0)]
    ):
        assert a == pytest.approx(b)

    trim_expt = _trim_scans_to_observations(deepcopy(experiments), reflections)
    new_im_ranges = [e.scan.get_image_range() for e in trim_expt]
    for a, b in zip(image_ranges, new_im_ranges):
        assert a == b

    sliced_ranges = [(r[0] + 20, r[1] - 20) for r in image_ranges]
    sliced = slice_reflections(reflections, sliced_ranges)

    trim_expt = _trim_scans_to_observations(deepcopy(experiments), sliced)
    new_array_ranges = [e.scan.get_array_range() for e in trim_expt]

    for i, e in enumerate(trim_expt):
        refs = sliced.select(sliced["id"] == i)
        bb = refs["shoebox"].bounding_boxes()
        z_min, z_max = bb.parts()[4:]
        assert new_array_ranges[i] == (min(z_min), max(z_max))


# =============================================================================
# 4. Port of tests/algorithms/refinement/test_prediction_parameters.py
# =============================================================================


def test_prediction_parameters_gpu():
    overrides = """geometry.parameters.crystal.a.length.range = 10 50
  geometry.parameters.crystal.b.length.range = 10 50
  geometry.parameters.crystal.c.length.range = 10 50"""

    master_phil = parse(geometry_phil)
    models = setup_geometry.Extract(master_phil, overrides)

    mydetector = models.detector
    mygonio = models.goniometer
    mycrystal = models.crystal
    mybeam = models.beam

    sequence_range = (0.0, math.pi / 5.0)
    sf = ScanFactory()
    myscan = sf.make_scan(
        image_range=(1, 720),
        exposure_times=0.1,
        oscillation=(0, 0.1),
        epochs=list(range(720)),
        deg=True,
    )

    det_param = DetectorParameterisationSinglePanel(mydetector)
    s0_param = BeamParameterisation(mybeam, mygonio)
    xlo_param = CrystalOrientationParameterisation(mycrystal)
    xluc_param = CrystalUnitCellParameterisation(mycrystal)
    gon_param = GoniometerParameterisation(mygonio, mybeam)

    experiments = ExperimentList()
    experiments.append(
        Experiment(
            beam=mybeam,
            detector=mydetector,
            goniometer=mygonio,
            scan=myscan,
            crystal=mycrystal,
            imageset=None,
        )
    )

    pred_param = XYPhiPredictionParameterisationGPU(
        experiments,
        detector_parameterisations=[det_param],
        beam_parameterisations=[s0_param],
        xl_orientation_parameterisations=[xlo_param],
        xl_unit_cell_parameterisations=[xluc_param],
        goniometer_parameterisations=[gon_param],
    )
    _patch_crystal_uc_parameterisations(pred_param)

    resolution = 2.0
    index_generator = IndexGenerator(
        mycrystal.get_unit_cell(),
        space_group(space_group_symbols(1).hall()).type(),
        resolution,
    )
    indices = index_generator.to_array()

    ray_predictor = ScansRayPredictor(experiments, sequence_range)
    obs_refs = ray_predictor(indices)
    intersects = ray_intersection(mydetector, obs_refs)
    obs_refs = obs_refs.select(intersects)

    ref_predictor = ScansExperimentsPredictorGPU(experiments)
    obs_refs["id"] = flex.int(len(obs_refs), 0)
    obs_refs = ref_predictor(obs_refs)
    obs_refs["xyzobs.mm.value"] = obs_refs["xyzcal.mm"]

    im_width = 0.1 * math.pi / 180.0
    px_size = mydetector[0].get_pixel_size()
    var_x = flex.double(len(obs_refs), (px_size[0] / 2.0) ** 2)
    var_y = flex.double(len(obs_refs), (px_size[1] / 2.0) ** 2)
    var_phi = flex.double(len(obs_refs), (im_width / 2.0) ** 2)
    obs_refs["xyzobs.mm.variance"] = flex.vec3_double(var_x, var_y, var_phi)

    refman = ReflectionManager(obs_refs, experiments, outlier_detector=None)
    refman.finalise()
    reflections = refman.get_matches()

    an_grads = pred_param.get_gradients(reflections)

    p_vals = pred_param.get_param_vals()
    deltas = [1.0e-7] * len(p_vals)

    for i, delta in enumerate(deltas):
        val = p_vals[i]
        p_vals[i] -= delta / 2.0
        pred_param.set_param_vals(p_vals)
        ref_predictor(reflections)
        rev_state = reflections["xyzcal.mm"].deep_copy()

        p_vals[i] += delta
        pred_param.set_param_vals(p_vals)
        ref_predictor(reflections)
        fwd_state = reflections["xyzcal.mm"].deep_copy()
        p_vals[i] = val

        fd = fwd_state - rev_state
        x_grads, y_grads, phi_grads = fd.parts()
        x_grads /= delta
        y_grads /= delta
        phi_grads /= delta

        assert x_grads == pytest.approx(an_grads[i]["dX_dp"], abs=5.0e-6)
        assert y_grads == pytest.approx(an_grads[i]["dY_dp"], abs=5.5e-6)
        assert phi_grads == pytest.approx(an_grads[i]["dphi_dp"], abs=5.0e-6)

    pred_param.set_param_vals(p_vals)


# =============================================================================
# 5. Port of tests/algorithms/refinement/test_scan_varying_prediction_parameters.py
# =============================================================================


def test_scan_varying_prediction_parameters_gpu():
    overrides = """geometry.parameters.crystal.a.length.range = 10 50
geometry.parameters.crystal.b.length.range = 10 50
geometry.parameters.crystal.c.length.range = 10 50"""

    master_phil = parse(geometry_phil)
    models = setup_geometry.Extract(master_phil, overrides)
    detector = models.detector
    goniometer = models.goniometer
    crystal = models.crystal
    beam = models.beam

    sf = ScanFactory()
    scan = sf.make_scan((1, 20), 0.5, (0, 0.5), list(range(20)))

    experiments = ExperimentList()
    experiments.append(
        Experiment(
            beam=beam,
            detector=detector,
            goniometer=goniometer,
            scan=scan,
            crystal=crystal,
            imageset=None,
        )
    )

    ref_predictor = ScansExperimentsPredictorGPU(experiments)
    det_param = ScanVaryingDetectorParameterisationSinglePanel(
        detector, scan.get_array_range(), 3
    )
    s0_param = ScanVaryingBeamParameterisation(
        beam, scan.get_array_range(), 3, goniometer
    )
    xlo_param = ScanVaryingCrystalOrientationParameterisation(
        crystal, scan.get_array_range(), 3
    )
    xluc_param = ScanVaryingCrystalUnitCellParameterisation(
        crystal, scan.get_array_range(), 3
    )
    gon_param = ScanVaryingGoniometerParameterisation(
        goniometer, scan.get_array_range(), 3, beam
    )

    sequence_range = scan.get_oscillation_range(deg=False)
    index_generator = IndexGenerator(
        crystal.get_unit_cell(),
        space_group(space_group_symbols(1).hall()).type(),
        2.0,
    )
    indices = index_generator.to_array()
    ray_predictor = ScansRayPredictor(experiments, sequence_range)
    obs_refs = ray_predictor(indices)
    intersects = ray_intersection(detector, obs_refs)
    obs_refs = obs_refs.select(intersects)
    obs_refs["id"] = flex.int(len(obs_refs), 0)
    obs_refs = ref_predictor(obs_refs)
    obs_refs["xyzobs.mm.value"] = obs_refs["xyzcal.mm"]

    im_width = 0.1 * pi / 180.0
    px_size = detector[0].get_pixel_size()
    var_x = flex.double(len(obs_refs), (px_size[0] / 2.0) ** 2)
    var_y = flex.double(len(obs_refs), (px_size[1] / 2.0) ** 2)
    var_phi = flex.double(len(obs_refs), (im_width / 2.0) ** 2)
    obs_refs["xyzobs.mm.variance"] = flex.vec3_double(var_x, var_y, var_phi)

    flex.set_random_seed(12407)
    reflections = obs_refs.select(flex.random_selection(len(obs_refs), 10))
    block_calculator = BlockCalculator(experiments, reflections)
    reflections = block_calculator.per_image()

    refman = ReflectionManager(reflections, experiments, outlier_detector=None)
    refman.finalise()

    pred_param = ScanVaryingPredictionParameterisationGPU(
        experiments,
        [det_param],
        [s0_param],
        [xlo_param],
        [xluc_param],
        [gon_param],
    )
    _patch_crystal_uc_parameterisations(pred_param)

    reflections = refman.get_matches()
    pred_param.compose(reflections)
    an_grads = pred_param.get_gradients(reflections)

    p_vals = pred_param.get_param_vals()
    deltas = [1.0e-7] * len(p_vals)

    for i, delta in enumerate(deltas):
        val = p_vals[i]
        p_vals[i] -= delta / 2.0
        pred_param.set_param_vals(p_vals)
        pred_param.compose(reflections)
        ref_predictor(reflections)
        rev_state = reflections["xyzcal.mm"].deep_copy()

        p_vals[i] += delta
        pred_param.set_param_vals(p_vals)
        pred_param.compose(reflections)
        ref_predictor(reflections)
        fwd_state = reflections["xyzcal.mm"].deep_copy()
        p_vals[i] = val

        fd = fwd_state - rev_state
        x_grads, y_grads, phi_grads = fd.parts()
        x_grads /= delta
        y_grads /= delta
        phi_grads /= delta

        for a, b in zip(x_grads, an_grads[i]["dX_dp"]):
            assert a == pytest.approx(b, abs=1e-5)
        for a, b in zip(y_grads, an_grads[i]["dY_dp"]):
            assert a == pytest.approx(b, abs=1e-5)
        for a, b in zip(phi_grads, an_grads[i]["dphi_dp"]):
            assert a == pytest.approx(b, abs=1e-5)

    pred_param.set_param_vals(p_vals)
    pred_param.compose(reflections)


# =============================================================================
# 6. Port of tests/algorithms/refinement/test_stills_prediction_parameters.py
# =============================================================================


class _StillsTestHelper:
    def __init__(self):
        overrides = """
    geometry.parameters.crystal.a.length.range=40 50;
    geometry.parameters.crystal.b.length.range=40 50;
    geometry.parameters.crystal.c.length.range=40 50;
    geometry.parameters.random_seed = 42"""

        master_phil = parse(geometry_phil)
        models = setup_geometry.Extract(master_phil, overrides)
        self.detector = models.detector
        self.gonio = models.goniometer
        self.crystal = models.crystal
        self.beam = models.beam

        self.stills_experiments = ExperimentList()
        self.stills_experiments.append(
            Experiment(
                beam=self.beam,
                detector=self.detector,
                crystal=self.crystal,
                imageset=None,
            )
        )

        self.det_param = DetectorParameterisationSinglePanel(self.detector)
        self.s0_param = BeamParameterisation(self.beam, self.gonio)
        self.xlo_param = CrystalOrientationParameterisation(self.crystal)
        self.xluc_param = CrystalUnitCellParameterisation(self.crystal)
        self.generate_reflections()

    def generate_reflections(self):
        sf = ScanFactory()
        self.scan = sf.make_scan(
            image_range=(1, 1),
            exposure_times=0.1,
            oscillation=(0, 3.0),
            epochs=list(range(1)),
            deg=True,
        )
        sequence_range = self.scan.get_oscillation_range(deg=False)
        experiments = ExperimentList()
        experiments.append(
            Experiment(
                beam=self.beam,
                detector=self.detector,
                goniometer=self.gonio,
                scan=self.scan,
                crystal=self.crystal,
                imageset=None,
            )
        )
        ray_predictor = ScansRayPredictor(experiments, sequence_range)
        index_generator = IndexGenerator(
            self.crystal.get_unit_cell(),
            space_group(space_group_symbols(1).hall()).type(),
            2.0,
        )
        indices = index_generator.to_array()
        rays = ray_predictor(indices)
        self.reflections = flex.reflection_table.empty_standard(len(rays))
        self.reflections.update(rays)
        self.reflections["xyzobs.mm.variance"] += (1e-3, 1e-3, 1e-6)

    def get_fd_gradients(self, pred_param, ref_predictor):
        p_vals = pred_param.get_param_vals()
        deltas = [1.0e-7] * len(p_vals)
        fd_grads = []
        p_names = pred_param.get_param_names()
        for i in range(len(deltas)):
            val = p_vals[i]
            p_vals[i] -= deltas[i] / 2.0
            pred_param.set_param_vals(p_vals)
            ref_predictor(self.reflections)
            x, y, _ = self.reflections["xyzcal.mm"].deep_copy().parts()
            delpsi = self.reflections["delpsical.rad"].deep_copy()
            rev_state = flex.vec3_double(x, y, delpsi)

            p_vals[i] += deltas[i]
            pred_param.set_param_vals(p_vals)
            ref_predictor(self.reflections)
            x, y, _ = self.reflections["xyzcal.mm"].deep_copy().parts()
            delpsi = self.reflections["delpsical.rad"].deep_copy()
            fwd_state = flex.vec3_double(x, y, delpsi)
            p_vals[i] = val

            fd = fwd_state - rev_state
            x_grads, y_grads, delpsi_grads = fd.parts()
            x_grads /= deltas[i]
            y_grads /= deltas[i]
            delpsi_grads /= deltas[i]
            fd_grads.append(
                {
                    "name": p_names[i],
                    "dX_dp": x_grads,
                    "dY_dp": y_grads,
                    "dDeltaPsi_dp": delpsi_grads,
                }
            )
        pred_param.set_param_vals(p_vals)
        return fd_grads


@pytest.fixture(scope="module")
def stills_tc():
    return _StillsTestHelper()


def test_stills_pred_param_gpu(stills_tc):
    pred_param = StillsPredictionParameterisationGPU(
        stills_tc.stills_experiments,
        detector_parameterisations=[stills_tc.det_param],
        beam_parameterisations=[stills_tc.s0_param],
        xl_orientation_parameterisations=[stills_tc.xlo_param],
        xl_unit_cell_parameterisations=[stills_tc.xluc_param],
    )
    _patch_crystal_uc_parameterisations(pred_param)
    ref_predictor = StillsExperimentsPredictorGPU(stills_tc.stills_experiments)
    ref_predictor(stills_tc.reflections)

    an_grads = pred_param.get_gradients(stills_tc.reflections)
    fd_grads = stills_tc.get_fd_gradients(pred_param, ref_predictor)

    for an_grad, fd_grad in zip(an_grads, fd_grads):
        for name in ["dX_dp", "dY_dp", "dDeltaPsi_dp"]:
            a = fd_grad[name]
            b = an_grad[name]
            abs_error = a - b
            fns = five_number_summary(abs_error)
            assert flex.max(flex.abs(abs_error)) < 0.0003
            iqr = fns[3] - fns[1]
            if iqr < 1.0e-10:
                continue
            sel = (abs_error < fns[3] + 1.5 * iqr) & (abs_error > fns[1] - 1.5 * iqr)
            tst = flex.max_index(flex.abs(abs_error.select(sel)))
            tst_val = abs_error.select(sel)[tst]
            assert abs(tst_val) < 0.00009


def test_spherical_relp_stills_pred_param_gpu(stills_tc):
    pred_param = SphericalRelpStillsPredictionParameterisationGPU(
        stills_tc.stills_experiments,
        detector_parameterisations=[stills_tc.det_param],
        beam_parameterisations=[stills_tc.s0_param],
        xl_orientation_parameterisations=[stills_tc.xlo_param],
        xl_unit_cell_parameterisations=[stills_tc.xluc_param],
    )
    _patch_crystal_uc_parameterisations(pred_param)
    ref_predictor = ExperimentsPredictorFactoryGPU.from_experiments(
        stills_tc.stills_experiments, force_stills=True, spherical_relp=True
    )
    ref_predictor(stills_tc.reflections)

    an_grads = pred_param.get_gradients(stills_tc.reflections)
    fd_grads = stills_tc.get_fd_gradients(pred_param, ref_predictor)

    for an_grad, fd_grad in zip(an_grads, fd_grads):
        for name in ["dX_dp", "dY_dp", "dDeltaPsi_dp"]:
            for a, b in zip(an_grad[name], fd_grad[name]):
                if name == "dDeltaPsi_dp":
                    assert a == pytest.approx(b, abs=5e-3)
                else:
                    assert a == pytest.approx(b, abs=5e-6)


# =============================================================================
# 7. Port of tests/algorithms/refinement/test_refinement_regression.py
# =============================================================================


def test_refinement_regression_gpu():
    override = """geometry.parameters
  {
    beam.wavelength.random=False
    beam.wavelength.value=1.0
    beam.direction.inclination.random=False
    crystal.a.length.random=False
    crystal.a.length.value=12.0
    crystal.a.direction.method=exactly
    crystal.a.direction.exactly.direction=1.0 0.002 -0.004
    crystal.b.length.random=False
    crystal.b.length.value=14.0
    crystal.b.direction.method=exactly
    crystal.b.direction.exactly.direction=-0.002 1.0 0.002
    crystal.c.length.random=False
    crystal.c.length.value=13.0
    crystal.c.direction.method=exactly
    crystal.c.direction.exactly.direction=0.002 -0.004 1.0
    detector.directions.method=exactly
    detector.directions.exactly.dir1=0.99 0.002 -0.004
    detector.directions.exactly.norm=0.002 -0.001 0.99
    detector.centre.method=exactly
    detector.centre.exactly.value=1.0 -0.5 199.0
  }"""

    master_phil = parse(f"{geometry_phil}\n{minimiser_phil}")
    models = setup_geometry.Extract(
        master_phil, local_overrides=override, verbose=False
    )

    mydetector = models.detector
    mygonio = models.goniometer
    mycrystal = models.crystal
    mybeam = models.beam

    det_param = DetectorParameterisationSinglePanel(mydetector)
    s0_param = BeamParameterisation(mybeam, mygonio)
    xlo_param = CrystalOrientationParameterisation(mycrystal)
    xluc_param = CrystalUnitCellParameterisation(mycrystal)
    s0_param.set_fixed([True, False, True])

    sf = ScanFactory()
    myscan = sf.make_scan(
        image_range=(1, 1800),
        exposure_times=0.1,
        oscillation=(0, 0.1),
        epochs=list(range(1800)),
        deg=True,
    )

    experiments = ExperimentList()
    experiments.append(
        Experiment(
            beam=mybeam,
            detector=mydetector,
            goniometer=mygonio,
            scan=myscan,
            crystal=mycrystal,
            imageset=None,
        )
    )

    pred_param = XYPhiPredictionParameterisationGPU(
        experiments, [det_param], [s0_param], [xlo_param], [xluc_param]
    )
    _patch_crystal_uc_parameterisations(pred_param)

    det_p_vals = det_param.get_param_vals()
    det_param.set_param_vals(
        [a + b for a, b in zip(det_p_vals, [1.0, 1.0, 1.0, 4.0, 4.0, 4.0])]
    )

    s0_p_vals = s0_param.get_param_vals()
    p_vals = list(s0_p_vals)
    p_vals[0] += 4.0
    s0_param.set_param_vals(p_vals)

    xlo_p_vals = xlo_param.get_param_vals()
    xlo_param.set_param_vals([a + b for a, b in zip(xlo_p_vals, [3.0, 3.0, 3.0])])

    xluc_p_vals = xluc_param.get_param_vals()
    cell_params = mycrystal.get_unit_cell().parameters()
    cell_params = [a + b for a, b in zip(cell_params, [0.1, -0.1, 0.1, 0.1, -0.1, 0.0])]
    new_uc = unit_cell(cell_params)
    newB = matrix.sqr(new_uc.fractionalization_matrix()).transpose()
    S = symmetrize_reduce_enlarge(mycrystal.get_space_group())
    S.set_orientation(orientation=newB)
    X = tuple([e * 1.0e5 for e in S.forward_independent_parameters()])
    xluc_param.set_param_vals(X)

    index_generator = IndexGenerator(
        mycrystal.get_unit_cell(),
        space_group(space_group_symbols(1).hall()).type(),
        2.0,
    )
    indices = index_generator.to_array()
    sequence_range = myscan.get_oscillation_range(deg=False)
    ray_predictor = ScansRayPredictor(experiments, sequence_range)
    obs_refs = ray_predictor(indices)
    intersects = ray_intersection(mydetector, obs_refs)
    obs_refs = obs_refs.select(intersects)

    ref_predictor = ScansExperimentsPredictorGPU(experiments)
    obs_refs["id"] = flex.int(len(obs_refs), 0)
    obs_refs = ref_predictor(obs_refs)
    obs_refs["xyzobs.mm.value"] = obs_refs["xyzcal.mm"]

    im_width = 0.1 * pi / 180.0
    px_size = mydetector[0].get_pixel_size()
    var_x = flex.double(len(obs_refs), (px_size[0] / 2.0) ** 2)
    var_y = flex.double(len(obs_refs), (px_size[1] / 2.0) ** 2)
    var_phi = flex.double(len(obs_refs), (im_width / 2.0) ** 2)
    obs_refs["xyzobs.mm.variance"] = flex.vec3_double(var_x, var_y, var_phi)
    assert len(obs_refs) == 1128

    s0_param.set_param_vals(s0_p_vals)
    det_param.set_param_vals(det_p_vals)
    xlo_param.set_param_vals(xlo_p_vals)
    xluc_param.set_param_vals(xluc_p_vals)

    refman = ReflectionManager(
        obs_refs, experiments, outlier_detector=None, close_to_spindle_cutoff=0.1
    )
    mytarget = LeastSquaresPositionalResidualWithRmsdCutoffGPU(
        experiments, ref_predictor, refman, pred_param, restraints_parameterisation=None
    )

    refiner = GaussNewtonIterationsGPU(
        target=mytarget, prediction_parameterisation=pred_param, log=None
    )
    refiner.run()
    assert mytarget.achieved()
    assert refiner.get_num_steps() == 1
    assert approx_equal(
        mytarget.rmsds(), (0.00508252354876, 0.00420954552156, 8.97303428289e-05)
    )

    s0_param.set_param_vals(s0_p_vals)
    det_param.set_param_vals(det_p_vals)
    xlo_param.set_param_vals(xlo_p_vals)
    xluc_param.set_param_vals(xluc_p_vals)

    refiner2 = LBFGScurvsGPU(
        target=mytarget, prediction_parameterisation=pred_param, log=None
    )
    refiner2.run()
    assert mytarget.achieved()
    assert refiner2.get_num_steps() == 9
    assert approx_equal(
        mytarget.rmsds(), (0.0558857700305, 0.0333446685335, 0.000347402754278)
    )


# =============================================================================
# 8. Port of tests/algorithms/refinement/test_multi_experiment_refinement.py
# =============================================================================


def test_multi_experiment_refinement_gpu():
    master_phil = parse(f"{geometry_phil}\n{minimiser_phil}")
    models1 = setup_geometry.Extract(
        master_phil, local_overrides="geometry.parameters.random_seed = 1"
    )
    crystal1 = models1.crystal

    models2 = setup_geometry.Extract(
        master_phil, local_overrides="geometry.parameters.random_seed = 2"
    )
    mydetector = models2.detector
    mygonio = models2.goniometer
    crystal2 = models2.crystal
    mybeam = models2.beam

    sf = ScanFactory()
    myscan = sf.make_scan(
        image_range=(1, 180),
        exposure_times=0.1,
        oscillation=(0, 0.1),
        epochs=list(range(180)),
        deg=True,
    )
    sequence_range = myscan.get_oscillation_range(deg=False)

    experiments = ExperimentList()
    experiments.append(
        Experiment(
            beam=mybeam,
            detector=mydetector,
            goniometer=mygonio,
            scan=myscan,
            crystal=crystal1,
            imageset=None,
        )
    )
    experiments.append(
        Experiment(
            beam=mybeam,
            detector=mydetector,
            goniometer=mygonio,
            scan=myscan,
            crystal=crystal2,
            imageset=None,
        )
    )

    det_param = DetectorParameterisationSinglePanel(mydetector)
    s0_param = BeamParameterisation(mybeam, mygonio)
    xl1o_param = CrystalOrientationParameterisation(crystal1)
    xl1uc_param = CrystalUnitCellParameterisation(crystal1)
    xl2o_param = CrystalOrientationParameterisation(crystal2)
    xl2uc_param = CrystalUnitCellParameterisation(crystal2)
    s0_param.set_fixed([True, False, True])

    det_p_vals = det_param.get_param_vals()
    det_param.set_param_vals(
        [a + b for a, b in zip(det_p_vals, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])]
    )
    s0_p_vals = s0_param.get_param_vals()
    p_vals = list(s0_p_vals)
    p_vals[0] += 2.0
    s0_param.set_param_vals(p_vals)

    xlo_p_vals = []
    for xlo in (xl1o_param, xl2o_param):
        p_vals = xlo.get_param_vals()
        xlo_p_vals.append(p_vals)
        xlo.set_param_vals([a + b for a, b in zip(p_vals, [2.0, 2.0, 2.0])])

    xluc_p_vals = []
    for xluc, xl in ((xl1uc_param, crystal1), (xl2uc_param, crystal2)):
        p_vals = xluc.get_param_vals()
        xluc_p_vals.append(p_vals)
        cell_params = xl.get_unit_cell().parameters()
        cell_params = [
            a + b for a, b in zip(cell_params, [0.1, 0.1, 0.1, 0.0, 0.0, 0.1])
        ]
        new_uc = unit_cell(cell_params)
        newB = matrix.sqr(new_uc.fractionalization_matrix()).transpose()
        S = symmetrize_reduce_enlarge(xl.get_space_group())
        S.set_orientation(orientation=newB)
        X = tuple([e * 1.0e5 for e in S.forward_independent_parameters()])
        xluc.set_param_vals(X)

    indices1 = IndexGenerator(
        crystal1.get_unit_cell(),
        space_group(space_group_symbols(1).hall()).type(),
        2.5,
    ).to_array()
    indices2 = IndexGenerator(
        crystal2.get_unit_cell(),
        space_group(space_group_symbols(1).hall()).type(),
        2.5,
    ).to_array()

    ray_predictor = ScansRayPredictor(experiments, sequence_range)
    obs_refs1 = ray_predictor(indices1, experiment_id=0)
    obs_refs1["id"] = flex.int(len(obs_refs1), 0)
    obs_refs2 = ray_predictor(indices2, experiment_id=1)
    obs_refs2["id"] = flex.int(len(obs_refs2), 1)

    obs_refs1 = obs_refs1.select(ray_intersection(mydetector, obs_refs1))
    obs_refs2 = obs_refs2.select(ray_intersection(mydetector, obs_refs2))

    ref_predictor = ScansExperimentsPredictorGPU(experiments)
    obs_refs1 = ref_predictor(obs_refs1)
    obs_refs2 = ref_predictor(obs_refs2)
    obs_refs1["xyzobs.mm.value"] = obs_refs1["xyzcal.mm"]
    obs_refs2["xyzobs.mm.value"] = obs_refs2["xyzcal.mm"]

    im_width = 0.1 * pi / 18.0
    px_size = mydetector[0].get_pixel_size()
    for refs in (obs_refs1, obs_refs2):
        var_x = flex.double(len(refs), (px_size[0] / 2.0) ** 2)
        var_y = flex.double(len(refs), (px_size[1] / 2.0) ** 2)
        var_phi = flex.double(len(refs), (im_width / 2.0) ** 2)
        refs["xyzobs.mm.variance"] = flex.vec3_double(var_x, var_y, var_phi)

    obs_refs1.extend(obs_refs2)
    obs_refs = obs_refs1

    s0_param.set_param_vals(s0_p_vals)
    det_param.set_param_vals(det_p_vals)
    xl1o_param.set_param_vals(xlo_p_vals[0])
    xl2o_param.set_param_vals(xlo_p_vals[1])
    xl1uc_param.set_param_vals(xluc_p_vals[0])
    xl2uc_param.set_param_vals(xluc_p_vals[1])

    params = refiner_phil_scope.fetch(source=parse("")).extract()
    refiner = RefinerFactoryGPU.from_parameters_data_experiments(
        params, obs_refs, experiments
    )
    refiner.run()

    params.refinement.parameterisation.scan_varying = True
    refiner = RefinerFactoryGPU.from_parameters_data_experiments(
        params, obs_refs, experiments
    )
    refiner.run()

    refined_experiments = refiner.get_experiments()
    sp = [xl.get_num_scan_points() for xl in refined_experiments.crystals()]
    assert sp.count(181) == 2


# =============================================================================
# 9. Port of tests/algorithms/refinement/test_parameter_auto_reduction.py
# =============================================================================


def test_parameter_auto_reduction_gpu():
    test = _StillsTestHelper()
    ref_predictor = StillsExperimentsPredictorGPU(test.stills_experiments)
    ref_predictor(test.reflections)
    refman = ReflectionManagerFactoryGPU.from_parameters_reflections_experiments(
        refman_phil_scope.extract(),
        test.reflections,
        test.stills_experiments,
        do_stills=True,
    )
    refman.finalise()
    pred_param = StillsPredictionParameterisationGPU(
        test.stills_experiments,
        detector_parameterisations=[test.det_param],
        beam_parameterisations=[test.s0_param],
        xl_orientation_parameterisations=[test.xlo_param],
        xl_unit_cell_parameterisations=[test.xluc_param],
    )
    _patch_crystal_uc_parameterisations(pred_param)

    assert len(refman.get_matches()) == 823

    options = ar_phil_scope.extract()
    options.min_nref_per_parameter = 792
    ar = AutoReduce(options, pred_param=pred_param, reflection_manager=refman)
    ar.check_and_fail()

    options.min_nref_per_parameter = 793
    ar = AutoReduce(options, pred_param=pred_param, reflection_manager=refman)
    with pytest.raises(DialsRefineConfigError):
        ar.check_and_fail()

    options.min_nref_per_parameter = 793
    ar = AutoReduce(options, pred_param=pred_param, reflection_manager=refman)
    ar.check_and_fix()
    xl_uc_params = pred_param.get_crystal_unit_cell_parameterisations()
    assert xl_uc_params[0].num_free() == 3
    assert xl_uc_params[0].get_param_names() == ["g_param_1", "g_param_2", "g_param_5"]


# =============================================================================
# 10. Port of tests/algorithms/refinement/test_stills_refinement.py
# =============================================================================


def test_stills_refinement_gpu():
    master_phil = parse(f"{geometry_phil}\n{minimiser_phil}")
    param = """
  geometry.parameters.crystal.a.length.range=40 50;
  geometry.parameters.crystal.b.length.range=40 50;
  geometry.parameters.crystal.c.length.range=40 50;
  geometry.parameters.random_seed = 42"""
    models = setup_geometry.Extract(master_phil, local_overrides=param)

    crystal = models.crystal
    mydetector = models.detector
    mygonio = models.goniometer
    mybeam = models.beam

    sf = ScanFactory()
    myscan = sf.make_scan(
        image_range=(1, 1),
        exposure_times=0.1,
        oscillation=(0, 1.5),
        epochs=list(range(1)),
        deg=True,
    )
    sequence_range = myscan.get_oscillation_range(deg=False)

    stills_experiments = ExperimentList()
    stills_experiments.append(
        Experiment(beam=mybeam, detector=mydetector, crystal=crystal, imageset=None)
    )
    scans_experiments = ExperimentList()
    scans_experiments.append(
        Experiment(
            beam=mybeam,
            detector=mydetector,
            crystal=crystal,
            goniometer=mygonio,
            scan=myscan,
            imageset=None,
        )
    )

    xlo_param = CrystalOrientationParameterisation(crystal)
    xluc_param = CrystalUnitCellParameterisation(crystal)

    xlo_p_vals = [xlo_param.get_param_vals()]
    xlo_param.set_param_vals([a + b for a, b in zip(xlo_p_vals[0], [5.0, 5.0, 5.0])])

    xluc_p_vals = [xluc_param.get_param_vals()]
    cell_params = crystal.get_unit_cell().parameters()
    cell_params = [a + b for a, b in zip(cell_params, [1.0, 1.0, -1.0, 0.0, 0.0, 0.5])]
    new_uc = unit_cell(cell_params)
    newB = matrix.sqr(new_uc.fractionalization_matrix()).transpose()
    S = symmetrize_reduce_enlarge(crystal.get_space_group())
    S.set_orientation(orientation=newB)
    X = tuple([e * 1.0e5 for e in S.forward_independent_parameters()])
    xluc_param.set_param_vals(X)

    target_crystal = deepcopy(crystal)

    indices = IndexGenerator(
        crystal.get_unit_cell(),
        space_group(space_group_symbols(1).hall()).type(),
        2.0,
    ).to_array()

    ref_predictor = ScansRayPredictor(scans_experiments, sequence_range)
    obs_refs = ref_predictor(indices, experiment_id=0)
    obs_refs = obs_refs.select(ray_intersection(mydetector, obs_refs))
    tmp = flex.reflection_table.empty_standard(len(obs_refs))
    tmp.update(obs_refs)
    obs_refs = tmp

    im_width = 0.1 * pi / 180.0
    px_size = mydetector[0].get_pixel_size()
    var_x = flex.double(len(obs_refs), (px_size[0] / 2.0) ** 2)
    var_y = flex.double(len(obs_refs), (px_size[1] / 2.0) ** 2)
    var_phi = flex.double(len(obs_refs), (im_width / 2.0) ** 2)
    obs_refs["xyzobs.mm.variance"] = flex.vec3_double(var_x, var_y, var_phi)

    stills_ref_predictor = StillsExperimentsPredictorGPU(stills_experiments)
    obs_refs_stills = stills_ref_predictor(obs_refs)
    obs_refs_stills["xyzobs.mm.value"] = obs_refs_stills["xyzcal.mm"]

    xlo_param.set_param_vals(xlo_p_vals[0])
    xluc_param.set_param_vals(xluc_p_vals[0])

    params = refiner_phil_scope.fetch(source=parse("")).extract()
    params.refinement.target.bin_size_fraction = 0.01
    params.refinement.parameterisation.beam.fix = "all"
    params.refinement.parameterisation.detector.fix = "all"
    refiner = RefinerFactoryGPU.from_parameters_data_experiments(
        params, obs_refs_stills, stills_experiments
    )
    history = refiner.run()
    assert len(history["rmsd"]) == 9

    refined_crystal = refiner.get_experiments()[0].crystal
    assert refined_crystal.get_unit_cell().is_similar_to(target_crystal.get_unit_cell())


# =============================================================================
# 11. Port of tests/algorithms/refinement/test_hierarchical_detector_refinement.py
# =============================================================================


def test_hierarchical_detector_refinement_gpu(dials_data):
    data_dir = dials_data("refinement_test_data")
    experiments_path = data_dir / "cspad-single-image.expt"
    experiments = ExperimentListFactory.from_serialized_format(
        experiments_path, check_format=False
    )
    im_set = experiments.imagesets()[0]
    detector = copy.deepcopy(im_set.get_detector())
    beam = im_set.get_beam()

    crystal = Crystal(
        (40.0, 0.0, 0.0), (0.0, 40.0, 0.0), (0.0, 0.0, 40.0), space_group_symbol="P1"
    )
    from dxtbx.model import GoniometerFactory

    goniometer = GoniometerFactory.known_axis((1.0, 0.0, 0.0))
    sf = ScanFactory()
    scan = sf.make_scan(
        image_range=(1, 1800),
        exposure_times=0.1,
        oscillation=(0, 0.1),
        epochs=list(range(1800)),
        deg=True,
    )
    sequence_range = scan.get_oscillation_range(deg=False)

    experiments = ExperimentList()
    experiments.append(
        Experiment(
            beam=beam,
            detector=detector,
            goniometer=goniometer,
            scan=scan,
            crystal=crystal,
            imageset=None,
        )
    )

    indices = IndexGenerator(
        crystal.get_unit_cell(),
        space_group(space_group_symbols(1).hall()).type(),
        2.0,
    ).to_array()
    ray_predictor = ScansRayPredictor(experiments, sequence_range)
    obs_refs = ray_predictor(indices)
    obs_refs = obs_refs.select(ray_intersection(detector, obs_refs))
    ref_predictor = ScansExperimentsPredictorGPU(experiments)
    obs_refs["id"] = flex.int(len(obs_refs), 0)
    obs_refs = ref_predictor(obs_refs)
    obs_refs["xyzobs.mm.value"] = obs_refs["xyzcal.mm"]

    im_width = 0.1 * math.pi / 180.0
    px_size = detector[0].get_pixel_size()
    var_x = flex.double(len(obs_refs), (px_size[0] / 2.0) ** 2)
    var_y = flex.double(len(obs_refs), (px_size[1] / 2.0) ** 2)
    var_phi = flex.double(len(obs_refs), (im_width / 2.0) ** 2)
    obs_refs["xyzobs.mm.variance"] = flex.vec3_double(var_x, var_y, var_phi)

    det_param = DetectorParameterisationHierarchical(detector, level=1)
    p_vals = list(det_param.get_param_vals())
    p_vals[1] += 2
    p_vals[2] -= 2
    p_vals[7] += 2
    p_vals[8] += 2
    p_vals[13] -= 2
    p_vals[14] += 2
    p_vals[19] -= 2
    p_vals[20] -= 2
    det_param.set_param_vals(p_vals)
    det_param = DetectorParameterisationHierarchical(detector, level=1)

    beam_param = BeamParameterisation(beam, goniometer)
    xlo_param = CrystalOrientationParameterisation(crystal)
    xluc_param = CrystalUnitCellParameterisation(crystal)
    beam_param.set_fixed([True] * 3)
    xluc_param.set_fixed([True] * 6)
    xlo_param.set_fixed([True] * 3)

    pred_param = XYPhiPredictionParameterisationGPU(
        experiments, [det_param], [beam_param], [xlo_param], [xluc_param]
    )
    param_reporter = ParameterReporter(
        [det_param], [beam_param], [xlo_param], [xluc_param]
    )
    refman = ReflectionManager(obs_refs, experiments, nref_per_degree=20)
    target = LeastSquaresPositionalResidualWithRmsdCutoffGPU(
        experiments,
        ref_predictor,
        refman,
        pred_param,
        restraints_parameterisation=None,
        frac_binsize_cutoff=0.0001,
    )
    refinery = LevenbergMarquardtIterationsGPU(
        target=target,
        prediction_parameterisation=pred_param,
        log=None,
        max_iterations=20,
    )
    refiner = RefinerGPU(
        experiments=experiments,
        pred_param=pred_param,
        param_reporter=param_reporter,
        refman=refman,
        target=target,
        refinery=refinery,
    )
    history = refiner.run()
    assert history.reason_for_termination == "RMSD target achieved"

    orig_det = im_set.get_detector()
    refined_det = refiner.get_experiments()[0].detector
    for op, rp in zip(orig_det, refined_det):
        o1 = matrix.col(op.get_origin())
        o2 = matrix.col(rp.get_origin())
        assert math.fabs(o1.length() - o2.length()) / o1.length() == pytest.approx(
            0, abs=1e-5
        )
        assert o1.accute_angle(o2) == pytest.approx(0, abs=1e-5)


# =============================================================================
# 12. Port of tests/algorithms/refinement/test_dials-423.py
# =============================================================================


def test_dials_423_gpu(dials_data):
    data_dir = dials_data("refinement_test_data")
    exp_file = data_dir / "dials-423.json"
    ref_file = data_dir / "dials-423.pickle"

    reflections = flex.reflection_table.from_file(ref_file)
    experiments = ExperimentListFactory.from_json_file(exp_file, check_format=False)
    params = refiner_phil_scope.fetch(source=phil.parse("")).extract()
    params.refinement.reflections.outlier.algorithm = "null"

    refiner = RefinerFactoryGPU.from_parameters_data_experiments(
        params, reflections, experiments
    )
    d1 = experiments[0].detector
    d2 = refiner.get_experiments()[0].detector
    assert d1.is_similar_to(d2)


# =============================================================================
# 13. Port of tests/algorithms/refinement/test_cspad_refinement.py & test_xfel_metrology.py
# =============================================================================


def test_cspad_refinement_gpu(dials_data, tmp_path):
    data_dir = dials_data("iterative_cspad_refinement")
    result = subprocess.run(
        [
            DIALS_GPU_REFINE,
            data_dir / "cspad_refined_experiments_step6_level2_300.json",
            data_dir / "cspad_reflections_step7_300.pickle",
            data_dir / "refine.phil",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    reg_exp = ExperimentListFactory.from_json_file(
        data_dir / "regression_experiments.json", check_format=False
    )
    ref_exp = ExperimentListFactory.from_json_file(
        tmp_path / "refined.expt", check_format=False
    )

    tol = 1e-5
    for b1, b2 in zip(reg_exp.beams(), ref_exp.beams()):
        assert b1.is_similar_to(
            b2,
            wavelength_tolerance=tol,
            direction_tolerance=tol,
            polarization_normal_tolerance=tol,
            polarization_fraction_tolerance=tol,
        )
        s0_1 = matrix.col(b1.get_unit_s0())
        s0_2 = matrix.col(b2.get_unit_s0())
        assert s0_1.accute_angle(s0_2, deg=True) < 0.0057
    for c1, c2 in zip(reg_exp.crystals(), ref_exp.crystals()):
        assert c1.is_similar_to(c2)
    for d1, d2 in zip(reg_exp.detectors(), ref_exp.detectors()):
        assert d1.is_similar_to(
            d2,
            fast_axis_tolerance=1e-4,
            slow_axis_tolerance=1e-4,
            origin_tolerance=1e-2,
        )


def test_xfel_constrained_refinement_gpu(dials_data):
    bevington = pytest.importorskip("scitbx.examples.bevington")
    if not hasattr(bevington, "non_linear_ls_eigen_wrapper"):
        pytest.skip("Skipping test as SparseLevMar engine not available")

    data_dir = dials_data("cspad_metrology")
    refl = flex.reflection_table.from_file(data_dir / "benchmark_level2d.pickle")
    expt = ExperimentListFactory.from_json_file(data_dir / "benchmark_level2d.json")

    refine_phil = phil.parse((data_dir / "refine.phil").read_text())
    constraint_phil = phil.parse(
        """
refinement {
  parameterisation {
    detector {
      fix_list=Tau2,Tau3
      constraints {
        parameter=Dist
      }
    }
  }
}
"""
    )
    params = refiner_phil_scope.fetch(sources=[refine_phil, constraint_phil]).extract()
    detector = expt.detectors()[0]
    initial_distances = [p.get_distance() for p in detector]

    refiner = RefinerFactoryGPU.from_parameters_data_experiments(params, refl, expt)
    refiner.run()
    detector = refiner.get_experiments().detectors()[0]
    final_distances = [p.get_distance() for p in detector]
    dist_diff = [a - b for a, b in zip(final_distances, initial_distances)]
    for d in dist_diff[1:]:
        assert d == pytest.approx(dist_diff[0])


# =============================================================================
# 14. Port of tests/algorithms/refinement/test_refine_multi_stills.py & test_sv_multi_panel_refinement.py
# =============================================================================


def test_multi_stills_gpu(dials_data, tmp_path):
    data_dir = dials_data("refinement_test_data")
    result = subprocess.run(
        [
            DIALS_GPU_REFINE,
            data_dir / "multi_stills_combined.json",
            data_dir / "multi_stills_combined.pickle",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    reg_exp = ExperimentListFactory.from_json_file(
        data_dir / "multi_stills_regression.json", check_format=False
    )
    ref_exp = ExperimentListFactory.from_json_file(
        tmp_path / "refined.expt", check_format=False
    )
    for c1, c2 in zip(reg_exp.crystals(), ref_exp.crystals()):
        assert c1.is_similar_to(c2)


def test_restrained_refinement_with_fixed_parameterisations_gpu(dials_data):
    user_phil = phil.parse(
        """
refinement {
  parameterisation {
    auto_reduction {
      min_nref_per_parameter = 90
      action = fail *fix remove
    }
    crystal {
      unit_cell {
        restraints {
          tie_to_target {
            values = 95 95 132 90 90 120
            sigmas = 1 1 1 0 0 0
            id = 0 1 2 3 4 5 6 7 8 9
          }
        }
      }
    }
  }
}
"""
    )
    working_params = refiner_phil_scope.fetch(source=user_phil).extract()
    data_dir = dials_data("refinement_test_data")
    experiments = ExperimentListFactory.from_json_file(
        data_dir / "multi_stills_combined.json", check_format=False
    )
    reflections = flex.reflection_table.from_file(
        data_dir / "multi_stills_combined.pickle"
    )
    refiner = RefinerFactoryGPU.from_parameters_data_experiments(
        working_params, reflections, experiments
    )
    history = refiner.run()
    rmsd_limits = (0.2044, 0.2220, 0.0063)
    for a, b in zip(history["rmsd"][-1], rmsd_limits):
        assert a < b


def test_scan_varying_refinement_of_a_multiple_panel_detector_gpu(dials_data, tmp_path):
    data_dir = dials_data("refinement_test_data")
    result = subprocess.run(
        [
            DIALS_GPU_REFINE,
            data_dir / "I23_24_panel.json",
            data_dir / "I23_24_panel.pickle",
            "scan_varying=true",
            "history=history.json",
            "outlier.separate_blocks=False",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    history = Journal.from_json_file(tmp_path / "history.json")
    final_rmsd = history["rmsd"][-1]
    assert final_rmsd[0] < 0.05
    assert final_rmsd[1] < 0.04
    assert final_rmsd[2] < 0.0002

    rt = flex.reflection_table.from_file(tmp_path / "refined.refl")
    uir = rt.get_flags(rt.flags.used_in_refinement)
    assert uir.count(True) == history["num_reflections"][-1]


# =============================================================================
# 15. Port of tests/command_line/test_refine.py
# =============================================================================


def test_i04_weak_data_gpu(dials_data, tmp_path):
    data_dir = dials_data("refinement_test_data")
    experiments_path = data_dir / "i04-weak.json"
    pickle_path = data_dir / "i04-weak.pickle"

    cmd = (
        DIALS_GPU_REFINE,
        "close_to_spindle_cutoff=0.05",
        "reflections_per_degree=100",
        "outlier.separate_blocks=False",
        "scan_varying=False",
        "reflections.outlier.nproc=1",
        experiments_path,
        pickle_path,
    )
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True)
    assert not result.returncode and not result.stderr

    reg_exp = ExperimentListFactory.from_json_file(
        data_dir / "i04-weak-regression.json", check_format=False
    )[0]
    ref_exp = ExperimentListFactory.from_json_file(
        tmp_path / "refined.expt", check_format=False
    )[0]

    assert reg_exp.crystal == ref_exp.crystal
    assert reg_exp.detector == ref_exp.detector
    assert reg_exp.beam == ref_exp.beam

    assert ref_exp.crystal.get_cell_parameter_sd() == pytest.approx(
        (0.0009903, 0.0009903, 0.0021227, 0.0, 0.0, 0.0), abs=1e-6
    )
    assert ref_exp.crystal.get_cell_volume_sd() == pytest.approx(23.8051262, abs=1e-6)


def test_scan_varying_refinement_rmsds_gpu(dials_data, tmp_path):
    data_dir = dials_data("refinement_test_data")
    experiments_path = data_dir / "from-xds.json"
    pickle_path = data_dir / "from-xds-all.pickle"

    result = subprocess.run(
        (
            DIALS_GPU_REFINE,
            experiments_path,
            pickle_path,
            "scan_varying=False",
            "reflections_per_degree=50",
            "outlier.algorithm=null",
            "close_to_spindle_cutoff=0.05",
        ),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    result = subprocess.run(
        (
            DIALS_GPU_REFINE,
            "refined.expt",
            pickle_path,
            "scan_varying=true",
            "output.history=history.json",
            "correlation_plot.filename=corrgram.pdf",
            "reflections_per_degree=50",
            "outlier.algorithm=null",
            "close_to_spindle_cutoff=0.05",
            "crystal.orientation.smoother.interval_width_degrees=36.0",
            "crystal.unit_cell.smoother.interval_width_degrees=36.0",
            "set_scan_varying_errors=True",
            "reflections.outlier.nproc=1",
        ),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    history = Journal.from_json_file(tmp_path / "history.json")
    expected_rmsds = [
        (0.088488398, 0.114583571, 0.001460382),
        (0.080489334, 0.086406517, 0.001284069),
        (0.078835086, 0.086052630, 0.001195882),
        (0.077476911, 0.086194611, 0.001161143),
        (0.076755840, 0.086090630, 0.001157239),
        (0.076586376, 0.085939462, 0.001155641),
        (0.076603722, 0.085878953, 0.001155065),
        (0.076611382, 0.085862959, 0.001154863),
        (0.076608732, 0.085856935, 0.001154384),
        (0.076605731, 0.085852271, 0.001153858),
        (0.076604576, 0.085852318, 0.001153643),
        (0.076603981, 0.085854175, 0.001153594),
    ]
    for a, b in zip(history["rmsd"], expected_rmsds):
        assert a == pytest.approx(b, abs=1e-6)

    rt = flex.reflection_table.from_file(tmp_path / "refined.refl")
    uir = rt.get_flags(rt.flags.used_in_refinement)
    assert uir.count(True) == history["num_reflections"][-1]


def test_scan_varying_with_automated_outlier_rejection_block_width_interval_width_gpu(
    dials_data, tmp_path
):
    data_dir = dials_data("refinement_test_data")
    experiments_path = data_dir / "from-xds.json"
    pickle_path = data_dir / "from-xds-all.pickle"

    result = subprocess.run(
        (
            DIALS_GPU_REFINE,
            experiments_path,
            pickle_path,
            "scan_varying=true",
            "max_iterations=5",
            "output.history=history.json",
            "crystal.orientation.smoother.interval_width_degrees=auto",
            "crystal.unit_cell.smoother.interval_width_degrees=auto",
            "reflections.outlier.nproc=1",
        ),
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    history = Journal.from_json_file(tmp_path / "history.json")
    expected_rmsds = [
        [0.6198424, 0.3534224, 0.0069824],
        [0.1729424, 0.1140850, 0.0047478],
        [0.0984923, 0.0844989, 0.0026512],
        [0.0696712, 0.0733906, 0.0014370],
        [0.0647380, 0.0721140, 0.0011680],
        [0.0633892, 0.0719455, 0.0010771],
    ]
    for a, b in zip(history["rmsd"], expected_rmsds):
        assert a == pytest.approx(b, abs=1e-6)

    ref_exp = ExperimentListFactory.from_json_file(
        tmp_path / "refined.expt", check_format=False
    )[0]
    unit_cell_params = ref_exp.crystal.get_unit_cell().parameters()
    assert unit_cell_params == pytest.approx(
        [42.27482, 42.27482, 39.66893, 90.00000, 90.00000, 90.00000], abs=1e-3
    )

    refined_refl = flex.reflection_table.from_file(tmp_path / "refined.refl")
    predicted = flex.reflection_table.from_predictions_multi([ref_exp])
    matched, reference, unmatched = predicted.match_with_reference(refined_refl)
    assert reference.size() > (0.997 * refined_refl.size())

    ann = AnnAdaptor(data=predicted["xyzcal.px"].as_double(), dim=3, k=1)
    ann.query(refined_refl["xyzcal.px"].as_double())
    assert (ann.distances < 0.5).count(True) > (0.998 * refined_refl.size())


@pytest.mark.parametrize("fix", ["cell", "orientation"])
def test_scan_varying_with_fixed_crystal_gpu(fix, dials_data, tmp_path):
    location = dials_data("multi_crystal_proteinase_k")
    refls = location / "reflections_1.pickle"
    expts = location / "experiments_1.json"

    result = subprocess.run(
        [DIALS_GPU_REFINE, expts, refls, f"crystal.fix={fix}"],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr


def test_scan_varying_missing_segments_multi_crystal_gpu(dials_data, tmp_path):
    location = dials_data("i19_1_pdteet_index")
    refls = location / "indexed.refl"
    expts = location / "indexed.expt"

    data = flex.reflection_table.from_file(refls)
    data = data.select(data.get_flags(data.flags.indexed))
    z = data["xyzobs.px.value"].parts()[2]
    sel = ((data["id"] == 0) & (z > 200)) | ((data["id"] == 1) & (z < 650))
    trimmed = data.select(sel)

    trimmed_filename = tmp_path / "indexed_trim.refl"
    trimmed.as_file(trimmed_filename)

    result = subprocess.run(
        [DIALS_GPU_REFINE, expts, trimmed_filename],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    el = ExperimentListFactory.from_json_file(
        tmp_path / "refined.expt", check_format=False
    )
    image_ranges = [e.scan.get_image_range() for e in el]
    assert image_ranges[0][1] == 850
    assert image_ranges[1][0] == 1


@pytest.mark.parametrize("gcb", ["None", "3000"])
def test_scan_varying_multi_scan_one_crystal_gpu(gcb, dials_data, tmp_path):
    location = dials_data("l_cysteine_dials_output")
    refls = location / "indexed.refl"
    expts = location / "indexed.expt"

    result = subprocess.run(
        [
            DIALS_GPU_REFINE,
            expts,
            refls,
            "output.history=history.json",
            "outlier.algorithm=tukey",
            "max_iterations=3",
            "unit_cell.smoother.interval_width_degrees=56",
            "orientation.smoother.interval_width_degrees=56",
            "gradient_calculation_blocksize=" + gcb,
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    el = ExperimentListFactory.from_json_file(
        tmp_path / "refined.expt", check_format=False
    )
    assert len(el.crystals()) == 4

    history = Journal.from_json_file(tmp_path / "history.json")
    expected_rmsds = [
        (0.1401658782847504, 0.2225931584837884, 0.002349912655443814),
        (0.12060230585178289, 0.1585977879739876, 0.002114318828411418),
        (0.10970832317567975, 0.1348574975434352, 0.001955034565537597),
        (0.10373159352273859, 0.12827852889951505, 0.0017901404193256304),
    ]
    for a, b in zip(history["rmsd"], expected_rmsds):
        assert a == pytest.approx(b, abs=1e-6)


def test_find_disjoint_sets_gpu():
    beam = Beam()
    shared_beam = []
    for _ in range(random.randint(1, 5)):
        crystal = Crystal((10, 0, 0, 0, 10, 0, 0, 0, 10), "P1")
        goniometer = Goniometer()
        detector = Detector()
        shared_beam.append(
            Experiment(
                beam=beam, goniometer=goniometer, crystal=crystal, detector=detector
            )
        )

    crystal = Crystal((10, 0, 0, 0, 10, 0, 0, 0, 10), "P1")
    shared_crystal = []
    for _ in range(random.randint(1, 5)):
        beam = Beam()
        goniometer = Goniometer()
        detector = Detector()
        shared_crystal.append(
            Experiment(
                beam=beam, goniometer=goniometer, crystal=crystal, detector=detector
            )
        )

    detector = Detector()
    shared_detector = []
    for _ in range(random.randint(1, 5)):
        beam = Beam()
        crystal = Crystal((10, 0, 0, 0, 10, 0, 0, 0, 10), "P1")
        goniometer = Goniometer()
        shared_detector.append(
            Experiment(
                beam=beam, goniometer=goniometer, crystal=crystal, detector=detector
            )
        )

    goniometer = Goniometer()
    shared_goniometer = []
    for _ in range(random.randint(1, 5)):
        beam = Beam()
        crystal = Crystal((10, 0, 0, 0, 10, 0, 0, 0, 10), "P1")
        detector = Detector()
        shared_goniometer.append(
            Experiment(
                beam=beam, goniometer=goniometer, crystal=crystal, detector=detector
            )
        )

    experiments = shared_beam + shared_crystal + shared_detector + shared_goniometer
    random.shuffle(experiments)
    experiments = ExperimentList(experiments)

    disjoint_sets = _find_disjoint_sets(experiments)
    assert len(disjoint_sets) == 4
    check = {
        len(shared_beam),
        len(shared_crystal),
        len(shared_detector),
        len(shared_goniometer),
    }
    for group in disjoint_sets:
        assert len(group) in check


def test_refinement_of_disjoint_sets_gpu(dials_data, tmp_path):
    location = dials_data("polyhedra_narrow_wedges")
    expts = [
        location / f"sweep_00{i}_experiments.json"
        for i in ["2", "3", "4", "5", "6", "7", "9"]
    ]
    refls = [
        location / f"sweep_00{i}_reflections.pickle"
        for i in ["2", "3", "4", "5", "6", "7", "9"]
    ]
    result = subprocess.run(
        [shutil.which("dials.combine_experiments")] + expts + refls,
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    result = subprocess.run(
        [
            DIALS_GPU_REFINE,
            "combined.expt",
            "combined.refl",
            "scan_varying=False",
            "output.reflections=separate.refl",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    separate_refl = flex.reflection_table.from_file(tmp_path / "separate.refl")
    separate_refl = separate_refl.select(
        separate_refl.get_flags(separate_refl.flags.used_in_refinement)
    )
    x_obs, y_obs, z_obs = separate_refl["xyzobs.mm.value"].parts()
    x_calc, y_calc, z_calc = separate_refl["xyzcal.mm"].parts()
    x_resid2 = (x_calc - x_obs) ** 2
    y_resid2 = (y_calc - y_obs) ** 2
    z_resid2 = (z_calc - z_obs) ** 2
    separate_rmsd_x, separate_rmsd_y, separate_rmsd_z = [], [], []
    for i in range(22):
        sel = separate_refl["id"] == i
        n = sel.count(True)
        separate_rmsd_x.append(math.sqrt(flex.sum(x_resid2.select(sel)) / n))
        separate_rmsd_y.append(math.sqrt(flex.sum(y_resid2.select(sel)) / n))
        separate_rmsd_z.append(math.sqrt(flex.sum(z_resid2.select(sel)) / n))

    result = subprocess.run(
        [
            DIALS_GPU_REFINE,
            "combined.expt",
            "combined.refl",
            "scan_varying=False",
            "separate_independent_sets=False",
            "output.reflections=joint.refl",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    joint_refl = flex.reflection_table.from_file(tmp_path / "joint.refl")
    joint_refl = joint_refl.select(
        joint_refl.get_flags(joint_refl.flags.used_in_refinement)
    )
    x_obs, y_obs, z_obs = joint_refl["xyzobs.mm.value"].parts()
    x_calc, y_calc, z_calc = joint_refl["xyzcal.mm"].parts()
    x_resid2 = (x_calc - x_obs) ** 2
    y_resid2 = (y_calc - y_obs) ** 2
    z_resid2 = (z_calc - z_obs) ** 2
    joint_rmsd_x, joint_rmsd_y, joint_rmsd_z = [], [], []
    for i in range(22):
        sel = joint_refl["id"] == i
        n = sel.count(True)
        joint_rmsd_x.append(math.sqrt(flex.sum(x_resid2.select(sel)) / n))
        joint_rmsd_y.append(math.sqrt(flex.sum(y_resid2.select(sel)) / n))
        joint_rmsd_z.append(math.sqrt(flex.sum(z_resid2.select(sel)) / n))

    assert (
        flex.max(flex.abs(flex.double(separate_rmsd_x) - flex.double(joint_rmsd_x)))
        < 0.01
    )
    assert (
        flex.max(flex.abs(flex.double(separate_rmsd_y) - flex.double(joint_rmsd_y)))
        < 0.01
    )
    assert (
        np.degrees(
            flex.max(flex.abs(flex.double(separate_rmsd_z) - flex.double(joint_rmsd_z)))
        )
        < 0.01
    )
