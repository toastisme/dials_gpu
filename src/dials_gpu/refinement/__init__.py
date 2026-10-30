from __future__ import annotations

from dials.algorithms.refinement import DialsRefineConfigError, DialsRefineRuntimeError
from dials_gpu.refinement.engine import (
    GaussNewtonIterationsGPU,
    Journal,
    LBFGScurvsGPU,
    LevenbergMarquardtIterationsGPU,
    SimpleLBFGSGPU,
    SparseLevenbergMarquardtIterationsGPU,
)
from dials_gpu.refinement.outlier_detection import CentroidOutlierFactoryGPU
from dials_gpu.refinement.parameterisation import (
    ScanVaryingPredictionParameterisationGPU,
    ScanVaryingPredictionParameterisationSparseGPU,
    SphericalRelpStillsPredictionParameterisationGPU,
    SphericalRelpStillsPredictionParameterisationSparseGPU,
    StillsPredictionParameterisationGPU,
    StillsPredictionParameterisationSparseGPU,
    XYPhiPredictionParameterisationGPU,
    XYPhiPredictionParameterisationSparseGPU,
    _patch_crystal_uc_parameterisations,
)
from dials_gpu.refinement.predictor import (
    ExperimentsPredictorFactoryGPU,
    ScansExperimentsPredictorGPU,
    StillsExperimentsPredictorGPU,
)
from dials_gpu.refinement.refiner import (
    Refiner,
    RefinerFactory,
    RefinerFactoryGPU,
    RefinerGPU,
    ReflectionManagerFactoryGPU,
    ScanVaryingRefiner,
    ScanVaryingRefinerGPU,
    phil_scope,
)
from dials_gpu.refinement.target import (
    LeastSquaresPositionalResidualWithRmsdCutoffGPU,
    LeastSquaresPositionalResidualWithRmsdCutoffSparseGPU,
    LeastSquaresStillsResidualWithRmsdCutoffGPU,
    LeastSquaresStillsResidualWithRmsdCutoffSparseGPU,
    TargetFactoryGPU,
)

__all__ = [
    "CentroidOutlierFactoryGPU",
    "DialsRefineConfigError",
    "DialsRefineRuntimeError",
    "ExperimentsPredictorFactoryGPU",
    "GaussNewtonIterationsGPU",
    "Journal",
    "LBFGScurvsGPU",
    "LeastSquaresPositionalResidualWithRmsdCutoffGPU",
    "LeastSquaresPositionalResidualWithRmsdCutoffSparseGPU",
    "LeastSquaresStillsResidualWithRmsdCutoffGPU",
    "LeastSquaresStillsResidualWithRmsdCutoffSparseGPU",
    "LevenbergMarquardtIterationsGPU",
    "Refiner",
    "RefinerFactory",
    "RefinerFactoryGPU",
    "RefinerGPU",
    "ReflectionManagerFactoryGPU",
    "ScanVaryingPredictionParameterisationGPU",
    "ScanVaryingPredictionParameterisationSparseGPU",
    "ScanVaryingRefiner",
    "ScanVaryingRefinerGPU",
    "ScansExperimentsPredictorGPU",
    "SimpleLBFGSGPU",
    "SparseLevenbergMarquardtIterationsGPU",
    "SphericalRelpStillsPredictionParameterisationGPU",
    "SphericalRelpStillsPredictionParameterisationSparseGPU",
    "StillsExperimentsPredictorGPU",
    "StillsPredictionParameterisationGPU",
    "StillsPredictionParameterisationSparseGPU",
    "TargetFactoryGPU",
    "XYPhiPredictionParameterisationGPU",
    "XYPhiPredictionParameterisationSparseGPU",
    "_patch_crystal_uc_parameterisations",
    "phil_scope",
]
