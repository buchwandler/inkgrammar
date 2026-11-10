"""Training-only models and numeric fitting utilities."""

from .checkpoint import FitCheckpoint, load_checkpoint, save_checkpoint
from .family_fit import (
    FitRunArtifacts,
    load_fit_reports,
    optimize_primitive_family,
    primitive_path,
)
from .models import (
    DuctusModel,
    FitAnchorRecord,
    FitAnchorsFile,
    FitCorpus,
    FitFamiliesFile,
    FitProject,
    FitSample,
    FitTopologyFile,
    GlyphRecipe,
    PrimitiveInstance,
    PrimitiveTemplate,
    ScriptGrammar,
    SourceLockFile,
    TopologyTransform,
)
from .objective import (
    LocalRegion,
    ObjectiveConfig,
    ObjectiveResult,
    ObjectiveWeights,
    evaluate_objective,
    soft_mask_loss,
    symmetric_distance_loss,
)
from .optimizer import (
    OptimizationResult,
    OptimizerConfig,
    checkpoint_from_result,
    optimize_parameters,
    resume_optimization,
)
from .parameterization import ParameterSpace, ParameterSpec
from .render import FittingRender, render_paths, render_traces
from .trace import CompiledPartTrace, compile_glyph_trace
from .workspace import (
    FitProjectBundle,
    MaterializationResult,
    load_fit_project,
    materialize_fit_project,
)

__all__ = [
    "FittingRender",
    "LocalRegion",
    "ObjectiveConfig",
    "ObjectiveResult",
    "ObjectiveWeights",
    "CompiledPartTrace",
    "DuctusModel",
    "FitCheckpoint",
    "FitCorpus",
    "FitProject",
    "FitSample",
    "GlyphRecipe",
    "ParameterSpec",
    "ParameterSpace",
    "PrimitiveInstance",
    "PrimitiveTemplate",
    "ScriptGrammar",
    "evaluate_objective",
    "render_paths",
    "render_traces",
    "soft_mask_loss",
    "symmetric_distance_loss",
    "load_checkpoint",
    "compile_glyph_trace",
    "OptimizerConfig",
    "OptimizationResult",
    "checkpoint_from_result",
    "optimize_parameters",
    "resume_optimization",
    "save_checkpoint",
    "FitAnchorRecord",
    "FitAnchorsFile",
    "FitFamiliesFile",
    "FitTopologyFile",
    "FitProjectBundle",
    "MaterializationResult",
    "SourceLockFile",
    "load_fit_project",
    "materialize_fit_project",
    "TopologyTransform",
    "FitRunArtifacts",
    "load_fit_reports",
    "optimize_primitive_family",
    "primitive_path",
]
