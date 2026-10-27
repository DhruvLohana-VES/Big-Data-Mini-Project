"""
Visualization and Publication Figures Package
"""
from .plots import (
    plot_fig1_study_pipeline,
    plot_fig2_annotation_workflow,
    plot_fig3_prompt_designs,
    plot_fig4_dataset_composition,
    plot_fig5_llm_human_agreement,
    plot_fig6_downstream_model_performance,
    plot_fig7_cost_quality_curve,
    plot_fig8_calibration_reliability,
    plot_fig9_error_analysis_matrix,
    generate_all_figures
)

__all__ = [
    "plot_fig1_study_pipeline",
    "plot_fig2_annotation_workflow",
    "plot_fig3_prompt_designs",
    "plot_fig4_dataset_composition",
    "plot_fig5_llm_human_agreement",
    "plot_fig6_downstream_model_performance",
    "plot_fig7_cost_quality_curve",
    "plot_fig8_calibration_reliability",
    "plot_fig9_error_analysis_matrix",
    "generate_all_figures"
]
