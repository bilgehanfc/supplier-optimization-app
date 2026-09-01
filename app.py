"""Streamlit supplier-selection portal.

Run locally with:
    streamlit run app.py

The Supplier Optimization page preserves the existing optimization UI and
imports all mathematical logic from ``supplier_selection_methods.py``.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from supplier_selection_methods import (
    goal_programming,
    preemptive_optimization,
    topsis,
    weighted_sum,
)


st.set_page_config(
    page_title="Supplier Selection Optimization",
    page_icon="📊",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Corporate branding and translations
# -----------------------------------------------------------------------------


LANGUAGE_OPTIONS = {"TR": "Türkçe", "EN": "English", "DE": "Deutsch"}

TRANSLATIONS: dict[str, dict[str, str]] = {
    "EN": {
        "language": "Language",
        "navigation": "Navigation",
        "page_homepage": "Homepage",
        "page_parts_database": "Parts Database",
        "page_supplier_database": "Supplier Database",
        "page_osa_assessment": "OSA Assessment",
        "page_supplier_optimization": "Supplier Optimization",
        "homepage_title": "Supplier Selection Optimization Portal",
        "homepage_caption": "A decision-support workspace for supplier, part, and OSA analysis.",
        "homepage_subtitle": "An optimization-based decision support system for supplier evaluation, ranking and selection.",
        "homepage_kpi_parts": "Parts",
        "homepage_kpi_candidate_suppliers": "Candidate Suppliers",
        "homepage_kpi_qualified_suppliers": "Qualified Suppliers",
        "homepage_kpi_optimization_models": "Optimization Models",
        "framework_title": "Supplier Selection Framework",
        "framework_osa_assessment": "OSA Assessment",
        "framework_supplier_qualification": "Supplier Qualification",
        "framework_cost_evaluation": "Cost Evaluation",
        "framework_quality_evaluation": "Quality Evaluation",
        "framework_delivery_evaluation": "Delivery Evaluation",
        "framework_supplier_ranking": "Supplier Ranking",
        "optimization_methods_title": "Optimization Methods",
        "method_best_when": "Best when",
        "method_description": "Description",
        "method_example": "Example",
        "method_weighted_sum_best_when": "Criteria are measurable and importance can be expressed with stable weights.",
        "method_weighted_sum_description": "Scores each supplier on normalized criteria and combines them into one weighted score.",
        "method_weighted_sum_example": "Best overall supplier when cost is 40%, quality is 35%, and delivery is 25%.",
        "method_topsis_best_when": "You want a balanced choice close to the ideal supplier.",
        "method_topsis_description": "Ranks suppliers by their distance from the ideal and anti-ideal solutions.",
        "method_topsis_example": "A supplier strong across all criteria but not necessarily first in any single one.",
        "method_goal_programming_best_when": "You have explicit cost, quality, and delivery targets.",
        "method_goal_programming_description": "Minimizes weighted deviations from target values while respecting capacities.",
        "method_goal_programming_example": "Achieve quality ≥ 90, delivery ≤ 7 days, and cost ≤ €110.",
        "method_preemptive_best_when": "Some goals must always take priority over others.",
        "method_preemptive_description": "Optimizes objectives lexicographically, protecting higher-priority goals first.",
        "method_preemptive_example": "Meet quality first, then delivery, then minimize cost.",
        "method_guide_title": "Which Method Should You Choose?",
        "method_guide_weighted_sum": "Choose Weighted Sum when criteria are measurable and their relative importance can be represented with reliable weights.",
        "method_guide_topsis": "Choose TOPSIS when you want a balanced supplier that is closest to the ideal profile across all criteria.",
        "method_guide_goal_programming": "Choose Goal Programming when cost, quality, and delivery targets must be met as closely as possible.",
        "method_guide_preemptive": "Choose Preemptive Optimization when business priorities are ranked and higher-priority goals cannot be sacrificed.",
        "kpi_total_suppliers": "Total Suppliers",
        "kpi_total_parts": "Total Parts",
        "kpi_active_assessments": "Active Assessments",
        "kpi_scenarios": "Optimization Scenarios",
        "osa_heading": "What are OSA criteria?",
        "osa_text": """
**On-Site Assessment (OSA)** criteria are structured measures used to evaluate a supplier during a physical or remote assessment. They can cover quality-management maturity, production capability, process controls, delivery performance, financial resilience, sustainability, safety, and compliance.

Each criterion should have a clear definition, scoring scale, evidence requirement, responsible assessor, and weighting policy. Consistent definitions make supplier comparisons more transparent and auditable.
""",
        "algorithms_heading": "Optimization methods",
        "algorithms_text": """
- **TOPSIS:** Ranks suppliers by their distance from an ideal supplier and an anti-ideal supplier.
- **Weighted Sum:** Converts criteria to comparable utilities and calculates a weighted overall score.
- **Preemptive Optimization:** Applies priorities lexicographically; higher-priority goals are protected before lower-priority goals.
- **Goal Programming:** Minimizes weighted deviations from quality, delivery, and cost targets.
""",
        "parts_title": "Parts Database",
        "parts_subtitle": "Manage part information used throughout supplier evaluation and optimization workflows.",
        "parts_kpi_total": "Total Parts",
        "parts_kpi_candidate_suppliers": "Candidate Suppliers",
        "parts_kpi_qualified_suppliers": "Qualified Suppliers",
        "parts_kpi_active_rfqs": "Active RFQs",
        "parts_filter_number": "Search Part Number",
        "parts_filter_number_placeholder": "e.g. DT-INT-001",
        "parts_filter_description": "Search Part Description",
        "parts_filter_description_placeholder": "e.g. Interior Trims",
        "parts_filter_plant": "Plant",
        "parts_filter_sop_year": "SOP Year",
        "parts_all_plants": "All Plants",
        "parts_all_years": "All Years",
        "parts_table_title": "Parts Catalogue",
        "parts_table_count": "Showing {shown} of {total} parts",
        "parts_select_hint": "Select a row to view complete part metadata and supplier coverage.",
        "parts_no_matches": "No parts match the selected filters.",
        "parts_detail_title": "Part Detail",
        "parts_candidate_suppliers": "Candidate Suppliers",
        "parts_qualified_suppliers": "Qualified Suppliers",
        "parts_no_selection": "Select a part from the table to open its detail panel.",
        "part_number": "Part Number",
        "part_description": "Part Description",
        "plant": "Plant",
        "sop_year": "SOP Year",
        "lifetime_years": "Lifetime (Years)",
        "annual_volume": "Annual Volume",
        "budget": "Budget (€)",
        "target_cost": "Target Cost (€)",
        "supplier_column_name": "Supplier Name",
        "supplier_column_location": "Location",
        "supplier_column_capacity": "Capacity",
        "supplier_column_part": "Supplied Part",
        "supplier_title": "Supplier Database",
        "supplier_subtitle": "Manage supplier qualification, cost, quality and delivery performance data.",
        "supplier_kpi_candidate_suppliers": "Candidate Suppliers",
        "supplier_kpi_qualified_suppliers": "Qualified Suppliers",
        "supplier_kpi_average_osa_score": "Average OSA Score",
        "supplier_kpi_countries_represented": "Countries Represented",
        "supplier_filter_part_number": "Part Number",
        "supplier_filter_part_description": "Part Description",
        "supplier_filter_name": "Supplier Name",
        "supplier_filter_country": "Country",
        "supplier_filter_osa_status": "OSA Status",
        "supplier_filter_part_number_placeholder": "e.g. DT-INT-001",
        "supplier_filter_part_description_placeholder": "e.g. Interior Trims",
        "supplier_filter_name_placeholder": "e.g. Grammer",
        "supplier_all_parts": "All Parts",
        "supplier_all_countries": "All Countries",
        "supplier_all_statuses": "All OSA Statuses",
        "supplier_table_title": "Supplier Comparison Catalogue",
        "supplier_table_count": "Showing {shown} of {total} supplier records",
        "supplier_select_hint": "Select a row to view the supplier profile, or choose records below for side-by-side comparison.",
        "supplier_no_matches": "No supplier records match the selected filters.",
        "supplier_detail_title": "Supplier Detail",
        "supplier_detail_no_selection": "Select a supplier record from the table to open its profile.",
        "supplier_section_basic_info": "Section 1 · Basic Information",
        "supplier_section_osa_info": "Section 2 · OSA Information",
        "supplier_section_commercial_info": "Section 3 · Commercial Information",
        "supplier_section_quality_performance": "Section 4 · Quality Performance",
        "supplier_section_delivery_performance": "Section 5 · Delivery Performance",
        "supplier_section_awarding_readiness": "Section 6 · Awarding Readiness",
        "supplier_name": "Supplier Name",
        "supplier_code": "Supplier Code",
        "country": "Country",
        "production_location": "Production Location",
        "supplier_plant": "Plant",
        "osa_score": "OSA Score",
        "osa_status": "OSA Status",
        "status_qualified": "Qualified",
        "status_not_qualified": "Not Qualified",
        "unit_price": "Unit Price (€)",
        "annual_cost": "Annual Cost (€)",
        "tooling_cost": "Tooling Cost (€)",
        "quality_score": "Quality Score",
        "defect_rate": "Defect Rate (%)",
        "warranty_claims": "Warranty Claims",
        "incoming_acceptance_rate": "Incoming Acceptance Rate (%)",
        "process_capability_cpk": "Process Capability (Cpk)",
        "cart_days": "Corrective Action Response Time (CART) (Days)",
        "delivery_score": "Delivery Score",
        "on_time_delivery": "On-Time Delivery (%)",
        "lead_time_days": "Lead Time (Days)",
        "delivery_accuracy": "Delivery Accuracy (%)",
        "readiness_standard_contract": "Standard Contract Acceptance",
        "readiness_quality_certificate": "Quality Certificate Status",
        "readiness_environmental_certificate": "Environmental Certificate Status",
        "readiness_supplier_risk": "Supplier Risk Status",
        "readiness_fsrm": "FSRM Status",
        "status_approved": "Approved",
        "status_review_required": "Review Required",
        "status_missing": "Missing",
        "supplier_comparison_title": "Supplier Comparison View",
        "supplier_comparison_select": "Select supplier records to compare",
        "supplier_comparison_hint": "Select up to three supplier records for a side-by-side comparison.",
        "osa_title": "OSA Assessment",
        "osa_placeholder": "This module is under development.",
        "osa_subtitle": "Supplier Qualification and Readiness Evaluation",
        "osa_kpi_assessed_suppliers": "Assessed Suppliers",
        "osa_kpi_qualified_suppliers": "Qualified Suppliers",
        "osa_kpi_not_qualified_suppliers": "Not Qualified Suppliers",
        "osa_kpi_threshold": "OSA Threshold",
        "osa_filter_part_number": "Part Number",
        "osa_filter_part_description": "Part Description",
        "osa_filter_supplier_name": "Supplier Name",
        "osa_filter_country": "Country",
        "osa_filter_part_description_placeholder": "e.g. Interior Trims",
        "osa_filter_supplier_name_placeholder": "e.g. Grammer",
        "osa_all_parts": "All Parts",
        "osa_all_countries": "All Countries",
        "osa_supplier_selection_title": "Supplier Selection",
        "osa_supplier_selection": "Select supplier record",
        "osa_supplier_selection_hint": "Choose a supplier record to review and adjust its six-category OSA assessment.",
        "osa_no_matches": "No supplier records match the selected filters.",
        "osa_supplier_info_title": "Supplier Information",
        "osa_category_title": "OSA Assessment Framework",
        "osa_weight": "Weight",
        "osa_subcriteria": "Subcriteria",
        "osa_category_quality_system": "Category 1 · Quality System",
        "osa_category_production_capability": "Category 2 · Production Capability",
        "osa_category_capacity_scalability": "Category 3 · Capacity & Scalability",
        "osa_category_technical_capability": "Category 4 · Technical Capability",
        "osa_category_logistics_supply_chain": "Category 5 · Logistics & Supply Chain",
        "osa_category_management_compliance": "Category 6 · Management & Compliance",
        "osa_sub_iatf_16949": "IATF 16949",
        "osa_sub_iso_9001": "ISO 9001",
        "osa_sub_spc": "SPC",
        "osa_sub_traceability": "Traceability",
        "osa_sub_pfmea": "PFMEA",
        "osa_sub_control_plan": "Control Plan",
        "osa_sub_8d": "8D Problem Solving",
        "osa_sub_standard_work": "Standard Work",
        "osa_sub_process_stability": "Process Stability",
        "osa_sub_oee": "OEE",
        "osa_sub_tpm": "TPM",
        "osa_sub_5s": "5S",
        "osa_sub_preventive_maintenance": "Preventive Maintenance",
        "osa_sub_production_flow": "Production Flow",
        "osa_sub_current_capacity": "Current Capacity",
        "osa_sub_available_capacity": "Available Capacity",
        "osa_sub_capacity_utilization": "Capacity Utilization",
        "osa_sub_additional_shift": "Additional Shift Capability",
        "osa_sub_scalability": "Scalability",
        "osa_sub_demand_growth": "Demand Growth Capability",
        "osa_sub_similar_product": "Similar Product Experience",
        "osa_sub_engineering_support": "Engineering Support",
        "osa_sub_validation": "Validation Capability",
        "osa_sub_testing": "Testing Capability",
        "osa_sub_manufacturing_technology": "Manufacturing Technology",
        "osa_sub_rd_support": "R&D Support",
        "osa_sub_material_flow": "Material Flow",
        "osa_sub_fifo": "FIFO",
        "osa_sub_inventory_management": "Inventory Management",
        "osa_sub_packaging_management": "Packaging Management",
        "osa_sub_delivery_capability": "Delivery Capability",
        "osa_sub_emergency_logistics": "Emergency Logistics Plan",
        "osa_sub_edi": "EDI Capability",
        "osa_sub_psc_rating": "PSC Rating",
        "osa_sub_saq_rating": "SAQ Rating",
        "osa_sub_corrective_actions": "Corrective Actions",
        "osa_sub_fsrm_status": "FSRM Status",
        "osa_sub_standard_contract": "Standard Contract Acceptance",
        "osa_sub_supplier_risk": "Supplier Risk Status",
        "osa_sub_business_continuity": "Business Continuity Plan",
        "osa_breakdown_title": "OSA Breakdown",
        "osa_breakdown_category": "Category",
        "osa_breakdown_weight": "Weight",
        "osa_breakdown_score": "Score",
        "osa_breakdown_contribution": "Contribution",
        "osa_formula": "OSA Score = Σ(Category Score × Weight)",
        "osa_compliance_title": "Compliance & Certification",
        "osa_compliance_fsrm": "FSRM Status",
        "osa_compliance_iatf": "IATF 16949",
        "osa_compliance_iso_14001": "ISO 14001",
        "osa_compliance_tisax": "TISAX",
        "osa_compliance_contract": "Standard Contract Acceptance",
        "status_watch": "Watch",
        "osa_awarding_title": "Awarding Readiness",
        "osa_ready_for_award": "Ready For Award",
        "osa_task_for_awarding": "Task For Awarding Required",
        "osa_risk_board_approval": "Supplier Risk Board Approval Required",
        "osa_actions_title": "Need for Action",
        "osa_action_missing_certificates": "Missing Certificates",
        "osa_action_open_corrective_actions": "Open Corrective Actions",
        "osa_action_missing_osa_ica": "Missing OSA / ICA",
        "osa_action_restructuring": "Restructuring Requirements",
        "osa_action_clear": "No immediate action",
        "osa_action_required": "Action required",
        "osa_result_title": "OSA Result",
        "osa_total_score": "Total OSA Score",
        "osa_qualified": "QUALIFIED",
        "osa_not_qualified": "NOT QUALIFIED",
        "osa_exclusion_note": "Suppliers below 70 are excluded from optimization.",
        "osa_critical_findings_title": "Critical Findings · Veto Rules",
        "osa_veto_missing_iatf": "Missing IATF 16949",
        "osa_veto_corrective_action": "Critical Open Corrective Action",
        "osa_veto_risk_board": "Supplier Risk Board Rejection",
        "osa_veto_capacity": "Insufficient Production Capacity",
        "osa_veto_business_continuity": "Missing Business Continuity Plan",
        "osa_veto_product_safety": "Product Safety Violation",
        "osa_veto_triggered": "Veto triggered",
        "osa_veto_clear": "Clear",
        "optimization_title": "Supplier Optimization",
        "optimization_description": "Evaluate, rank and optimize suppliers using advanced multi-criteria decision-making methods.",
        "workflow_parts_database": "Parts Database",
        "workflow_supplier_database": "Supplier Database",
        "workflow_osa_assessment": "OSA Assessment",
        "workflow_supplier_optimization": "Supplier Optimization",
        "workflow_results_reports": "Results & Reports",
        "optimization_step_part_selection": "Step 1 · Part Selection",
        "optimization_step_method_selection": "Step 2 · Optimization Method",
        "optimization_step_data_import": "Step 3 · Supplier Data Import & Preview",
        "optimization_step_qualified_filter": "Step 4 · Qualified Supplier Filter",
        "optimization_step_criteria": "Step 5 · Criteria Construction",
        "optimization_step_configuration": "Step 6 · Method Configuration",
        "optimization_step_run": "Step 7 · Run Optimization",
        "optimization_step_results": "Step 8 · Optimization Results",
        "optimization_step_visualization": "Step 9 · Visualization",
        "optimization_step_decision": "Step 10 · Decision Summary",
        "optimization_step_reports": "Step 11 · Report Export",
        "optimization_part_selection": "Select a part",
        "optimization_part_selection_hint": "Choose the part to define demand, budget, and the supplier pool under evaluation.",
        "optimization_all_parts": "Select a part",
        "optimization_method_selection": "Select an optimization method",
        "optimization_method_selection_hint": "Choose the decision logic that best matches the business situation.",
        "optimization_method_weighted_sum_summary": "Balances all criteria simultaneously.",
        "optimization_method_weighted_sum_best_when": "Best when trade-offs between criteria are acceptable.",
        "optimization_method_topsis_summary": "Ranks suppliers based on distance from the ideal supplier.",
        "optimization_method_topsis_best_when": "Best when the primary objective is ranking alternatives.",
        "optimization_method_goal_programming_summary": "Finds the supplier allocation that best satisfies predefined targets.",
        "optimization_method_goal_programming_best_when": "Best when cost, quality, and delivery targets must be met.",
        "optimization_method_preemptive_summary": "Applies strict priorities where higher-priority criteria dominate.",
        "optimization_method_preemptive_best_when": "Best when business priorities cannot be traded off.",
        "optimization_upload_label": "Upload supplier dataset",
        "optimization_upload_help": "Supported formats: XLSX, XLS, and CSV.",
        "optimization_default_data": "Using the built-in supplier dataset until a file is uploaded.",
        "optimization_preview_title": "Supplier Dataset Preview",
        "optimization_preview_caption": "The preview includes supplier performance fields and the OSA qualification gate.",
        "optimization_expected_columns": "Expected columns: Part Number, Supplier, Annual Cost, Defect Rate, Warranty Claims, Incoming Acceptance Rate, Process Capability, Corrective Action Response Time, On-Time Delivery, Lead Time, Delivery Accuracy, OSA Score, Country.",
        "optimization_missing_required_columns": "The uploaded dataset is missing required columns: {columns}",
        "optimization_invalid_numeric_columns": "These optimization fields contain blank or non-numeric values: {columns}",
        "optimization_finite_numeric": "Optimization input values must be finite.",
        "optimization_invalid_capacity": "Capacity values must be non-negative and finite.",
        "optimization_missing_annual_volume": "Annual Volume could not be inferred for these uploaded parts: {parts}. Add an Annual Volume column or provide a matching Part Description.",
        "optimization_upload_error": "The uploaded dataset could not be prepared: {error}",
        "optimization_capacity_assumption": "Capacity was not supplied, so Annual Volume is used as a conservative allocation capacity. Capacity is a constraint only, not an optimization criterion.",
        "optimization_part_match_fallback": "No exact Part Number match was found. Supplier rows were matched using Part Description.",
        "optimization_gate_title": "Qualified Supplier Filter · Gatekeeper Rule",
        "optimization_osa_threshold": "OSA Threshold",
        "optimization_use_only_qualified": "☑ Use Only Qualified Suppliers",
        "optimization_osa_gate_note": "OSA is not an optimization criterion. It is used strictly as a qualification gate; suppliers with OSA Score < 70 are excluded when this rule is enabled.",
        "optimization_three_criteria_note": "Optimization uses only Cost, Quality, and Delivery. Capacity and Technical Capability are evaluated within OSA and are not optimization criteria.",
        "optimization_criteria_title": "Criteria Construction",
        "optimization_cost_definition": "Cost = Annual Supplier Cost; no pre-normalization is applied.",
        "optimization_quality_definition": "Quality Score = weighted subcriteria calculation from Defect Rate, Warranty Claims, Incoming Acceptance Rate, and Process Capability.",
        "optimization_delivery_definition": "Delivery Score = weighted subcriteria calculation from On-Time Delivery, Lead Time, and Delivery Accuracy.",
        "optimization_method_configuration": "Method Configuration",
        "optimization_weights_title": "Criterion Weights",
        "optimization_weight_cost": "Cost weight",
        "optimization_weight_quality": "Quality weight",
        "optimization_weight_delivery": "Delivery weight",
        "optimization_weight_total": "Weight total",
        "optimization_weight_validation": "Weights must sum to 1.00 before optimization can run.",
        "optimization_topsis_title": "TOPSIS Configuration",
        "optimization_topsis_cost": "Cost is configured as a cost criterion (lower is better).",
        "optimization_topsis_benefits": "Quality and Delivery are configured as benefit criteria (higher is better).",
        "optimization_topsis_normalization": "The existing TOPSIS backend applies Euclidean normalization and ideal/anti-ideal ranking.",
        "optimization_targets_title": "Target Values",
        "optimization_target_cost": "Target Cost (€)",
        "optimization_target_cost_unit": "Target Cost uses the same annual-cost units as the uploaded supplier dataset.",
        "optimization_target_quality": "Target Quality Score",
        "optimization_target_delivery": "Target Delivery Score",
        "optimization_preemptive_title": "Priority Ordering",
        "optimization_priority_1": "Priority 1",
        "optimization_priority_2": "Priority 2",
        "optimization_priority_3": "Priority 3",
        "optimization_priority_cost": "Cost",
        "optimization_priority_quality": "Quality",
        "optimization_priority_delivery": "Delivery",
        "optimization_priority_note": "The selected order documents the business priority intent; the existing preemptive solver implementation remains unchanged.",
        "optimization_run_button": "🚀 Run Optimization",
        "optimization_no_qualified_suppliers": "No suppliers remain after applying the OSA qualification gate.",
        "optimization_no_part_data": "No supplier data is available for the selected part.",
        "optimization_duplicate_suppliers": "Each supplier must appear only once for the selected part.",
        "optimization_priority_validation": "Priority 1, Priority 2, and Priority 3 must be different.",
        "optimization_results_title": "Optimization Results",
        "optimization_recommended_supplier": "Recommended Supplier",
        "optimization_final_score": "Final Score",
        "optimization_supplier_ranking": "Supplier Ranking",
        "optimization_ranking_bar_chart": "Supplier Ranking Bar Chart",
        "optimization_radar_chart": "Criteria Radar Chart",
        "optimization_no_results": "Run the optimization to display results and visualizations.",
        "optimization_decision_summary": "Decision Summary",
        "optimization_method_used": "Method Used",
        "optimization_selected_part": "Selected Part",
        "optimization_evaluated_suppliers": "Evaluated Suppliers",
        "optimization_qualified_suppliers": "Qualified Suppliers",
        "optimization_cost_score": "Cost",
        "optimization_quality_score": "Quality",
        "optimization_delivery_score": "Delivery",
        "optimization_osa_score": "OSA",
        "optimization_export_reports": "Export Reports",
        "optimization_export_pdf": "📄 Export PDF Report",
        "optimization_export_excel": "📊 Export Excel Report",
        "optimization_pdf_placeholder": "PDF report generation is ready for integration with the corporate reporting service.",
        "optimization_excel_error": "Excel report could not be generated: {error}",
        "optimization_report_generated_at": "Generated at",
        "optimization_results_csv": "Download optimization results (CSV)",
        "optimization_radar_cost": "Cost Performance",
        "optimization_radar_quality": "Quality",
        "optimization_radar_delivery": "Delivery",
        "optimization_radar_normalized_note": "0–100 normalized view",
        "choose_analysis": "Choose an analysis",
        "add_supplier_data": "Add supplier data",
        "optimization_method": "Optimization method",
        "supplier_data": "Supplier data",
        "supplier_data_caption": "Review or edit the values that will be used in this analysis.",
        "model_configuration": "Model configuration",
        "model_configuration_caption": "Weights normalize automatically. Set whether higher or lower is better.",
        "sample_caption": "Using the sample dataset · Upload CSV or Excel to replace it",
        "using_file": "Using",
        "analysis_help": "Rank individual suppliers or optimize a capacity-constrained split award.",
        "allocation_settings": "Allocation settings",
        "capacity_warning": "Add a Capacity column to use this method.",
        "demand_units": "Demand units",
        "cost_criterion": "Cost criterion",
        "quality_criterion": "Quality criterion",
        "delivery_criterion": "Delivery criterion",
        "minimum_quality_target": "Minimum quality target",
        "maximum_delivery_target": "Maximum delivery target",
        "maximum_cost_target": "Maximum cost target",
        "goal_weights": "Goal-programming weights",
        "quality_goal_weight": "Quality goal weight",
        "delivery_goal_weight": "Delivery goal weight",
        "cost_goal_weight": "Cost goal weight",
        "run_analysis": "Run analysis",
        "ranking_results": "Ranking results",
        "higher_score": "Higher {score} values indicate a better supplier.",
        "solution_metrics": "Solution metrics",
        "lexicographic_results": "Lexicographic priority results",
        "download_results": "Download final results (CSV)",
        "upload_supplier_data": "Upload supplier data",
        "upload_supplier_data_help": "Use columns such as Supplier, Cost, Quality, Delivery, and Capacity.",
        "error_read_upload": "Could not read the uploaded file: {error}",
        "error_empty_criterion": "At least one criterion is required; Capacity is not a criterion.",
        "error_capacity_required": "Add a Capacity column before running this method.",
        "error_capacity_column": "Preemptive Optimization and Goal Programming require a Capacity column.",
        "error_weight_required": "At least one criterion weight must be greater than zero.",
        "error_goal_weights": "At least one goal-programming weight must be greater than zero.",
        "error_analysis": "Analysis could not be completed: {error}",
        "choose_method_review": "Choose a method, review the model configuration, then run the analysis.",
        "error_table_supplier_column": "The table must contain a Supplier column.",
        "error_enter_supplier": "Enter at least one supplier.",
        "error_supplier_blank": "Supplier names cannot be blank.",
        "error_supplier_unique": "Supplier names must be non-empty and unique.",
        "error_numeric_criterion": "Add at least one numeric criterion column.",
        "error_invalid_numeric_columns": "These columns contain blank or non-numeric values: {columns}",
        "error_finite_numeric": "Numeric values must be finite.",
        "criterion_weight": "{criterion} weight",
        "criterion_direction": "{criterion} direction",
        "impact_benefit": "Benefit",
        "impact_cost": "Cost",
        "score_topsis": "TOPSIS score",
        "score_weighted_sum": "Weighted-sum score",
        "preemptive_allocation": "Preemptive allocation",
        "goal_programming_allocation": "Goal-programming allocation",
        "metric_average_cost": "Average Cost",
        "metric_average_quality": "Average Quality",
        "metric_average_delivery": "Average Delivery",
        "metric_quality_shortfall": "Quality Shortfall",
        "metric_delivery_excess": "Delivery Excess",
        "metric_optimal_quality_shortfall": "Optimal Quality Shortfall",
        "metric_optimal_delivery_excess_after_p1": "Optimal Delivery Excess after P1",
        "metric_normalized_objective": "Normalized Objective",
        "column_score": "Score",
        "column_rank": "Rank",
        "column_allocation_units": "Allocation Units",
        "column_allocation_share": "Allocation Share",
        "column_value": "Value",
        "method_topsis": "TOPSIS",
        "method_weighted_sum": "Weighted Sum",
        "method_preemptive": "Preemptive Optimization",
        "method_goal_programming": "Goal Programming",
    },
    "TR": {
        "language": "Dil",
        "navigation": "Gezinme",
        "page_homepage": "Ana Sayfa",
        "page_parts_database": "Parça Veritabanı",
        "page_supplier_database": "Tedarikçi Veritabanı",
        "page_osa_assessment": "OSA Değerlendirmesi",
        "page_supplier_optimization": "Tedarikçi Optimizasyonu",
        "homepage_title": "Tedarikçi Seçimi Optimizasyon Portalı",
        "homepage_caption": "Tedarikçi, parça ve OSA analizi için karar destek çalışma alanı.",
        "homepage_subtitle": "Tedarikçi değerlendirme, sıralama ve seçimi için optimizasyon tabanlı karar destek sistemi.",
        "homepage_kpi_parts": "Parça",
        "homepage_kpi_candidate_suppliers": "Aday Tedarikçiler",
        "homepage_kpi_qualified_suppliers": "Nitelikli Tedarikçiler",
        "homepage_kpi_optimization_models": "Optimizasyon Modelleri",
        "framework_title": "Tedarikçi Seçim Çerçevesi",
        "framework_osa_assessment": "OSA Değerlendirmesi",
        "framework_supplier_qualification": "Tedarikçi Yeterlilik Değerlendirmesi",
        "framework_cost_evaluation": "Maliyet Değerlendirmesi",
        "framework_quality_evaluation": "Kalite Değerlendirmesi",
        "framework_delivery_evaluation": "Teslimat Değerlendirmesi",
        "framework_supplier_ranking": "Tedarikçi Sıralaması",
        "optimization_methods_title": "Optimizasyon Yöntemleri",
        "method_best_when": "En uygun olduğu durum",
        "method_description": "Açıklama",
        "method_example": "Örnek",
        "method_weighted_sum_best_when": "Kriterler ölçülebilir olduğunda ve önem düzeyi sabit ağırlıklarla ifade edilebildiğinde.",
        "method_weighted_sum_description": "Her tedarikçiyi normalize edilmiş kriterlerle puanlar ve tek bir ağırlıklı puanda birleştirir.",
        "method_weighted_sum_example": "Maliyet %40, kalite %35 ve teslimat %25 olduğunda en iyi genel tedarikçiyi bulmak.",
        "method_topsis_best_when": "İdeal tedarikçiye yakın, dengeli bir seçim istediğinizde.",
        "method_topsis_description": "Tedarikçileri ideal ve ideal olmayan çözümlere olan uzaklıklarına göre sıralar.",
        "method_topsis_example": "Tek bir kriterde ilk sırada olmasa da tüm kriterlerde güçlü olan bir tedarikçi.",
        "method_goal_programming_best_when": "Açık maliyet, kalite ve teslimat hedefleriniz olduğunda.",
        "method_goal_programming_description": "Kapasiteleri dikkate alarak hedef değerlerden sapmaları ağırlıklı biçimde en aza indirir.",
        "method_goal_programming_example": "Kalite ≥ 90, teslimat ≤ 7 gün ve maliyet ≤ 110 € hedeflerine ulaşmak.",
        "method_preemptive_best_when": "Bazı hedeflerin diğerlerine göre her zaman öncelikli olması gerektiğinde.",
        "method_preemptive_description": "Amaçları sözlük sırasıyla optimize eder ve yüksek öncelikli hedefleri önce korur.",
        "method_preemptive_example": "Önce kaliteyi, sonra teslimatı karşılamak ve ardından maliyeti azaltmak.",
        "method_guide_title": "Hangi Yöntemi Seçmelisiniz?",
        "method_guide_weighted_sum": "Kriterler ölçülebilir olduğunda ve göreli önemleri güvenilir ağırlıklarla ifade edilebildiğinde Ağırlıklı Toplamı seçin.",
        "method_guide_topsis": "Tüm kriterlerde ideal profile en yakın dengeli tedarikçiyi istediğinizde TOPSIS'i seçin.",
        "method_guide_goal_programming": "Maliyet, kalite ve teslimat hedeflerine mümkün olduğunca yaklaşmanız gerektiğinde Hedef Programlamayı seçin.",
        "method_guide_preemptive": "İş öncelikleri sıralı olduğunda ve yüksek öncelikli hedeflerden vazgeçilemediğinde Öncelikli Optimizasyonu seçin.",
        "kpi_total_suppliers": "Toplam Tedarikçi",
        "kpi_total_parts": "Toplam Parça",
        "kpi_active_assessments": "Aktif Değerlendirme",
        "kpi_scenarios": "Optimizasyon Senaryosu",
        "osa_heading": "OSA kriterleri nedir?",
        "osa_text": """
**Yerinde Değerlendirme (OSA)** kriterleri, bir tedarikçiyi fiziksel veya uzaktan değerlendirmek için kullanılan yapılandırılmış ölçütlerdir. Kalite yönetimi, üretim yetkinliği, proses kontrolleri, teslimat performansı, finansal dayanıklılık, sürdürülebilirlik, iş güvenliği ve uyum gibi alanları kapsayabilir.

Her kriterin net bir tanımı, puanlama ölçeği, kanıt gereksinimi, sorumlu değerlendiricisi ve ağırlıklandırma politikası olmalıdır. Tutarlı tanımlar, tedarikçi karşılaştırmalarını daha şeffaf ve denetlenebilir kılar.
""",
        "algorithms_heading": "Optimizasyon yöntemleri",
        "algorithms_text": """
- **TOPSIS:** Tedarikçileri ideal ve ideal olmayan çözümlere olan uzaklıklarına göre sıralar.
- **Ağırlıklı Toplam:** Kriterleri karşılaştırılabilir faydalara dönüştürür ve ağırlıklı toplam puan hesaplar.
- **Öncelikli Optimizasyon:** Öncelikleri sözlük sırasıyla uygular; yüksek öncelikli hedefleri korur.
- **Hedef Programlama:** Kalite, teslimat ve maliyet hedeflerinden ağırlıklı sapmaları en aza indirir.
""",
        "parts_title": "Parça Veritabanı",
        "parts_subtitle": "Tedarikçi değerlendirme ve optimizasyon süreçlerinde kullanılan parça bilgilerini yönetin.",
        "parts_kpi_total": "Toplam Parça",
        "parts_kpi_candidate_suppliers": "Aday Tedarikçiler",
        "parts_kpi_qualified_suppliers": "Nitelikli Tedarikçiler",
        "parts_kpi_active_rfqs": "Aktif RFQ",
        "parts_filter_number": "Parça Numarasında Ara",
        "parts_filter_number_placeholder": "örn. DT-INT-001",
        "parts_filter_description": "Parça Açıklamasında Ara",
        "parts_filter_description_placeholder": "örn. Interior Trims",
        "parts_filter_plant": "Fabrika",
        "parts_filter_sop_year": "SOP Yılı",
        "parts_all_plants": "Tüm Fabrikalar",
        "parts_all_years": "Tüm Yıllar",
        "parts_table_title": "Parça Kataloğu",
        "parts_table_count": "Toplam {total} parçanın {shown} tanesi gösteriliyor",
        "parts_select_hint": "Parça metadatasını ve tedarikçi kapsamını görmek için bir satır seçin.",
        "parts_no_matches": "Seçilen filtrelerle eşleşen parça bulunamadı.",
        "parts_detail_title": "Parça Detayı",
        "parts_candidate_suppliers": "Aday Tedarikçiler",
        "parts_qualified_suppliers": "Nitelikli Tedarikçiler",
        "parts_no_selection": "Detay panelini açmak için tablodan bir parça seçin.",
        "part_number": "Parça Numarası",
        "part_description": "Parça Açıklaması",
        "plant": "Fabrika",
        "sop_year": "SOP Yılı",
        "lifetime_years": "Kullanım Ömrü (Yıl)",
        "annual_volume": "Yıllık Hacim",
        "budget": "Bütçe (€)",
        "target_cost": "Hedef Maliyet (€)",
        "supplier_column_name": "Tedarikçi Adı",
        "supplier_column_location": "Konum",
        "supplier_column_capacity": "Kapasite",
        "supplier_column_part": "Tedarik Edilen Parça",
        "supplier_title": "Tedarikçi Veritabanı",
        "supplier_subtitle": "Tedarikçi yeterlilik, maliyet, kalite ve teslimat performansı verilerini yönetin.",
        "supplier_kpi_candidate_suppliers": "Aday Tedarikçiler",
        "supplier_kpi_qualified_suppliers": "Nitelikli Tedarikçiler",
        "supplier_kpi_average_osa_score": "Ortalama OSA Puanı",
        "supplier_kpi_countries_represented": "Temsil Edilen Ülke",
        "supplier_filter_part_number": "Parça Numarası",
        "supplier_filter_part_description": "Parça Açıklaması",
        "supplier_filter_name": "Tedarikçi Adı",
        "supplier_filter_country": "Ülke",
        "supplier_filter_osa_status": "OSA Durumu",
        "supplier_filter_part_number_placeholder": "örn. DT-INT-001",
        "supplier_filter_part_description_placeholder": "örn. Interior Trims",
        "supplier_filter_name_placeholder": "örn. Grammer",
        "supplier_all_parts": "Tüm Parçalar",
        "supplier_all_countries": "Tüm Ülkeler",
        "supplier_all_statuses": "Tüm OSA Durumları",
        "supplier_table_title": "Tedarikçi Karşılaştırma Kataloğu",
        "supplier_table_count": "Toplam {total} tedarikçi kaydının {shown} tanesi gösteriliyor",
        "supplier_select_hint": "Tedarikçi profilini görmek için bir satır seçin veya aşağıdan yan yana karşılaştırma için kayıtları seçin.",
        "supplier_no_matches": "Seçilen filtrelerle eşleşen tedarikçi kaydı bulunamadı.",
        "supplier_detail_title": "Tedarikçi Detayı",
        "supplier_detail_no_selection": "Profil panelini açmak için tablodan bir tedarikçi kaydı seçin.",
        "supplier_section_basic_info": "Bölüm 1 · Temel Bilgiler",
        "supplier_section_osa_info": "Bölüm 2 · OSA Bilgileri",
        "supplier_section_commercial_info": "Bölüm 3 · Ticari Bilgiler",
        "supplier_section_quality_performance": "Bölüm 4 · Kalite Performansı",
        "supplier_section_delivery_performance": "Bölüm 5 · Teslimat Performansı",
        "supplier_section_awarding_readiness": "Bölüm 6 · Ödüllendirme Hazırlığı",
        "supplier_name": "Tedarikçi Adı",
        "supplier_code": "Tedarikçi Kodu",
        "country": "Ülke",
        "production_location": "Üretim Yeri",
        "supplier_plant": "Fabrika",
        "osa_score": "OSA Puanı",
        "osa_status": "OSA Durumu",
        "status_qualified": "Nitelikli",
        "status_not_qualified": "Nitelikli Değil",
        "unit_price": "Birim Fiyat (€)",
        "annual_cost": "Yıllık Maliyet (€)",
        "tooling_cost": "Kalıp Maliyeti (€)",
        "quality_score": "Kalite Puanı",
        "defect_rate": "Hata Oranı (%)",
        "warranty_claims": "Garanti Talepleri",
        "incoming_acceptance_rate": "Giriş Kabul Oranı (%)",
        "process_capability_cpk": "Proses Yeterliliği (Cpk)",
        "cart_days": "Düzeltici Faaliyet Yanıt Süresi (CART) (Gün)",
        "delivery_score": "Teslimat Puanı",
        "on_time_delivery": "Zamanında Teslimat (%)",
        "lead_time_days": "Teslimat Süresi (Gün)",
        "delivery_accuracy": "Teslimat Doğruluğu (%)",
        "readiness_standard_contract": "Standart Sözleşme Kabulü",
        "readiness_quality_certificate": "Kalite Sertifikası Durumu",
        "readiness_environmental_certificate": "Çevre Sertifikası Durumu",
        "readiness_supplier_risk": "Tedarikçi Risk Durumu",
        "readiness_fsrm": "FSRM Durumu",
        "status_approved": "Onaylandı",
        "status_review_required": "İnceleme Gerekli",
        "status_missing": "Eksik",
        "supplier_comparison_title": "Tedarikçi Karşılaştırma Görünümü",
        "supplier_comparison_select": "Karşılaştırılacak tedarikçi kayıtlarını seçin",
        "supplier_comparison_hint": "Yan yana karşılaştırma için en fazla üç tedarikçi kaydı seçin.",
        "osa_title": "OSA Değerlendirmesi",
        "osa_placeholder": "Bu modül geliştirme aşamasındadır.",
        "osa_subtitle": "Tedarikçi Yeterlilik ve Hazırlık Değerlendirmesi",
        "osa_kpi_assessed_suppliers": "Değerlendirilen Tedarikçiler",
        "osa_kpi_qualified_suppliers": "Nitelikli Tedarikçiler",
        "osa_kpi_not_qualified_suppliers": "Nitelikli Olmayan Tedarikçiler",
        "osa_kpi_threshold": "OSA Eşiği",
        "osa_filter_part_number": "Parça Numarası",
        "osa_filter_part_description": "Parça Açıklaması",
        "osa_filter_supplier_name": "Tedarikçi Adı",
        "osa_filter_country": "Ülke",
        "osa_filter_part_description_placeholder": "örn. Interior Trims",
        "osa_filter_supplier_name_placeholder": "örn. Grammer",
        "osa_all_parts": "Tüm Parçalar",
        "osa_all_countries": "Tüm Ülkeler",
        "osa_supplier_selection_title": "Tedarikçi Seçimi",
        "osa_supplier_selection": "Tedarikçi kaydı seçin",
        "osa_supplier_selection_hint": "Altı kategorili OSA değerlendirmesini incelemek ve düzenlemek için bir tedarikçi kaydı seçin.",
        "osa_no_matches": "Seçilen filtrelerle eşleşen tedarikçi kaydı bulunamadı.",
        "osa_supplier_info_title": "Tedarikçi Bilgileri",
        "osa_category_title": "OSA Değerlendirme Çerçevesi",
        "osa_weight": "Ağırlık",
        "osa_subcriteria": "Alt Kriterler",
        "osa_category_quality_system": "Kategori 1 · Kalite Sistemi",
        "osa_category_production_capability": "Kategori 2 · Üretim Yetkinliği",
        "osa_category_capacity_scalability": "Kategori 3 · Kapasite ve Ölçeklenebilirlik",
        "osa_category_technical_capability": "Kategori 4 · Teknik Yetkinlik",
        "osa_category_logistics_supply_chain": "Kategori 5 · Lojistik ve Tedarik Zinciri",
        "osa_category_management_compliance": "Kategori 6 · Yönetim ve Uyum",
        "osa_sub_iatf_16949": "IATF 16949",
        "osa_sub_iso_9001": "ISO 9001",
        "osa_sub_spc": "SPC",
        "osa_sub_traceability": "İzlenebilirlik",
        "osa_sub_pfmea": "PFMEA",
        "osa_sub_control_plan": "Kontrol Planı",
        "osa_sub_8d": "8D Problem Çözme",
        "osa_sub_standard_work": "Standart İş",
        "osa_sub_process_stability": "Proses Stabilitesi",
        "osa_sub_oee": "OEE",
        "osa_sub_tpm": "TPM",
        "osa_sub_5s": "5S",
        "osa_sub_preventive_maintenance": "Önleyici Bakım",
        "osa_sub_production_flow": "Üretim Akışı",
        "osa_sub_current_capacity": "Mevcut Kapasite",
        "osa_sub_available_capacity": "Kullanılabilir Kapasite",
        "osa_sub_capacity_utilization": "Kapasite Kullanımı",
        "osa_sub_additional_shift": "Ek Vardiya Yetkinliği",
        "osa_sub_scalability": "Ölçeklenebilirlik",
        "osa_sub_demand_growth": "Talep Artışı Yetkinliği",
        "osa_sub_similar_product": "Benzer Ürün Deneyimi",
        "osa_sub_engineering_support": "Mühendislik Desteği",
        "osa_sub_validation": "Validasyon Yetkinliği",
        "osa_sub_testing": "Test Yetkinliği",
        "osa_sub_manufacturing_technology": "Üretim Teknolojisi",
        "osa_sub_rd_support": "Ar-Ge Desteği",
        "osa_sub_material_flow": "Malzeme Akışı",
        "osa_sub_fifo": "FIFO",
        "osa_sub_inventory_management": "Stok Yönetimi",
        "osa_sub_packaging_management": "Ambalaj Yönetimi",
        "osa_sub_delivery_capability": "Teslimat Yetkinliği",
        "osa_sub_emergency_logistics": "Acil Lojistik Planı",
        "osa_sub_edi": "EDI Yetkinliği",
        "osa_sub_psc_rating": "PSC Değerlendirmesi",
        "osa_sub_saq_rating": "SAQ Değerlendirmesi",
        "osa_sub_corrective_actions": "Düzeltici Faaliyetler",
        "osa_sub_fsrm_status": "FSRM Durumu",
        "osa_sub_standard_contract": "Standart Sözleşme Kabulü",
        "osa_sub_supplier_risk": "Tedarikçi Risk Durumu",
        "osa_sub_business_continuity": "İş Sürekliliği Planı",
        "osa_breakdown_title": "OSA Dağılımı",
        "osa_breakdown_category": "Kategori",
        "osa_breakdown_weight": "Ağırlık",
        "osa_breakdown_score": "Puan",
        "osa_breakdown_contribution": "Katkı",
        "osa_formula": "OSA Puanı = Σ(Kategori Puanı × Ağırlık)",
        "osa_compliance_title": "Uyum ve Sertifikasyon",
        "osa_compliance_fsrm": "FSRM Durumu",
        "osa_compliance_iatf": "IATF 16949",
        "osa_compliance_iso_14001": "ISO 14001",
        "osa_compliance_tisax": "TISAX",
        "osa_compliance_contract": "Standart Sözleşme Kabulü",
        "status_watch": "İzleme",
        "osa_awarding_title": "Ödüllendirme Hazırlığı",
        "osa_ready_for_award": "Ödüle Hazır",
        "osa_task_for_awarding": "Ödüllendirme İçin Görev Gerekli",
        "osa_risk_board_approval": "Tedarikçi Risk Kurulu Onayı Gerekli",
        "osa_actions_title": "Aksiyon İhtiyacı",
        "osa_action_missing_certificates": "Eksik Sertifikalar",
        "osa_action_open_corrective_actions": "Açık Düzeltici Faaliyetler",
        "osa_action_missing_osa_ica": "Eksik OSA / ICA",
        "osa_action_restructuring": "Yeniden Yapılandırma Gereklilikleri",
        "osa_action_clear": "Acil aksiyon yok",
        "osa_action_required": "Aksiyon gerekli",
        "osa_result_title": "OSA Sonucu",
        "osa_total_score": "Toplam OSA Puanı",
        "osa_qualified": "NİTELİKLİ",
        "osa_not_qualified": "NİTELİKLİ DEĞİL",
        "osa_exclusion_note": "70 puanın altındaki tedarikçiler optimizasyona dahil edilmez.",
        "osa_critical_findings_title": "Kritik Bulgular · Veto Kuralları",
        "osa_veto_missing_iatf": "Eksik IATF 16949",
        "osa_veto_corrective_action": "Kritik Açık Düzeltici Faaliyet",
        "osa_veto_risk_board": "Tedarikçi Risk Kurulu Reddi",
        "osa_veto_capacity": "Yetersiz Üretim Kapasitesi",
        "osa_veto_business_continuity": "Eksik İş Sürekliliği Planı",
        "osa_veto_product_safety": "Ürün Güvenliği İhlali",
        "osa_veto_triggered": "Veto tetiklendi",
        "osa_veto_clear": "Temiz",
        "optimization_title": "Tedarikçi Optimizasyonu",
        "optimization_description": "Gelişmiş çok kriterli karar verme yöntemleriyle tedarikçileri değerlendirin, sıralayın ve optimize edin.",
        "workflow_parts_database": "Parça Veritabanı",
        "workflow_supplier_database": "Tedarikçi Veritabanı",
        "workflow_osa_assessment": "OSA Değerlendirmesi",
        "workflow_supplier_optimization": "Tedarikçi Optimizasyonu",
        "workflow_results_reports": "Sonuçlar ve Raporlar",
        "optimization_step_part_selection": "Adım 1 · Parça Seçimi",
        "optimization_step_method_selection": "Adım 2 · Optimizasyon Yöntemi",
        "optimization_step_data_import": "Adım 3 · Tedarikçi Verisi İçe Aktarma ve Önizleme",
        "optimization_step_qualified_filter": "Adım 4 · Nitelikli Tedarikçi Filtresi",
        "optimization_step_criteria": "Adım 5 · Kriter Oluşturma",
        "optimization_step_configuration": "Adım 6 · Yöntem Yapılandırması",
        "optimization_step_run": "Adım 7 · Optimizasyonu Çalıştırma",
        "optimization_step_results": "Adım 8 · Optimizasyon Sonuçları",
        "optimization_step_visualization": "Adım 9 · Görselleştirme",
        "optimization_step_decision": "Adım 10 · Karar Özeti",
        "optimization_step_reports": "Adım 11 · Rapor Dışa Aktarma",
        "optimization_part_selection": "Parça seçin",
        "optimization_part_selection_hint": "Değerlendirilecek talebi, bütçeyi ve tedarikçi havuzunu belirlemek için parçayı seçin.",
        "optimization_all_parts": "Parça seçin",
        "optimization_method_selection": "Optimizasyon yöntemi seçin",
        "optimization_method_selection_hint": "İş durumuna en uygun karar mantığını seçin.",
        "optimization_method_weighted_sum_summary": "Tüm kriterleri aynı anda dengeler.",
        "optimization_method_weighted_sum_best_when": "Kriterler arasındaki ödünleşimler kabul edilebilir olduğunda uygundur.",
        "optimization_method_topsis_summary": "Tedarikçileri ideal tedarikçiye uzaklıklarına göre sıralar.",
        "optimization_method_topsis_best_when": "Birincil amaç alternatifleri sıralamak olduğunda uygundur.",
        "optimization_method_goal_programming_summary": "Önceden belirlenen hedefleri en iyi karşılayan tedarikçi dağılımını bulur.",
        "optimization_method_goal_programming_best_when": "Maliyet, kalite ve teslimat hedeflerine ulaşılması gerektiğinde uygundur.",
        "optimization_method_preemptive_summary": "Yüksek öncelikli kriterlerin baskın olduğu katı öncelikler uygular.",
        "optimization_method_preemptive_best_when": "İş öncelikleri arasında ödünleşim yapılamadığında uygundur.",
        "optimization_upload_label": "Tedarikçi veri kümesi yükleyin",
        "optimization_upload_help": "Desteklenen formatlar: XLSX, XLS ve CSV.",
        "optimization_default_data": "Dosya yüklenene kadar yerleşik tedarikçi veri kümesi kullanılıyor.",
        "optimization_preview_title": "Tedarikçi Veri Kümesi Önizlemesi",
        "optimization_preview_caption": "Önizleme, tedarikçi performans alanlarını ve OSA yeterlilik kapısını içerir.",
        "optimization_expected_columns": "Beklenen sütunlar: Part Number, Supplier, Annual Cost, Defect Rate, Warranty Claims, Incoming Acceptance Rate, Process Capability, Corrective Action Response Time, On-Time Delivery, Lead Time, Delivery Accuracy, OSA Score, Country.",
        "optimization_missing_required_columns": "Yüklenen veri kümesinde gerekli sütunlar eksik: {columns}",
        "optimization_invalid_numeric_columns": "Şu optimizasyon alanlarında boş veya sayısal olmayan değerler var: {columns}",
        "optimization_finite_numeric": "Optimizasyon girdi değerleri sonlu olmalıdır.",
        "optimization_invalid_capacity": "Capacity değerleri negatif olamaz ve sonlu olmalıdır.",
        "optimization_missing_annual_volume": "Yüklenen şu parçalar için Annual Volume belirlenemedi: {parts}. Annual Volume sütunu ekleyin veya eşleşen bir Part Description sağlayın.",
        "optimization_upload_error": "Yüklenen veri kümesi hazırlanamadı: {error}",
        "optimization_capacity_assumption": "Capacity verilmediği için muhafazakâr dağıtım kapasitesi olarak Annual Volume kullanılıyor. Capacity yalnızca kısıttır, optimizasyon kriteri değildir.",
        "optimization_part_match_fallback": "Tam bir Part Number eşleşmesi bulunamadı. Tedarikçi satırları Part Description kullanılarak eşleştirildi.",
        "optimization_gate_title": "Nitelikli Tedarikçi Filtresi · Kapı Kuralı",
        "optimization_osa_threshold": "OSA Eşiği",
        "optimization_use_only_qualified": "☑ Yalnızca Nitelikli Tedarikçileri Kullan",
        "optimization_osa_gate_note": "OSA bir optimizasyon kriteri değildir. Yalnızca yeterlilik kapısı olarak kullanılır; bu kural etkin olduğunda OSA Puanı < 70 olan tedarikçiler dışlanır.",
        "optimization_three_criteria_note": "Optimizasyon yalnızca Maliyet, Kalite ve Teslimat kriterlerini kullanır. Kapasite ve Teknik Yetkinlik OSA içinde değerlendirilir ve optimizasyon kriteri değildir.",
        "optimization_criteria_title": "Kriter Oluşturma",
        "optimization_cost_definition": "Maliyet = Yıllık Tedarikçi Maliyeti; ön normalizasyon uygulanmaz.",
        "optimization_quality_definition": "Kalite Puanı = Hata Oranı, Garanti Talepleri, Giriş Kabul Oranı ve Proses Yeterliliğinden ağırlıklı alt kriter hesabı.",
        "optimization_delivery_definition": "Teslimat Puanı = Zamanında Teslimat, Teslimat Süresi ve Teslimat Doğruluğundan ağırlıklı alt kriter hesabı.",
        "optimization_method_configuration": "Yöntem Yapılandırması",
        "optimization_weights_title": "Kriter Ağırlıkları",
        "optimization_weight_cost": "Maliyet ağırlığı",
        "optimization_weight_quality": "Kalite ağırlığı",
        "optimization_weight_delivery": "Teslimat ağırlığı",
        "optimization_weight_total": "Ağırlık toplamı",
        "optimization_weight_validation": "Optimizasyon çalışmadan önce ağırlıkların toplamı 1,00 olmalıdır.",
        "optimization_topsis_title": "TOPSIS Yapılandırması",
        "optimization_topsis_cost": "Maliyet, maliyet kriteri olarak yapılandırılır (düşük değer daha iyidir).",
        "optimization_topsis_benefits": "Kalite ve Teslimat, fayda kriteri olarak yapılandırılır (yüksek değer daha iyidir).",
        "optimization_topsis_normalization": "Mevcut TOPSIS arka ucu Öklid normalizasyonu ve ideal/ideal olmayan çözüm sıralamasını uygular.",
        "optimization_targets_title": "Hedef Değerler",
        "optimization_target_cost": "Hedef Maliyet (€)",
        "optimization_target_cost_unit": "Hedef Maliyet, yüklenen tedarikçi veri kümesiyle aynı yıllık maliyet birimini kullanır.",
        "optimization_target_quality": "Hedef Kalite Puanı",
        "optimization_target_delivery": "Hedef Teslimat Puanı",
        "optimization_preemptive_title": "Öncelik Sıralaması",
        "optimization_priority_1": "Öncelik 1",
        "optimization_priority_2": "Öncelik 2",
        "optimization_priority_3": "Öncelik 3",
        "optimization_priority_cost": "Maliyet",
        "optimization_priority_quality": "Kalite",
        "optimization_priority_delivery": "Teslimat",
        "optimization_priority_note": "Seçilen sıra iş önceliği niyetini belgeler; mevcut öncelikli çözücü uygulaması değiştirilmeden korunur.",
        "optimization_run_button": "🚀 Optimizasyonu Çalıştır",
        "optimization_no_qualified_suppliers": "OSA yeterlilik kapısı uygulandıktan sonra tedarikçi kalmadı.",
        "optimization_no_part_data": "Seçilen parça için tedarikçi verisi bulunamadı.",
        "optimization_duplicate_suppliers": "Seçilen parça için her tedarikçi yalnızca bir kez bulunmalıdır.",
        "optimization_priority_validation": "Öncelik 1, Öncelik 2 ve Öncelik 3 birbirinden farklı olmalıdır.",
        "optimization_results_title": "Optimizasyon Sonuçları",
        "optimization_recommended_supplier": "Önerilen Tedarikçi",
        "optimization_final_score": "Nihai Puan",
        "optimization_supplier_ranking": "Tedarikçi Sıralaması",
        "optimization_ranking_bar_chart": "Tedarikçi Sıralaması Çubuk Grafiği",
        "optimization_radar_chart": "Kriter Radar Grafiği",
        "optimization_no_results": "Sonuçları ve görselleştirmeleri görmek için optimizasyonu çalıştırın.",
        "optimization_decision_summary": "Karar Özeti",
        "optimization_method_used": "Kullanılan Yöntem",
        "optimization_selected_part": "Seçilen Parça",
        "optimization_evaluated_suppliers": "Değerlendirilen Tedarikçiler",
        "optimization_qualified_suppliers": "Nitelikli Tedarikçiler",
        "optimization_cost_score": "Maliyet",
        "optimization_quality_score": "Kalite",
        "optimization_delivery_score": "Teslimat",
        "optimization_osa_score": "OSA",
        "optimization_export_reports": "Raporları Dışa Aktar",
        "optimization_export_pdf": "📄 PDF Raporunu Dışa Aktar",
        "optimization_export_excel": "📊 Excel Raporunu Dışa Aktar",
        "optimization_pdf_placeholder": "PDF rapor oluşturma, kurumsal raporlama servisine entegrasyon için hazırdır.",
        "optimization_excel_error": "Excel raporu oluşturulamadı: {error}",
        "optimization_report_generated_at": "Oluşturulma zamanı",
        "optimization_results_csv": "Optimizasyon sonuçlarını indir (CSV)",
        "optimization_radar_cost": "Maliyet Performansı",
        "optimization_radar_quality": "Kalite",
        "optimization_radar_delivery": "Teslimat",
        "optimization_radar_normalized_note": "0–100 normalize görünüm",
        "choose_analysis": "Analiz seçin",
        "add_supplier_data": "Tedarikçi verisi ekleyin",
        "optimization_method": "Optimizasyon yöntemi",
        "supplier_data": "Tedarikçi verisi",
        "supplier_data_caption": "Bu analizde kullanılacak değerleri inceleyin veya düzenleyin.",
        "model_configuration": "Model yapılandırması",
        "model_configuration_caption": "Ağırlıklar otomatik olarak normalize edilir. Yüksek veya düşük değerin daha iyi olduğunu seçin.",
        "sample_caption": "Örnek veri kümesi kullanılıyor · Değiştirmek için CSV veya Excel yükleyin",
        "using_file": "Kullanılıyor",
        "analysis_help": "Tekil tedarikçileri sıralayın veya kapasite kısıtlı bölüştürmeyi optimize edin.",
        "allocation_settings": "Dağıtım ayarları",
        "capacity_warning": "Bu yöntemi kullanmak için Capacity sütunu ekleyin.",
        "demand_units": "Talep birimi",
        "cost_criterion": "Maliyet kriteri",
        "quality_criterion": "Kalite kriteri",
        "delivery_criterion": "Teslimat kriteri",
        "minimum_quality_target": "Minimum kalite hedefi",
        "maximum_delivery_target": "Maksimum teslimat hedefi",
        "maximum_cost_target": "Maksimum maliyet hedefi",
        "goal_weights": "Hedef programlama ağırlıkları",
        "quality_goal_weight": "Kalite hedefi ağırlığı",
        "delivery_goal_weight": "Teslimat hedefi ağırlığı",
        "cost_goal_weight": "Maliyet hedefi ağırlığı",
        "run_analysis": "Analizi çalıştır",
        "ranking_results": "Sıralama sonuçları",
        "higher_score": "Daha yüksek {score} değeri daha iyi tedarikçiyi gösterir.",
        "solution_metrics": "Çözüm metrikleri",
        "lexicographic_results": "Sözlük sıralı öncelik sonuçları",
        "download_results": "Nihai sonuçları indir (CSV)",
        "upload_supplier_data": "Tedarikçi verisi yükleyin",
        "upload_supplier_data_help": "Supplier, Cost, Quality, Delivery ve Capacity gibi sütunları kullanın.",
        "error_read_upload": "Yüklenen dosya okunamadı: {error}",
        "error_empty_criterion": "En az bir kriter gereklidir; Capacity bir kriter değildir.",
        "error_capacity_required": "Bu yöntemi çalıştırmadan önce Capacity sütunu ekleyin.",
        "error_capacity_column": "Öncelikli Optimizasyon ve Hedef Programlama için Capacity sütunu gereklidir.",
        "error_weight_required": "En az bir kriter ağırlığı sıfırdan büyük olmalıdır.",
        "error_goal_weights": "En az bir hedef programlama ağırlığı sıfırdan büyük olmalıdır.",
        "error_analysis": "Analiz tamamlanamadı: {error}",
        "choose_method_review": "Bir yöntem seçin, model yapılandırmasını inceleyin ve analizi çalıştırın.",
        "error_table_supplier_column": "Tabloda Supplier sütunu bulunmalıdır.",
        "error_enter_supplier": "En az bir tedarikçi girin.",
        "error_supplier_blank": "Tedarikçi adları boş bırakılamaz.",
        "error_supplier_unique": "Tedarikçi adları boş olamaz ve benzersiz olmalıdır.",
        "error_numeric_criterion": "En az bir sayısal kriter sütunu ekleyin.",
        "error_invalid_numeric_columns": "Şu sütunlarda boş veya sayısal olmayan değerler var: {columns}",
        "error_finite_numeric": "Sayısal değerler sonlu olmalıdır.",
        "criterion_weight": "{criterion} ağırlığı",
        "criterion_direction": "{criterion} yönü",
        "impact_benefit": "Fayda",
        "impact_cost": "Maliyet",
        "score_topsis": "TOPSIS puanı",
        "score_weighted_sum": "Ağırlıklı toplam puanı",
        "preemptive_allocation": "Öncelikli dağıtım",
        "goal_programming_allocation": "Hedef programlama dağıtımı",
        "metric_average_cost": "Ortalama Maliyet",
        "metric_average_quality": "Ortalama Kalite",
        "metric_average_delivery": "Ortalama Teslimat",
        "metric_quality_shortfall": "Kalite Açığı",
        "metric_delivery_excess": "Teslimat Fazlası",
        "metric_optimal_quality_shortfall": "En İyi Kalite Açığı",
        "metric_optimal_delivery_excess_after_p1": "P1 Sonrası En İyi Teslimat Fazlası",
        "metric_normalized_objective": "Normalize Edilmiş Amaç",
        "column_score": "Puan",
        "column_rank": "Sıra",
        "column_allocation_units": "Dağıtım Birimleri",
        "column_allocation_share": "Dağıtım Payı",
        "column_value": "Değer",
        "method_topsis": "TOPSIS",
        "method_weighted_sum": "Ağırlıklı Toplam",
        "method_preemptive": "Öncelikli Optimizasyon",
        "method_goal_programming": "Hedef Programlama",
    },
    "DE": {
        "language": "Sprache",
        "navigation": "Navigation",
        "page_homepage": "Startseite",
        "page_parts_database": "Teiledatenbank",
        "page_supplier_database": "Lieferantendatenbank",
        "page_osa_assessment": "OSA-Bewertung",
        "page_supplier_optimization": "Lieferantenoptimierung",
        "homepage_title": "Portal zur Lieferantenauswahl-Optimierung",
        "homepage_caption": "Entscheidungsunterstützung für Lieferanten-, Teile- und OSA-Analysen.",
        "homepage_subtitle": "Ein optimierungsbasiertes Entscheidungsunterstützungssystem zur Lieferantenbewertung, -rangfolge und -auswahl.",
        "homepage_kpi_parts": "Teile",
        "homepage_kpi_candidate_suppliers": "Kandidatenlieferanten",
        "homepage_kpi_qualified_suppliers": "Qualifizierte Lieferanten",
        "homepage_kpi_optimization_models": "Optimierungsmodelle",
        "framework_title": "Rahmenwerk zur Lieferantenauswahl",
        "framework_osa_assessment": "OSA-Bewertung",
        "framework_supplier_qualification": "Lieferantenqualifikation",
        "framework_cost_evaluation": "Kostenbewertung",
        "framework_quality_evaluation": "Qualitätsbewertung",
        "framework_delivery_evaluation": "Lieferbewertung",
        "framework_supplier_ranking": "Lieferantenranking",
        "optimization_methods_title": "Optimierungsmethoden",
        "method_best_when": "Am besten geeignet, wenn",
        "method_description": "Beschreibung",
        "method_example": "Beispiel",
        "method_weighted_sum_best_when": "Kriterien messbar sind und ihre Bedeutung durch stabile Gewichte ausgedrückt werden kann.",
        "method_weighted_sum_description": "Bewertet jeden Lieferanten anhand normalisierter Kriterien und fasst sie zu einem gewichteten Gesamtscore zusammen.",
        "method_weighted_sum_example": "Bester Gesamtlieferant bei 40 % Kosten, 35 % Qualität und 25 % Lieferung.",
        "method_topsis_best_when": "Sie eine ausgewogene Auswahl nahe am idealen Lieferanten wünschen.",
        "method_topsis_description": "Ordnet Lieferanten nach ihrer Entfernung zur idealen und anti-idealen Lösung.",
        "method_topsis_example": "Ein Lieferant, der über alle Kriterien hinweg stark ist, ohne bei einem einzelnen Kriterium führend zu sein.",
        "method_goal_programming_best_when": "Sie konkrete Kosten-, Qualitäts- und Lieferziele haben.",
        "method_goal_programming_description": "Minimiert gewichtete Abweichungen von Zielwerten unter Berücksichtigung der Kapazitäten.",
        "method_goal_programming_example": "Qualität ≥ 90, Lieferung ≤ 7 Tage und Kosten ≤ 110 € erreichen.",
        "method_preemptive_best_when": "Einige Ziele immer Vorrang vor anderen haben müssen.",
        "method_preemptive_description": "Optimiert Ziele lexikografisch und schützt höher priorisierte Ziele zuerst.",
        "method_preemptive_example": "Zuerst Qualität erfüllen, dann Lieferung und anschließend Kosten minimieren.",
        "method_guide_title": "Welche Methode sollten Sie wählen?",
        "method_guide_weighted_sum": "Wählen Sie die gewichtete Summe, wenn Kriterien messbar sind und ihre relative Bedeutung mit zuverlässigen Gewichten dargestellt werden kann.",
        "method_guide_topsis": "Wählen Sie TOPSIS, wenn Sie einen ausgewogenen Lieferanten wünschen, der dem idealen Profil über alle Kriterien hinweg am nächsten kommt.",
        "method_guide_goal_programming": "Wählen Sie Zielprogrammierung, wenn Kosten-, Qualitäts- und Lieferziele möglichst genau erreicht werden müssen.",
        "method_guide_preemptive": "Wählen Sie präemptive Optimierung, wenn Geschäftsprioritäten geordnet sind und höher priorisierte Ziele nicht geopfert werden dürfen.",
        "kpi_total_suppliers": "Lieferanten gesamt",
        "kpi_total_parts": "Teile gesamt",
        "kpi_active_assessments": "Aktive Bewertungen",
        "kpi_scenarios": "Optimierungsszenarien",
        "osa_heading": "Was sind OSA-Kriterien?",
        "osa_text": """
**On-Site Assessment (OSA)**-Kriterien sind strukturierte Messgrößen zur physischen oder remote durchgeführten Bewertung eines Lieferanten. Sie können Qualitätsmanagement, Produktionsfähigkeit, Prozesskontrollen, Lieferleistung, finanzielle Stabilität, Nachhaltigkeit, Arbeitssicherheit und Compliance abdecken.

Jedes Kriterium sollte eine klare Definition, Bewertungsskala, Nachweisanforderung, verantwortliche Person und Gewichtung enthalten. Einheitliche Definitionen machen Lieferantenvergleiche transparenter und nachvollziehbarer.
""",
        "algorithms_heading": "Optimierungsmethoden",
        "algorithms_text": """
- **TOPSIS:** Bewertet Lieferanten anhand ihrer Distanz zu einer idealen und einer Anti-Ideal-Lösung.
- **Gewichtete Summe:** Überführt Kriterien in vergleichbare Nutzenwerte und berechnet einen gewichteten Gesamtscore.
- **Präemptive Optimierung:** Wendet Prioritäten lexikografisch an und schützt höher priorisierte Ziele.
- **Zielprogrammierung:** Minimiert gewichtete Abweichungen von Qualitäts-, Liefer- und Kostenzielen.
""",
        "parts_title": "Teiledatenbank",
        "parts_subtitle": "Verwalten Sie Teileinformationen für Lieferantenbewertung und Optimierungsabläufe.",
        "parts_kpi_total": "Teile gesamt",
        "parts_kpi_candidate_suppliers": "Kandidatenlieferanten",
        "parts_kpi_qualified_suppliers": "Qualifizierte Lieferanten",
        "parts_kpi_active_rfqs": "Aktive RFQs",
        "parts_filter_number": "Teilenummer suchen",
        "parts_filter_number_placeholder": "z. B. DT-INT-001",
        "parts_filter_description": "Teilebeschreibung suchen",
        "parts_filter_description_placeholder": "z. B. Interior Trims",
        "parts_filter_plant": "Werk",
        "parts_filter_sop_year": "SOP-Jahr",
        "parts_all_plants": "Alle Werke",
        "parts_all_years": "Alle Jahre",
        "parts_table_title": "Teilekatalog",
        "parts_table_count": "{shown} von {total} Teilen werden angezeigt",
        "parts_select_hint": "Wählen Sie eine Zeile aus, um vollständige Teiledaten und Lieferantenabdeckung zu sehen.",
        "parts_no_matches": "Keine Teile entsprechen den ausgewählten Filtern.",
        "parts_detail_title": "Teiledetails",
        "parts_candidate_suppliers": "Kandidatenlieferanten",
        "parts_qualified_suppliers": "Qualifizierte Lieferanten",
        "parts_no_selection": "Wählen Sie ein Teil aus der Tabelle, um das Detailpanel zu öffnen.",
        "part_number": "Teilenummer",
        "part_description": "Teilebeschreibung",
        "plant": "Werk",
        "sop_year": "SOP-Jahr",
        "lifetime_years": "Lebensdauer (Jahre)",
        "annual_volume": "Jahresvolumen",
        "budget": "Budget (€)",
        "target_cost": "Zielkosten (€)",
        "supplier_column_name": "Lieferantenname",
        "supplier_column_location": "Standort",
        "supplier_column_capacity": "Kapazität",
        "supplier_column_part": "Geliefertes Teil",
        "supplier_title": "Lieferantendatenbank",
        "supplier_subtitle": "Verwalten Sie Daten zur Lieferantenqualifikation sowie zu Kosten-, Qualitäts- und Lieferleistung.",
        "supplier_kpi_candidate_suppliers": "Kandidatenlieferanten",
        "supplier_kpi_qualified_suppliers": "Qualifizierte Lieferanten",
        "supplier_kpi_average_osa_score": "Durchschnittlicher OSA-Score",
        "supplier_kpi_countries_represented": "Vertretene Länder",
        "supplier_filter_part_number": "Teilenummer",
        "supplier_filter_part_description": "Teilebeschreibung",
        "supplier_filter_name": "Lieferantenname",
        "supplier_filter_country": "Land",
        "supplier_filter_osa_status": "OSA-Status",
        "supplier_filter_part_number_placeholder": "z. B. DT-INT-001",
        "supplier_filter_part_description_placeholder": "z. B. Interior Trims",
        "supplier_filter_name_placeholder": "z. B. Grammer",
        "supplier_all_parts": "Alle Teile",
        "supplier_all_countries": "Alle Länder",
        "supplier_all_statuses": "Alle OSA-Status",
        "supplier_table_title": "Lieferantenvergleichskatalog",
        "supplier_table_count": "{shown} von {total} Lieferantendatensätzen werden angezeigt",
        "supplier_select_hint": "Wählen Sie eine Zeile für das Lieferantenprofil oder Datensätze für den Seitenvergleich.",
        "supplier_no_matches": "Keine Lieferantendatensätze entsprechen den ausgewählten Filtern.",
        "supplier_detail_title": "Lieferantendetails",
        "supplier_detail_no_selection": "Wählen Sie einen Lieferantendatensatz aus der Tabelle, um das Profil zu öffnen.",
        "supplier_section_basic_info": "Abschnitt 1 · Basisinformationen",
        "supplier_section_osa_info": "Abschnitt 2 · OSA-Informationen",
        "supplier_section_commercial_info": "Abschnitt 3 · Kommerzielle Informationen",
        "supplier_section_quality_performance": "Abschnitt 4 · Qualitätsleistung",
        "supplier_section_delivery_performance": "Abschnitt 5 · Lieferleistung",
        "supplier_section_awarding_readiness": "Abschnitt 6 · Vergabebereitschaft",
        "supplier_name": "Lieferantenname",
        "supplier_code": "Lieferantencode",
        "country": "Land",
        "production_location": "Produktionsstandort",
        "supplier_plant": "Werk",
        "osa_score": "OSA-Score",
        "osa_status": "OSA-Status",
        "status_qualified": "Qualifiziert",
        "status_not_qualified": "Nicht qualifiziert",
        "unit_price": "Stückpreis (€)",
        "annual_cost": "Jahreskosten (€)",
        "tooling_cost": "Werkzeugkosten (€)",
        "quality_score": "Qualitätsscore",
        "defect_rate": "Fehlerquote (%)",
        "warranty_claims": "Gewährleistungsfälle",
        "incoming_acceptance_rate": "Eingangsannahmequote (%)",
        "process_capability_cpk": "Prozessfähigkeit (Cpk)",
        "cart_days": "Reaktionszeit für Korrekturmaßnahmen (CART) (Tage)",
        "delivery_score": "Lieferscore",
        "on_time_delivery": "Termintreue (%)",
        "lead_time_days": "Lieferzeit (Tage)",
        "delivery_accuracy": "Liefergenauigkeit (%)",
        "readiness_standard_contract": "Akzeptanz des Standardvertrags",
        "readiness_quality_certificate": "Status Qualitätszertifikat",
        "readiness_environmental_certificate": "Status Umweltzertifikat",
        "readiness_supplier_risk": "Lieferantenrisikostatus",
        "readiness_fsrm": "FSRM-Status",
        "status_approved": "Genehmigt",
        "status_review_required": "Prüfung erforderlich",
        "status_missing": "Fehlt",
        "supplier_comparison_title": "Lieferantenvergleich",
        "supplier_comparison_select": "Lieferantendatensätze zum Vergleich auswählen",
        "supplier_comparison_hint": "Wählen Sie bis zu drei Lieferantendatensätze für einen Seitenvergleich.",
        "osa_title": "OSA-Bewertung",
        "osa_placeholder": "Dieses Modul befindet sich in Entwicklung.",
        "osa_subtitle": "Bewertung von Lieferantenqualifikation und Vergabebereitschaft",
        "osa_kpi_assessed_suppliers": "Bewertete Lieferanten",
        "osa_kpi_qualified_suppliers": "Qualifizierte Lieferanten",
        "osa_kpi_not_qualified_suppliers": "Nicht qualifizierte Lieferanten",
        "osa_kpi_threshold": "OSA-Schwelle",
        "osa_filter_part_number": "Teilenummer",
        "osa_filter_part_description": "Teilebeschreibung",
        "osa_filter_supplier_name": "Lieferantenname",
        "osa_filter_country": "Land",
        "osa_filter_part_description_placeholder": "z. B. Interior Trims",
        "osa_filter_supplier_name_placeholder": "z. B. Grammer",
        "osa_all_parts": "Alle Teile",
        "osa_all_countries": "Alle Länder",
        "osa_supplier_selection_title": "Lieferantenauswahl",
        "osa_supplier_selection": "Lieferantendatensatz auswählen",
        "osa_supplier_selection_hint": "Wählen Sie einen Lieferantendatensatz, um die sechskategorige OSA-Bewertung zu prüfen und anzupassen.",
        "osa_no_matches": "Keine Lieferantendatensätze entsprechen den ausgewählten Filtern.",
        "osa_supplier_info_title": "Lieferanteninformationen",
        "osa_category_title": "OSA-Bewertungsrahmen",
        "osa_weight": "Gewicht",
        "osa_subcriteria": "Unterkriterien",
        "osa_category_quality_system": "Kategorie 1 · Qualitätssystem",
        "osa_category_production_capability": "Kategorie 2 · Produktionsfähigkeit",
        "osa_category_capacity_scalability": "Kategorie 3 · Kapazität und Skalierbarkeit",
        "osa_category_technical_capability": "Kategorie 4 · Technische Fähigkeit",
        "osa_category_logistics_supply_chain": "Kategorie 5 · Logistik und Lieferkette",
        "osa_category_management_compliance": "Kategorie 6 · Management und Compliance",
        "osa_sub_iatf_16949": "IATF 16949",
        "osa_sub_iso_9001": "ISO 9001",
        "osa_sub_spc": "SPC",
        "osa_sub_traceability": "Rückverfolgbarkeit",
        "osa_sub_pfmea": "PFMEA",
        "osa_sub_control_plan": "Control Plan",
        "osa_sub_8d": "8D-Problemlösung",
        "osa_sub_standard_work": "Standardarbeit",
        "osa_sub_process_stability": "Prozessstabilität",
        "osa_sub_oee": "OEE",
        "osa_sub_tpm": "TPM",
        "osa_sub_5s": "5S",
        "osa_sub_preventive_maintenance": "Vorbeugende Instandhaltung",
        "osa_sub_production_flow": "Produktionsfluss",
        "osa_sub_current_capacity": "Aktuelle Kapazität",
        "osa_sub_available_capacity": "Verfügbare Kapazität",
        "osa_sub_capacity_utilization": "Kapazitätsauslastung",
        "osa_sub_additional_shift": "Fähigkeit für zusätzliche Schichten",
        "osa_sub_scalability": "Skalierbarkeit",
        "osa_sub_demand_growth": "Fähigkeit zur Nachfrageausweitung",
        "osa_sub_similar_product": "Erfahrung mit ähnlichen Produkten",
        "osa_sub_engineering_support": "Engineering-Unterstützung",
        "osa_sub_validation": "Validierungsfähigkeit",
        "osa_sub_testing": "Testfähigkeit",
        "osa_sub_manufacturing_technology": "Fertigungstechnologie",
        "osa_sub_rd_support": "F&E-Unterstützung",
        "osa_sub_material_flow": "Materialfluss",
        "osa_sub_fifo": "FIFO",
        "osa_sub_inventory_management": "Bestandsmanagement",
        "osa_sub_packaging_management": "Verpackungsmanagement",
        "osa_sub_delivery_capability": "Lieferfähigkeit",
        "osa_sub_emergency_logistics": "Notfalllogistikplan",
        "osa_sub_edi": "EDI-Fähigkeit",
        "osa_sub_psc_rating": "PSC-Bewertung",
        "osa_sub_saq_rating": "SAQ-Bewertung",
        "osa_sub_corrective_actions": "Korrekturmaßnahmen",
        "osa_sub_fsrm_status": "FSRM-Status",
        "osa_sub_standard_contract": "Akzeptanz des Standardvertrags",
        "osa_sub_supplier_risk": "Lieferantenrisikostatus",
        "osa_sub_business_continuity": "Business-Continuity-Plan",
        "osa_breakdown_title": "OSA-Aufschlüsselung",
        "osa_breakdown_category": "Kategorie",
        "osa_breakdown_weight": "Gewicht",
        "osa_breakdown_score": "Score",
        "osa_breakdown_contribution": "Beitrag",
        "osa_formula": "OSA-Score = Σ(Kategoriescore × Gewicht)",
        "osa_compliance_title": "Compliance und Zertifizierung",
        "osa_compliance_fsrm": "FSRM-Status",
        "osa_compliance_iatf": "IATF 16949",
        "osa_compliance_iso_14001": "ISO 14001",
        "osa_compliance_tisax": "TISAX",
        "osa_compliance_contract": "Akzeptanz des Standardvertrags",
        "status_watch": "Beobachtung",
        "osa_awarding_title": "Vergabebereitschaft",
        "osa_ready_for_award": "Bereit für Vergabe",
        "osa_task_for_awarding": "Aufgabe für Vergabe erforderlich",
        "osa_risk_board_approval": "Genehmigung des Lieferantenrisikorats erforderlich",
        "osa_actions_title": "Handlungsbedarf",
        "osa_action_missing_certificates": "Fehlende Zertifikate",
        "osa_action_open_corrective_actions": "Offene Korrekturmaßnahmen",
        "osa_action_missing_osa_ica": "Fehlende OSA / ICA",
        "osa_action_restructuring": "Anforderungen zur Umstrukturierung",
        "osa_action_clear": "Kein sofortiger Handlungsbedarf",
        "osa_action_required": "Maßnahme erforderlich",
        "osa_result_title": "OSA-Ergebnis",
        "osa_total_score": "OSA-Gesamtscore",
        "osa_qualified": "QUALIFIZIERT",
        "osa_not_qualified": "NICHT QUALIFIZIERT",
        "osa_exclusion_note": "Lieferanten unter 70 werden von der Optimierung ausgeschlossen.",
        "osa_critical_findings_title": "Kritische Befunde · Vetoregeln",
        "osa_veto_missing_iatf": "Fehlendes IATF 16949",
        "osa_veto_corrective_action": "Kritische offene Korrekturmaßnahme",
        "osa_veto_risk_board": "Ablehnung durch den Lieferantenrisikorat",
        "osa_veto_capacity": "Unzureichende Produktionskapazität",
        "osa_veto_business_continuity": "Fehlender Business-Continuity-Plan",
        "osa_veto_product_safety": "Verstoß gegen Produktsicherheit",
        "osa_veto_triggered": "Veto ausgelöst",
        "osa_veto_clear": "Unauffällig",
        "optimization_title": "Lieferantenoptimierung",
        "optimization_description": "Lieferanten mit fortgeschrittenen multikriteriellen Entscheidungsverfahren bewerten, rangieren und optimieren.",
        "workflow_parts_database": "Teiledatenbank",
        "workflow_supplier_database": "Lieferantendatenbank",
        "workflow_osa_assessment": "OSA-Bewertung",
        "workflow_supplier_optimization": "Lieferantenoptimierung",
        "workflow_results_reports": "Ergebnisse und Berichte",
        "optimization_step_part_selection": "Schritt 1 · Teileauswahl",
        "optimization_step_method_selection": "Schritt 2 · Optimierungsmethode",
        "optimization_step_data_import": "Schritt 3 · Lieferantendatenimport und Vorschau",
        "optimization_step_qualified_filter": "Schritt 4 · Filter qualifizierter Lieferanten",
        "optimization_step_criteria": "Schritt 5 · Kriterienkonstruktion",
        "optimization_step_configuration": "Schritt 6 · Methodenkonfiguration",
        "optimization_step_run": "Schritt 7 · Optimierung starten",
        "optimization_step_results": "Schritt 8 · Optimierungsergebnisse",
        "optimization_step_visualization": "Schritt 9 · Visualisierung",
        "optimization_step_decision": "Schritt 10 · Entscheidungszusammenfassung",
        "optimization_step_reports": "Schritt 11 · Berichtsexport",
        "optimization_part_selection": "Teil auswählen",
        "optimization_part_selection_hint": "Wählen Sie das Teil, um Bedarf, Budget und Lieferantenpool festzulegen.",
        "optimization_all_parts": "Teil auswählen",
        "optimization_method_selection": "Optimierungsmethode auswählen",
        "optimization_method_selection_hint": "Wählen Sie die Entscheidungslogik, die am besten zur Geschäftssituation passt.",
        "optimization_method_weighted_sum_summary": "Balanciert alle Kriterien gleichzeitig.",
        "optimization_method_weighted_sum_best_when": "Geeignet, wenn Zielkonflikte zwischen Kriterien akzeptabel sind.",
        "optimization_method_topsis_summary": "Rangiert Lieferanten nach ihrer Distanz zum idealen Lieferanten.",
        "optimization_method_topsis_best_when": "Geeignet, wenn die Rangfolge von Alternativen das Hauptziel ist.",
        "optimization_method_goal_programming_summary": "Findet die Lieferantenverteilung, die vorgegebene Ziele am besten erfüllt.",
        "optimization_method_goal_programming_best_when": "Geeignet, wenn Kosten-, Qualitäts- und Lieferziele erfüllt werden müssen.",
        "optimization_method_preemptive_summary": "Wendet strikte Prioritäten an, wobei höhere Prioritäten dominieren.",
        "optimization_method_preemptive_best_when": "Geeignet, wenn Geschäftsprioritäten nicht gegeneinander abgewogen werden dürfen.",
        "optimization_upload_label": "Lieferantendatensatz hochladen",
        "optimization_upload_help": "Unterstützte Formate: XLSX, XLS und CSV.",
        "optimization_default_data": "Bis zum Upload einer Datei wird der integrierte Lieferantendatensatz verwendet.",
        "optimization_preview_title": "Vorschau des Lieferantendatensatzes",
        "optimization_preview_caption": "Die Vorschau enthält Lieferantenleistungsfelder und das OSA-Qualifikationsgate.",
        "optimization_expected_columns": "Erwartete Spalten: Part Number, Supplier, Annual Cost, Defect Rate, Warranty Claims, Incoming Acceptance Rate, Process Capability, Corrective Action Response Time, On-Time Delivery, Lead Time, Delivery Accuracy, OSA Score, Country.",
        "optimization_missing_required_columns": "Im hochgeladenen Datensatz fehlen erforderliche Spalten: {columns}",
        "optimization_invalid_numeric_columns": "Diese Optimierungsfelder enthalten leere oder nicht numerische Werte: {columns}",
        "optimization_finite_numeric": "Optimierungseingaben müssen endlich sein.",
        "optimization_invalid_capacity": "Capacity-Werte müssen nicht negativ und endlich sein.",
        "optimization_missing_annual_volume": "Annual Volume konnte für diese hochgeladenen Teile nicht ermittelt werden: {parts}. Fügen Sie eine Annual-Volume-Spalte hinzu oder geben Sie eine passende Part Description an.",
        "optimization_upload_error": "Der hochgeladene Datensatz konnte nicht vorbereitet werden: {error}",
        "optimization_capacity_assumption": "Ohne Capacity wird Annual Volume als konservative Verteilungskapazität verwendet. Capacity ist nur eine Nebenbedingung, kein Optimierungskriterium.",
        "optimization_part_match_fallback": "Keine exakte Part-Number-Übereinstimmung gefunden. Die Lieferantenzeilen wurden über Part Description zugeordnet.",
        "optimization_gate_title": "Filter qualifizierter Lieferanten · Gatekeeper-Regel",
        "optimization_osa_threshold": "OSA-Schwelle",
        "optimization_use_only_qualified": "☑ Nur qualifizierte Lieferanten verwenden",
        "optimization_osa_gate_note": "OSA ist kein Optimierungskriterium. Es dient ausschließlich als Qualifikationsgate; bei aktivierter Regel werden Lieferanten mit OSA-Score < 70 ausgeschlossen.",
        "optimization_three_criteria_note": "Die Optimierung verwendet nur Kosten, Qualität und Lieferung. Kapazität und technische Fähigkeit werden innerhalb der OSA bewertet und sind keine Optimierungskriterien.",
        "optimization_criteria_title": "Kriterienkonstruktion",
        "optimization_cost_definition": "Kosten = jährliche Lieferantenkosten; keine Vorabnormalisierung.",
        "optimization_quality_definition": "Qualitätsscore = gewichtete Unterkriterien aus Fehlerquote, Gewährleistungsfällen, Eingangsannahmequote und Prozessfähigkeit.",
        "optimization_delivery_definition": "Lieferscore = gewichtete Unterkriterien aus Termintreue, Lieferzeit und Liefergenauigkeit.",
        "optimization_method_configuration": "Methodenkonfiguration",
        "optimization_weights_title": "Kriteriengewichte",
        "optimization_weight_cost": "Kostenanteil",
        "optimization_weight_quality": "Qualitätsanteil",
        "optimization_weight_delivery": "Lieferanteil",
        "optimization_weight_total": "Gewichtssumme",
        "optimization_weight_validation": "Die Gewichte müssen vor dem Start der Optimierung 1,00 ergeben.",
        "optimization_topsis_title": "TOPSIS-Konfiguration",
        "optimization_topsis_cost": "Kosten werden als Kostenkriterium konfiguriert (niedriger ist besser).",
        "optimization_topsis_benefits": "Qualität und Lieferung werden als Nutzenkriterien konfiguriert (höher ist besser).",
        "optimization_topsis_normalization": "Das bestehende TOPSIS-Backend verwendet euklidische Normalisierung sowie Ideal-/Anti-Ideal-Ranking.",
        "optimization_targets_title": "Zielwerte",
        "optimization_target_cost": "Zielkosten (€)",
        "optimization_target_cost_unit": "Die Zielkosten verwenden dieselbe jährliche Kosteneinheit wie der hochgeladene Lieferantendatensatz.",
        "optimization_target_quality": "Zielqualitätsscore",
        "optimization_target_delivery": "Ziellieferscore",
        "optimization_preemptive_title": "Prioritätenfolge",
        "optimization_priority_1": "Priorität 1",
        "optimization_priority_2": "Priorität 2",
        "optimization_priority_3": "Priorität 3",
        "optimization_priority_cost": "Kosten",
        "optimization_priority_quality": "Qualität",
        "optimization_priority_delivery": "Lieferung",
        "optimization_priority_note": "Die ausgewählte Reihenfolge dokumentiert die Geschäftspriorität; die bestehende präemptive Solver-Implementierung bleibt unverändert.",
        "optimization_run_button": "🚀 Optimierung starten",
        "optimization_no_qualified_suppliers": "Nach Anwendung des OSA-Qualifikationsgates sind keine Lieferanten übrig.",
        "optimization_no_part_data": "Für das ausgewählte Teil sind keine Lieferantendaten verfügbar.",
        "optimization_duplicate_suppliers": "Für das ausgewählte Teil darf jeder Lieferant nur einmal vorkommen.",
        "optimization_priority_validation": "Priorität 1, Priorität 2 und Priorität 3 müssen unterschiedlich sein.",
        "optimization_results_title": "Optimierungsergebnisse",
        "optimization_recommended_supplier": "Empfohlener Lieferant",
        "optimization_final_score": "Endscore",
        "optimization_supplier_ranking": "Lieferantenranking",
        "optimization_ranking_bar_chart": "Balkendiagramm des Lieferantenrankings",
        "optimization_radar_chart": "Radar-Diagramm der Kriterien",
        "optimization_no_results": "Starten Sie die Optimierung, um Ergebnisse und Visualisierungen zu sehen.",
        "optimization_decision_summary": "Entscheidungszusammenfassung",
        "optimization_method_used": "Verwendete Methode",
        "optimization_selected_part": "Ausgewähltes Teil",
        "optimization_evaluated_suppliers": "Bewertete Lieferanten",
        "optimization_qualified_suppliers": "Qualifizierte Lieferanten",
        "optimization_cost_score": "Kosten",
        "optimization_quality_score": "Qualität",
        "optimization_delivery_score": "Lieferung",
        "optimization_osa_score": "OSA",
        "optimization_export_reports": "Berichte exportieren",
        "optimization_export_pdf": "📄 PDF-Bericht exportieren",
        "optimization_export_excel": "📊 Excel-Bericht exportieren",
        "optimization_pdf_placeholder": "Die PDF-Berichterstellung ist für die Integration in den Corporate-Reporting-Service vorbereitet.",
        "optimization_excel_error": "Der Excel-Bericht konnte nicht erstellt werden: {error}",
        "optimization_report_generated_at": "Erstellt am",
        "optimization_results_csv": "Optimierungsergebnisse herunterladen (CSV)",
        "optimization_radar_cost": "Kostenperformance",
        "optimization_radar_quality": "Qualität",
        "optimization_radar_delivery": "Lieferung",
        "optimization_radar_normalized_note": "Normalisierte Ansicht 0–100",
        "choose_analysis": "Analyse auswählen",
        "add_supplier_data": "Lieferantendaten hinzufügen",
        "optimization_method": "Optimierungsmethode",
        "supplier_data": "Lieferantendaten",
        "supplier_data_caption": "Prüfen oder bearbeiten Sie die Werte für diese Analyse.",
        "model_configuration": "Modellkonfiguration",
        "model_configuration_caption": "Gewichte werden automatisch normalisiert. Wählen Sie die bevorzugte Richtung.",
        "sample_caption": "Beispieldatensatz wird verwendet · CSV oder Excel zum Ersetzen hochladen",
        "using_file": "Verwendet",
        "analysis_help": "Lieferanten einzeln bewerten oder eine kapazitätsbeschränkte Aufteilung optimieren.",
        "allocation_settings": "Verteilungseinstellungen",
        "capacity_warning": "Fügen Sie eine Capacity-Spalte hinzu, um diese Methode zu verwenden.",
        "demand_units": "Bedarfseinheiten",
        "cost_criterion": "Kostenkriterium",
        "quality_criterion": "Qualitätskriterium",
        "delivery_criterion": "Lieferkriterium",
        "minimum_quality_target": "Mindestqualitätsziel",
        "maximum_delivery_target": "Maximales Lieferziel",
        "maximum_cost_target": "Maximales Kostenziel",
        "goal_weights": "Gewichte der Zielprogrammierung",
        "quality_goal_weight": "Gewicht Qualitätsziel",
        "delivery_goal_weight": "Gewicht Lieferziel",
        "cost_goal_weight": "Gewicht Kostenziel",
        "run_analysis": "Analyse starten",
        "ranking_results": "Ranking-Ergebnisse",
        "higher_score": "Höhere {score}-Werte zeigen den besseren Lieferanten.",
        "solution_metrics": "Lösungsmetriken",
        "lexicographic_results": "Ergebnisse der lexikografischen Prioritäten",
        "download_results": "Endergebnisse herunterladen (CSV)",
        "upload_supplier_data": "Lieferantendaten hochladen",
        "upload_supplier_data_help": "Verwenden Sie Spalten wie Supplier, Cost, Quality, Delivery und Capacity.",
        "error_read_upload": "Die hochgeladene Datei konnte nicht gelesen werden: {error}",
        "error_empty_criterion": "Mindestens ein Kriterium ist erforderlich; Capacity ist kein Kriterium.",
        "error_capacity_required": "Fügen Sie vor dem Start dieser Methode eine Capacity-Spalte hinzu.",
        "error_capacity_column": "Präemptive Optimierung und Zielprogrammierung erfordern eine Capacity-Spalte.",
        "error_weight_required": "Mindestens ein Kriteriengewicht muss größer als null sein.",
        "error_goal_weights": "Mindestens ein Zielprogrammierungsgewicht muss größer als null sein.",
        "error_analysis": "Die Analyse konnte nicht abgeschlossen werden: {error}",
        "choose_method_review": "Wählen Sie eine Methode, prüfen Sie die Modellkonfiguration und starten Sie die Analyse.",
        "error_table_supplier_column": "Die Tabelle muss eine Supplier-Spalte enthalten.",
        "error_enter_supplier": "Geben Sie mindestens einen Lieferanten ein.",
        "error_supplier_blank": "Lieferantennamen dürfen nicht leer sein.",
        "error_supplier_unique": "Lieferantennamen müssen nicht leer und eindeutig sein.",
        "error_numeric_criterion": "Fügen Sie mindestens eine numerische Kriterienspalte hinzu.",
        "error_invalid_numeric_columns": "Diese Spalten enthalten leere oder nicht numerische Werte: {columns}",
        "error_finite_numeric": "Numerische Werte müssen endlich sein.",
        "criterion_weight": "Gewicht für {criterion}",
        "criterion_direction": "Richtung für {criterion}",
        "impact_benefit": "Nutzen",
        "impact_cost": "Kosten",
        "score_topsis": "TOPSIS-Score",
        "score_weighted_sum": "Score der gewichteten Summe",
        "preemptive_allocation": "Präemptive Verteilung",
        "goal_programming_allocation": "Zielprogrammierungs-Verteilung",
        "metric_average_cost": "Durchschnittskosten",
        "metric_average_quality": "Durchschnittsqualität",
        "metric_average_delivery": "Durchschnittliche Lieferung",
        "metric_quality_shortfall": "Qualitätsdefizit",
        "metric_delivery_excess": "Lieferüberschreitung",
        "metric_optimal_quality_shortfall": "Optimales Qualitätsdefizit",
        "metric_optimal_delivery_excess_after_p1": "Optimale Lieferüberschreitung nach P1",
        "metric_normalized_objective": "Normalisiertes Ziel",
        "column_score": "Score",
        "column_rank": "Rang",
        "column_allocation_units": "Verteilungseinheiten",
        "column_allocation_share": "Verteilungsanteil",
        "column_value": "Wert",
        "method_topsis": "TOPSIS",
        "method_weighted_sum": "Gewichtete Summe",
        "method_preemptive": "Präemptive Optimierung",
        "method_goal_programming": "Zielprogrammierung",
    },
}

if "language" not in st.session_state:
    st.session_state["language"] = "EN"

PAGE_KEYS = [
    "homepage",
    "parts_database",
    "supplier_database",
    "osa_assessment",
    "supplier_optimization",
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "homepage"

if st.session_state["current_page"] not in PAGE_KEYS:
    st.session_state["current_page"] = "homepage"

active_language_code = st.session_state["language"]


def t(key: str, **values: Any) -> str:
    """Return the selected-language translation with English fallback."""
    language = st.session_state.get("language", "EN")
    text = TRANSLATIONS.get(language, TRANSLATIONS["EN"]).get(
        key, TRANSLATIONS["EN"].get(key, key)
    )
    return text.format(**values) if values else text


def corporate_table_style(data: pd.DataFrame) -> Any:
    """Apply a high-contrast corporate palette to displayed dataframes."""
    return (
        data.style
        .set_properties(
            **{
                "background-color": "#10273B",
                "border-color": "#315B85",
                "color": "#FFFFFF",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#0B1F3A"),
                        ("color", "#FFFFFF"),
                        ("font-weight", "700"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [("color", "#FFFFFF")],
                },
            ]
        )
    )


METRIC_TRANSLATION_KEYS = {
    "average_cost": "metric_average_cost",
    "average_quality": "metric_average_quality",
    "average_delivery": "metric_average_delivery",
    "quality_shortfall": "metric_quality_shortfall",
    "delivery_excess": "metric_delivery_excess",
    "optimal_quality_shortfall": "metric_optimal_quality_shortfall",
    "optimal_delivery_excess_after_P1": "metric_optimal_delivery_excess_after_p1",
    "normalized_objective": "metric_normalized_objective",
}


def translate_metric_label(name: str) -> str:
    """Translate known optimization outputs without changing backend keys."""
    key = METRIC_TRANSLATION_KEYS.get(name)
    return t(key) if key else name.replace("_", " ").title()


# Wikimedia Commons hosts the Mercedes-Benz Star SVG; CSS inversion renders it white.
MERCEDES_LOGO_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/3/32/"
    "Mercedes-Benz_Star_2022.svg"
)

st.markdown(
    f"""
    <style>
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
            font-family: Arial, Helvetica, sans-serif;
        }}
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stMain"], .main, .block-container {{
            background-color: #081827 !important;
            color: #FFFFFF !important;
        }}
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stHeader"],
        [data-testid="stSubheader"],
        [data-testid="stCaptionContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stDataFrame"],
        [data-testid="stDataEditor"] {{
            color: #FFFFFF !important;
            opacity: 1 !important;
        }}
        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
            background-color: #10273B !important;
        }}
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"],
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stAlert"] p,
        [data-testid="stException"] {{
            color: #FFFFFF !important;
            opacity: 1 !important;
        }}
        .block-container {{
            margin: 0 !important;
            padding-top: 104px;
        }}
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {{
            display: none !important;
        }}
        div[data-testid="stElementContainer"]:has(.corporate-top-bar) {{
            height: 0 !important;
            margin: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
        }}
        div[data-testid="stElementContainer"]:has(.st-key-corporate-navigation) {{
            margin: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
        }}
        .st-key-corporate-navigation {{
            align-items: center;
            border-bottom: 1px solid color-mix(in srgb, currentColor 18%, transparent);
            margin: 0 !important;
            padding: 0 !important;
        }}
        .st-key-corporate-navigation [data-testid="stElementContainer"],
        .st-key-corporate-navigation [data-testid="stVerticalBlock"] {{
            margin: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
        }}
        .st-key-corporate-navigation [data-testid="stHorizontalBlock"] {{
            align-items: center;
            gap: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .st-key-corporate-navigation [data-testid="column"] {{
            margin: 0 !important;
            padding: 0 !important;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] {{
            margin: 0 !important;
            padding: 0 !important;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] > label {{
            display: none !important;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] {{
            align-items: center !important;
            display: flex !important;
            flex-wrap: nowrap !important;
            gap: 0 !important;
            margin: 0 !important;
            max-width: 100%;
            overflow-x: auto;
            padding: 0 !important;
            scrollbar-width: none;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"]::-webkit-scrollbar {{
            display: none;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label {{
            align-items: center !important;
            background: transparent !important;
            border: 0 !important;
            border-radius: 0 !important;
            color: #FFFFFF !important;
            cursor: pointer;
            display: flex !important;
            flex: 0 0 auto !important;
            font-weight: 600;
            font-size: 0.84rem;
            line-height: 1.2;
            margin: 0 0.55rem 0 0 !important;
            min-height: 2.75rem;
            padding: 5px 15px !important;
            position: relative;
            white-space: nowrap;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label::after {{
            color: #FFFFFF !important;
            content: " / ";
            font-weight: 400;
            position: absolute;
            right: -0.4rem;
            top: 50%;
            transform: translateY(-50%);
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label:last-child::after {{
            content: none;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label:hover,
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {{
            background: #00ADEF !important;
            color: #FFFFFF !important;
            padding: 5px 15px !important;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label:hover > div,
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) > div {{
            background: #00ADEF !important;
            color: #FFFFFF !important;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label:hover p,
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p {{
            color: #FFFFFF !important;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label:hover::after,
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked)::after {{
            color: #FFFFFF !important;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label:has(input:focus-visible) {{
            outline: 2px solid #0078c8 !important;
            outline-offset: 2px;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] input[type="radio"] {{
            height: 0 !important;
            opacity: 0 !important;
            position: absolute !important;
            width: 0 !important;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label > div {{
            align-items: center !important;
            background: transparent !important;
            border: 0 !important;
            color: inherit !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] p {{
            color: inherit !important;
            margin: 0 !important;
        }}
        .st-key-corporate-navigation [data-testid="stButton"] {{
            margin: 0 !important;
            width: 100%;
        }}
        .st-key-corporate-navigation [data-testid="stButton"] button {{
            align-items: center;
            background: transparent !important;
            border: 0 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            color: #FFFFFF !important;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            min-height: 2.75rem;
            padding: 5px 10px !important;
            white-space: nowrap;
        }}
        .st-key-corporate-navigation [data-testid="stButton"] button:hover {{
            background: #00ADEF !important;
            color: #FFFFFF !important;
        }}
        .st-key-corporate-navigation [data-testid="stButton"] button:focus-visible {{
            outline: 2px solid #0078c8 !important;
            outline-offset: 2px;
        }}
        .st-key-language_{active_language_code} button {{
            background: #00ADEF !important;
            color: #FFFFFF !important;
            font-weight: 700;
        }}
        .st-key-corporate-navigation .language-separator {{
            align-items: center;
            color: #FFFFFF !important;
            display: flex;
            font-size: 0.78rem;
            height: 2.75rem;
            justify-content: center;
        }}
        .corporate-top-bar {{
            align-items: center;
            background: #000000 !important;
            color: #FFFFFF !important;
            display: flex;
            box-sizing: border-box;
            flex-direction: row;
            height: 104px;
            justify-content: space-between;
            left: 0;
            padding: 18px 48px;
            position: fixed;
            right: 0;
            top: 0;
            z-index: 1000000;
            pointer-events: none;
            line-height: 1;
        }}
        .corporate-top-bar__logo {{
            display: block;
            filter: brightness(0) invert(1);
            flex: 0 0 auto;
            height: 58px;
            object-fit: contain;
            width: 58px;
        }}
        .corporate-top-bar__brand {{
            color: #FFFFFF !important;
            font-family: Georgia, 'Times New Roman', serif !important;
            font-size: 1.45rem;
            letter-spacing: 0.02em;
            line-height: 1;
            margin: 0;
            white-space: nowrap;
        }}
        @media (max-width: 768px) {{
            .block-container {{
                padding-top: 88px;
            }}
            .corporate-top-bar {{
                height: 88px;
                padding: 14px 20px;
            }}
            .corporate-top-bar__logo {{
                height: 48px;
                width: 48px;
            }}
            .corporate-top-bar__brand {{
                font-size: 1.1rem;
            }}
            .st-key-corporate-navigation [data-testid="stRadio"] [role="radiogroup"] > label {{
                font-size: 0.7rem;
                margin-right: 0.35rem !important;
                padding-left: 0.45rem !important;
                padding-right: 0.45rem !important;
            }}
            .st-key-corporate-navigation [data-testid="stButton"] button {{
                font-size: 0.68rem;
            }}
        }}
    </style>
    <div class="corporate-top-bar" aria-label="Mercedes-Benz Türk">
        <img class="corporate-top-bar__logo" src="{MERCEDES_LOGO_URL}"
             alt="Mercedes-Benz logo" />
        <div class="corporate-top-bar__brand">Mercedes-Benz Türk</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Shared data and utility functions
# -----------------------------------------------------------------------------


def default_supplier_data() -> pd.DataFrame:
    """Return the dummy dataset used by the core algorithm module."""
    return pd.DataFrame(
        {
            "Supplier": ["Supplier A", "Supplier B", "Supplier C"],
            "Cost": [100.0, 120.0, 90.0],
            "Quality": [80.0, 90.0, 75.0],
            "Delivery": [10.0, 7.0, 12.0],
            "Capacity": [50.0, 80.0, 70.0],
        }
    )


@st.cache_data(show_spinner=False)
def read_supplier_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read CSV/XLSX input without changing the original uploaded file."""
    if file_name.lower().endswith(".csv"):
        data = pd.read_csv(io.BytesIO(file_bytes))
    else:
        data = pd.read_excel(io.BytesIO(file_bytes))
    return standardize_supplier_table(data)


def standardize_supplier_table(data: pd.DataFrame) -> pd.DataFrame:
    """Make common supplier-column variations usable by the app."""
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]

    supplier_column = next(
        (
            column
            for column in data.columns
            if column.lower() in {"supplier", "supplier name", "vendor", "vendor name"}
        ),
        None,
    )

    # Excel files often contain the former index as an "Unnamed: 0" column.
    if supplier_column is None and len(data.columns) > 0:
        first_column = data.columns[0]
        if first_column.lower().startswith("unnamed"):
            supplier_column = first_column

    if supplier_column is not None:
        data = data.rename(columns={supplier_column: "Supplier"})
    else:
        data.insert(0, "Supplier", [f"Supplier {i + 1}" for i in range(len(data))])

    capacity_column = next(
        (column for column in data.columns if column.lower() == "capacity"), None
    )
    if capacity_column is not None and capacity_column != "Capacity":
        data = data.rename(columns={capacity_column: "Capacity"})

    return data


def inferred_impact(column: str) -> str:
    """Choose a useful default impact for familiar cost-like column names."""
    cost_words = ("cost", "price", "fee", "time", "day", "delivery", "lead", "risk")
    return "cost" if any(word in column.lower() for word in cost_words) else "benefit"


def default_column(columns: list[str], preferred: tuple[str, ...]) -> str:
    """Return the first preferred column that exists, otherwise the first column."""
    for column in preferred:
        if column in columns:
            return column
    return columns[0]


def prepare_numeric_data(edited_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the editable table and return criteria data plus numeric values."""
    if "Supplier" not in edited_data.columns:
        raise ValueError(t("error_table_supplier_column"))
    if edited_data.empty:
        raise ValueError(t("error_enter_supplier"))
    if edited_data["Supplier"].isna().any():
        raise ValueError(t("error_supplier_blank"))

    supplier_names = edited_data["Supplier"].astype(str).str.strip()
    if (supplier_names == "").any() or supplier_names.duplicated().any():
        raise ValueError(t("error_supplier_unique"))

    numeric = edited_data.drop(columns=["Supplier"]).apply(pd.to_numeric, errors="coerce")
    if numeric.empty:
        raise ValueError(t("error_numeric_criterion"))
    if numeric.isna().any().any():
        invalid_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(t("error_invalid_numeric_columns", columns=invalid_columns))
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(t("error_finite_numeric"))

    numeric.index = supplier_names
    numeric.index.name = "Supplier"
    criteria = numeric.drop(columns=["Capacity"], errors="ignore")
    return criteria, numeric


def result_for_download(
    result: pd.DataFrame,
    metrics: pd.Series | None = None,
    stages: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Convert displayed results into one CSV-friendly final output table."""
    output = result.reset_index()
    if "index" in output.columns:
        output = output.rename(columns={"index": "Supplier"})

    # Allocation metrics are repeated as metadata columns so the single CSV
    # contains both supplier-level results and the solution summary.
    if metrics is not None:
        for name, value in metrics.items():
            output[name] = value
    if stages is not None:
        for name, value in stages.items():
            output[name] = value
    return output


# -----------------------------------------------------------------------------
# Homepage and database pages
# -----------------------------------------------------------------------------


def homepage_page() -> None:
    """Render the corporate Homepage without changing portal functionality."""
    st.markdown(
        """
        <style>
            .homepage-hero {
                background: linear-gradient(135deg, #0b1f3a 0%, #163f6b 100%);
                border-radius: 0.75rem;
                box-shadow: 0 14px 32px rgba(11, 31, 58, 0.18);
                color: #FFFFFF;
                margin: 0 0 1.5rem;
                padding: 2.4rem 2.75rem;
            }
            .homepage-hero h1 {
                color: #FFFFFF !important;
                font-size: clamp(2rem, 4vw, 3.1rem);
                font-weight: 720;
                letter-spacing: -0.04em;
                line-height: 1.05;
                margin: 0;
            }
            .homepage-hero p {
                color: #FFFFFF !important;
                font-size: 1.05rem;
                line-height: 1.6;
                margin: 0.9rem 0 0;
                max-width: 58rem;
            }
            .homepage-section-heading {
                color: #FFFFFF !important;
                font-size: 1.35rem;
                font-weight: 720;
                letter-spacing: -0.02em;
                margin: 2rem 0 0.85rem;
            }
            .st-key-homepage_content [data-testid="stMetric"] {
                background: #10273B;
                border: 1px solid #315B85;
                border-radius: 0.65rem;
                padding: 1rem 1.1rem;
            }
            .st-key-homepage_content [data-testid="stMetricLabel"],
            .st-key-homepage_content [data-testid="stMetricValue"],
            .st-key-homepage_content [data-testid="stMetricDelta"] {
                color: #FFFFFF !important;
                opacity: 1 !important;
            }
            .homepage-flow {
                align-items: stretch;
                display: flex;
                justify-content: center;
                margin: 0;
            }
            .homepage-flow-step {
                align-items: center;
                background: #10273B;
                border: 1px solid #315B85;
                border-radius: 0.55rem;
                color: #FFFFFF !important;
                display: flex;
                flex-direction: column;
                font-size: 0.78rem;
                font-weight: 700;
                justify-content: center;
                line-height: 1.25;
                min-height: 5.1rem;
                padding: 0.55rem 0.35rem;
                text-align: center;
            }
            .homepage-flow-number {
                align-items: center;
                background: #00ADEF;
                border-radius: 50%;
                color: #FFFFFF !important;
                display: flex;
                font-size: 0.65rem;
                height: 1.35rem;
                justify-content: center;
                margin-bottom: 0.45rem;
                width: 1.35rem;
            }
            .homepage-flow-arrow {
                align-items: center;
                color: #FFFFFF !important;
                display: flex;
                font-size: 1.35rem;
                height: 100%;
                justify-content: center;
            }
            .method-card {
                background: #10273B;
                border-radius: 0.55rem;
                color: #FFFFFF !important;
                min-height: 14rem;
                padding: 1.15rem;
            }
            .method-card-accent {
                border-radius: 999px;
                height: 0.3rem;
                margin-bottom: 1rem;
                width: 2.8rem;
            }
            .method-card h3 {
                color: #FFFFFF !important;
                font-size: 1.2rem;
                margin: 0 0 1rem;
            }
            .method-card-copy p {
                color: #FFFFFF !important;
                font-size: 0.9rem;
                line-height: 1.45;
                margin: 0.55rem 0 0;
            }
            .method-card-copy strong {
                color: #FFFFFF !important;
                font-weight: 800;
            }
            .homepage-guide {
                background: #10273B;
                border-left: 4px solid #00ADEF;
                border-radius: 0.25rem 0.65rem 0.65rem 0.25rem;
                color: #FFFFFF !important;
                padding: 1rem 1.3rem 1rem 1.1rem;
            }
            .homepage-guide ul {
                margin: 0;
                padding-left: 1.15rem;
            }
            .homepage-guide li {
                color: #FFFFFF !important;
                line-height: 1.55;
                margin: 0.45rem 0;
            }
            @media (max-width: 768px) {
                .homepage-hero {
                    padding: 1.8rem 1.35rem;
                }
                .homepage-flow-step {
                    font-size: 0.64rem;
                    min-height: 4.5rem;
                }
                .homepage-flow-arrow {
                    font-size: 0.95rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="homepage_content"):
        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------
        st.markdown(
            f"""
            <div class="homepage-hero">
                <h1>{t("homepage_title")}</h1>
                <p>{t("homepage_subtitle")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------------
        # KPI dashboard
        # ------------------------------------------------------------------
        kpi_columns = st.columns(4)
        kpis = [
            (t("homepage_kpi_parts"), "15"),
            (t("homepage_kpi_candidate_suppliers"), "45"),
            (t("homepage_kpi_qualified_suppliers"), "38"),
            (t("homepage_kpi_optimization_models"), "4"),
        ]
        for kpi_column, (label, value) in zip(kpi_columns, kpis):
            with kpi_column:
                with st.container(border=True):
                    st.metric(label, value)

        # ------------------------------------------------------------------
        # Supplier selection framework
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="homepage-section-heading">{t("framework_title")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            flow_columns = st.columns(11, gap="small")
            flow_items = [
                ("01", "framework_osa_assessment"),
                ("➔", None),
                ("02", "framework_supplier_qualification"),
                ("➔", None),
                ("03", "framework_cost_evaluation"),
                ("➔", None),
                ("04", "framework_quality_evaluation"),
                ("➔", None),
                ("05", "framework_delivery_evaluation"),
                ("➔", None),
                ("06", "framework_supplier_ranking"),
            ]
            for flow_column, (marker, label_key) in zip(flow_columns, flow_items):
                with flow_column:
                    if label_key is None:
                        st.markdown(
                            '<div class="homepage-flow-arrow">➔</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="homepage-flow-step">
                                <div class="homepage-flow-number">{marker}</div>
                                <div>{t(label_key)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # ------------------------------------------------------------------
        # Optimization method cards: two columns across two rows
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="homepage-section-heading">{t("optimization_methods_title")}</div>',
            unsafe_allow_html=True,
        )

        def render_method_card(
            slug: str,
            title: str,
            accent: str,
            best_when: str,
            description: str,
            example: str,
        ) -> None:
            """Render one consistent, corporate-styled optimization card."""
            with st.container(border=True, key=f"homepage_method_{slug}"):
                st.markdown(
                    f"""
                    <div class="method-card">
                        <div class="method-card-accent" style="background: {accent};"></div>
                        <h3>{title}</h3>
                        <div class="method-card-copy">
                            <p><strong>{t("method_best_when")}:</strong> {best_when}</p>
                            <p><strong>{t("method_description")}:</strong> {description}</p>
                            <p><strong>{t("method_example")}:</strong> {example}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        first_method_column, second_method_column = st.columns(2, gap="large")
        with first_method_column:
            render_method_card(
                "weighted_sum",
                t("method_weighted_sum"),
                "#0078c8",
                t("method_weighted_sum_best_when"),
                t("method_weighted_sum_description"),
                t("method_weighted_sum_example"),
            )
        with second_method_column:
            render_method_card(
                "topsis",
                t("method_topsis"),
                "#4d83b3",
                t("method_topsis_best_when"),
                t("method_topsis_description"),
                t("method_topsis_example"),
            )

        third_method_column, fourth_method_column = st.columns(2, gap="large")
        with third_method_column:
            render_method_card(
                "goal_programming",
                t("method_goal_programming"),
                "#315b85",
                t("method_goal_programming_best_when"),
                t("method_goal_programming_description"),
                t("method_goal_programming_example"),
            )
        with fourth_method_column:
            render_method_card(
                "preemptive",
                t("method_preemptive"),
                "#0b1f3a",
                t("method_preemptive_best_when"),
                t("method_preemptive_description"),
                t("method_preemptive_example"),
            )

        # ------------------------------------------------------------------
        # Method selection guide
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="homepage-section-heading">{t("method_guide_title")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="homepage-guide">
                    <ul>
                        <li>{t("method_guide_weighted_sum")}</li>
                        <li>{t("method_guide_topsis")}</li>
                        <li>{t("method_guide_goal_programming")}</li>
                        <li>{t("method_guide_preemptive")}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def parts_database_page() -> None:
    """Render the corporate parts catalogue and selected-part detail view."""
    st.markdown(
        """
        <style>
            .parts-hero {
                background: linear-gradient(135deg, #0b1f3a 0%, #163f6b 100%);
                border-radius: 0.75rem;
                box-shadow: 0 14px 32px rgba(11, 31, 58, 0.16);
                color: #FFFFFF;
                margin: 0 0 1.5rem;
                padding: 2.2rem 2.6rem;
            }
            .parts-hero h1 {
                color: #FFFFFF !important;
                font-size: clamp(2rem, 4vw, 3rem);
                font-weight: 720;
                letter-spacing: -0.04em;
                line-height: 1.05;
                margin: 0;
            }
            .parts-hero p {
                color: #FFFFFF !important;
                font-size: 1.04rem;
                line-height: 1.55;
                margin: 0.85rem 0 0;
                max-width: 58rem;
            }
            .parts-section-heading {
                color: #FFFFFF !important;
                font-size: 1.3rem;
                font-weight: 720;
                letter-spacing: -0.02em;
                margin: 1.8rem 0 0.8rem;
            }
            .st-key-parts_database_content [data-testid="stMetric"] {
                background: #10273B;
                border: 1px solid #315B85;
                border-radius: 0.65rem;
                padding: 1rem 1.1rem;
            }
            .st-key-parts_database_content [data-testid="stMetricLabel"],
            .st-key-parts_database_content [data-testid="stMetricValue"],
            .st-key-parts_database_content [data-testid="stMetricDelta"] {
                color: #FFFFFF !important;
                opacity: 1 !important;
            }
            .parts-detail-eyebrow {
                color: #FFFFFF !important;
                font-size: 0.76rem;
                font-weight: 800;
                letter-spacing: 0.1em;
                margin-bottom: 0.35rem;
                text-transform: uppercase;
            }
            .parts-detail-heading {
                color: #FFFFFF !important;
                font-size: 1.45rem;
                font-weight: 720;
                line-height: 1.2;
                margin: 0 0 0.8rem;
            }
            .parts-supplier-heading {
                color: #FFFFFF !important;
                font-size: 1rem;
                font-weight: 720;
                margin: 0.25rem 0 0.45rem;
            }
            .parts-supplier-list {
                background: #10273B;
                border-left: 3px solid #00ADEF;
                border-radius: 0.2rem 0.5rem 0.5rem 0.2rem;
                color: #FFFFFF !important;
                line-height: 1.55;
                margin-bottom: 1.2rem;
                padding: 0.75rem 1rem;
            }
            .parts-supplier-list li {
                color: #FFFFFF !important;
            }
            .parts-supplier-list ul {
                margin: 0;
                padding-left: 1.15rem;
            }
            .parts-supplier-list li {
                margin: 0.22rem 0;
            }
            @media (max-width: 768px) {
                .parts-hero {
                    padding: 1.8rem 1.35rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="parts_database_content"):
        # ------------------------------------------------------------------
        # Page header
        # ------------------------------------------------------------------
        st.markdown(
            f"""
            <div class="parts-hero">
                <h1>{t("parts_title")}</h1>
                <p>{t("parts_subtitle")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------------
        # KPI dashboard
        # ------------------------------------------------------------------
        kpi_columns = st.columns(4)
        kpis = [
            (t("parts_kpi_total"), "15"),
            (t("parts_kpi_candidate_suppliers"), "45"),
            (t("parts_kpi_qualified_suppliers"), "38"),
            (t("parts_kpi_active_rfqs"), "15"),
        ]
        for kpi_column, (label, value) in zip(kpi_columns, kpis):
            with kpi_column:
                with st.container(border=True):
                    st.metric(label, value)

        # ------------------------------------------------------------------
        # Part catalogue: representative Daimler Truck-style sample data
        # ------------------------------------------------------------------
        parts_catalogue = pd.DataFrame(
            [
                ["DT-INT-001", "Interior Trims", "Aksaray", 2025, 10, 18000, 135.00, 118.00],
                ["DT-STW-002", "Stowage Box Above Windshield", "Aksaray", 2025, 10, 12000, 96.00, 82.00],
                ["DT-STW-003", "Rear Stowage Box", "Wörth", 2026, 10, 12000, 110.00, 95.00],
                ["DT-RHT-004", "Roof Hatch", "Mannheim", 2025, 12, 8500, 265.00, 230.00],
                ["DT-HOR-005", "Horn", "Gaggenau", 2025, 12, 22000, 48.00, 39.00],
                ["DT-BPI-006", "B-Pillar", "Aksaray", 2026, 10, 14000, 188.00, 165.00],
                ["DT-SAD-007", "Storage Above Door", "Aksaray", 2026, 10, 13500, 124.00, 106.00],
                ["DT-API-008", "A-Pillar", "Aksaray", 2026, 10, 14000, 172.00, 149.00],
                ["DT-TV-009", "TV Bracket", "Wörth", 2027, 8, 5000, 142.00, 120.00],
                ["DT-CIN-010", "Cab Insulation", "Mannheim", 2025, 10, 11500, 215.00, 184.00],
                ["DT-ATP-011", "Attachment Parts", "Gaggenau", 2025, 12, 22000, 72.00, 58.00],
                ["DT-SVI-012", "Sun Visor", "Aksaray", 2025, 10, 18000, 61.00, 49.00],
                ["DT-SBL-013", "Sun Blind", "Aksaray", 2026, 10, 16000, 84.00, 70.00],
                ["DT-SWB-014", "Steering Wheel Buttons", "Gaggenau", 2027, 8, 21000, 53.00, 43.00],
                ["DT-INL-015", "Interior Lighting", "Mannheim", 2026, 10, 20000, 92.00, 76.00],
            ],
            columns=[
                "Part Number",
                "Part Description",
                "Plant",
                "SOP Year",
                "Lifetime (Years)",
                "Annual Volume",
                "Budget (€)",
                "Target Cost (€)",
            ],
        )

        supplier_coverage = {
            "DT-INT-001": ("Eissmann Automotive; Grammer; Brose", "Eissmann Automotive; Grammer"),
            "DT-STW-002": ("Borgers; Grammer; Novares", "Borgers; Grammer"),
            "DT-STW-003": ("Borgers; Novares; Eissmann Automotive", "Borgers; Novares"),
            "DT-RHT-004": ("Webasto; Roof Systems; Inalfa", "Webasto; Inalfa"),
            "DT-HOR-005": ("Hella; Forvia; Bosch", "Hella; Bosch"),
            "DT-BPI-006": ("Brose; Lear; Eissmann Automotive", "Brose; Lear"),
            "DT-SAD-007": ("Grammer; Brose; Novares", "Grammer; Brose"),
            "DT-API-008": ("Eissmann Automotive; Lear; Forvia", "Eissmann Automotive; Lear"),
            "DT-TV-009": ("Kongsberg; Brose; Novares", "Kongsberg; Novares"),
            "DT-CIN-010": ("Borgers; Autoneum; Adler Pelzer", "Autoneum; Adler Pelzer"),
            "DT-ATP-011": ("Gestamp; Benteler; Kirchhoff", "Benteler; Kirchhoff"),
            "DT-SVI-012": ("Kongsberg; Grupo Antolin; Grammer", "Grupo Antolin; Grammer"),
            "DT-SBL-013": ("Brose; Grupo Antolin; Webasto", "Brose; Grupo Antolin"),
            "DT-SWB-014": ("Preh; Kostal; ZF", "Preh; Kostal"),
            "DT-INL-015": ("Hella; Forvia; Marelli", "Hella; Forvia"),
        }

        part_column_keys = {
            "Part Number": "part_number",
            "Part Description": "part_description",
            "Plant": "plant",
            "SOP Year": "sop_year",
            "Lifetime (Years)": "lifetime_years",
            "Annual Volume": "annual_volume",
            "Budget (€)": "budget",
            "Target Cost (€)": "target_cost",
        }

        st.markdown(
            f'<div class="parts-section-heading">{t("parts_table_title")}</div>',
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------------
        # Filter controls
        # ------------------------------------------------------------------
        with st.container(border=True, key="parts_filters"):
            number_column, description_column, plant_column, year_column = st.columns(4)
            with number_column:
                number_query = st.text_input(
                    t("parts_filter_number"),
                    placeholder=t("parts_filter_number_placeholder"),
                    key="parts_filter_number_input",
                ).strip().lower()
            with description_column:
                description_query = st.text_input(
                    t("parts_filter_description"),
                    placeholder=t("parts_filter_description_placeholder"),
                    key="parts_filter_description_input",
                ).strip().lower()
            with plant_column:
                plant_options = [t("parts_all_plants")] + sorted(
                    parts_catalogue["Plant"].unique().tolist()
                )
                plant_filter = st.selectbox(
                    t("parts_filter_plant"),
                    plant_options,
                    key="parts_filter_plant_input",
                )
            with year_column:
                year_options = [t("parts_all_years")] + [
                    str(year) for year in sorted(parts_catalogue["SOP Year"].unique())
                ]
                year_filter = st.selectbox(
                    t("parts_filter_sop_year"),
                    year_options,
                    key="parts_filter_year_input",
                )

        filtered_parts = parts_catalogue.copy()
        if number_query:
            filtered_parts = filtered_parts[
                filtered_parts["Part Number"].str.lower().str.contains(number_query, na=False)
            ]
        if description_query:
            filtered_parts = filtered_parts[
                filtered_parts["Part Description"].str.lower().str.contains(
                    description_query, na=False
                )
            ]
        if plant_filter != t("parts_all_plants"):
            filtered_parts = filtered_parts[filtered_parts["Plant"] == plant_filter]
        if year_filter != t("parts_all_years"):
            filtered_parts = filtered_parts[filtered_parts["SOP Year"] == int(year_filter)]

        st.caption(
            t(
                "parts_table_count",
                shown=len(filtered_parts),
                total=len(parts_catalogue),
            )
        )

        if filtered_parts.empty:
            st.warning(t("parts_no_matches"))
            return

        st.caption(t("parts_select_hint"))
        table_state = st.dataframe(
            corporate_table_style(
                filtered_parts.rename(
                    columns={column: t(key) for column, key in part_column_keys.items()}
                )
            ),
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="parts_catalogue_table",
        )

        # Streamlit returns a dataframe selection state for interactive tables.
        # Keep the selected part number in session state so it remains visible
        # while the user reviews the page.
        selection_state = getattr(table_state, "selection", None)
        selected_rows = getattr(selection_state, "rows", [])
        if isinstance(selection_state, dict):
            selected_rows = selection_state.get("rows", [])

        selected_part_number = st.session_state.get("selected_part_number")
        if selected_rows:
            selected_part_number = filtered_parts.iloc[int(selected_rows[0])]["Part Number"]
            st.session_state["selected_part_number"] = selected_part_number

        if selected_part_number not in set(filtered_parts["Part Number"]):
            selected_part_number = None

        # ------------------------------------------------------------------
        # Part detail panel
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="parts-section-heading">{t("parts_detail_title")}</div>',
            unsafe_allow_html=True,
        )
        if selected_part_number is None:
            st.info(t("parts_no_selection"))
            return

        selected_part = parts_catalogue.loc[
            parts_catalogue["Part Number"] == selected_part_number
        ].iloc[0]
        candidate_suppliers, qualified_suppliers = supplier_coverage[selected_part_number]

        with st.container(border=True, key="parts_detail_panel"):
            metadata_column, supplier_column = st.columns([1.45, 1])
            with metadata_column:
                st.markdown(
                    f"""
                    <div class="parts-detail-eyebrow">{selected_part["Part Number"]}</div>
                    <div class="parts-detail-heading">{selected_part["Part Description"]}</div>
                    """,
                    unsafe_allow_html=True,
                )
                metadata = pd.DataFrame(
                    {
                        "Field": [t(part_column_keys[field]) for field in parts_catalogue.columns],
                        "Value": [selected_part[field] for field in parts_catalogue.columns],
                    }
                )
                st.dataframe(
                    corporate_table_style(metadata),
                    width="stretch",
                    hide_index=True,
                )

            with supplier_column:
                st.markdown(
                    f'<div class="parts-supplier-heading">{t("parts_candidate_suppliers")}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class=\"parts-supplier-list\"><ul>"
                    + "".join(
                        f"<li>{supplier.strip()}</li>"
                        for supplier in candidate_suppliers.split(";")
                    )
                    + "</ul></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="parts-supplier-heading">{t("parts_qualified_suppliers")}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class=\"parts-supplier-list\"><ul>"
                    + "".join(
                        f"<li>{supplier.strip()}</li>"
                        for supplier in qualified_suppliers.split(";")
                    )
                    + "</ul></div>",
                    unsafe_allow_html=True,
                )


def supplier_database_page() -> None:
    """Render the supplier catalogue, profile detail, and comparison views."""
    st.markdown(
        """
        <style>
            .supplier-hero {
                background: linear-gradient(135deg, #0b1f3a 0%, #163f6b 100%);
                border-radius: 0.75rem;
                box-shadow: 0 14px 32px rgba(11, 31, 58, 0.18);
                color: #FFFFFF;
                margin: 0 0 1.5rem;
                padding: 2.2rem 2.6rem;
            }
            .supplier-hero h1 {
                color: #FFFFFF !important;
                font-size: clamp(2rem, 4vw, 3rem);
                font-weight: 720;
                letter-spacing: -0.04em;
                line-height: 1.05;
                margin: 0;
            }
            .supplier-hero p {
                color: #FFFFFF !important;
                font-size: 1.04rem;
                line-height: 1.55;
                margin: 0.85rem 0 0;
                max-width: 58rem;
            }
            .supplier-section-heading {
                color: #FFFFFF !important;
                font-size: 1.3rem;
                font-weight: 720;
                letter-spacing: -0.02em;
                margin: 1.8rem 0 0.8rem;
            }
            .st-key-supplier_database_content [data-testid="stMetric"] {
                background: #10273B;
                border: 1px solid #315B85;
                border-radius: 0.65rem;
                padding: 1rem 1.1rem;
            }
            .st-key-supplier_database_content [data-testid="stMetricLabel"],
            .st-key-supplier_database_content [data-testid="stMetricValue"],
            .st-key-supplier_database_content [data-testid="stMetricDelta"] {
                color: #FFFFFF !important;
                opacity: 1 !important;
            }
            .supplier-profile-heading {
                color: #FFFFFF !important;
                font-size: 1.45rem;
                font-weight: 720;
                line-height: 1.2;
                margin: 0;
            }
            .supplier-profile-subheading {
                color: #FFFFFF !important;
                font-size: 0.95rem;
                margin: 0.35rem 0 0;
                opacity: 1;
            }
            .supplier-info-item {
                background: #0B1F3A;
                border: 1px solid #315B85;
                border-radius: 0.45rem;
                margin-bottom: 0.6rem;
                min-height: 3.35rem;
                padding: 0.65rem 0.8rem;
            }
            .supplier-info-label {
                color: #FFFFFF !important;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                margin-bottom: 0.22rem;
                text-transform: uppercase;
            }
            .supplier-info-value {
                color: #FFFFFF !important;
                font-size: 0.94rem;
                font-weight: 600;
                line-height: 1.35;
            }
            .supplier-status-badge {
                border-radius: 999px;
                display: inline-block;
                font-size: 0.82rem;
                font-weight: 800;
                line-height: 1.2;
                padding: 0.5rem 0.8rem;
            }
            .supplier-status-badge.qualified,
            .supplier-status-badge.approved {
                background: #176B4D;
                color: #FFFFFF !important;
            }
            .supplier-status-badge.not-qualified,
            .supplier-status-badge.missing {
                background: #9E2A2B;
                color: #FFFFFF !important;
            }
            .supplier-status-badge.review {
                background: #8A5A00;
                color: #FFFFFF !important;
            }
            .supplier-comparison-card {
                background: #10273B;
                border: 1px solid #315B85;
                border-radius: 0.65rem;
                color: #FFFFFF !important;
                min-height: 100%;
                padding: 1rem;
            }
            .supplier-comparison-card h3 {
                color: #FFFFFF !important;
                font-size: 1.05rem;
                margin: 0 0 0.2rem;
            }
            .supplier-comparison-card .comparison-part {
                color: #FFFFFF !important;
                font-size: 0.78rem;
                margin: 0 0 0.9rem;
            }
            .supplier-comparison-row {
                border-top: 1px solid #315B85;
                display: flex;
                justify-content: space-between;
                gap: 0.75rem;
                padding: 0.42rem 0;
            }
            .supplier-comparison-label,
            .supplier-comparison-value {
                color: #FFFFFF !important;
                font-size: 0.79rem;
                line-height: 1.35;
            }
            .supplier-comparison-label {
                font-weight: 600;
            }
            .supplier-comparison-value {
                font-weight: 800;
                text-align: right;
            }
            @media (max-width: 768px) {
                .supplier-hero {
                    padding: 1.8rem 1.35rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Keep the catalogue consistent with the 15 parts defined on Parts Database.
    # Each part intentionally has exactly three candidate supplier records.
    part_records = [
        ("DT-INT-001", "Interior Trims", "Aksaray", 2025, 10, 18000, 118.00, ("Eissmann Automotive", "Grammer", "Brose")),
        ("DT-STW-002", "Stowage Box Above Windshield", "Aksaray", 2025, 10, 12000, 82.00, ("Borgers", "Grammer", "Novares")),
        ("DT-STW-003", "Rear Stowage Box", "Wörth", 2026, 10, 12000, 95.00, ("Borgers", "Novares", "Eissmann Automotive")),
        ("DT-RHT-004", "Roof Hatch", "Mannheim", 2025, 12, 8500, 230.00, ("Webasto", "Roof Systems", "Inalfa")),
        ("DT-HOR-005", "Horn", "Gaggenau", 2025, 12, 22000, 39.00, ("Hella", "Forvia", "Bosch")),
        ("DT-BPI-006", "B-Pillar", "Aksaray", 2026, 10, 14000, 165.00, ("Brose", "Lear", "Eissmann Automotive")),
        ("DT-SAD-007", "Storage Above Door", "Aksaray", 2026, 10, 13500, 106.00, ("Grammer", "Brose", "Novares")),
        ("DT-API-008", "A-Pillar", "Aksaray", 2026, 10, 14000, 149.00, ("Eissmann Automotive", "Lear", "Forvia")),
        ("DT-TV-009", "TV Bracket", "Wörth", 2027, 8, 5000, 120.00, ("Kongsberg", "Brose", "Novares")),
        ("DT-CIN-010", "Cab Insulation", "Mannheim", 2025, 10, 11500, 184.00, ("Borgers", "Autoneum", "Adler Pelzer")),
        ("DT-ATP-011", "Attachment Parts", "Gaggenau", 2025, 12, 22000, 58.00, ("Gestamp", "Benteler", "Kirchhoff")),
        ("DT-SVI-012", "Sun Visor", "Aksaray", 2025, 10, 18000, 49.00, ("Kongsberg", "Grupo Antolin", "Grammer")),
        ("DT-SBL-013", "Sun Blind", "Aksaray", 2026, 10, 16000, 70.00, ("Brose", "Grupo Antolin", "Webasto")),
        ("DT-SWB-014", "Steering Wheel Buttons", "Gaggenau", 2027, 8, 21000, 43.00, ("Preh", "Kostal", "ZF")),
        ("DT-INL-015", "Interior Lighting", "Mannheim", 2026, 10, 20000, 76.00, ("Hella", "Forvia", "Marelli")),
    ]

    supplier_names = list(dict.fromkeys(
        supplier_name
        for part in part_records
        for supplier_name in part[-1]
    ))
    countries = ["Germany", "Türkiye", "France", "Belgium", "Austria", "Italy", "Spain"]
    production_locations = [
        "Böblingen", "Amberg", "Coburg", "Bursa", "Stuttgart", "Paris", "Brussels",
        "Graz", "Turin", "Barcelona", "Munich", "Wörth", "Lyon", "Madrid", "Prague",
        "Mannheim", "Gaggenau", "Aksaray", "Brno", "Zaragoza", "Köln", "Bologna", "Valencia",
    ]

    # Seven illustrative records are below the strict OSA qualification limit.
    not_qualified_records = {
        (0, 2), (1, 1), (3, 2), (5, 0), (8, 1), (11, 2), (14, 1)
    }
    supplier_rows: list[dict[str, Any]] = []
    for part_index, part in enumerate(part_records):
        (
            part_number,
            part_description,
            plant,
            _sop_year,
            _lifetime_years,
            annual_volume,
            target_cost,
            candidate_names,
        ) = part
        for candidate_position, supplier_name in enumerate(candidate_names):
            supplier_index = supplier_names.index(supplier_name)
            osa_score = 76 + ((supplier_index * 5 + part_index * 3 + candidate_position * 2) % 17)
            if (part_index, candidate_position) in not_qualified_records:
                osa_score = 62 + ((supplier_index + part_index) % 6)

            quality_score = 80 + ((supplier_index * 4 + part_index * 3 + candidate_position) % 16)
            delivery_score = 80 + ((supplier_index * 3 + part_index * 2 + candidate_position * 2) % 16)
            unit_price = round(
                target_cost * (1.01 + candidate_position * 0.018 + (supplier_index % 4) * 0.006),
                2,
            )
            annual_cost = round(unit_price * annual_volume, 2)
            tooling_cost = round(
                85000 + ((supplier_index * 17500 + part_index * 6500) % 140000),
                2,
            )

            supplier_rows.append(
                {
                    "Record ID": f"{part_number}|{supplier_name}",
                    "Part Number": part_number,
                    "Part Description": part_description,
                    "Supplier Name": supplier_name,
                    "Supplier Code": f"SUP-{supplier_index + 1:03d}",
                    "Country": countries[supplier_index % len(countries)],
                    "Production Location": production_locations[supplier_index],
                    "Plant": plant,
                    "OSA Score": osa_score,
                    "OSA Status": "Qualified" if osa_score >= 70 else "Not Qualified",
                    "Unit Price (€)": unit_price,
                    "Annual Cost (€)": annual_cost,
                    "Tooling Cost (€)": tooling_cost,
                    "Annual Volume": annual_volume,
                    "Target Cost (€)": target_cost,
                    "Quality Score": quality_score,
                    "Defect Rate (%)": round(max(0.15, 2.8 - quality_score * 0.022), 2),
                    "Warranty Claims": max(1, int(round(12 - quality_score / 10 + part_index % 3))),
                    "Incoming Acceptance Rate (%)": round(min(99.8, quality_score + 5.5), 2),
                    "Process Capability (Cpk)": round(1.15 + quality_score / 100 * 0.65, 2),
                    "CART (Days)": round(max(1.5, 8.5 - quality_score * 0.04), 1),
                    "Delivery Score": delivery_score,
                    "On-Time Delivery (%)": round(min(99.5, delivery_score + 3.5), 2),
                    "Lead Time (Days)": round(
                        4.5 + (100 - delivery_score) * 0.08 + candidate_position * 0.4,
                        1,
                    ),
                    "Delivery Accuracy (%)": round(min(99.7, delivery_score + 2.0), 2),
                    "Standard Contract Acceptance": "Approved" if osa_score >= 75 else "Review Required",
                    "Quality Certificate Status": "Approved" if quality_score >= 85 else "Review Required",
                    "Environmental Certificate Status": (
                        "Missing" if supplier_index % 9 == 0
                        else "Approved" if supplier_index % 3 else "Review Required"
                    ),
                    "Supplier Risk Status": "Approved" if osa_score >= 80 else "Review Required",
                    "FSRM Status": (
                        "Approved" if osa_score >= 80 and quality_score >= 85
                        else "Missing" if supplier_index % 5 == 0
                        else "Review Required"
                    ),
                }
            )

    supplier_data = pd.DataFrame(supplier_rows)

    def status_badge(status: str) -> str:
        """Return a translated, high-contrast status badge label."""
        badge_map = {
            "Qualified": ("qualified", f"✅ {t('status_qualified')}"),
            "Not Qualified": ("not-qualified", f"❌ {t('status_not_qualified')}"),
            "Approved": ("approved", f"✅ {t('status_approved')}"),
            "Review Required": ("review", f"⚠️ {t('status_review_required')}"),
            "Missing": ("missing", f"❌ {t('status_missing')}"),
        }
        badge_class, label = badge_map[status]
        return f'<span class="supplier-status-badge {badge_class}">{label}</span>'

    def status_text(status: str) -> str:
        """Return a translated plain-text status for dataframes and filters."""
        status_keys = {
            "Qualified": "status_qualified",
            "Not Qualified": "status_not_qualified",
            "Approved": "status_approved",
            "Review Required": "status_review_required",
            "Missing": "status_missing",
        }
        return t(status_keys[status])

    def render_info_grid(items: list[tuple[str, str]], columns_count: int = 2) -> None:
        """Render labeled metadata cards while keeping all text pure white."""
        for start in range(0, len(items), columns_count):
            row_items = items[start : start + columns_count]
            columns = st.columns(columns_count)
            for column, (label, value) in zip(columns, row_items):
                with column:
                    st.markdown(
                        f"""
                        <div class="supplier-info-item">
                            <div class="supplier-info-label">{label}</div>
                            <div class="supplier-info-value">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    def render_numeric_metrics(
        items: list[tuple[str, str]],
        columns_count: int,
    ) -> None:
        """Render compact metric rows inside a supplier detail section."""
        for start in range(0, len(items), columns_count):
            row_items = items[start : start + columns_count]
            columns = st.columns(columns_count)
            for column, (label, value) in zip(columns, row_items):
                column.metric(label, value)

    # ------------------------------------------------------------------
    # Page header and KPI dashboard
    # ------------------------------------------------------------------
    with st.container(key="supplier_database_content"):
        st.markdown(
            f"""
            <div class="supplier-hero">
                <h1>{t("supplier_title")}</h1>
                <p>{t("supplier_subtitle")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        kpi_columns = st.columns(4)
        kpis = [
            (t("supplier_kpi_candidate_suppliers"), str(len(supplier_data))),
            (
                t("supplier_kpi_qualified_suppliers"),
                str(int((supplier_data["OSA Score"] >= 70).sum())),
            ),
            (t("supplier_kpi_average_osa_score"), str(round(supplier_data["OSA Score"].mean()))),
            (t("supplier_kpi_countries_represented"), str(supplier_data["Country"].nunique())),
        ]
        for kpi_column, (label, value) in zip(kpi_columns, kpis):
            with kpi_column:
                with st.container(border=True):
                    st.metric(label, value)

        # ------------------------------------------------------------------
        # Filter section
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="supplier-section-heading">{t("supplier_table_title")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="supplier_filters"):
            part_number_column, part_description_column, name_column, country_column, status_column = st.columns(5)
            with part_number_column:
                part_number_filter = st.selectbox(
                    t("supplier_filter_part_number"),
                    [t("supplier_all_parts")] + sorted(supplier_data["Part Number"].unique()),
                    key="supplier_filter_part_number_input",
                )
            with part_description_column:
                part_description_filter = st.text_input(
                    t("supplier_filter_part_description"),
                    placeholder=t("supplier_filter_part_description_placeholder"),
                    key="supplier_filter_part_description_input",
                ).strip().lower()
            with name_column:
                supplier_name_filter = st.text_input(
                    t("supplier_filter_name"),
                    placeholder=t("supplier_filter_name_placeholder"),
                    key="supplier_filter_name_input",
                ).strip().lower()
            with country_column:
                country_filter = st.selectbox(
                    t("supplier_filter_country"),
                    [t("supplier_all_countries")] + sorted(supplier_data["Country"].unique()),
                    key="supplier_filter_country_input",
                )
            with status_column:
                status_options = [
                    ("all", t("supplier_all_statuses")),
                    ("Qualified", f"✅ {t('status_qualified')}"),
                    ("Not Qualified", f"❌ {t('status_not_qualified')}"),
                ]
                selected_status_key, _ = st.selectbox(
                    t("supplier_filter_osa_status"),
                    status_options,
                    format_func=lambda option: option[1],
                    key="supplier_filter_osa_status_input",
                )

        filtered_suppliers = supplier_data.copy()
        if part_number_filter != t("supplier_all_parts"):
            filtered_suppliers = filtered_suppliers[
                filtered_suppliers["Part Number"] == part_number_filter
            ]
        if part_description_filter:
            filtered_suppliers = filtered_suppliers[
                filtered_suppliers["Part Description"].str.lower().str.contains(
                    part_description_filter,
                    na=False,
                )
            ]
        if supplier_name_filter:
            filtered_suppliers = filtered_suppliers[
                filtered_suppliers["Supplier Name"].str.lower().str.contains(
                    supplier_name_filter,
                    na=False,
                )
            ]
        if country_filter != t("supplier_all_countries"):
            filtered_suppliers = filtered_suppliers[
                filtered_suppliers["Country"] == country_filter
            ]
        if selected_status_key != "all":
            filtered_suppliers = filtered_suppliers[
                filtered_suppliers["OSA Status"] == selected_status_key
            ]

        st.caption(
            t(
                "supplier_table_count",
                shown=len(filtered_suppliers),
                total=len(supplier_data),
            )
        )
        if filtered_suppliers.empty:
            st.warning(t("supplier_no_matches"))
            return

        st.caption(t("supplier_select_hint"))
        table_columns = [
            "Part Number",
            "Part Description",
            "Supplier Name",
            "Country",
            "Production Location",
            "OSA Score",
            "OSA Status",
            "Annual Cost (€)",
            "Quality Score",
            "Delivery Score",
        ]
        table_column_keys = {
            "Part Number": "part_number",
            "Part Description": "part_description",
            "Supplier Name": "supplier_column_name",
            "Country": "country",
            "Production Location": "production_location",
            "OSA Score": "osa_score",
            "OSA Status": "osa_status",
            "Annual Cost (€)": "annual_cost",
            "Quality Score": "quality_score",
            "Delivery Score": "delivery_score",
        }
        displayed_suppliers = filtered_suppliers[table_columns].copy()
        displayed_suppliers["OSA Status"] = displayed_suppliers["OSA Status"].map(status_text)
        displayed_suppliers = displayed_suppliers.rename(
            columns={column: t(key) for column, key in table_column_keys.items()}
        )
        table_state = st.dataframe(
            corporate_table_style(displayed_suppliers),
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="supplier_catalogue_table",
        )

        selection_state = getattr(table_state, "selection", None)
        selected_rows = getattr(selection_state, "rows", [])
        if isinstance(selection_state, dict):
            selected_rows = selection_state.get("rows", [])

        selected_record_id = st.session_state.get("selected_supplier_record_id")
        if selected_rows:
            selected_record_id = filtered_suppliers.iloc[int(selected_rows[0])]["Record ID"]
            st.session_state["selected_supplier_record_id"] = selected_record_id
        if selected_record_id not in set(filtered_suppliers["Record ID"]):
            selected_record_id = None

        selected_record = None
        if selected_record_id is not None:
            selected_record = supplier_data.loc[
                supplier_data["Record ID"] == selected_record_id
            ].iloc[0]

        # ------------------------------------------------------------------
        # Supplier detail panel: six clearly separated profile sections
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="supplier-section-heading">{t("supplier_detail_title")}</div>',
            unsafe_allow_html=True,
        )
        if selected_record is None:
            st.info(t("supplier_detail_no_selection"))
        else:
            with st.container(border=True, key="supplier_detail_panel"):
                st.markdown(
                    f"""
                    <div class="supplier-profile-heading">{selected_record['Supplier Name']}</div>
                    <p class="supplier-profile-subheading">
                        {selected_record['Part Number']} · {selected_record['Part Description']}
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="supplier-section-heading">{t("supplier_section_basic_info")}</div>',
                    unsafe_allow_html=True,
                )
                render_info_grid(
                    [
                        (t("supplier_name"), str(selected_record["Supplier Name"])),
                        (t("supplier_code"), str(selected_record["Supplier Code"])),
                        (t("country"), str(selected_record["Country"])),
                        (t("production_location"), str(selected_record["Production Location"])),
                        (t("supplier_plant"), str(selected_record["Plant"])),
                        (t("part_number"), str(selected_record["Part Number"])),
                        (t("part_description"), str(selected_record["Part Description"])),
                    ]
                )

                st.markdown(
                    f'<div class="supplier-section-heading">{t("supplier_section_osa_info")}</div>',
                    unsafe_allow_html=True,
                )
                osa_column, status_column = st.columns([1, 2])
                with osa_column:
                    st.metric(t("osa_score"), f"{selected_record['OSA Score']:.0f} / 100")
                with status_column:
                    st.markdown(
                        f'<div class="supplier-info-label">{t("osa_status")}</div>{status_badge(selected_record["OSA Status"])}',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="supplier-section-heading">{t("supplier_section_commercial_info")}</div>',
                    unsafe_allow_html=True,
                )
                render_numeric_metrics(
                    [
                        (t("unit_price"), f"€{selected_record['Unit Price (€)']:,.2f}"),
                        (t("annual_cost"), f"€{selected_record['Annual Cost (€)']:,.0f}"),
                        (t("tooling_cost"), f"€{selected_record['Tooling Cost (€)']:,.0f}"),
                        (t("annual_volume"), f"{selected_record['Annual Volume']:,.0f}"),
                        (t("target_cost"), f"€{selected_record['Target Cost (€)']:,.2f}"),
                    ],
                    5,
                )

                st.markdown(
                    f'<div class="supplier-section-heading">{t("supplier_section_quality_performance")}</div>',
                    unsafe_allow_html=True,
                )
                render_numeric_metrics(
                    [
                        (t("quality_score"), f"{selected_record['Quality Score']:.0f} / 100"),
                        (t("defect_rate"), f"{selected_record['Defect Rate (%)']:.2f}%"),
                        (t("warranty_claims"), f"{selected_record['Warranty Claims']:,}"),
                        (t("incoming_acceptance_rate"), f"{selected_record['Incoming Acceptance Rate (%)']:.2f}%"),
                        (t("process_capability_cpk"), f"{selected_record['Process Capability (Cpk)']:.2f}"),
                        (t("cart_days"), f"{selected_record['CART (Days)']:.1f}"),
                    ],
                    3,
                )

                st.markdown(
                    f'<div class="supplier-section-heading">{t("supplier_section_delivery_performance")}</div>',
                    unsafe_allow_html=True,
                )
                render_numeric_metrics(
                    [
                        (t("delivery_score"), f"{selected_record['Delivery Score']:.0f} / 100"),
                        (t("on_time_delivery"), f"{selected_record['On-Time Delivery (%)']:.2f}%"),
                        (t("lead_time_days"), f"{selected_record['Lead Time (Days)']:.1f}"),
                        (t("delivery_accuracy"), f"{selected_record['Delivery Accuracy (%)']:.2f}%"),
                    ],
                    4,
                )

                st.markdown(
                    f'<div class="supplier-section-heading">{t("supplier_section_awarding_readiness")}</div>',
                    unsafe_allow_html=True,
                )
                readiness_fields = [
                    (t("readiness_standard_contract"), "Standard Contract Acceptance"),
                    (t("readiness_quality_certificate"), "Quality Certificate Status"),
                    (t("readiness_environmental_certificate"), "Environmental Certificate Status"),
                    (t("readiness_supplier_risk"), "Supplier Risk Status"),
                    (t("readiness_fsrm"), "FSRM Status"),
                ]
                readiness_columns = st.columns(5)
                for column, (label, field) in zip(readiness_columns, readiness_fields):
                    with column:
                        st.markdown(
                            f'<div class="supplier-info-label">{label}</div>{status_badge(selected_record[field])}',
                            unsafe_allow_html=True,
                        )

        # ------------------------------------------------------------------
        # Side-by-side supplier comparison view
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="supplier-section-heading">{t("supplier_comparison_title")}</div>',
            unsafe_allow_html=True,
        )
        comparison_options = {
            f"{row['Supplier Name']} · {row['Part Number']}": row["Record ID"]
            for _, row in filtered_suppliers.iterrows()
        }
        selected_comparison_labels = st.multiselect(
            t("supplier_comparison_select"),
            list(comparison_options),
            default=(
                [f"{selected_record['Supplier Name']} · {selected_record['Part Number']}"
                 ] if selected_record is not None else []
            ),
            key="supplier_comparison_selection",
        )
        st.caption(t("supplier_comparison_hint"))

        if len(selected_comparison_labels) > 3:
            selected_comparison_labels = selected_comparison_labels[:3]
            st.warning(t("supplier_comparison_hint"))

        comparison_ids = [comparison_options[label] for label in selected_comparison_labels]
        comparison_records = supplier_data[
            supplier_data["Record ID"].isin(comparison_ids)
        ]
        if comparison_records.empty:
            st.info(t("supplier_detail_no_selection"))
        else:
            comparison_columns = st.columns(len(comparison_records))
            for column, (_, row) in zip(comparison_columns, comparison_records.iterrows()):
                with column:
                    st.markdown(
                        f"""
                        <div class="supplier-comparison-card">
                            <h3>{row['Supplier Name']}</h3>
                            <p class="comparison-part">{row['Part Number']} · {row['Part Description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.container(border=True):
                        st.metric(t("osa_score"), f"{row['OSA Score']:.0f} / 100")
                        st.metric(t("annual_cost"), f"€{row['Annual Cost (€)']:,.0f}")
                        st.metric(t("quality_score"), f"{row['Quality Score']:.0f} / 100")
                        st.metric(t("delivery_score"), f"{row['Delivery Score']:.0f} / 100")
                        comparison_values = [
                            (t("defect_rate"), f"{row['Defect Rate (%)']:.2f}%"),
                            (t("incoming_acceptance_rate"), f"{row['Incoming Acceptance Rate (%)']:.2f}%"),
                            (t("process_capability_cpk"), f"{row['Process Capability (Cpk)']:.2f}"),
                            (t("cart_days"), f"{row['CART (Days)']:.1f}"),
                            (t("on_time_delivery"), f"{row['On-Time Delivery (%)']:.2f}%"),
                            (t("lead_time_days"), f"{row['Lead Time (Days)']:.1f}"),
                            (t("delivery_accuracy"), f"{row['Delivery Accuracy (%)']:.2f}%"),
                        ]
                        st.markdown(
                            "".join(
                                f"<div class=\"supplier-comparison-row\"><span class=\"supplier-comparison-label\">{label}</span><span class=\"supplier-comparison-value\">{value}</span></div>"
                                for label, value in comparison_values
                            ),
                            unsafe_allow_html=True,
                        )


def osa_assessment_page() -> None:
    """Render the interactive six-category OSA qualification workspace."""
    st.markdown(
        """
        <style>
            .osa-hero {
                background: linear-gradient(135deg, #0b1f3a 0%, #163f6b 100%);
                border-radius: 0.75rem;
                box-shadow: 0 14px 32px rgba(11, 31, 58, 0.18);
                color: #FFFFFF;
                margin: 0 0 1.5rem;
                padding: 2.2rem 2.6rem;
            }
            .osa-hero h1 {
                color: #FFFFFF !important;
                font-size: clamp(2rem, 4vw, 3rem);
                font-weight: 720;
                letter-spacing: -0.04em;
                line-height: 1.05;
                margin: 0;
            }
            .osa-hero p {
                color: #FFFFFF !important;
                font-size: 1.04rem;
                line-height: 1.55;
                margin: 0.85rem 0 0;
                max-width: 58rem;
            }
            .osa-section-heading {
                color: #FFFFFF !important;
                font-size: 1.3rem;
                font-weight: 720;
                letter-spacing: -0.02em;
                margin: 1.8rem 0 0.8rem;
            }
            .st-key-osa_assessment_content [data-testid="stMetric"] {
                background: #10273B;
                border: 1px solid #315B85;
                border-radius: 0.65rem;
                padding: 1rem 1.1rem;
            }
            .st-key-osa_assessment_content [data-testid="stMetricLabel"],
            .st-key-osa_assessment_content [data-testid="stMetricValue"],
            .st-key-osa_assessment_content [data-testid="stMetricDelta"] {
                color: #FFFFFF !important;
                opacity: 1 !important;
            }
            .osa-info-item {
                background: #0B1F3A;
                border: 1px solid #315B85;
                border-radius: 0.45rem;
                margin-bottom: 0.6rem;
                min-height: 3.35rem;
                padding: 0.65rem 0.8rem;
            }
            .osa-info-label,
            .osa-card-label,
            .osa-veto-label {
                color: #FFFFFF !important;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                margin-bottom: 0.22rem;
                text-transform: uppercase;
            }
            .osa-info-value {
                color: #FFFFFF !important;
                font-size: 0.94rem;
                font-weight: 600;
                line-height: 1.35;
            }
            .osa-category-card {
                background: #10273B;
                border: 1px solid #315B85;
                border-radius: 0.65rem;
                color: #FFFFFF !important;
                min-height: 18rem;
                padding: 1rem 1.05rem;
            }
            .osa-category-card h3 {
                color: #FFFFFF !important;
                font-size: 1.02rem;
                line-height: 1.3;
                margin: 0 0 0.7rem;
            }
            .osa-category-card p,
            .osa-category-card li,
            .osa-category-card strong {
                color: #FFFFFF !important;
            }
            .osa-category-card ul {
                margin: 0.35rem 0 0;
                padding-left: 1.1rem;
            }
            .osa-category-card li {
                font-size: 0.81rem;
                line-height: 1.4;
                margin: 0.18rem 0;
            }
            .osa-formula {
                background: #0B1F3A;
                border-left: 4px solid #00ADEF;
                border-radius: 0.25rem 0.55rem 0.55rem 0.25rem;
                color: #FFFFFF !important;
                font-size: 1rem;
                line-height: 1.6;
                margin-top: 0.85rem;
                padding: 0.8rem 1rem;
            }
            .osa-result-panel {
                background: linear-gradient(135deg, #0B1F3A 0%, #163F6B 100%);
                border: 1px solid #00ADEF;
                border-radius: 0.65rem;
                color: #FFFFFF !important;
                padding: 1.25rem;
            }
            .osa-result-score {
                color: #FFFFFF !important;
                font-size: 2.4rem;
                font-weight: 800;
                line-height: 1;
                margin: 0.15rem 0 0.75rem;
            }
            .osa-result-panel p,
            .osa-result-panel strong {
                color: #FFFFFF !important;
            }
            .osa-status-badge {
                border-radius: 999px;
                display: inline-block;
                font-size: 0.8rem;
                font-weight: 800;
                line-height: 1.2;
                padding: 0.48rem 0.75rem;
            }
            .osa-status-badge.approved,
            .osa-status-badge.qualified,
            .osa-status-badge.ready,
            .osa-status-badge.clear {
                background: #176B4D;
                color: #FFFFFF !important;
            }
            .osa-status-badge.watch,
            .osa-status-badge.task {
                background: #8A5A00;
                color: #FFFFFF !important;
            }
            .osa-status-badge.missing,
            .osa-status-badge.not-qualified,
            .osa-status-badge.risk,
            .osa-status-badge.triggered {
                background: #9E2A2B;
                color: #FFFFFF !important;
            }
            .osa-compliance-card,
            .osa-action-card,
            .osa-veto-card {
                background: #10273B;
                border: 1px solid #315B85;
                border-radius: 0.55rem;
                color: #FFFFFF !important;
                min-height: 5.4rem;
                padding: 0.75rem;
            }
            .osa-compliance-card .osa-card-label,
            .osa-action-card .osa-card-label,
            .osa-veto-card .osa-veto-label {
                min-height: 2.1rem;
            }
            .osa-action-card.required,
            .osa-veto-card.triggered {
                background: #4A1F24;
                border-color: #E04F5F;
            }
            .osa-action-card.clear,
            .osa-veto-card.clear {
                border-color: #176B4D;
            }
            .osa-veto-card {
                min-height: 6.4rem;
            }
            .osa-veto-card p,
            .osa-action-card p {
                color: #FFFFFF !important;
                font-size: 0.78rem;
                line-height: 1.3;
                margin: 0.5rem 0 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # The OSA catalogue uses the same 15 parts and three-candidate coverage
    # as the Supplier Database: 15 parts x 3 candidate supplier records.
    part_records = [
        ("DT-INT-001", "Interior Trims", "Aksaray", ("Eissmann Automotive", "Grammer", "Brose")),
        ("DT-STW-002", "Stowage Box Above Windshield", "Aksaray", ("Borgers", "Grammer", "Novares")),
        ("DT-STW-003", "Rear Stowage Box", "Wörth", ("Borgers", "Novares", "Eissmann Automotive")),
        ("DT-RHT-004", "Roof Hatch", "Mannheim", ("Webasto", "Roof Systems", "Inalfa")),
        ("DT-HOR-005", "Horn", "Gaggenau", ("Hella", "Forvia", "Bosch")),
        ("DT-BPI-006", "B-Pillar", "Aksaray", ("Brose", "Lear", "Eissmann Automotive")),
        ("DT-SAD-007", "Storage Above Door", "Aksaray", ("Grammer", "Brose", "Novares")),
        ("DT-API-008", "A-Pillar", "Aksaray", ("Eissmann Automotive", "Lear", "Forvia")),
        ("DT-TV-009", "TV Bracket", "Wörth", ("Kongsberg", "Brose", "Novares")),
        ("DT-CIN-010", "Cab Insulation", "Mannheim", ("Borgers", "Autoneum", "Adler Pelzer")),
        ("DT-ATP-011", "Attachment Parts", "Gaggenau", ("Gestamp", "Benteler", "Kirchhoff")),
        ("DT-SVI-012", "Sun Visor", "Aksaray", ("Kongsberg", "Grupo Antolin", "Grammer")),
        ("DT-SBL-013", "Sun Blind", "Aksaray", ("Brose", "Grupo Antolin", "Webasto")),
        ("DT-SWB-014", "Steering Wheel Buttons", "Gaggenau", ("Preh", "Kostal", "ZF")),
        ("DT-INL-015", "Interior Lighting", "Mannheim", ("Hella", "Forvia", "Marelli")),
    ]
    supplier_names = list(dict.fromkeys(
        supplier_name
        for part in part_records
        for supplier_name in part[-1]
    ))
    countries = ["Germany", "Türkiye", "France", "Belgium", "Austria", "Italy", "Spain"]
    production_locations = [
        "Böblingen", "Amberg", "Coburg", "Bursa", "Stuttgart", "Paris", "Brussels",
        "Graz", "Turin", "Barcelona", "Munich", "Wörth", "Lyon", "Madrid", "Prague",
        "Mannheim", "Gaggenau", "Aksaray", "Brno", "Zaragoza", "Köln", "Bologna", "Valencia",
    ]
    not_qualified_records = {
        (0, 2), (1, 1), (3, 2), (5, 0), (8, 1), (11, 2), (14, 1)
    }
    categories = [
        {
            "key": "quality_system",
            "title_key": "osa_category_quality_system",
            "weight": 0.20,
            "subcriteria": [
                "osa_sub_iatf_16949", "osa_sub_iso_9001", "osa_sub_spc", "osa_sub_traceability",
                "osa_sub_pfmea", "osa_sub_control_plan", "osa_sub_8d",
            ],
        },
        {
            "key": "production_capability",
            "title_key": "osa_category_production_capability",
            "weight": 0.20,
            "subcriteria": [
                "osa_sub_standard_work", "osa_sub_process_stability", "osa_sub_oee", "osa_sub_tpm",
                "osa_sub_5s", "osa_sub_preventive_maintenance", "osa_sub_production_flow",
            ],
        },
        {
            "key": "capacity_scalability",
            "title_key": "osa_category_capacity_scalability",
            "weight": 0.15,
            "subcriteria": [
                "osa_sub_current_capacity", "osa_sub_available_capacity", "osa_sub_capacity_utilization",
                "osa_sub_additional_shift", "osa_sub_scalability", "osa_sub_demand_growth",
            ],
        },
        {
            "key": "technical_capability",
            "title_key": "osa_category_technical_capability",
            "weight": 0.15,
            "subcriteria": [
                "osa_sub_similar_product", "osa_sub_engineering_support", "osa_sub_validation",
                "osa_sub_testing", "osa_sub_manufacturing_technology", "osa_sub_rd_support",
            ],
        },
        {
            "key": "logistics_supply_chain",
            "title_key": "osa_category_logistics_supply_chain",
            "weight": 0.15,
            "subcriteria": [
                "osa_sub_material_flow", "osa_sub_fifo", "osa_sub_inventory_management",
                "osa_sub_packaging_management", "osa_sub_delivery_capability", "osa_sub_emergency_logistics",
                "osa_sub_edi",
            ],
        },
        {
            "key": "management_compliance",
            "title_key": "osa_category_management_compliance",
            "weight": 0.15,
            "subcriteria": [
                "osa_sub_psc_rating", "osa_sub_saq_rating", "osa_sub_corrective_actions",
                "osa_sub_fsrm_status", "osa_sub_standard_contract", "osa_sub_supplier_risk",
                "osa_sub_business_continuity",
            ],
        },
    ]

    supplier_rows: list[dict[str, Any]] = []
    for part_index, (part_number, part_description, plant, candidates) in enumerate(part_records):
        for candidate_position, supplier_name in enumerate(candidates):
            supplier_index = supplier_names.index(supplier_name)
            category_scores = {
                "quality_system": 88 + ((supplier_index * 3 + part_index + candidate_position) % 8),
                "production_capability": 85 + ((supplier_index * 5 + part_index + candidate_position) % 10),
                "capacity_scalability": 82 + ((supplier_index * 2 + part_index * 3 + candidate_position) % 11),
                "technical_capability": 84 + ((supplier_index * 4 + part_index + candidate_position * 2) % 10),
                "logistics_supply_chain": 83 + ((supplier_index * 3 + part_index * 2 + candidate_position) % 11),
                "management_compliance": 82 + ((supplier_index * 5 + part_index * 2 + candidate_position) % 12),
            }
            if (part_index, candidate_position) in not_qualified_records:
                category_scores = {
                    category["key"]: 62 + ((supplier_index + part_index + category_index) % 8)
                    for category_index, category in enumerate(categories)
                }

            osa_score = sum(
                category_scores[category["key"]] * category["weight"]
                for category in categories
            )
            critical_open_corrective_action = (
                (supplier_index + part_index + candidate_position) % 11 == 7
            )
            supplier_risk_board_rejection = (
                (supplier_index + part_index + candidate_position) % 13 == 9
            )
            capacity_sufficient = (
                (part_index, candidate_position) not in not_qualified_records
                and (supplier_index + part_index) % 13 != 4
            )
            business_continuity_plan = (
                "Missing" if (supplier_index + part_index) % 10 == 3 else "Approved"
            )
            product_safety_violation = (
                (supplier_index + part_index + candidate_position) % 17 == 5
            )
            open_corrective_actions = (
                3 if critical_open_corrective_action
                else (supplier_index + part_index + candidate_position) % 3
            )
            osa_ica_status = (
                "Missing" if (supplier_index + part_index) % 8 == 0 else "Available"
            )

            supplier_rows.append(
                {
                    "Record ID": f"{part_number}|{supplier_name}",
                    "Part Number": part_number,
                    "Part Description": part_description,
                    "Supplier Name": supplier_name,
                    "Supplier Code": f"SUP-{supplier_index + 1:03d}",
                    "Country": countries[supplier_index % len(countries)],
                    "Production Location": production_locations[supplier_index],
                    "Plant": plant,
                    "OSA Score": osa_score,
                    "OSA Status": "Qualified" if osa_score >= 70 else "Not Qualified",
                    **{
                        f"score_{category_key}": score
                        for category_key, score in category_scores.items()
                    },
                    "FSRM Status": "Approved" if category_scores["management_compliance"] >= 84 else "Watch",
                    "IATF 16949": (
                        "Missing" if (part_index, candidate_position) in {(1, 1), (5, 0)} else "Approved"
                    ),
                    "ISO 14001": "Watch" if supplier_index % 5 == 0 else "Approved",
                    "TISAX": "Watch" if supplier_index % 4 == 0 else "Approved",
                    "Standard Contract Acceptance": (
                        "Approved" if category_scores["management_compliance"] >= 86 else "Watch"
                    ),
                    "Open Corrective Actions": open_corrective_actions,
                    "OSA / ICA Status": osa_ica_status,
                    "Restructuring Requirement": (supplier_index + part_index) % 11 == 0,
                    "Critical Open Corrective Action": critical_open_corrective_action,
                    "Supplier Risk Board Rejection": supplier_risk_board_rejection,
                    "Capacity Sufficient": capacity_sufficient,
                    "Business Continuity Plan": business_continuity_plan,
                    "Product Safety Violation": product_safety_violation,
                }
            )

    osa_data = pd.DataFrame(supplier_rows)

    def status_badge(status: str) -> str:
        """Return a translated badge for OSA/compliance statuses."""
        badge_map = {
            "Approved": ("approved", f"✅ {t('status_approved')}"),
            "Watch": ("watch", f"⚠️ {t('status_watch')}"),
            "Missing": ("missing", f"❌ {t('status_missing')}"),
            "Qualified": ("qualified", f"✅ {t('osa_qualified')}"),
            "Not Qualified": ("not-qualified", f"❌ {t('osa_not_qualified')}"),
        }
        badge_class, label = badge_map[status]
        return f'<span class="osa-status-badge {badge_class}">{label}</span>'

    def render_info_grid(items: list[tuple[str, str]], columns_count: int = 3) -> None:
        """Render basic supplier metadata as high-contrast information cards."""
        for start in range(0, len(items), columns_count):
            row_items = items[start : start + columns_count]
            columns = st.columns(columns_count)
            for column, (label, value) in zip(columns, row_items):
                with column:
                    st.markdown(
                        f"""
                        <div class="osa-info-item">
                            <div class="osa-info-label">{label}</div>
                            <div class="osa-info-value">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    def render_status_card(label: str, status: str, card_class: str = "") -> None:
        """Render a translated status card with an accessible color/state label."""
        st.markdown(
            f"""
            <div class="osa-compliance-card {card_class}">
                <div class="osa-card-label">{label}</div>
                {status_badge(status)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(key="osa_assessment_content"):
        # ------------------------------------------------------------------
        # Page header and KPI dashboard
        # ------------------------------------------------------------------
        st.markdown(
            f"""
            <div class="osa-hero">
                <h1>{t("osa_title")}</h1>
                <p>{t("osa_subtitle")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        kpi_columns = st.columns(4)
        kpis = [
            (t("osa_kpi_assessed_suppliers"), str(len(osa_data))),
            (t("osa_kpi_qualified_suppliers"), str(int((osa_data["OSA Score"] >= 70).sum()))),
            (t("osa_kpi_not_qualified_suppliers"), str(int((osa_data["OSA Score"] < 70).sum()))),
            (t("osa_kpi_threshold"), "70"),
        ]
        for column, (label, value) in zip(kpi_columns, kpis):
            with column:
                with st.container(border=True):
                    st.metric(label, value)

        # ------------------------------------------------------------------
        # Filter section and supplier selection
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="osa-section-heading">{t("osa_supplier_selection_title")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="osa_filters"):
            part_number_column, part_description_column, supplier_column, country_column = st.columns(4)
            with part_number_column:
                part_number_filter = st.selectbox(
                    t("osa_filter_part_number"),
                    [t("osa_all_parts")] + sorted(osa_data["Part Number"].unique()),
                    key="osa_filter_part_number_input",
                )
            with part_description_column:
                part_description_filter = st.text_input(
                    t("osa_filter_part_description"),
                    placeholder=t("osa_filter_part_description_placeholder"),
                    key="osa_filter_part_description_input",
                ).strip().lower()
            with supplier_column:
                supplier_name_filter = st.text_input(
                    t("osa_filter_supplier_name"),
                    placeholder=t("osa_filter_supplier_name_placeholder"),
                    key="osa_filter_supplier_name_input",
                ).strip().lower()
            with country_column:
                country_filter = st.selectbox(
                    t("osa_filter_country"),
                    [t("osa_all_countries")] + sorted(osa_data["Country"].unique()),
                    key="osa_filter_country_input",
                )

        filtered_osa = osa_data.copy()
        if part_number_filter != t("osa_all_parts"):
            filtered_osa = filtered_osa[filtered_osa["Part Number"] == part_number_filter]
        if part_description_filter:
            filtered_osa = filtered_osa[
                filtered_osa["Part Description"].str.lower().str.contains(
                    part_description_filter,
                    na=False,
                )
            ]
        if supplier_name_filter:
            filtered_osa = filtered_osa[
                filtered_osa["Supplier Name"].str.lower().str.contains(
                    supplier_name_filter,
                    na=False,
                )
            ]
        if country_filter != t("osa_all_countries"):
            filtered_osa = filtered_osa[filtered_osa["Country"] == country_filter]

        if filtered_osa.empty:
            st.warning(t("osa_no_matches"))
            return

        st.caption(t("osa_supplier_selection_hint"))
        supplier_options = {
            f"{row['Supplier Name']} · {row['Part Number']}": row["Record ID"]
            for _, row in filtered_osa.iterrows()
        }
        selected_supplier_label = st.selectbox(
            t("osa_supplier_selection"),
            list(supplier_options),
            key="osa_selected_supplier_record",
        )
        selected_record_id = supplier_options[selected_supplier_label]
        selected_record = filtered_osa.loc[
            filtered_osa["Record ID"] == selected_record_id
        ].iloc[0]

        overview_columns = [
            "Part Number", "Part Description", "Supplier Name", "Country",
            "Production Location", "OSA Score", "OSA Status",
        ]
        overview_keys = {
            "Part Number": "part_number",
            "Part Description": "part_description",
            "Supplier Name": "supplier_column_name",
            "Country": "country",
            "Production Location": "production_location",
            "OSA Score": "osa_score",
            "OSA Status": "osa_status",
        }
        overview = filtered_osa[overview_columns].copy()
        overview["OSA Status"] = overview["OSA Status"].map(
            lambda status: t("osa_qualified") if status == "Qualified" else t("osa_not_qualified")
        )
        st.dataframe(
            corporate_table_style(
                overview.rename(columns={column: t(key) for column, key in overview_keys.items()})
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown(
            f'<div class="osa-section-heading">{t("osa_supplier_info_title")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="osa_supplier_info"):
            render_info_grid(
                [
                    (t("supplier_name"), str(selected_record["Supplier Name"])),
                    (t("supplier_code"), str(selected_record["Supplier Code"])),
                    (t("part_number"), str(selected_record["Part Number"])),
                    (t("part_description"), str(selected_record["Part Description"])),
                    (t("country"), str(selected_record["Country"])),
                    (t("production_location"), str(selected_record["Production Location"])),
                ]
            )

        # ------------------------------------------------------------------
        # Six-category interactive OSA assessment framework
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="osa-section-heading">{t("osa_category_title")}</div>',
            unsafe_allow_html=True,
        )
        assessed_scores: dict[str, float] = {}
        category_columns = st.columns(2, gap="large")
        for category_index, category in enumerate(categories):
            with category_columns[category_index % 2]:
                with st.container(border=True, key=f"osa_category_card_{category['key']}"):
                    st.markdown(
                        f"""
                        <div class="osa-category-card">
                            <h3>{t(category['title_key'])}</h3>
                            <div class="osa-card-label">{t('osa_weight')}: {category['weight']:.0%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    default_score = float(selected_record[f"score_{category['key']}"])
                    assessed_scores[category["key"]] = st.slider(
                        f"{t('osa_score')} · {t(category['title_key'])}",
                        min_value=0.0,
                        max_value=100.0,
                        value=default_score,
                        step=0.1,
                        key=f"osa_score_{selected_record_id}_{category['key']}",
                    )
                    st.markdown(
                        f"""
                        <div class="osa-category-card">
                            <div class="osa-card-label">{t('osa_subcriteria')}</div>
                            <ul>
                                {''.join(f'<li>{t(subcriterion)}</li>' for subcriterion in category['subcriteria'])}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ------------------------------------------------------------------
        # OSA breakdown and qualification gate
        # ------------------------------------------------------------------
        total_osa_score = sum(
            assessed_scores[category["key"]] * category["weight"]
            for category in categories
        )
        breakdown = pd.DataFrame(
            {
                t("osa_breakdown_category"): [t(category["title_key"]) for category in categories],
                t("osa_breakdown_weight"): [f"{category['weight']:.0%}" for category in categories],
                t("osa_breakdown_score"): [assessed_scores[category["key"]] for category in categories],
                t("osa_breakdown_contribution"): [
                    assessed_scores[category["key"]] * category["weight"]
                    for category in categories
                ],
            }
        )
        st.markdown(
            f'<div class="osa-section-heading">{t("osa_breakdown_title")}</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(corporate_table_style(breakdown), width="stretch", hide_index=True)
        st.markdown(
            f'<div class="osa-formula"><strong>{t("osa_formula")}</strong></div>',
            unsafe_allow_html=True,
        )

        qualified = total_osa_score >= 70
        with st.container(border=True, key="osa_result_panel"):
            st.markdown(
                f'<div class="osa-section-heading">{t("osa_result_title")}</div>',
                unsafe_allow_html=True,
            )
            result_column, qualification_column = st.columns([1, 2])
            with result_column:
                st.markdown(
                    f'<div class="osa-card-label">{t("osa_total_score")}</div><div class="osa-result-score">{total_osa_score:.1f} / 100</div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(max(total_osa_score / 100, 0.0), 1.0))
            with qualification_column:
                qualification_class = "qualified" if qualified else "not-qualified"
                qualification_label = t("osa_qualified") if qualified else t("osa_not_qualified")
                qualification_icon = "✅" if qualified else "❌"
                st.markdown(
                    f'<div class="osa-card-label">{t("osa_kpi_threshold")}: 70</div><span class="osa-status-badge {qualification_class}">{qualification_icon} {qualification_label}</span><p>{t("osa_exclusion_note")}</p>',
                    unsafe_allow_html=True,
                )

        # ------------------------------------------------------------------
        # Critical findings and veto rules
        # ------------------------------------------------------------------
        critical_rules = [
            (t("osa_veto_missing_iatf"), selected_record["IATF 16949"] == "Missing"),
            (t("osa_veto_corrective_action"), bool(selected_record["Critical Open Corrective Action"])),
            (t("osa_veto_risk_board"), bool(selected_record["Supplier Risk Board Rejection"])),
            (t("osa_veto_capacity"), not bool(selected_record["Capacity Sufficient"])),
            (t("osa_veto_business_continuity"), selected_record["Business Continuity Plan"] == "Missing"),
            (t("osa_veto_product_safety"), bool(selected_record["Product Safety Violation"])),
        ]
        critical_findings = [label for label, triggered in critical_rules if triggered]
        st.markdown(
            f'<div class="osa-section-heading">{t("osa_critical_findings_title")}</div>',
            unsafe_allow_html=True,
        )
        veto_columns = st.columns(3, gap="medium")
        for index, (label, triggered) in enumerate(critical_rules):
            with veto_columns[index % 3]:
                state_class = "triggered" if triggered else "clear"
                state_label = t("osa_veto_triggered") if triggered else t("osa_veto_clear")
                state_icon = "❌" if triggered else "✅"
                st.markdown(
                    f"""
                    <div class="osa-veto-card {state_class}">
                        <div class="osa-veto-label">{label}</div>
                        <span class="osa-status-badge {state_class}">{state_icon} {state_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ------------------------------------------------------------------
        # Compliance, awarding readiness, and action tracking
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="osa-section-heading">{t("osa_compliance_title")}</div>',
            unsafe_allow_html=True,
        )
        compliance_items = [
            (t("osa_compliance_fsrm"), selected_record["FSRM Status"]),
            (t("osa_compliance_iatf"), selected_record["IATF 16949"]),
            (t("osa_compliance_iso_14001"), selected_record["ISO 14001"]),
            (t("osa_compliance_tisax"), selected_record["TISAX"]),
            (t("osa_compliance_contract"), selected_record["Standard Contract Acceptance"]),
        ]
        compliance_columns = st.columns(5, gap="small")
        for column, (label, status) in zip(compliance_columns, compliance_items):
            with column:
                render_status_card(label, status)

        compliance_needs_review = any(status != "Approved" for _, status in compliance_items)
        action_flags = [
            (
                t("osa_action_missing_certificates"),
                any(status == "Missing" for _, status in compliance_items),
            ),
            (
                t("osa_action_open_corrective_actions"),
                int(selected_record["Open Corrective Actions"]) > 0,
            ),
            (
                t("osa_action_missing_osa_ica"),
                selected_record["OSA / ICA Status"] == "Missing",
            ),
            (
                t("osa_action_restructuring"),
                bool(selected_record["Restructuring Requirement"]),
            ),
        ]
        if critical_findings:
            awarding_key = "osa_risk_board_approval"
            awarding_class = "risk"
            awarding_icon = "❌"
        elif compliance_needs_review or any(flag for _, flag in action_flags) or not qualified:
            awarding_key = "osa_task_for_awarding"
            awarding_class = "task"
            awarding_icon = "⚠️"
        else:
            awarding_key = "osa_ready_for_award"
            awarding_class = "ready"
            awarding_icon = "✅"

        st.markdown(
            f'<div class="osa-section-heading">{t("osa_awarding_title")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(
                f'<span class="osa-status-badge {awarding_class}">{awarding_icon} {t(awarding_key)}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="osa-section-heading">{t("osa_actions_title")}</div>',
            unsafe_allow_html=True,
        )
        action_columns = st.columns(4, gap="small")
        for column, (label, required) in zip(action_columns, action_flags):
            with column:
                state_class = "required" if required else "clear"
                state_icon = "⚠️" if required else "✅"
                state_label = t("osa_action_required") if required else t("osa_action_clear")
                st.markdown(
                    f"""
                    <div class="osa-action-card {state_class}">
                        <div class="osa-card-label">{label}</div>
                        <p>{state_icon} {state_label}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# -----------------------------------------------------------------------------
# Existing Supplier Optimization UI
# -----------------------------------------------------------------------------


def display_ranking(result: pd.DataFrame, score_title: str) -> None:
    """Display ranking table and a basic score chart."""
    st.subheader(t("ranking_results"))
    displayed_result = result.rename(
        columns={
            "score": t("column_score"),
            "rank": t("column_rank"),
        }
    ).copy()
    displayed_result.index.name = t("supplier_column_name")
    st.dataframe(corporate_table_style(displayed_result), width="stretch")
    st.caption(t("higher_score", score=score_title))
    chart_data = result.sort_values("score")[["score"]]
    st.bar_chart(chart_data, horizontal=True)


def display_allocation(
    allocation: pd.DataFrame,
    metrics: pd.Series,
    title: str,
) -> None:
    """Display allocation results, metrics, and a basic allocation chart."""
    st.subheader(title)
    visible_allocation = allocation[allocation["allocation_units"] > 1e-8]
    displayed_allocation = visible_allocation.rename(
        columns={
            "allocation_units": t("column_allocation_units"),
            "allocation_share": t("column_allocation_share"),
        }
    ).copy()
    displayed_allocation.index.name = t("supplier_column_name")
    st.dataframe(corporate_table_style(displayed_allocation), width="stretch")
    st.bar_chart(
        allocation.sort_values("allocation_units")[["allocation_units"]],
        horizontal=True,
    )

    st.subheader(t("solution_metrics"))
    metric_items = list(metrics.items())
    for start in range(0, len(metric_items), 4):
        metric_columns = st.columns(min(4, len(metric_items) - start))
        for column, (name, value) in zip(metric_columns, metric_items[start : start + 4]):
            column.metric(translate_metric_label(name), f"{value:,.4f}")


def run_analysis(
    method: str,
    criteria: pd.DataFrame,
    weights: dict[str, float],
    impacts: dict[str, str],
    numeric_data: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Dispatch the selected UI configuration to the core algorithm functions."""
    if method == "TOPSIS":
        result = topsis(criteria, weights, impacts)
        display_ranking(result, t("score_topsis"))
        return result_for_download(result)

    if method == "Weighted Sum":
        result = weighted_sum(criteria, weights, impacts)
        display_ranking(result, t("score_weighted_sum"))
        return result_for_download(result)

    capacity_column = settings["capacity_column"]
    if capacity_column is None:
        raise ValueError(t("error_capacity_column"))

    capacities = numeric_data[capacity_column].to_dict()
    common_arguments = {
        "data": criteria,
        "demand_units": settings["demand_units"],
        "capacities": capacities,
        "cost_col": settings["cost_column"],
        "quality_col": settings["quality_column"],
        "delivery_col": settings["delivery_column"],
    }

    if method == "Preemptive Optimization":
        allocation, metrics, stages = preemptive_optimization(
            **common_arguments,
            quality_target=settings["quality_target"],
            delivery_target=settings["delivery_target"],
        )
        display_allocation(allocation, metrics, t("preemptive_allocation"))
        st.subheader(t("lexicographic_results"))
        stage_values = pd.Series(
            {translate_metric_label(name): value for name, value in stages.items()},
            name=t("column_value"),
        ).to_frame()
        st.dataframe(corporate_table_style(stage_values), width="stretch")
        return result_for_download(allocation, metrics, stages)

    allocation, metrics = goal_programming(
        **common_arguments,
        quality_target=settings["quality_target"],
        delivery_target=settings["delivery_target"],
        cost_target=settings["cost_target"],
        goal_weights=settings["goal_weights"],
    )
    display_allocation(allocation, metrics, t("goal_programming_allocation"))
    return result_for_download(allocation, metrics)


def supplier_optimization_page() -> None:
    """Render the five-page portal's part-driven supplier optimization workflow."""
    from datetime import datetime
    from html import escape

    st.markdown(
        """
        <style>
            .stApp {
                background-image: radial-gradient(
                    circle at 8% 0%,
                    color-mix(in srgb, currentColor 5%, transparent) 0,
                    transparent 32rem
                );
            }
            .block-container {
                max-width: 1440px;
                padding-top: calc(9rem + env(safe-area-inset-top));
                padding-bottom: 4rem;
            }
            .optimization-hero {
                background: linear-gradient(135deg, #0B1F3A 0%, #163F6B 100%);
                border-radius: 0.75rem;
                box-shadow: 0 14px 32px rgba(11, 31, 58, 0.18);
                color: #FFFFFF;
                margin: 0 0 1.4rem;
                padding: 2.25rem 2.6rem;
            }
            .optimization-hero h1 {
                color: #FFFFFF !important;
                font-size: clamp(2rem, 4vw, 3rem);
                font-weight: 750;
                letter-spacing: -0.04em;
                line-height: 1.05;
                margin: 0;
            }
            .optimization-hero p {
                color: #FFFFFF !important;
                font-size: 1.04rem;
                line-height: 1.55;
                margin: 0.85rem 0 0;
                max-width: 62rem;
            }
            .optimization-section-heading {
                color: #FFFFFF !important;
                font-size: 1.3rem;
                font-weight: 750;
                letter-spacing: -0.02em;
                margin: 1.7rem 0 0.8rem;
            }
            .optimization-workflow {
                align-items: stretch;
                display: flex;
                gap: 0.35rem;
                justify-content: center;
                margin: 0 0 1.5rem;
                overflow-x: auto;
                padding-bottom: 0.15rem;
            }
            .optimization-workflow-step {
                background: #10273B;
                border: 1px solid #315B85;
                border-radius: 0.35rem;
                color: #FFFFFF !important;
                flex: 1 0 10rem;
                font-size: 0.78rem;
                font-weight: 700;
                line-height: 1.25;
                padding: 0.75rem 0.6rem;
                text-align: center;
            }
            .optimization-workflow-step.active {
                background: #00ADEF;
                border-color: #00ADEF;
                color: #FFFFFF !important;
            }
            .optimization-workflow-arrow {
                align-items: center;
                color: #FFFFFF !important;
                display: flex;
                font-size: 1.2rem;
                justify-content: center;
            }
            .optimization-method-card {
                background: #10273B;
                border: 1px solid #315B85;
                border-radius: 0.6rem;
                color: #FFFFFF !important;
                min-height: 9.2rem;
                padding: 0.95rem 1rem;
            }
            .optimization-method-card.selected {
                background: #163F6B;
                border: 2px solid #00ADEF;
            }
            .optimization-method-card h3,
            .optimization-method-card p,
            .optimization-method-card strong {
                color: #FFFFFF !important;
            }
            .optimization-method-card h3 {
                font-size: 1rem;
                margin: 0 0 0.5rem;
            }
            .optimization-method-card p {
                font-size: 0.82rem;
                line-height: 1.35;
                margin: 0.32rem 0;
            }
            .optimization-info-item {
                background: #0B1F3A;
                border: 1px solid #315B85;
                border-radius: 0.45rem;
                margin-bottom: 0.6rem;
                min-height: 3.3rem;
                padding: 0.65rem 0.8rem;
            }
            .optimization-info-label,
            .optimization-card-label {
                color: #FFFFFF !important;
                font-size: 0.72rem;
                font-weight: 750;
                letter-spacing: 0.04em;
                margin-bottom: 0.22rem;
                text-transform: uppercase;
            }
            .optimization-info-value {
                color: #FFFFFF !important;
                font-size: 0.95rem;
                font-weight: 650;
                line-height: 1.35;
            }
            .optimization-note,
            .optimization-formula,
            .optimization-recommended-card,
            .optimization-summary-card {
                background: #10273B;
                border: 1px solid #315B85;
                border-radius: 0.6rem;
                color: #FFFFFF !important;
                padding: 1rem 1.1rem;
            }
            .optimization-note {
                border-left: 4px solid #00ADEF;
                line-height: 1.5;
                margin: 0.7rem 0;
            }
            .optimization-note strong,
            .optimization-note p,
            .optimization-formula strong,
            .optimization-summary-card strong,
            .optimization-summary-card p {
                color: #FFFFFF !important;
            }
            .optimization-note p,
            .optimization-summary-card p {
                font-size: 0.9rem;
                margin: 0.35rem 0 0;
            }
            .optimization-formula {
                border-left: 4px solid #00ADEF;
                font-size: 1rem;
                line-height: 1.55;
                margin-top: 0.8rem;
            }
            .optimization-recommended-card {
                background: linear-gradient(135deg, #0B1F3A 0%, #163F6B 100%);
                border: 1px solid #00ADEF;
                min-height: 15rem;
            }
            .optimization-recommended-card h2,
            .optimization-recommended-card h3,
            .optimization-recommended-card p,
            .optimization-recommended-card strong {
                color: #FFFFFF !important;
            }
            .optimization-recommended-card h2 {
                font-size: 1.1rem;
                margin: 0 0 0.7rem;
            }
            .optimization-recommended-name {
                color: #FFFFFF !important;
                font-size: 1.65rem;
                font-weight: 800;
                line-height: 1.15;
                margin: 0.3rem 0 0.9rem;
            }
            .optimization-recommended-card p {
                font-size: 0.87rem;
                line-height: 1.4;
                margin: 0.35rem 0;
            }
            .optimization-summary-card {
                min-height: 100%;
            }
            .optimization-summary-card h3 {
                color: #FFFFFF !important;
                font-size: 1.02rem;
                margin: 0 0 0.8rem;
            }
            .optimization-summary-row {
                border-top: 1px solid #315B85;
                display: flex;
                gap: 0.7rem;
                justify-content: space-between;
                padding: 0.45rem 0;
            }
            .optimization-summary-row span {
                color: #FFFFFF !important;
                font-size: 0.82rem;
            }
            .optimization-summary-row span:last-child {
                font-weight: 800;
                text-align: right;
            }
            .optimization-report-meta {
                color: #FFFFFF !important;
                font-size: 0.85rem;
                margin: 0.4rem 0 0.9rem;
            }
            .optimization-run-button [data-testid="stBaseButton-primary"] {
                background: #00ADEF !important;
                border-color: #00ADEF !important;
                color: #FFFFFF !important;
                font-size: 1rem;
                font-weight: 800;
                min-height: 3rem;
            }
            .optimization-run-button [data-testid="stBaseButton-primary"]:hover {
                background: #0078D6 !important;
                border-color: #0078D6 !important;
            }
            .st-key-optimization_method_choice [data-testid="stRadio"] > label {
                color: #FFFFFF !important;
            }
            .st-key-optimization_method_choice [data-testid="stRadio"] [role="radiogroup"] {
                gap: 0.5rem !important;
            }
            .st-key-optimization_method_choice [data-testid="stRadio"] [role="radiogroup"] > label {
                background: #10273B !important;
                border: 1px solid #315B85 !important;
                border-radius: 0.45rem !important;
                color: #FFFFFF !important;
                padding: 0.65rem 0.8rem !important;
            }
            .st-key-optimization_method_choice [data-testid="stRadio"] [role="radiogroup"] > label:hover,
            .st-key-optimization_method_choice [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
                background: #163F6B !important;
                border-color: #00ADEF !important;
                color: #FFFFFF !important;
            }
            .st-key-optimization_method_choice [data-testid="stRadio"] [role="radiogroup"] > label p,
            .st-key-optimization_method_choice [data-testid="stRadio"] [role="radiogroup"] > label div {
                color: #FFFFFF !important;
            }
            @media (max-width: 768px) {
                .block-container {
                    padding-top: calc(7.5rem + env(safe-area-inset-top));
                }
                .optimization-hero {
                    padding: 1.8rem 1.35rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Shared part master used to drive Step 1 and the default supplier dataset.
    part_records = [
        ("DT-INT-001", "Interior Trims", "Aksaray", 2025, 10, 18000, 135.00, 118.00, ("Eissmann Automotive", "Grammer", "Brose")),
        ("DT-STW-002", "Stowage Box Above Windshield", "Aksaray", 2025, 10, 12000, 96.00, 82.00, ("Borgers", "Grammer", "Novares")),
        ("DT-STW-003", "Rear Stowage Box", "Wörth", 2026, 10, 12000, 110.00, 95.00, ("Borgers", "Novares", "Eissmann Automotive")),
        ("DT-RHT-004", "Roof Hatch", "Mannheim", 2025, 12, 8500, 265.00, 230.00, ("Webasto", "Roof Systems", "Inalfa")),
        ("DT-HOR-005", "Horn", "Gaggenau", 2025, 12, 22000, 48.00, 39.00, ("Hella", "Forvia", "Bosch")),
        ("DT-BPI-006", "B-Pillar", "Aksaray", 2026, 10, 14000, 188.00, 165.00, ("Brose", "Lear", "Eissmann Automotive")),
        ("DT-SAD-007", "Storage Above Door", "Aksaray", 2026, 10, 13500, 124.00, 106.00, ("Grammer", "Brose", "Novares")),
        ("DT-API-008", "A-Pillar", "Aksaray", 2026, 10, 14000, 172.00, 149.00, ("Eissmann Automotive", "Lear", "Forvia")),
        ("DT-TV-009", "TV Bracket", "Wörth", 2027, 8, 5000, 142.00, 120.00, ("Kongsberg", "Brose", "Novares")),
        ("DT-CIN-010", "Cab Insulation", "Mannheim", 2025, 10, 11500, 215.00, 184.00, ("Borgers", "Autoneum", "Adler Pelzer")),
        ("DT-ATP-011", "Attachment Parts", "Gaggenau", 2025, 12, 22000, 72.00, 58.00, ("Gestamp", "Benteler", "Kirchhoff")),
        ("DT-SVI-012", "Sun Visor", "Aksaray", 2025, 10, 18000, 61.00, 49.00, ("Kongsberg", "Grupo Antolin", "Grammer")),
        ("DT-SBL-013", "Sun Blind", "Aksaray", 2026, 10, 16000, 84.00, 70.00, ("Brose", "Grupo Antolin", "Webasto")),
        ("DT-SWB-014", "Steering Wheel Buttons", "Gaggenau", 2027, 8, 21000, 53.00, 43.00, ("Preh", "Kostal", "ZF")),
        ("DT-INL-015", "Interior Lighting", "Mannheim", 2026, 10, 20000, 92.00, 76.00, ("Hella", "Forvia", "Marelli")),
    ]
    part_catalogue = pd.DataFrame(
        [
            {
                "Part Number": record[0],
                "Part Description": record[1],
                "Plant": record[2],
                "SOP Year": record[3],
                "Lifetime (Years)": record[4],
                "Annual Volume": record[5],
                "Budget (€)": record[6],
                "Target Cost (€)": record[7],
            }
            for record in part_records
        ]
    )

    supplier_names = list(dict.fromkeys(
        supplier_name
        for record in part_records
        for supplier_name in record[-1]
    ))
    countries = ["Germany", "Türkiye", "France", "Belgium", "Austria", "Italy", "Spain"]
    not_qualified_records = {
        (0, 2), (1, 1), (3, 2), (5, 0), (8, 1), (11, 2), (14, 1)
    }

    def build_default_dataset() -> pd.DataFrame:
        """Create a complete 15-part x 3-supplier sample for the portal."""
        rows: list[dict[str, Any]] = []
        for part_index, record in enumerate(part_records):
            (
                part_number,
                _description,
                _plant,
                _sop_year,
                _lifetime_years,
                annual_volume,
                _budget,
                target_cost,
                candidates,
            ) = record
            for candidate_position, supplier_name in enumerate(candidates):
                supplier_index = supplier_names.index(supplier_name)
                osa_score = 76 + ((supplier_index * 5 + part_index * 3 + candidate_position * 2) % 17)
                if (part_index, candidate_position) in not_qualified_records:
                    osa_score = 62 + ((supplier_index + part_index) % 6)

                defect_rate = round(0.45 + ((supplier_index + part_index + candidate_position) % 8) * 0.18, 2)
                warranty_claims = 1 + ((supplier_index * 2 + part_index + candidate_position) % 6)
                incoming_acceptance = round(96.2 - ((supplier_index + part_index) % 8) * 0.8, 2)
                cpk = round(1.28 + ((supplier_index * 3 + part_index + candidate_position) % 8) * 0.06, 2)
                cart_days = round(2.6 + ((supplier_index + candidate_position) % 6) * 0.45, 1)
                on_time_delivery = round(96.0 - ((supplier_index * 2 + part_index) % 8) * 0.9, 2)
                lead_time = round(4.2 + ((supplier_index + part_index + candidate_position) % 6) * 0.5, 1)
                delivery_accuracy = round(97.0 - ((supplier_index + part_index * 2) % 7) * 0.85, 2)
                unit_price = round(
                    target_cost * (1.01 + candidate_position * 0.018 + (supplier_index % 4) * 0.006),
                    2,
                )
                rows.append(
                    {
                        "Part Number": part_number,
                        "Supplier": supplier_name,
                        "Annual Volume": annual_volume,
                        "Annual Cost": round(unit_price * annual_volume, 2),
                        "Defect Rate": defect_rate,
                        "Warranty Claims": warranty_claims,
                        "Incoming Acceptance Rate": incoming_acceptance,
                        "Process Capability": cpk,
                        "Corrective Action Response Time": cart_days,
                        "On-Time Delivery": on_time_delivery,
                        "Lead Time": lead_time,
                        "Delivery Accuracy": delivery_accuracy,
                        "OSA Score": osa_score,
                        "Country": countries[supplier_index % len(countries)],
                        # Used only as a capacity constraint by the existing LP methods.
                        "Capacity": round(annual_volume * (0.45 + candidate_position * 0.12 + (supplier_index % 3) * 0.03), 2),
                    }
                )
        return pd.DataFrame(rows)

    def canonical_column_key(column: Any) -> str:
        """Normalize uploaded headers for forgiving Excel/CSV aliases."""
        return "".join(character.lower() for character in str(column) if character.isalnum())

    def normalize_optimization_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Map common upload aliases to the optimization schema and validate inputs."""
        data = data.copy()
        aliases = {
            "partnumber": "Part Number",
            "part": "Part Number",
            "partdescription": "Part Description",
            "description": "Part Description",
            "supplier": "Supplier",
            "suppliername": "Supplier",
            "vendor": "Supplier",
            "annualcost": "Annual Cost",
            "annualsuppliercost": "Annual Cost",
            "cost": "Annual Cost",
            "defectrate": "Defect Rate",
            "warrantyclaims": "Warranty Claims",
            "incomingacceptancerate": "Incoming Acceptance Rate",
            "processcapability": "Process Capability",
            "processcapabilitycpk": "Process Capability",
            "correctiveactionresponsetime": "Corrective Action Response Time",
            "correctiveactionresponsetimedays": "Corrective Action Response Time",
            "correctiveactionresponsetimecartdays": "Corrective Action Response Time",
            "cartdays": "Corrective Action Response Time",
            "ontimedelivery": "On-Time Delivery",
            "leadtime": "Lead Time",
            "leadtimedays": "Lead Time",
            "deliveryaccuracy": "Delivery Accuracy",
            "osascore": "OSA Score",
            "country": "Country",
            "annualvolume": "Annual Volume",
            "capacity": "Capacity",
        }
        rename_map: dict[Any, str] = {}
        existing_columns = set(data.columns)
        for column in data.columns:
            target = aliases.get(canonical_column_key(column))
            if target and target not in existing_columns:
                rename_map[column] = target
        data = data.rename(columns=rename_map)

        required_columns = [
            "Part Number", "Supplier", "Annual Cost", "Defect Rate", "Warranty Claims",
            "Incoming Acceptance Rate", "Process Capability", "Corrective Action Response Time",
            "On-Time Delivery", "Lead Time", "Delivery Accuracy", "OSA Score", "Country",
        ]
        missing_columns = [column for column in required_columns if column not in data.columns]
        if missing_columns:
            raise ValueError(t("optimization_missing_required_columns", columns=missing_columns))

        data["Part Number"] = data["Part Number"].astype(str).str.strip()
        data["Supplier"] = data["Supplier"].astype(str).str.strip()
        numeric_columns = [column for column in required_columns if column not in {"Part Number", "Supplier", "Country"}]
        for column in numeric_columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[numeric_columns].isna().any().any():
            invalid_columns = data[numeric_columns].columns[data[numeric_columns].isna().any()].tolist()
            raise ValueError(t("optimization_invalid_numeric_columns", columns=invalid_columns))
        if not np.isfinite(data[numeric_columns].to_numpy()).all():
            raise ValueError(t("optimization_finite_numeric"))

        part_volume_map = dict(
            zip(
                part_catalogue["Part Number"].astype(str).str.strip(),
                part_catalogue["Annual Volume"],
            )
        )
        part_description_volume_map = dict(
            zip(
                part_catalogue["Part Description"].astype(str).str.strip().str.casefold(),
                part_catalogue["Annual Volume"],
            )
        )
        mapped_volume = data["Part Number"].map(part_volume_map)
        if "Part Description" in data.columns:
            description_volume = data["Part Description"].astype(str).str.strip().str.casefold().map(
                part_description_volume_map
            )
            mapped_volume = mapped_volume.fillna(description_volume)

        if "Annual Volume" not in data.columns:
            data["Annual Volume"] = mapped_volume
        else:
            data["Annual Volume"] = pd.to_numeric(data["Annual Volume"], errors="coerce")
            data["Annual Volume"] = data["Annual Volume"].fillna(mapped_volume)
        if data["Annual Volume"].isna().any():
            missing_parts = data.loc[data["Annual Volume"].isna(), "Part Number"].drop_duplicates().tolist()
            raise ValueError(t("optimization_missing_annual_volume", parts=missing_parts))
        capacity_assumed = "Capacity" not in data.columns
        if capacity_assumed:
            data["Capacity"] = data["Annual Volume"]
        data["Capacity"] = pd.to_numeric(data["Capacity"], errors="coerce")
        if data["Capacity"].isna().any() or (data["Capacity"] < 0).any():
            raise ValueError(t("optimization_invalid_capacity"))
        return data, capacity_assumed

    def render_info_grid(items: list[tuple[str, str]], columns_count: int = 3) -> None:
        """Render compact high-contrast metadata cards."""
        for start in range(0, len(items), columns_count):
            row_items = items[start : start + columns_count]
            columns = st.columns(columns_count)
            for column, (label, value) in zip(columns, row_items):
                with column:
                    st.markdown(
                        f"""
                        <div class="optimization-info-item">
                            <div class="optimization-info-label">{label}</div>
                            <div class="optimization-info-value">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    def construct_criteria(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Construct only Cost, Quality, and Delivery optimization criteria."""
        data = data.copy()
        defect_component = np.clip(100 - data["Defect Rate"] * 12, 0, 100)
        warranty_component = np.clip(100 - data["Warranty Claims"] * 8, 0, 100)
        cpk_component = np.clip((data["Process Capability"] - 1.0) / 0.8 * 100, 0, 100)
        quality_score = (
            0.25 * defect_component
            + 0.20 * warranty_component
            + 0.30 * data["Incoming Acceptance Rate"]
            + 0.25 * cpk_component
        )
        lead_time_component = np.clip(100 - data["Lead Time"] * 8, 0, 100)
        delivery_score = (
            0.45 * data["On-Time Delivery"]
            + 0.20 * lead_time_component
            + 0.35 * data["Delivery Accuracy"]
        )
        supplier_index = pd.Index(data["Supplier"].astype(str), name="Supplier")
        criteria = pd.DataFrame(
            {
                "Cost": data["Annual Cost"].to_numpy(dtype=float),
                "Quality": quality_score.to_numpy(dtype=float),
                "Delivery": delivery_score.to_numpy(dtype=float),
            },
            index=supplier_index,
        )
        components = data.copy()
        components["Quality"] = quality_score.to_numpy(dtype=float)
        components["Delivery"] = delivery_score.to_numpy(dtype=float)
        return criteria, components

    def radar_svg(labels: list[str], values: list[float]) -> str:
        """Render a dependency-free SVG radar chart for Cost, Quality, Delivery."""
        import math

        width, height = 430, 340
        center_x, center_y, radius = 215, 160, 100
        angles = [-math.pi / 2, -math.pi / 2 + 2 * math.pi / 3, -math.pi / 2 + 4 * math.pi / 3]

        def point(value: float, angle: float) -> tuple[float, float]:
            bounded = min(max(value, 0.0), 100.0) / 100
            return center_x + radius * bounded * math.cos(angle), center_y + radius * bounded * math.sin(angle)

        def point_string(scale: float) -> str:
            return " ".join(
                f"{center_x + radius * scale * math.cos(angle):.1f},{center_y + radius * scale * math.sin(angle):.1f}"
                for angle in angles
            )

        axis_lines = "".join(
            f'<line x1="{center_x}" y1="{center_y}" x2="{center_x + radius * math.cos(angle):.1f}" y2="{center_y + radius * math.sin(angle):.1f}" stroke="#315B85" stroke-width="1" />'
            for angle in angles
        )
        rings = "".join(
            f'<polygon points="{point_string(scale)}" fill="none" stroke="#315B85" stroke-width="1" />'
            for scale in (0.25, 0.5, 0.75, 1.0)
        )
        value_points = " ".join(
            f"{point(value, angle)[0]:.1f},{point(value, angle)[1]:.1f}"
            for value, angle in zip(values, angles)
        )
        label_positions = [
            (center_x, center_y - radius - 20, "middle"),
            (center_x + radius + 24, center_y + radius * 0.58, "start"),
            (center_x - radius - 24, center_y + radius * 0.58, "end"),
        ]
        label_svg = "".join(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="#FFFFFF" font-size="12" font-weight="700">{escape(label)}</text>'
            for label, (x, y, anchor) in zip(labels, label_positions)
        )
        return f"""
        <svg class="optimization-radar" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(t('optimization_radar_chart'))}">
            <rect width="100%" height="100%" rx="12" fill="#10273B" />
            {rings}
            {axis_lines}
            <polygon points="{value_points}" fill="#00ADEF" fill-opacity="0.35" stroke="#00ADEF" stroke-width="3" />
            {label_svg}
            <text x="215" y="318" text-anchor="middle" fill="#FFFFFF" font-size="11">{escape(t('optimization_radar_normalized_note'))}</text>
        </svg>
        """

    default_data = build_default_dataset()
    part_lookup = part_catalogue.set_index("Part Number").to_dict("index")

    with st.container(key="optimization_portal_content"):
        # ------------------------------------------------------------------
        # Header and workflow overview
        # ------------------------------------------------------------------
        st.markdown(
            f"""
            <div class="optimization-hero">
                <h1>{t("optimization_title")}</h1>
                <p>{t("optimization_description")}</p>
            </div>
            <div class="optimization-workflow" aria-label="Workflow overview">
                <div class="optimization-workflow-step">{t("workflow_parts_database")}</div>
                <div class="optimization-workflow-arrow">➔</div>
                <div class="optimization-workflow-step">{t("workflow_supplier_database")}</div>
                <div class="optimization-workflow-arrow">➔</div>
                <div class="optimization-workflow-step">{t("workflow_osa_assessment")}</div>
                <div class="optimization-workflow-arrow">➔</div>
                <div class="optimization-workflow-step active">{t("workflow_supplier_optimization")}</div>
                <div class="optimization-workflow-arrow">➔</div>
                <div class="optimization-workflow-step">{t("workflow_results_reports")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------------
        # Step 1: Part selection and metadata
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="optimization-section-heading">{t("optimization_step_part_selection")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="optimization_part_selection_panel"):
            st.caption(t("optimization_part_selection_hint"))
            selected_part_number = st.selectbox(
                t("optimization_part_selection"),
                part_catalogue["Part Number"].tolist(),
                format_func=lambda value: f"{value} · {part_lookup[value]['Part Description']}",
                key="optimization_selected_part",
            )
            selected_part = part_lookup[selected_part_number]
            default_part_data = default_data[default_data["Part Number"] == selected_part_number]
            render_info_grid(
                [
                    (t("part_number"), selected_part_number),
                    (t("part_description"), str(selected_part["Part Description"])),
                    (t("annual_volume"), f"{selected_part['Annual Volume']:,.0f}"),
                    (t("budget"), f"€{selected_part['Budget (€)']:,.2f}"),
                    (t("supplier_kpi_candidate_suppliers"), f"{len(default_part_data)}"),
                    (
                        t("supplier_kpi_qualified_suppliers"),
                        f"{int((default_part_data['OSA Score'] >= 70).sum())}",
                    ),
                ]
            )

        # ------------------------------------------------------------------
        # Step 2: Method selection cards
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="optimization-section-heading">{t("optimization_step_method_selection")}</div>',
            unsafe_allow_html=True,
        )
        method_options = ["Weighted Sum", "TOPSIS", "Goal Programming", "Preemptive Optimization"]
        method_keys = {
            "Weighted Sum": "method_weighted_sum",
            "TOPSIS": "method_topsis",
            "Goal Programming": "method_goal_programming",
            "Preemptive Optimization": "method_preemptive",
        }
        method_card_info = [
            ("Weighted Sum", "optimization_method_weighted_sum_summary", "optimization_method_weighted_sum_best_when"),
            ("TOPSIS", "optimization_method_topsis_summary", "optimization_method_topsis_best_when"),
            ("Goal Programming", "optimization_method_goal_programming_summary", "optimization_method_goal_programming_best_when"),
            ("Preemptive Optimization", "optimization_method_preemptive_summary", "optimization_method_preemptive_best_when"),
        ]
        with st.container(border=True, key="optimization_method_choice"):
            selected_method = st.radio(
                t("optimization_method_selection"),
                method_options,
                format_func=lambda option: t(method_keys[option]),
                horizontal=True,
                key="optimization_method_choice_radio",
            )
            st.caption(t("optimization_method_selection_hint"))
            method_card_columns = st.columns(4, gap="small")
            for column, (method_name, summary_key, best_when_key) in zip(method_card_columns, method_card_info):
                with column:
                    selected_class = "selected" if selected_method == method_name else ""
                    st.markdown(
                        f"""
                        <div class="optimization-method-card {selected_class}">
                            <h3>{t(method_keys[method_name])}</h3>
                            <p><strong>{t('method_description')}:</strong> {t(summary_key)}</p>
                            <p><strong>{t('method_best_when')}:</strong> {t(best_when_key)}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ------------------------------------------------------------------
        # Step 3: Data import and preview
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="optimization-section-heading">{t("optimization_step_data_import")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="optimization_data_import_panel"):
            uploaded_file = st.file_uploader(
                t("optimization_upload_label"),
                type=["xlsx", "xls", "csv"],
                help=t("optimization_upload_help"),
                key="optimization_supplier_upload",
            )
            if uploaded_file is None:
                source_data = default_data.copy()
                st.caption(t("optimization_default_data"))
                source_signature = "default"
                capacity_assumed = False
            else:
                try:
                    uploaded_data = read_supplier_file(uploaded_file.name, uploaded_file.getvalue())
                    source_data, capacity_assumed = normalize_optimization_dataset(uploaded_data)
                    source_signature = f"{uploaded_file.name}:{len(source_data)}"
                    st.caption(f"{t('using_file')} {uploaded_file.name}")
                except Exception as error:
                    st.error(t("optimization_upload_error", error=error))
                    st.stop()

            st.markdown(
                f'<div class="optimization-card-label">{t("optimization_preview_title")}</div>',
                unsafe_allow_html=True,
            )
            st.caption(t("optimization_preview_caption"))
            st.caption(t("optimization_expected_columns"))
            preview_columns = [
                "Part Number", "Supplier", "Annual Cost", "Defect Rate", "Warranty Claims",
                "Incoming Acceptance Rate", "Process Capability", "Corrective Action Response Time",
                "On-Time Delivery", "Lead Time", "Delivery Accuracy", "OSA Score", "Country",
            ]
            preview_keys = {
                "Part Number": "part_number",
                "Supplier": "supplier_column_name",
                "Annual Cost": "annual_cost",
                "Defect Rate": "defect_rate",
                "Warranty Claims": "warranty_claims",
                "Incoming Acceptance Rate": "incoming_acceptance_rate",
                "Process Capability": "process_capability_cpk",
                "Corrective Action Response Time": "cart_days",
                "On-Time Delivery": "on_time_delivery",
                "Lead Time": "lead_time_days",
                "Delivery Accuracy": "delivery_accuracy",
                "OSA Score": "osa_score",
                "Country": "country",
            }
            st.dataframe(
                corporate_table_style(
                    source_data[preview_columns].rename(
                        columns={column: t(key) for column, key in preview_keys.items()}
                    ).head(100)
                ),
                width="stretch",
                hide_index=True,
            )
            if capacity_assumed:
                st.info(t("optimization_capacity_assumption"))

        # ------------------------------------------------------------------
        # Step 4: OSA gatekeeper rule
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="optimization-section-heading">{t("optimization_step_qualified_filter")}</div>',
            unsafe_allow_html=True,
        )
        selected_part_data = source_data[source_data["Part Number"] == selected_part_number].copy()
        matched_by_description = False
        if selected_part_data.empty and "Part Description" in source_data.columns:
            selected_description_key = str(selected_part["Part Description"]).strip().casefold()
            description_keys = source_data["Part Description"].astype(str).str.strip().str.casefold()
            selected_part_data = source_data[description_keys == selected_description_key].copy()
            matched_by_description = not selected_part_data.empty
        if matched_by_description:
            st.info(t("optimization_part_match_fallback"))
        if selected_part_data.empty:
            st.warning(t("optimization_no_part_data"))
            st.stop()

        with st.container(border=True, key="optimization_osa_gate_panel"):
            gate_metric_column, gate_checkbox_column = st.columns([1, 2])
            with gate_metric_column:
                st.metric(t("optimization_osa_threshold"), "70")
            with gate_checkbox_column:
                use_only_qualified = st.checkbox(
                    t("optimization_use_only_qualified"),
                    value=True,
                    key="optimization_use_only_qualified",
                )
            st.markdown(
                f'<div class="optimization-note"><strong>{t("optimization_gate_title")}</strong><p>{t("optimization_osa_gate_note")}</p><p>{t("optimization_three_criteria_note")}</p></div>',
                unsafe_allow_html=True,
            )
            qualified_part_data = selected_part_data[selected_part_data["OSA Score"] >= 70].copy()
            evaluated_data = qualified_part_data if use_only_qualified else selected_part_data
            st.caption(
                f"{t('optimization_evaluated_suppliers')}: {len(evaluated_data)} · "
                f"{t('optimization_qualified_suppliers')}: {len(qualified_part_data)}"
            )
            if evaluated_data.empty:
                st.error(t("optimization_no_qualified_suppliers"))
                st.stop()

        if evaluated_data["Supplier"].duplicated().any():
            st.error(t("optimization_duplicate_suppliers"))
            st.stop()

        # ------------------------------------------------------------------
        # Step 5: Construct Cost, Quality, and Delivery criteria only
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="optimization-section-heading">{t("optimization_step_criteria")}</div>',
            unsafe_allow_html=True,
        )
        criteria_data, scored_data = construct_criteria(evaluated_data)
        with st.container(border=True, key="optimization_criteria_panel"):
            st.markdown(
                f'<div class="optimization-card-label">{t("optimization_criteria_title")}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="optimization-note">
                    <p><strong>{t('optimization_cost_definition')}</strong></p>
                    <p><strong>{t('optimization_quality_definition')}</strong></p>
                    <p><strong>{t('optimization_delivery_definition')}</strong></p>
                    <p>{t('optimization_three_criteria_note')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------------
        # Step 6: Conditional method configuration
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="optimization-section-heading">{t("optimization_step_configuration")}</div>',
            unsafe_allow_html=True,
        )
        method_weights = {"Cost": 0.40, "Quality": 0.35, "Delivery": 0.25}
        # Goal Programming compares this target against the Annual Cost criterion,
        # so initialize it in the same units as the selected supplier dataset.
        target_cost = float(evaluated_data["Annual Cost"].min())
        target_quality = 85.0
        target_delivery = 90.0
        priority_order = ["Cost", "Quality", "Delivery"]
        weights_valid = True
        priority_valid = True
        with st.container(border=True, key="optimization_method_configuration_panel"):
            if selected_method == "Weighted Sum":
                st.markdown(f"##### {t('optimization_weights_title')}")
                weight_columns = st.columns(3)
                weight_labels = {
                    "Cost": "optimization_weight_cost",
                    "Quality": "optimization_weight_quality",
                    "Delivery": "optimization_weight_delivery",
                }
                for column, criterion in zip(weight_columns, ["Cost", "Quality", "Delivery"]):
                    with column:
                        method_weights[criterion] = st.number_input(
                            t(weight_labels[criterion]),
                            min_value=0.0,
                            max_value=1.0,
                            value=method_weights[criterion],
                            step=0.05,
                            format="%.2f",
                            key=f"optimization_weight_{criterion.lower()}",
                        )
                weight_total = sum(method_weights.values())
                st.metric(t("optimization_weight_total"), f"{weight_total:.2f}")
                weights_valid = bool(np.isclose(weight_total, 1.0, atol=1e-6))
                if not weights_valid:
                    st.warning(t("optimization_weight_validation"))

            elif selected_method == "TOPSIS":
                st.markdown(f"##### {t('optimization_topsis_title')}")
                topsis_columns = st.columns(3)
                criterion_display_keys = {
                    "Cost": "optimization_cost_score",
                    "Quality": "optimization_quality_score",
                    "Delivery": "optimization_delivery_score",
                }
                for column, criterion in zip(topsis_columns, ["Cost", "Quality", "Delivery"]):
                    with column:
                        column.metric(t(criterion_display_keys[criterion]), f"{method_weights[criterion]:.0%}")
                st.markdown(
                    f"""
                    <div class="optimization-note">
                        <p>{t('optimization_topsis_cost')}</p>
                        <p>{t('optimization_topsis_benefits')}</p>
                        <p>{t('optimization_topsis_normalization')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            elif selected_method == "Goal Programming":
                st.markdown(f"##### {t('optimization_targets_title')}")
                target_columns = st.columns(3)
                with target_columns[0]:
                    target_cost = st.number_input(
                        t("optimization_target_cost"),
                        min_value=0.0,
                        value=target_cost,
                        step=1.0,
                        key="optimization_target_cost_input",
                    )
                with target_columns[1]:
                    target_quality = st.number_input(
                        t("optimization_target_quality"),
                        min_value=0.0,
                        max_value=100.0,
                        value=85.0,
                        step=1.0,
                        key="optimization_target_quality_input",
                    )
                with target_columns[2]:
                    target_delivery = st.number_input(
                        t("optimization_target_delivery"),
                        min_value=0.0,
                        max_value=100.0,
                        value=90.0,
                        step=1.0,
                        key="optimization_target_delivery_input",
                    )
                st.caption(t("optimization_target_cost_unit"))

            else:
                st.markdown(f"##### {t('optimization_preemptive_title')}")
                priority_columns = st.columns(3)
                priority_keys = [
                    "optimization_priority_1",
                    "optimization_priority_2",
                    "optimization_priority_3",
                ]
                criterion_labels = {
                    "Cost": "optimization_priority_cost",
                    "Quality": "optimization_priority_quality",
                    "Delivery": "optimization_priority_delivery",
                }
                priority_order = []
                for column, label_key, default_index in zip(
                    priority_columns,
                    priority_keys,
                    [0, 1, 2],
                ):
                    with column:
                        priority_order.append(
                            st.selectbox(
                                t(label_key),
                                ["Cost", "Quality", "Delivery"],
                                index=default_index,
                                format_func=lambda value: t(criterion_labels[value]),
                                key=f"optimization_{label_key}",
                            )
                        )
                priority_valid = len(set(priority_order)) == 3
                if not priority_valid:
                    st.warning(t("optimization_priority_validation"))
                st.caption(t("optimization_priority_note"))

        # ------------------------------------------------------------------
        # Steps 7–9: Run, results, and visualization
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="optimization-section-heading">{t("optimization_step_run")}</div>',
            unsafe_allow_html=True,
        )
        signature = (
            source_signature,
            selected_part_number,
            selected_method,
            bool(use_only_qualified),
            tuple(round(method_weights[criterion], 6) for criterion in ["Cost", "Quality", "Delivery"]),
            round(target_cost, 6),
            round(target_quality, 6),
            round(target_delivery, 6),
            tuple(priority_order),
            tuple(scored_data["Supplier"].astype(str)),
        )
        with st.container(key="optimization_run_button"):
            run_button = st.button(
                t("optimization_run_button"),
                type="primary",
                width="stretch",
                disabled=not weights_valid or not priority_valid,
            )

        if run_button:
            try:
                supplier_index = pd.Index(scored_data["Supplier"].astype(str), name="Supplier")
                impacts = {"Cost": "cost", "Quality": "benefit", "Delivery": "benefit"}
                if selected_method == "Weighted Sum":
                    backend_result = weighted_sum(criteria_data, method_weights, impacts)
                    ranking = pd.DataFrame(
                        {
                            "Rank": backend_result["rank"].astype(int).to_numpy(),
                            "Supplier": backend_result.index.astype(str),
                            "Final Score": backend_result["score"].to_numpy(dtype=float) * 100,
                        }
                    )
                    allocation = None
                elif selected_method == "TOPSIS":
                    backend_result = topsis(criteria_data, method_weights, impacts)
                    ranking = pd.DataFrame(
                        {
                            "Rank": backend_result["rank"].astype(int).to_numpy(),
                            "Supplier": backend_result.index.astype(str),
                            "Final Score": backend_result["score"].to_numpy(dtype=float) * 100,
                        }
                    )
                    allocation = None
                else:
                    # The existing LP methods expect a lower-is-better delivery
                    # value, so the displayed Delivery Score is converted to
                    # (100 - Delivery Score) only at this backend boundary.
                    allocation_data = pd.DataFrame(
                        {
                            "Cost": criteria_data["Cost"].to_numpy(dtype=float),
                            "Quality": criteria_data["Quality"].to_numpy(dtype=float),
                            "Delivery": 100 - criteria_data["Delivery"].to_numpy(dtype=float),
                        },
                        index=supplier_index,
                    )
                    capacities = dict(zip(supplier_index, scored_data["Capacity"].astype(float)))
                    demand_units = float(selected_part["Annual Volume"])
                    if selected_method == "Preemptive Optimization":
                        allocation, _metrics, _stages = preemptive_optimization(
                            data=allocation_data,
                            demand_units=demand_units,
                            capacities=capacities,
                            quality_target=target_quality,
                            delivery_target=100 - target_delivery,
                            cost_col="Cost",
                            quality_col="Quality",
                            delivery_col="Delivery",
                        )
                    else:
                        allocation, _metrics = goal_programming(
                            data=allocation_data,
                            demand_units=demand_units,
                            capacities=capacities,
                            quality_target=target_quality,
                            delivery_target=100 - target_delivery,
                            cost_target=target_cost,
                            goal_weights={"quality": 0.45, "delivery": 0.35, "cost": 0.20},
                            cost_col="Cost",
                            quality_col="Quality",
                            delivery_col="Delivery",
                        )
                    ranking = pd.DataFrame(
                        {
                            "Rank": allocation["allocation_share"].rank(
                                method="min", ascending=False
                            ).astype(int).to_numpy(),
                            "Supplier": allocation.index.astype(str),
                            "Final Score": allocation["allocation_share"].to_numpy(dtype=float) * 100,
                        }
                    ).sort_values(["Rank", "Final Score"], ascending=[True, False])

                recommended_supplier = str(ranking.iloc[0]["Supplier"])
                recommended_row = scored_data.loc[
                    scored_data["Supplier"].astype(str) == recommended_supplier
                ].iloc[0]
                st.session_state["supplier_optimization_result"] = {
                    "signature": signature,
                    "method": selected_method,
                    "ranking": ranking.reset_index(drop=True),
                    "recommended_supplier": recommended_supplier,
                    "recommended_final_score": float(ranking.iloc[0]["Final Score"]),
                    "selected_part": selected_part_number,
                    "evaluated_data": scored_data.reset_index(drop=True),
                    "qualified_count": int((selected_part_data["OSA Score"] >= 70).sum()),
                    "candidate_count": len(selected_part_data),
                    "recommended_row": recommended_row.to_dict(),
                    "allocation": allocation,
                }
            except Exception as error:
                st.error(t("error_analysis", error=error))

        result_payload = st.session_state.get("supplier_optimization_result")
        if not result_payload or result_payload.get("signature") != signature:
            st.info(t("optimization_no_results"))
            return

        ranking = result_payload["ranking"]
        recommended_supplier = result_payload["recommended_supplier"]
        recommended_row = result_payload["recommended_row"]
        method_name = result_payload["method"]
        result_part = part_lookup[result_payload["selected_part"]]

        st.markdown(
            f'<div class="optimization-section-heading">{t("optimization_step_results")}</div>',
            unsafe_allow_html=True,
        )
        results_column, ranking_column = st.columns([0.9, 1.6], gap="large")
        with results_column:
            st.markdown(
                f"""
                <div class="optimization-recommended-card">
                    <h2>✅ {t("optimization_recommended_supplier")}</h2>
                    <div class="optimization-recommended-name">{escape(recommended_supplier)}</div>
                    <p><strong>{t('country')}:</strong> {escape(str(recommended_row['Country']))}</p>
                    <p><strong>{t('optimization_final_score')}:</strong> {result_payload['recommended_final_score']:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ranking_column:
            st.markdown(
                f'<div class="optimization-card-label">{t("optimization_supplier_ranking")}</div>',
                unsafe_allow_html=True,
            )
            ranking_display = ranking.rename(
                columns={
                    "Rank": t("column_rank"),
                    "Supplier": t("supplier_column_name"),
                    "Final Score": t("optimization_final_score"),
                }
            )
            st.dataframe(
                corporate_table_style(ranking_display),
                width="stretch",
                hide_index=True,
            )

        st.markdown(
            f'<div class="optimization-section-heading">{t("optimization_step_visualization")}</div>',
            unsafe_allow_html=True,
        )
        chart_column, radar_column = st.columns(2, gap="large")
        with chart_column:
            st.markdown(
                f'<div class="optimization-card-label">{t("optimization_ranking_bar_chart")}</div>',
                unsafe_allow_html=True,
            )
            st.bar_chart(
                ranking.sort_values("Final Score").set_index("Supplier")[["Final Score"]],
                horizontal=True,
            )
        with radar_column:
            st.markdown(
                f'<div class="optimization-card-label">{t("optimization_radar_chart")}</div>',
                unsafe_allow_html=True,
            )
            evaluated_costs = result_payload["evaluated_data"]["Annual Cost"].astype(float)
            recommended_cost = float(recommended_row["Annual Cost"])
            cost_performance = float(evaluated_costs.min() / recommended_cost * 100) if recommended_cost > 0 else 0.0
            st.markdown(
                radar_svg(
                    [
                        t("optimization_radar_cost"),
                        t("optimization_radar_quality"),
                        t("optimization_radar_delivery"),
                    ],
                    [cost_performance, float(recommended_row["Quality"]), float(recommended_row["Delivery"])],
                ),
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------------
        # Steps 10–11: Decision summary and report export
        # ------------------------------------------------------------------
        st.markdown(
            f'<div class="optimization-section-heading">{t("optimization_step_decision")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="optimization_decision_summary_panel"):
            st.markdown(
                f'<div class="optimization-card-label">{t("optimization_decision_summary")}</div>',
                unsafe_allow_html=True,
            )
            render_info_grid(
                [
                    (t("optimization_method_used"), t(method_keys[method_name])),
                    (t("optimization_selected_part"), f"{result_part['Part Number']} · {result_part['Part Description']}"),
                    (t("optimization_evaluated_suppliers"), str(result_payload["candidate_count"])),
                    (t("optimization_qualified_suppliers"), str(result_payload["qualified_count"])),
                    (t("optimization_cost_score"), f"€{float(recommended_row['Annual Cost']):,.0f}"),
                    (t("optimization_quality_score"), f"{float(recommended_row['Quality']):.2f}"),
                    (t("optimization_delivery_score"), f"{float(recommended_row['Delivery']):.2f}"),
                    (t("optimization_osa_score"), f"{float(recommended_row['OSA Score']):.1f}"),
                ]
            )

        st.markdown(
            f'<div class="optimization-section-heading">{t("optimization_step_reports")}</div>',
            unsafe_allow_html=True,
        )
        evaluated_report = result_payload["evaluated_data"].copy()
        report_data = evaluated_report.merge(
            ranking[["Supplier", "Rank", "Final Score"]],
            on="Supplier",
            how="left",
        )
        report_data.insert(0, "Part Number", result_part["Part Number"])
        report_data.insert(1, "Part Description", result_part["Part Description"])
        report_data.insert(2, "Method", t(method_keys[method_name]))
        generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        report_data["Generated At"] = generated_at
        report_columns = [
            "Part Number", "Part Description", "Method", "Rank", "Supplier", "Country",
            "Final Score", "Annual Cost", "Quality", "Delivery", "OSA Score", "Generated At",
        ]
        report_data = report_data[report_columns]
        report_display = report_data.rename(
            columns={
                "Part Number": t("part_number"),
                "Part Description": t("part_description"),
                "Method": t("optimization_method_used"),
                "Rank": t("column_rank"),
                "Supplier": t("supplier_column_name"),
                "Country": t("country"),
                "Final Score": t("optimization_final_score"),
                "Annual Cost": t("annual_cost"),
                "Quality": t("optimization_quality_score"),
                "Delivery": t("optimization_delivery_score"),
                "OSA Score": t("optimization_osa_score"),
                "Generated At": t("optimization_report_generated_at"),
            }
        )

        with st.container(border=True, key="optimization_report_export_panel"):
            st.markdown(
                f'<div class="optimization-card-label">{t("optimization_export_reports")}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="optimization-report-meta">{t("optimization_report_generated_at")}: {generated_at}</div>',
                unsafe_allow_html=True,
            )
            csv_bytes = report_display.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                t("optimization_results_csv"),
                data=csv_bytes,
                file_name="supplier_optimization_results.csv",
                mime="text/csv",
                width="stretch",
            )
            export_column, excel_column = st.columns(2, gap="small")
            with export_column:
                if st.button(t("optimization_export_pdf"), width="stretch", key="optimization_pdf_button"):
                    st.info(t("optimization_pdf_placeholder"))
            with excel_column:
                excel_buffer = io.BytesIO()
                excel_bytes = None
                excel_error = None
                try:
                    with pd.ExcelWriter(excel_buffer) as writer:
                        pd.DataFrame(
                            {
                                "Field": [
                                    t("optimization_method_used"),
                                    t("optimization_selected_part"),
                                    t("optimization_recommended_supplier"),
                                    t("optimization_final_score"),
                                    t("optimization_report_generated_at"),
                                ],
                                "Value": [
                                    t(method_keys[method_name]),
                                    f"{result_part['Part Number']} · {result_part['Part Description']}",
                                    recommended_supplier,
                                    f"{result_payload['recommended_final_score']:.2f}",
                                    generated_at,
                                ],
                            }
                        ).to_excel(writer, sheet_name="Summary", index=False)
                        report_display.to_excel(writer, sheet_name="Ranking", index=False)
                        report_display.to_excel(writer, sheet_name="Performance", index=False)
                    excel_bytes = excel_buffer.getvalue()
                except Exception as error:
                    excel_error = error
                if excel_bytes is not None:
                    st.download_button(
                        t("optimization_export_excel"),
                        data=excel_bytes,
                        file_name="supplier_optimization_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width="stretch",
                    )
                elif excel_error is not None:
                    st.error(t("optimization_excel_error", error=excel_error))


# -----------------------------------------------------------------------------
# Polished horizontal language selection and navigation: exactly five pages
# -----------------------------------------------------------------------------


with st.container(key="corporate_navigation"):
    navigation_column, language_column = st.columns([6, 1.1], gap="small")

    page_labels = [t(f"page_{key}") for key in PAGE_KEYS]
    with navigation_column:
        # A single horizontal radio provides routing and keyboard accessibility.
        # CSS above removes the circles and presents it as a compact link row.
        selected_page = st.radio(
            t("navigation"),
            page_labels,
            index=PAGE_KEYS.index(st.session_state["current_page"]),
            horizontal=True,
            label_visibility="collapsed",
            key=f"page_navigation_{st.session_state['language']}",
        )
        selected_page_key = PAGE_KEYS[page_labels.index(selected_page)]
        st.session_state["current_page"] = selected_page_key

    # Inline language links sit on the far right of the same navigation row.
    with language_column:
        language_columns = st.columns([1, 0.35, 1, 0.35, 1], gap="small")
        language_layout = (
            ("TR", language_columns[0]),
            ("|", language_columns[1]),
            ("EN", language_columns[2]),
            ("|", language_columns[3]),
            ("DE", language_columns[4]),
        )

        for language_code, language_column in language_layout:
            with language_column:
                if language_code == "|":
                    st.markdown(
                        '<span class="language-separator">|</span>',
                        unsafe_allow_html=True,
                    )
                elif st.button(
                    language_code,
                    key=f"language_{language_code}",
                    type="tertiary",
                    width="stretch",
                    help=LANGUAGE_OPTIONS[language_code],
                ):
                    st.session_state["language"] = language_code
                    st.rerun()


if selected_page_key == "homepage":
    homepage_page()
elif selected_page_key == "parts_database":
    parts_database_page()
elif selected_page_key == "supplier_database":
    supplier_database_page()
elif selected_page_key == "osa_assessment":
    osa_assessment_page()
elif selected_page_key == "supplier_optimization":
    supplier_optimization_page()
