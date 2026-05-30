"""
Role Meta Descriptions for Position Embeddings.

This module contains detailed text descriptions of each agent role
for embedding-based differentiation. The descriptions are used to
generate distinct position embeddings for different roles.
"""

from typing import Dict


# Detailed role descriptions for position embeddings
ROLE_META_DESC: Dict[str, str] = {
    'Mathematical Analyst': """
Mathematical Analyst role specialized in breaking down complex math problems.
Responsibilities: Identify mathematical structure, define variables, set up equations.
Strengths: Problem decomposition, variable identification, equation formulation.
Approach: Systematic analysis before solution, clear variable definitions.
Best for: Initial problem understanding and mathematical modeling.
""".strip(),

    'Math Solver': """
Math Solver role specialized in executing mathematical computations.
Responsibilities: Solve equations, perform calculations, verify numerical results.
Strengths: Accurate computation, step-by-step solutions, arithmetic precision.
Approach: Methodical calculation with verification at each step.
Best for: Carrying out calculations and deriving numerical answers.
""".strip(),

    'Inspector': """
Inspector role specialized in reviewing and validating solutions.
Responsibilities: Check solution correctness, identify errors, verify final answers.
Strengths: Error detection, logical verification, quality assurance.
Approach: Critical review of reasoning and calculations, edge case checking.
Best for: Final verification and error catching before submission.
""".strip(),

    'Project Manager': """
Project Manager role specialized in design patterns and code structure.
Responsibilities: Coordinate solution strategy, organize workflow, manage approach.
Strengths: Strategic planning, workflow organization, high-level design.
Approach: Top-down planning and coordination of solution components.
Best for: Complex problems requiring structured multi-step approaches.
""".strip(),

    'Algorithm Designer': """
Algorithm Designer role specialized in algorithm design and pseudocode.
Responsibilities: Design algorithms, create pseudocode, plan computational steps.
Strengths: Algorithm optimization, logical flow design, complexity analysis.
Approach: Design-first thinking with clear algorithmic specifications.
Best for: Problems requiring novel algorithms or optimization.
""".strip(),

    'Test Analyst': """
Test Analyst role specialized in test cases and edge conditions.
Responsibilities: Identify edge cases, design test scenarios, validate solutions.
Strengths: Edge case discovery, boundary testing, comprehensive validation.
Approach: Systematic test case generation and solution stress-testing.
Best for: Ensuring solution robustness across all possible inputs.
""".strip(),

    'Programming Expert': """
Programming Expert role specialized in writing executable code.
Responsibilities: Implement algorithms, write clean code, handle edge cases.
Strengths: Code generation, syntax correctness, implementation skills.
Approach: Convert problem specifications into working, tested code.
Best for: Tasks requiring actual code implementation.
""".strip(),

    'Bug Fixer': """
Bug Fixer role specialized in debugging and code correction.
Responsibilities: Identify bugs, fix errors, improve code robustness.
Strengths: Debugging, error analysis, code improvement.
Approach: Analyze failing code, identify root causes, apply fixes.
Best for: Correcting issues in existing code implementations.
""".strip(),

    'Knowlegable Expert': """
Knowledgeable Expert role specialized in domain knowledge and research.
Responsibilities: Provide domain expertise, search for relevant information.
Strengths: Deep domain knowledge, information retrieval, expert insights.
Approach: Knowledge-driven problem solving with expert references.
Best for: Problems requiring specialized domain knowledge.
""".strip(),

    'Critic': """
Critic role specialized in critical analysis and feedback.
Responsibilities: Critique solutions, identify weaknesses, suggest improvements.
Strengths: Critical thinking, weakness identification, constructive feedback.
Approach: Adversarial analysis to find flaws and improve quality.
Best for: Improving solution quality through critical review.
""".strip(),

    'Mathematician': """
Mathematician role for STEM questions.
Responsibilities: Handle mathematical reasoning, calculations, formal logic.
Strengths: Numerical computation, formal proofs, quantitative analysis.
Approach: Apply mathematical principles systematically.
Best for: Math, physics, engineering, and quantitative questions.
""".strip(),

    'Psychologist': """
Psychologist role for behavioral and social questions.
Responsibilities: Analyze human behavior, cognitive processes, social dynamics.
Strengths: Understanding human nature, psychological theories, research methods.
Approach: Apply psychological frameworks to understand phenomena.
Best for: Psychology, sociology, and human behavior questions.
""".strip(),

    'Historian': """
Historian role for historical and cultural questions.
Responsibilities: Provide historical context, analyze events, identify patterns.
Strengths: Chronological knowledge, cause-effect analysis, cultural awareness.
Approach: Place questions in historical context and analyze significance.
Best for: History, political science, and cultural studies questions.
""".strip(),

    'Doctor': """
Medical Doctor role for health and biology questions.
Responsibilities: Medical knowledge, biological processes, health advice.
Strengths: Anatomy, physiology, disease mechanisms, treatments.
Approach: Apply medical and biological knowledge systematically.
Best for: Medicine, biology, and health-related questions.
""".strip(),

    'Lawyer': """
Lawyer role for legal and ethical questions.
Responsibilities: Legal reasoning, case analysis, ethical frameworks.
Strengths: Legal principles, argumentation, precedent analysis.
Approach: Apply legal and ethical frameworks to analyze situations.
Best for: Law, ethics, and governance questions.
""".strip(),

    'Economist': """
Economist role for economic and business questions.
Responsibilities: Economic analysis, market dynamics, financial reasoning.
Strengths: Economic theories, quantitative analysis, market understanding.
Approach: Apply economic principles to analyze scenarios.
Best for: Economics, business, finance questions.
""".strip(),

    'Programmer': """
Programmer role for computer science questions.
Responsibilities: Algorithm design, coding concepts, system architecture.
Strengths: Programming logic, data structures, computational thinking.
Approach: Apply CS fundamentals to solve problems.
Best for: Computer science, engineering, and technology questions.
""".strip(),

    'AnalyzeAgent': """
Analysis Agent role specialized in understanding and analyzing problems.
Responsibilities: Analyze problem statements, extract key information, identify constraints.
Strengths: Comprehension, information extraction, constraint identification.
Approach: Thorough reading and analysis before solution attempt.
Best for: Complex problems requiring careful initial analysis.
""".strip(),

    'CodeWriting': """
Code Writing role specialized in generating and implementing code.
Responsibilities: Write code, implement algorithms, produce executable solutions.
Strengths: Code generation, syntax correctness, implementation skills.
Approach: Convert problem specifications into working code.
Best for: Programming problems requiring code implementation.
""".strip(),

    # Default fallback for unknown roles
    'default': """
General problem-solving agent role.
Responsibilities: Analyze problems and provide solutions.
Strengths: General reasoning and problem-solving capabilities.
Approach: Systematic analysis and solution generation.
Best for: General tasks without specialized requirements.
""".strip(),
}


def get_role_description(role: str) -> str:
    """
    Get the description for a role, with fallback to default.

    Args:
        role: The role name (e.g., "Math Solver", "Inspector")

    Returns:
        The detailed description string for the role
    """
    return ROLE_META_DESC.get(role, ROLE_META_DESC['default'])
