from typing import Union, Dict, Any, List
import itertools

from HieraMAS.prompt.prompt_set import PromptSet
from HieraMAS.prompt.prompt_set_registry import PromptSetRegistry
from HieraMAS.prompt.common import get_combine_materials


roles = itertools.cycle(['Knowlegable Expert',
                        #  'Wiki Searcher',
                         'Critic',
                         'Mathematician',
                         'Psychologist',
                         'Historian',
                         'Doctor',
                         'Lawyer',
                         'Economist',
                         'Programmer'])

ROLE_DESCRIPTION = {
"Knowlegable Expert":
"""
You are a knowlegable expert in question answering.
Please give several key entities that need to be searched in wikipedia to solve the problem, for example: catfish effect, broken window effect, Shakespeare.
If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it
""",
"Wiki Searcher":
"""
You will be given a question and a wikipedia overview of the key entities within it.
Please refer to them step by step to give your answer.
And point out potential issues in other agent's analysis.
""",
"Critic":
"""
You are an excellent critic.
Please point out potential issues in other agent's analysis point by point.
""",
"Mathematician":
"""
You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.
""",
"Psychologist":
"""
You are a psychologist.
You are good at psychology, sociology, and philosophy.
You give people scientific suggestions that will make them feel better.
""",
"Historian":
"""
You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.
""",
"Doctor":
"""
You are a doctor and come up with creative treatments for illnesses or diseases.
You are able to recommend conventional medicines, herbal remedies and other natural alternatives. 
You also consider the patient's age, lifestyle and medical history when providing your recommendations.
""",
"Lawyer":
"""
You are good at law, politics, and history.
""",
"Economist":
"""
You are good at economics, finance, and business.
You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.
""",
"Programmer":
"""
You are good at computer science, engineering, and physics.
You have experience in designing and developing computer software and hardware.
""",
"Fake":
"""
You are a liar who only tell lies.
""",
}

ROLE_CONNECTION = [('Knowlegable Expert','Mathematician'),
                   ('Knowlegable Expert','Economist'),
                   ('Knowlegable Expert','Lawyer'),
                   ('Knowlegable Expert','Critic'),
                   ('Knowlegable Expert','Psychologist'),
                   ('Knowlegable Expert','Doctor'),
                   ('Knowlegable Expert','Historian'),
                   ('Knowlegable Expert','Programmer'),
                   ('Knowlegable Expert','Critic'),
                   ('Mathematician','Critic'),
                   ('Mathematician','Critic'),
                   ('Psychologist','Critic'),
                   ('Economist','Lawyer'),
                   ('Lawyer','Critic'),
                   ('Critic','Psychologist'),
                   ('Psychologist','Doctor'),
                   ('Doctor','Historian'),
                   ('Historian','Knowlegable Expert'),
                   ('Programmer','Mathematician'),
                   ('Programmer','Knowlegable Expert'),
                    ('Mathematician','Programmer'),
                    ('Programmer','Economist'),
                    ('Economist','Psychologist'),
                    ('Psychologist','Knowlegable Expert'),
                    ('Critic','Historian'),
                    ('Historian','Economist'),
                    ('Lawyer','Knowlegable Expert'),
                    ('Doctor','Lawyer'),
                    ('Mathematician','Doctor'),
                    ('Programmer','Critic'),
                    ('Economist','Doctor'),
                    ('Lawyer','Critic'),
                    ('Psychologist','Lawyer'),
                    ('Historian','Mathematician'),
                    ('Programmer','Doctor'),
                    ('Doctor','Psychologist'),
                    ('Historian','Programmer'),
                    ('Critic','Economist')]

# Role-specific aggregation prompts for MultiNode (synthesis + answer output)
MULTI_AGENT_AGGREGATION_PROMPTS = {
    "Knowlegable Expert": """--- Your Task ---
Synthesize all expert analyses above and provide the best combined analysis:
1. Identify key concepts and entities mentioned across analyses
2. Check factual accuracy of each analysis
3. Combine the strongest reasoning points into a coherent analysis

Based on your synthesis, provide your final answer.
The first line of your reply must contain only one letter (A, B, C, or D).
""",

    "Critic": """--- Your Task ---
Synthesize all critiques above:
1. Evaluate the validity of each critique's points
2. Identify the strongest counter-arguments
3. Determine which answer best survives critical scrutiny

Based on your synthesis, provide your final answer.
The first line of your reply must contain only one letter (A, B, C, or D).
""",

    "Mathematician": """--- Your Task ---
Synthesize all mathematical analyses above:
1. Check mathematical correctness of each reasoning
2. Verify calculations and logical steps
3. Combine the strongest mathematical arguments

Based on your synthesis, provide your final answer.
The first line of your reply must contain only one letter (A, B, C, or D).
""",

    "Psychologist": """--- Your Task ---
Synthesize all psychological perspectives above:
1. Consider cognitive and behavioral factors mentioned
2. Evaluate the scientific validity of each analysis
3. Combine insights into a coherent psychological perspective

Based on your synthesis, provide your final answer.
The first line of your reply must contain only one letter (A, B, C, or D).
""",

    "Historian": """--- Your Task ---
Synthesize all historical analyses above:
1. Check historical accuracy and context
2. Evaluate the evidence and reasoning
3. Combine the strongest historical insights

Based on your synthesis, provide your final answer.
The first line of your reply must contain only one letter (A, B, C, or D).
""",

    "Doctor": """--- Your Task ---
Synthesize all medical/health perspectives above:
1. Check medical accuracy of each analysis
2. Consider relevant biological and health factors
3. Combine the strongest medical insights

Based on your synthesis, provide your final answer.
The first line of your reply must contain only one letter (A, B, C, or D).
""",

    "Lawyer": """--- Your Task ---
Synthesize all legal analyses above:
1. Check legal accuracy and reasoning
2. Consider relevant laws, precedents, and principles
3. Combine the strongest legal arguments

Based on your synthesis, provide your final answer.
The first line of your reply must contain only one letter (A, B, C, or D).
""",

    "Economist": """--- Your Task ---
Synthesize all economic analyses above:
1. Check economic reasoning and principles
2. Consider market dynamics and financial factors
3. Combine the strongest economic insights

Based on your synthesis, provide your final answer.
The first line of your reply must contain only one letter (A, B, C, or D).
""",

    "Programmer": """--- Your Task ---
Synthesize all technical analyses above:
1. Check technical accuracy of each analysis
2. Consider computer science and engineering principles
3. Combine the strongest technical insights

Based on your synthesis, provide your final answer.
The first line of your reply must contain only one letter (A, B, C, or D).
""",
}

# Baseline direct prompts for aggregator-only mode (no worker outputs to synthesize)
BASELINE_DIRECT_PROMPTS = {
    "Knowlegable Expert": """You are a knowledgeable expert in question answering.
Analyze this multiple choice question and provide your answer.
Only one answer out of the offered 4 is correct.
Your reply must include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter (A, B, C, or D).
Reply in less than 100 words.""",

    "Critic": """You are an excellent critic and analytical thinker.
Analyze this multiple choice question, evaluate each option critically, and provide your answer.
Only one answer out of the offered 4 is correct.
Your reply must include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter (A, B, C, or D).
Reply in less than 100 words.""",

    "Mathematician": """You are a mathematician who is good at math games, arithmetic calculation, and logical reasoning.
Analyze this multiple choice question and provide your answer.
Only one answer out of the offered 4 is correct.
Your reply must include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter (A, B, C, or D).
Reply in less than 100 words.""",

    "Psychologist": """You are a psychologist with expertise in psychology, sociology, and philosophy.
Analyze this multiple choice question and provide your answer.
Only one answer out of the offered 4 is correct.
Your reply must include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter (A, B, C, or D).
Reply in less than 100 words.""",

    "Historian": """You are a historian who researches and analyzes cultural, economic, political, and social events.
Analyze this multiple choice question and provide your answer.
Only one answer out of the offered 4 is correct.
Your reply must include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter (A, B, C, or D).
Reply in less than 100 words.""",

    "Doctor": """You are a medical doctor with expertise in health, biology, and medical sciences.
Analyze this multiple choice question and provide your answer.
Only one answer out of the offered 4 is correct.
Your reply must include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter (A, B, C, or D).
Reply in less than 100 words.""",

    "Lawyer": """You are a lawyer with expertise in law, politics, and history.
Analyze this multiple choice question and provide your answer.
Only one answer out of the offered 4 is correct.
Your reply must include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter (A, B, C, or D).
Reply in less than 100 words.""",

    "Economist": """You are an economist with expertise in economics, finance, and business.
Analyze this multiple choice question and provide your answer.
Only one answer out of the offered 4 is correct.
Your reply must include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter (A, B, C, or D).
Reply in less than 100 words.""",

    "Programmer": """You are a programmer with expertise in computer science, engineering, and physics.
Analyze this multiple choice question and provide your answer.
Only one answer out of the offered 4 is correct.
Your reply must include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter (A, B, C, or D).
Reply in less than 100 words.""",
}

@PromptSetRegistry.register('mmlu')
class MMLUPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option qestion answering.
    """
    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers."
    
    def get_role_connection(self):
        return ROLE_CONNECTION
    
    def get_description(self,role):
        return ROLE_DESCRIPTION[role]
    
    @staticmethod
    def get_constraint():
        return """
            I will ask you a question.
            I will also give you 4 answers enumerated as A, B, C and D.
            Only one answer out of the offered 4 is correct.
            You must choose the correct answer to the question.
            Your response must be one of the 4 letters: A, B, C or D,
            corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C or D)
        """
    
    @staticmethod
    def get_analyze_constraint(role):
        return ROLE_DESCRIPTION[role] if role in ROLE_DESCRIPTION.keys() else ""+ """
I will ask you a question and 4 answers enumerated as A, B, C and D.
Only one answer out of the offered 4 is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : A, B, C or D)
"""
    
    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a question.
        I will also give you 4 answers enumerated as A, B, C and D.
        Only one answer out of the offered 4 is correct.
        You must choose the correct answer to the question.
        Your response must be one of the 4 letters: A, B, C or D,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Let's analyze step by step and synthesize the best answer.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """
    
    @staticmethod
    def get_format():
        return NotImplementedError

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The first line of your reply must contain only one letter(for example : A, B, C or D)
                """
    # @staticmethod
    # def get_adversarial_answer_prompt(question):
    #     return f"""Randomly output a letter from ABCD on the first line.
    #             Then output any gibberish paragraph on the same topic as the following question: {question}.
    #             The first line of your reply must contain only one letter(for example : A, B, C or D)
    #             """
    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
    
    @staticmethod
    def get_decision_few_shot():
        return ""
    
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            answer = answer[0] # Try to format the answer by taking the first letter
        return answer

    @staticmethod
    def get_baseline_prompt(role: str) -> str:
        """
        Get baseline prompt for aggregator-only mode (direct answering without workers).

        Args:
            role: The role name

        Returns:
            Baseline prompt string for the role
        """
        if role in BASELINE_DIRECT_PROMPTS:
            return BASELINE_DIRECT_PROMPTS[role]
        # Fallback to a generic expert prompt
        return BASELINE_DIRECT_PROMPTS.get("Knowlegable Expert", "")

    @staticmethod
    def get_multi_agent_aggregation_prompt(task: str, agent_outputs: List[Dict[str, str]], role: str = "Knowlegable Expert") -> str:
        """
        Generate role-specific prompt for aggregating multiple MMLU analyses.

        Args:
            task: The original question with options
            agent_outputs: List of {"role": str, "output": str} from worker agents
            role: The role of workers (determines aggregation style)

        Returns:
            Formatted prompt for the aggregator to synthesize analyses

        Raises:
            ValueError: If role is not found in MULTI_AGENT_AGGREGATION_PROMPTS
        """
        if role not in MULTI_AGENT_AGGREGATION_PROMPTS:
            raise ValueError(f"Unknown role '{role}' for MMLU aggregation. "
                            f"Valid roles: {list(MULTI_AGENT_AGGREGATION_PROMPTS.keys())}")

        prompt = f"Question:\n{task}\n\n"
        prompt += f"Multiple {role}s have independently provided analyses:\n\n"

        for i, output in enumerate(agent_outputs, 1):
            prompt += f"--- Analysis {i} ---\n{output['output']}\n\n"

        prompt += MULTI_AGENT_AGGREGATION_PROMPTS[role]

        return prompt