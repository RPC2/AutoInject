import logging
from typing import List

import numpy as np

from rlpi.attack.learners.base import AdaptiveAttackLearner
from rlpi.attack.templates.strings import SPECIAL_TOKENS_LIST
from rlpi.attack.templates.task_modifier import InjectionTaskModifier
from rlpi.attack.templates.templates import (
    AGENT_ATTACK_TEMPLATES,
    format_attack_prompt,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("RandomAttackLearner")


class RandomAttackLearner(AdaptiveAttackLearner[InjectionTaskModifier]):
    """
    A learner that randomly selects templates and modifiers for attacks.
    This serves as a baseline to compare against more sophisticated learners.
    """

    def __init__(
        self,
        default_agent_name: str = "Assistant",
        default_persona_name: str = "AI",
        *args,
        **kwargs,
    ):
        super().__init__(
            initial_tokens=SPECIAL_TOKENS_LIST,
            learning_rate=0.0,
            exploration_rate=0.0,
        )

        self.templates_dict = AGENT_ATTACK_TEMPLATES
        self.template_indices = list(self.templates_dict.keys())
        self.modifier_list = SPECIAL_TOKENS_LIST

        self.default_agent_name = default_agent_name
        self.default_persona_name = default_persona_name

    def modify_tasks(
        self, tasks: List[InjectionTaskModifier], user_task: str
    ) -> None:
        """
        Randomly select template and modifiers for each task.
        """
        for task in tasks:
            malicious_goal = task._original_goal
            # Randomly select template and modifiers
            selected_template_idx = np.random.choice(
                list(self.templates_dict.keys())
            )
            selected_mod1_str = np.random.choice(self.modifier_list)
            selected_mod2_str = np.random.choice(self.modifier_list)

            # Format the final prompt
            final_prompt_str = format_attack_prompt(
                template_id=selected_template_idx,
                malicious_goal=malicious_goal,
                modifier_1=selected_mod1_str,
                modifier_2=selected_mod2_str,
                agent_name=self.default_agent_name,
                original_user_task=user_task,
                persona_name=self.default_persona_name,
            )

            task.set_formatted_goal(formatted_prompt=final_prompt_str)

            logger.debug(
                f"Random selection - Template: {selected_template_idx}, "
                f"Mod1: {selected_mod1_str}, Mod2: {selected_mod2_str}"
            )

    def update_scores(self, success_rate: float, utility_score: float) -> None:
        """
        No-op implementation as random selection doesn't learn from scores.
        """
        pass

    def save_model(self, path: str) -> None:
        """
        No-op implementation as there's no model to save.
        """
        pass
