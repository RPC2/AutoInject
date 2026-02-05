from agentdojo.default_suites.v1.slack.injection_tasks import (  # noqa: F401 - Register tasks
    SlackInjectionTask,
)
from agentdojo.default_suites.v1.slack.task_suite import (
    task_suite as slack_task_suite,
)
from agentdojo.default_suites.v1.slack.user_tasks import (  # noqa: F401 - Register tasks
    SlackUserTask,
)

__all__ = ["slack_task_suite"]
