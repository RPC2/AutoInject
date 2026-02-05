from agentdojo.default_suites.v1.banking.injection_tasks import (  # noqa: F401 - Register tasks
    BankingInjectionTask,
)
from agentdojo.default_suites.v1.banking.task_suite import (
    task_suite as banking_task_suite,
)
from agentdojo.default_suites.v1.banking.user_tasks import (  # noqa: F401 - Register tasks
    BankingUserTask,
)

__all__ = ["banking_task_suite"]
