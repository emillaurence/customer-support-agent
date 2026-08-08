"""Tools the Bookly agent can call.

Flat and stateless by design: each takes plain arguments, reads its own data, and
returns a typed result. The orchestrator owns the session; tools do not.

Six tools, and two small modules behind them: `fixtures` reads the mock order
data, and `eligibility_tokens` holds the server-side record of what each
eligibility token permits. Policy is read from Neo4j through `agent.graph`.

Every tool works as an ordinary Python function, with no model involved — which
is what the tests exercise.
"""

from tools.check_return_eligibility import check_return_eligibility
from tools.escalate_to_human import EscalationResult, escalate_to_human
from tools.initiate_return import ReturnBlockedError, ReturnResult, initiate_return
from tools.lookup_order import OrderDetails, Shipment, lookup_order
from tools.search_policy import PolicyMatch, PolicySearchResult, search_policy
from tools.verify_identity import VerifyIdentityResult, verify_identity

__all__ = [
    "EscalationResult",
    "OrderDetails",
    "PolicyMatch",
    "PolicySearchResult",
    "ReturnBlockedError",
    "ReturnResult",
    "Shipment",
    "VerifyIdentityResult",
    "check_return_eligibility",
    "escalate_to_human",
    "initiate_return",
    "lookup_order",
    "search_policy",
    "verify_identity",
]
