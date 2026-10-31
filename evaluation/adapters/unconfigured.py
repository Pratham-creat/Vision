class AIEvaluationUnavailable(RuntimeError):
    """Raised when no compatible model adapter has been configured."""


def unavailable_message() -> str:
    return "AI evaluation not configured. Configure a compatible model adapter and endpoint to evaluate predictions."
