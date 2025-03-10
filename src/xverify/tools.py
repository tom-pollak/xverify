"""
Tools for the verifier, modified from willccbb/verifiers
"""


def calculator(
    expression: str,  # A mathematical expression using only numbers and basic operators (+,-,*,/,**,())
) -> str:  # The result of the calculation or an error message
    """Evaluates a single line of Python math expression. No imports or variables allowed.

    Examples:
        {"expression": "2 + 2"} -> "4"
        {"expression": "3 * (17 + 4)"} -> "63"
        {"expression": "100 / 5"} -> "20.0"
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid characters in expression"

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def search(
    query: str,  # The search query string
    num_results: int = 5,  # Number of results to return (default: 5, max: 10)
) -> str:  # Formatted string with bullet points of top results, each with title and brief summary
    """Searches DuckDuckGo and returns concise summaries of top results.

    Examples:
        {"query": "who invented the lightbulb", "num_results": 3}
    """
    from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=min(num_results, 10)))
            if not results:
                return "No results found"

            summaries = []
            for r in results:
                title = r["title"]
                snippet = r["body"][:200].rsplit(".", 1)[0] + "."
                summaries.append(f"• {title}\n  {snippet}")

            return "\n\n".join(summaries)
    except Exception as e:
        return f"Error: {str(e)}"
