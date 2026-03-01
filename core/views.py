from django.shortcuts import render
from .market_utils import get_market_analysis

def index(request):
    symbol = request.GET.get("symbol", "AAPL").strip()
    analysis = None
    error = None

    if symbol:
        try:
            analysis = get_market_analysis(symbol)
            if not analysis.get("markov") and not analysis.get("tech"):
                error = f"Unable to fetch data for symbol: {symbol}"
                analysis = None
        except Exception as e:
            error = f"An error occurred: {str(e)}"
            analysis = None

    context = {
        "analysis": analysis,
        "error": error,
        "symbol": symbol
    }
    return render(request, "core/index.html", context)