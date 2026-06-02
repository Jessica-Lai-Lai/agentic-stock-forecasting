import os
import json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_forecast(history_context: str, horizon: int = 4) -> dict:
    """
    Generate stock price forecast using an LLM.

    Input:
    - history_context: structured historical context from context_agent.py
    - horizon: number of future weeks to forecast

    Output:
    - A dictionary containing:
      macro_reasoning
      micro_reasoning
      final_forecast
    """

    prompt = f"""
You are a lightweight stock forecasting agent.

Your task:
Use the historical stock price data and news context to forecast the next {horizon} weekly closing prices.

Historical context:
{history_context}

Instructions:
1. Analyze the overall macro trend.
2. Analyze week-by-week micro movements.
3. Forecast exactly {horizon} future weekly closing prices.
4. Be conservative and avoid unrealistic jumps.
5. Output valid JSON only.

The JSON must follow this structure:
{{
  "macro_reasoning": "Explain the overall trend in 2-4 sentences.",
  "micro_reasoning": [
    {{
      "week": 1,
      "movement": "up",
      "reason": "Explain why week 1 may move this way.",
      "forecast": 133.5
    }}
  ],
  "final_forecast": [133.5, 134.2, 135.0, 135.8]
}}
"""

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )

    output_text = response.output_text

    try:
        result = json.loads(output_text)
    except json.JSONDecodeError:
        raise ValueError(f"Model did not return valid JSON:\n{output_text}")

    if "macro_reasoning" not in result:
        raise ValueError("Missing macro_reasoning in model output.")

    if "micro_reasoning" not in result:
        raise ValueError("Missing micro_reasoning in model output.")

    if "final_forecast" not in result:
        raise ValueError("Missing final_forecast in model output.")

    if len(result["final_forecast"]) != horizon:
        raise ValueError(
            f"Expected {horizon} forecast values, "
            f"but got {len(result['final_forecast'])}."
        )

    return result



def save_forecast_output(
    result: dict,
    output_path: str = "outputs/forecast_output.json") -> str:
    """
    Save forecast agent output to a JSON file.

    This file will be used as the input for evaluation.py.
    """

    os.makedirs("outputs", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return output_path


def main():
    from data_loader import load_stock_data, split_train_test
    from context_agent import build_full_context

    #df = load_stock_data("data/sample_stock_data.csv")
    df = load_stock_data("data/real_stock_data.csv")
    train_df, test_df = split_train_test(df, train_size=8, test_size=4)

    history_context = build_full_context(train_df)

    forecast_result = generate_forecast(history_context, horizon=4)

    print(json.dumps(forecast_result, indent=2))


    output_path = save_forecast_output(forecast_result)

    print(f"\nForecast output saved to: {output_path}")


if __name__ == "__main__":
    main()