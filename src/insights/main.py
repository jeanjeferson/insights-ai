#!/usr/bin/env python
import sys
import warnings

from datetime import datetime, timedelta

from insights.crew import Insights

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Definindo as datas para a consulta (últimos 2 anos)
data_fim = datetime.now().strftime('%Y-%m-%d')
data_inicio = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

def run():
    """
    Run the crew.
    """
    inputs = {
        'data_inicio': data_inicio,
        'data_fim': data_fim
    }
    
    try:
        Insights().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        Insights().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Insights().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        Insights().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
