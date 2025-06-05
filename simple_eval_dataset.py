from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

case1 = Case(  
    name='simple_case',
    inputs='What is the capital of France?',
    expected_output='Paris',
    metadata={'difficulty': 'easy'},
)


class MyEvaluator(Evaluator[str, str]):
    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if ctx.output == ctx.expected_output:
            return 1.0
        elif (
            isinstance(ctx.output, str)
            and ctx.expected_output.lower() in ctx.output.lower()
        ):
            return 0.8
        else:
            return 0.0


dataset = Dataset(
    cases=[case1],
    evaluators=[IsInstance(type_name='str'), MyEvaluator()],  
)


async def guess_city_100(question: str) -> str:  
    return 'Paris'
report = dataset.evaluate_sync(guess_city_100)  
report.print(include_input=True, include_output=True, include_durations=False) 

async def guess_city_80(question: str) -> str:  
    return 'Parisx'
report = dataset.evaluate_sync(guess_city_80)  
report.print(include_input=True, include_output=True, include_durations=False) 

async def guess_city_0(question: str) -> str:  
    return 'London'
report = dataset.evaluate_sync(guess_city_0)  
report.print(include_input=True, include_output=True, include_durations=False) 

 
