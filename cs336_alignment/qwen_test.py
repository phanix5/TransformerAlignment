from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the united states is",
    "1234343 + 324243="
]

llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"])

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated Text: {generated_text!r}")
