import gpt4all
llm_name = "ggml-model-gpt4all-falcon-q4_0.bin"
falcon = gpt4all.GPT4All(llm_name, allow_download=False, model_path='./models/')
