from gpt_inference import ChatCompletionGPTModel

def gptscore(input, output, gpt_model=None, api_key=None):
    """
    These models are supported by the chat completions API

    GPT-3.5 Models
    "gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct"
    
    GPT-4 Models
    "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview", "gpt-4-0125-preview", "gpt-4-1106-preview", 
    "gpt-4", "gpt-4-0613", "gpt-4-0314"
    
    GPT-4o mini Models
    "gpt-4o-mini", "gpt-4o-mini-2024-07-18"
    
    GPT-4o Models
    "gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13", "chatgpt-4o-latest"
    
    o1-preview and o1-mini
    "o1-preview", "o1-preview-2024-09-12", "o1-mini", "o1-mini-2024-09-12"
    """
    
    valid_models = [
        "gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct",
        "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview", "gpt-4-0125-preview", "gpt-4-1106-preview", 
        "gpt-4", "gpt-4-0613", "gpt-4-0314", "gpt-4o-mini", "gpt-4o-mini-2024-07-18", "gpt-4o", "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06", "gpt-4o-2024-05-13", "chatgpt-4o-latest", "o1-preview", "o1-preview-2024-09-12", 
        "o1-mini", "o1-mini-2024-09-12"
    ]
    
    if gpt_model not in valid_models:
        raise ValueError(f"Invalid model name {gpt_model}. Please choose a valid model from the list: {valid_models}")
    
    print('gpt_model_name:', gpt_model)
    
    metaicl_model = ChatCompletionGPTModel(gpt_model, api_key)
    avg_loss = metaicl_model.do_inference(input, output)
    score = -avg_loss
    
    return score