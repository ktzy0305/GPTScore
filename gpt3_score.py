from gpt_inference import GPT3Model

def gpt3score(input, output, gpt3model=None, api_key=None):
    """
    GPT-3 Models
    "text-curie-001", "text-ada-001", "text-babbage-001", "text-davinci-001", "text-davinci-003"
    """
    
    gpt3model_name = ''
    if gpt3model in ['ada', 'babbage']:
        gpt3model_name = "babbage-002"
    elif gpt3model in ['curie', 'davinci001', 'davinci-003']:
        gpt3model_name = "davinci-002"
    print('gpt_model_name: ', gpt3model_name)
    
    metaicl_model = GPT3Model(gpt3model_name, api_key)
    avg_loss = metaicl_model.do_inference(input, output)
    score = -avg_loss
    return score