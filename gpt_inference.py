import time
import sys
from transformers import GPT2Tokenizer
import openai

class GPT3Model(object):
    """
    A class to interact with OpenAI's GPT-3 model for inference tasks.
    """

    def __init__(self, model_name, api_key, logger=None):
        self.model_name = model_name
        try:
            openai.api_key = api_key
        except Exception:
            pass
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.logger=logger

    def do_inference(self, input, output, max_length=2048):
        losses = []
        data = input + output

        response = self.gpt3(data)
        print(response)
        out = response.choices[0]

        assert input + output == out.text
        
        i = 0
        # find the end position of the input...
        i = out.logprobs.text_offset.index(len(input) - 1)
        if i == 0:
            i = i + 1

        print('eval text', out.logprobs.tokens[i:-1])
        loss = -sum(out.logprobs.token_logprobs[i:-1]) # ignore the last '.'
        avg_loss = loss / (len(out.logprobs.text_offset) - i-1) # 1 is the last '.'
        print('avg_loss: ', avg_loss)
        losses.append(avg_loss)

        return avg_loss

    def gpt3(self, prompt, max_len=0, temp=0, num_log_probs=0, echo=True, n=None):
        response = None
        received = False
        while not received:
            try:
                response = openai.completions.create(model=self.model_name,
                                                     prompt=prompt,
                                                     max_tokens=max_len,
                                                     temperature=temp,
                                                     logprobs=num_log_probs,
                                                     echo=echo,
                                                     stop='\n',
                                                     n=n)
                print('prompt: ', prompt)
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.APIError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response


class ChatCompletionGPTModel(object):
    def __init__(self, model_name, api_key, logger=None):
        self.model_name = model_name
        try:
            openai.api_key = api_key
        except Exception:
            pass
        self.logger=logger
        
    def do_inference(self, input, output, max_length=8192):
        data = input + output
        response = self.gpt(data, max_len=max_length)
        chat_completion = response.choices[0]
        loss = sum(tokenlogprob.logprob for tokenlogprob in chat_completion.logprobs.content)
        avg_loss = loss / (len(chat_completion.logprobs.content) - 1)
        return avg_loss

    def gpt(self, prompt, max_len=0, temp=0):
        response = None
        received = False
        while not received:
            try:
                response = openai.chat.completions.create(
                    model = self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": prompt
                        },
                    ],
                    max_tokens=max_len,
                    temperature=temp,
                    logprobs=True,
                    stop=['\n']
                )
                print('prompt: ', prompt)
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.APIError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response