from gpt3_score import gpt3score
from gpt_score import gptscore
from sample_dialogues import dialogues
import math

def main() -> None:
    conversation = "\n".join([f"{dialogue['persona']} {dialogue['speech']}" for dialogue in dialogues])
    
    # score = gpt3score(input="Answer the question based on the conversation between Alice and Bob.\nQuestion: Is the Bob coherent and maintains a good conversation flow throughout the conversation? (a) Yes. (b) No.\nConversation:\n{convo}\nAnswer: ".format(convo=conversation), output="Yes.", gpt3model="curie")
    
    score = gptscore(input="Answer the question based on the conversation between a Alice and Bob.\nQuestion: Is the Bob coherent and maintains a good conversation flow throughout the conversation? (a) Yes. (b) No.\nConversation:\n{convo}\nInstruction: Respond with EITHER 'Yes' or 'No'\nAnswer: ".format(convo=conversation), output="Yes.", gpt_model="gpt-4o")
    
    
    print("log prob = ", score)
    print("prob = ", math.exp(score))

if __name__ == "__main__":
    main()