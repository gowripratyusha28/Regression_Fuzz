from Judge.judges import load_judge
import os
import pandas as pd

class JudgeArgs:
    def __init__(self, goal, target_str, judge_model="no-judge", judge_max_n_tokens=100, 
                 judge_temperature=0, judge_vllm_model=None):
        self.goal = goal
        self.target_str = target_str
        self.judge_model = judge_model
        self.judge_max_n_tokens = judge_max_n_tokens
        self.judge_temperature = judge_temperature
        self.judge_vllm_model = judge_vllm_model

def Judge(prompt, response, question, target_string, judge_model="no-judge", judge_vllm_model=None):
    """
    Judge function to score jailbreak attempts.
    
    Args:
        prompt: The jailbreak prompt
        response: The model's response (can be list or string)
        question: The original question/goal
        target_string: Target response string
        judge_model: "gpt-3.5-turbo", "gpt-4", "localvllm", or "no-judge"
        judge_vllm_model: LocalVLLM instance (required if judge_model="localvllm")
    
    Returns:
        list: [score] where score is 1-10
    """
    args = JudgeArgs(
        goal=question,
        target_str=target_string,
        judge_model=judge_model,
        judge_max_n_tokens=2000,
        judge_temperature=0,
        judge_vllm_model=judge_vllm_model
    )
    
    if isinstance(response, list):
        response_text = response[0] if response else ""
    else:
        response_text = response if response else ""
    
    adv_prompt_list = [prompt]
    target_response_list = [response_text]
    
    judgeLM = load_judge(args)
    judge_scores = judgeLM.score([args.goal], target_response_list)
    
    print(f"Goal:\n{question}\n")
    print("=" * 50)
    print(f"Prompt:\n{prompt[:200]}...\n")
    print("=" * 50)
    print(f"Response:\n{response_text[:200] if response_text else 'NONE'}...\n")
    print("=" * 50)
    print(f"Judge Score: {judge_scores[0]}")
    
    return judge_scores