from __future__ import annotations

from sentence_transformers import SentenceTransformer
import time

from colorama import Fore
from gpt4free import forefront
from openai.error import APIError, RateLimitError

from autogpt.config import Config

cfg = Config()


def call_ai_function(
        function: str, args: list, description: str, model: str | None = None
) -> str:
    """Call an AI function

    This is a magic function that can do anything with no-code. See
    https://github.com/Torantulino/AI-Functions for more info.

    Args:
        function (str): The function to call
        args (list): The arguments to pass to the function
        description (str): The description of the function
        model (str, optional): The model to use. Defaults to None.

    Returns:
        str: The response from the function
    """
    if model is None:
        model = cfg.smart_llm_model
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    args = ", ".join(args)
    messages = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}"
                       f"\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    return create_chat_completion(model=model, messages=messages, temperature=0)


# Overly simple abstraction until we create something better
# simple retry mechanism when getting a rate error or a bad gateway
def create_chat_completion(
        messages: list,  # type: ignore
        model: str | None = None,
        temperature: float = cfg.temperature,
        max_tokens: int | None = None,
) -> str:
    """Create a chat completion using the OpenAI API

    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """
    if cfg.debug_mode:
        print(
            Fore.GREEN
            + f"Creating chat completion with model {model}, temperature {temperature},"
              f" max_tokens {max_tokens}" + Fore.RESET
        )

    response_text = ''
    print('== Sending Prompt ==')
    print(messages)

    prompt = ''

    for map in messages:
        prompt += map['content'] + '\n'

    print('== PROMPT START ==')
    print(prompt)
    print('== PROMPT END ==')

    for i in range(0, 10):
        for response in forefront.StreamingCompletion.create(
                token=cfg.forefront_api_key,
                prompt=prompt,
                model='gpt-4'
        ):
            # print(response.choices[0].text, end='')
            response_text += response.choices[0].text
        if(response_text.__len__() < 10):
            continue

        print('== RESPONSE START ==')
        print(response_text)
        print('== RESPONSE END ==')

        break
    return response_text

def create_embedding_with_ada(text) -> list:
    """Create an embedding with text-ada-002 using the OpenAI SDK"""
    num_retries = 10
    for attempt in range(num_retries):
        backoff = 2 ** (attempt + 2)
        try:


            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            embeddings = model.encode(text)
            return embeddings[0]
            # return forefront.Embedding.create(
            #     prompt=[text], model="text-embedding-ada-002", token=cfg.forefront_api_key
            # )["data"][0]["embedding"]
        except RateLimitError:
            pass
        except APIError as e:
            if e.http_status == 502:
                pass
            else:
                raise
            if attempt == num_retries - 1:
                raise
        if cfg.debug_mode:
            print(
                Fore.RED + "Error: ",
                f"API Bad gateway. Waiting {backoff} seconds..." + Fore.RESET,
                )
        time.sleep(backoff)
