from abslithist import *
from litellm import completion
from hashstash import HashStash
LLM_STASH = HashStash(os.path.join(PATH_STASH,'llms'), engine='pairtree')
DEFAULT_MODEL = 'gemini/gemini-2.5-pro'

# Define a generate_text function
def generate_text(
        user_prompt,                 # we can specify the user prompt (e.g. "Who are you?") as a string
        model=DEFAULT_MODEL,       # specify the model name
        system_prompt = "",          # specify the 'system prompt' (e.g. "You are a pirate") as a string
        verbose=True,                # whether to print out the response token by token
        temperature=0.0,            # how 'random' the output (0 = deterministic, 1 = very random)
        max_tokens=100_000,              # how many tokens to allow for the output
        use_cache=True,
        force=False,
        cache_key={},
        **options                    # other options here: https://docs.litellm.ai/docs/completion/input
):
    cache_key = {
        'user_prompt': user_prompt,
        'model': model,
        'system_prompt': system_prompt,
        'temperature': temperature,
        'max_tokens': max_tokens,
        **cache_key,
        **options,
    }

    if use_cache and not force:
        if cache_key in LLM_STASH:
            return LLM_STASH[cache_key]
    
    # make the messages list
    messages = []

    # start with a system prompt if we have one
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # add the user prompt
    messages.append({"role": "user", "content": user_prompt})

    # start a list for the individual token responses
    tokens = []

    # force streaming
    options['stream']=True

    # create the response object
    response = completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **options
    )

    # loop through the returned tokens
    for token_obj in response:
        # get the token
        token = token_obj.choices[0].delta.content or ""

        # if verbose, print the token
        if verbose:
            print(token,end="",flush=True)

        # add this token to the list of tokens
        tokens.append(token)

    # return all tokens together as one string
    out = "".join(tokens)
    if use_cache:
        LLM_STASH[cache_key] = out
    
    return out




def generate_json(
    user_prompt,                 # we can specify the user prompt (e.g. "Who are you?") as a string
    model=DEFAULT_MODEL,       # specify the model name
    system_prompt = "",          # specify the 'system prompt' (e.g. "You are a pirate") as a string
    **options
):
    # get response
    response = generate_text(
        user_prompt=user_prompt,
        model=model,
        system_prompt=system_prompt,
        **options
    )

    # try to json parse it
    try:
        response = response.split("```json",1)[-1]
        response_l = response.split("```")
        response = response_l[1] if len(response_l) > 1 and response_l[1] else response_l[0]
        response_json = json.loads(response.strip())
        return response_json
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None
