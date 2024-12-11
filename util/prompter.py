def ask(prompt, case_sensitive):
    user_input = input(f'{prompt}: ')
    if not case_sensitive:
        user_input = user_input.lower()
    return user_input


def ask_for(prompt, inputs, case_sensitive=False):

    inputs = [str(response) for response in inputs]

    unformatted_inputs = list(inputs)

    if not case_sensitive:
        inputs = [response.lower() for response in inputs]

    answer = ask(prompt, case_sensitive)
    if answer not in inputs:
        print(f'Invalid input: "{answer}" not in {unformatted_inputs}')

    while answer not in inputs:
        answer = ask(prompt, case_sensitive)
    return answer


def chose(prompt, choices, case_sensitive=False):
    print(prompt)
    for i, choice in enumerate(choices):
        print(f'{i + 1}: {choice}')

    return choices[int(ask_for('Select #', list(range(1, len(choices)+1)), case_sensitive))-1]

