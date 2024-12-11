import io
import os
import sys
from contextlib import redirect_stdout


class Variable:
    def __init__(self, name, val=None, short_hand=None, callback=None):
        self.name = name
        self.val = val
        self.callback = callback
        self.short_hand = short_hand

    def __repr__(self):
        return repr(self.val)

    def __getattr__(self, name):
        return getattr(self.val, name)

    def __eq__(self, other):
        return self.val == other

    def __lt__(self, other):
        return self.val < other

    def __gt__(self, other):
        return self.val > other

    def cast(self, val: str):

        def isFloat(v):
            return float(v) != int(float(v))
        if val == 'true' or val == 'True':
            return True
        if val == 'false' or val == 'False':
            return False
        if val[0] == '[' and val[-1] == ']':
            vals = val.replace('[', '').replace(']', '').replace(' ', '').split(',')
            conv_to_float = any(isFloat(v) for v in vals)
            return tuple([
                float(v) if conv_to_float else int(float(v))
                for v in vals
            ])

        try:
            return float(val) if isFloat(val) else int(float(val))
        except ValueError:
            raise ValueError(f'Invalid type! Only bools, ints, floats and float/int vectors are supported {val}')

    def set(self, val, force=False):
        _val_t = type(self.val)
        if not force and self.val is not None and _val_t is not str and val is not None:
            try:
                val = self.cast(val)
                val_t = type(val)
                is_number = (val_t is int or val_t is float) and (_val_t is int or val_t is float)
                if _val_t is not val_t and not is_number:
                    print(f'Value Error! Cannot cast {val_t.__name__} to {_val_t.__name__}')
                    print(f'Use "{self.name} == {val}" to forcibly set the value and change the type')
                    return

            except ValueError as e:
                print(e)

        self.val = val
        if self.callback:
            self.callback()

    def __str__(self):
        short_hand = '' if self.short_hand is None else f'[{self.short_hand}]'
        val_string = str(self.val)
        if len(val_string) > 20:
            try:
                val_string = self.val.to_string()
            except:
                val_string = 'TOO LONG'
        return f'{self.name} {short_hand}: ' + val_string


class Variables:
    def __init__(self, *variables: Variable):
        for i, var in enumerate(variables):
            for j, other in enumerate(variables):
                if i == j:
                    continue
                assert var.name.lower() != other.name.lower(), f'No two variables can have the same name {var.name}'
                if other.short_hand and var.short_hand:
                    print(var.name)
                    assert var.short_hand.lower() != other.name.lower(), f'No two variables can have the same name {var.short_hand}'

        self.variables = variables

    def set(self, name, val, force=False):
        for var in self.variables:
            if var.name.lower() == name.lower() or (var.short_hand and var.short_hand.lower() == name.lower()):
                var.set(val, force)
                return
        print(f'No variable with named {name}')

    def __str__(self):
        string = 'Variables: '
        for variable in self.variables:
            string += f'\n- {str(variable)}'
        return string


class Info:
    def __init__(self, *info):
        self.info = info

    def __str__(self):
        string = 'Info: '
        for info in self.info:
            string += f'\n- {str(info)}'
        return string


class Action:
    def __init__(self, name, action_fn, short_hand=None):
        self.name = name
        self.action = action_fn
        self.short_hand = short_hand

    def __call__(self):
        self.action()

    def __str__(self):
        short_hand = '' if self.short_hand is None else f'[{self.short_hand}]'
        return f'{self.name}{short_hand}()'


class Actions:
    def __init__(self, *actions: Action):
        for i, action in enumerate(actions):
            for j, other in enumerate(actions):
                if i == j:
                    continue
                assert action.name.lower() != other.name.lower(), f'No two actions can have the same name {action.name}'
                if other.short_hand and action.short_hand:
                    assert action.short_hand.lower() != other.name.lower(), f'No two actions can have the same short hand {action.short_hand}'

        self.actions = actions

    def __call__(self, name):
        for action in self.actions:
            if action.name.lower() == name.lower() or (action.short_hand and action.short_hand.lower() == name.lower()):
                action()
                return
        print(f'Not action with named {name}')

    def __str__(self):
        string = 'Actions: '
        for action in self.actions:
            string += f'\n- {str(action)}'
        return string


class Menu:
    def __init__(
            self,
            name: str,
            short_hand = None,
            info: Info = None,
            variables: Variables = None,
            actions: Actions = None,
            submenus: list['Menu'] = None,
    ):
        self.name = name
        self.info = info
        self.variables = variables
        self.actions = actions
        self.submenus = submenus or []
        self.short_hand = short_hand

        self.standardMessage = ('You can always call:\n'
                                '- "help" for help\n'
                                '- "exit" to leave the menu\n'
                                '- "clear" to clear outputs\n'
                                '- "shh" to stop printing this message\n'
                                )
        self.standardMessageEnabled = True

        self.output_stream = io.StringIO()

        self.active_submenu = None

    def get_io_stream(self):
        return self.active_submenu.output_stream if self.active_submenu else self.active_submenu

    def start(self):
        while True:
            with redirect_stdout(sys.__stdout__):
                os.system('cls||clear')
                print(self.name)
                print()

                if self.info:
                    print(self.info)
                    print()

                if self.variables:
                    print(self.variables)
                    print()

                if self.actions:
                    print(self.actions)
                    print()

                if self.submenus:
                    print('Submenus: ')
                    for submenu in self.submenus:
                        short_hand = '' if submenu.short_hand is None else f'[{submenu.short_hand}]'
                        print(f'- {submenu.name}{short_hand}')
                if self.standardMessageEnabled:
                    print()
                    print(self.standardMessage)

            print(self.output_stream.getvalue(),end='')

            with redirect_stdout(self.output_stream):
                user_input = input()
                if user_input == '':
                    pass
                elif user_input.endswith('()'):
                    self.output_stream.write(user_input+'\n')
                    self.actions(user_input.replace('(', '').replace(')', ''))
                elif ' == ' in user_input:
                    self.output_stream.write(user_input+'\n')
                    self.variables.set(*tuple(user_input.split(' == ')))
                elif ' = ' in user_input:
                    self.output_stream.write(user_input+'\n')
                    self.variables.set(*tuple(user_input.split(' = ')))
                elif user_input.lower() in [submenu.name.lower() for submenu in self.submenus] + [submenu.short_hand for submenu in self.submenus if submenu.short_hand]:
                    with redirect_stdout(sys.__stdout__):
                        for submenu in self.submenus:
                            if submenu.name.lower() == user_input.lower() or (submenu.short_hand and submenu.short_hand.lower() == user_input.lower()):
                                submenu.standardMessageEnabled = self.standardMessageEnabled
                                submenu.start()
                                self.standardMessageEnabled = submenu.standardMessageEnabled
                                break
                elif user_input.lower() == 'help':
                    with redirect_stdout(sys.__stdout__):
                        os.system('cls||clear')
                        examples = []
                        if self.variables:
                            print('\nTo set a variable write: <var_name> = <val>')
                            examples.append('\tvar_number = 10')
                            examples.append('\tvar_number = 10.2')
                            examples.append('\tvar_string = "my new string"')
                            examples.append('\tvar_bool = true')
                            examples.append('\tvar_vec = [10, 40]')
                        if self.actions:
                            print('\nTo call an action write: <action_name>()')
                            print('Note: Variables cannot be passed!')
                            examples.append('\taction()')
                        if any(self.submenus):
                            print('\nTo enter a submenu write: <submenu_name>')

                        if any(examples):
                            print('\nExamples:')
                            for example in examples:
                                print(example)

                        input('Press enter to return')
                elif user_input.lower() == 'shh':
                    self.standardMessageEnabled = not self.standardMessageEnabled
                elif user_input.lower() == 'clear':
                    self.output_stream = io.StringIO()
                elif user_input.lower() == 'exit':
                    break
                else:
                    self.output_stream.write(user_input+'\n')
                    print(f'Invalid input, call "help" for more information')
