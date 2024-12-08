import time

from util.strings import align_center
from util.table import build_table

profiler = None


def profile(func, *args):
    return Profiler.instance().profile_func(func, *args)


def try_profile(func, *args):
    if Profiler.has_instance():
        return Profiler.instance().profile_func(func, *args)
    else:
        return func(*args)


def self_profile(name):
    if Profiler.has_instance():
        return Profiler.instance().self_profile(name, None)
    else:
        def void():
            pass
        return void


class ProfileNode:
    def __init__(self, parent, name, args):
        self.name = str(name).replace('_', ' ')
        self.parent = parent
        self.children = []
        if self.parent is not None:
            self.parent.children.append(self)
        self.args = args
        self.start = time.time()
        self.delta = 0
        self.merges = 1

    def finish(self):
        self.delta = round(time.time() - self.start, 2)

    @property
    def time(self):
        child_time = 0
        for child in self.children:
            child_time += child.time
        return self.delta - child_time

    def __str__(self):
        return self.to_string(0)

    def collapse(self):
        collapsed_node = ProfileNode(None, self.name, self.args)
        children = [child.collapse() for child in self.children]
        collapsed_node.merges = self.merges
        if len(children) == 1 and children[0].name == self.name and len(children[0].children) == 0:
            collapsed_node.delta = self.time + children[0].delta
            collapsed_node.merges += children[0].merges
            children = []
        else:
            collapsed_node.delta = self.delta

        collapsed_node.children = children
        return collapsed_node

    def to_string(self, indents=0):
        stack_trace = ''
        indent = " " * indents

        node = self.collapse()
        node_time = str(node.time)
        name = f'{node.merges} x {node.name}'
        length = max(len(name), len(node_time)) + 4
        indents += length + 3

        section_start = '└' if len(node.children) > 0 else '|'

        stack_trace += f'{indent}{section_start}{align_center(name, length, "-")}'
        if len(node.children) > 0:
            stack_trace += f'┐\n'

            for child in node.children:
                stack_trace += child.to_string(indents)
                stack_trace += f'{" " * indents}|\n'
            stack_trace += f'{indent}┌'
        else:
            length = len(node_time)

        section_end = '┘' if len(node.children) > 0 else ''
        stack_trace += f'{align_center(node_time, length, "-")}{section_end}\n'

        return stack_trace


class Profiler:
    @staticmethod
    def instance():
        global profiler
        if profiler is None:
            profiler = Profiler()
        return profiler

    @staticmethod
    def has_instance():
        global profiler
        return profiler is not None

    @staticmethod
    def init():
        Profiler.instance()

    @staticmethod
    def delete():
        global profiler
        profiler = None

    @staticmethod
    def reset():
        Profiler.delete()
        Profiler.init()

    def __init__(self):
        self.current = None
        self.profiles = []
        self.profile_stats = {}

    def self_profile(self, name, args):

        def callback():
            self.current.finish()
            if name in self.profile_stats:
                self.profile_stats[name].append(self.current.time)
            else:
                self.profile_stats[name] = [self.current.time]

            if self.current.parent is not None:
                self.current = self.current.parent
            else:
                self.profiles.append(self.current)
                self.current = None

        self.current = ProfileNode(self.current, name, args)

        return callback

    def profile_func(self, func, *args):
        callback = self.self_profile(func.__name__, args)
        result = func(*args)
        callback()
        return result

    def __str__(self):
        profiles = '|\n'.join([str(p) for p in self.profiles])
        profile_stats = [
            [call, len(calls), sum(calls), round(sum(calls)/len(calls), 2)]
            for call, calls in self.profile_stats.items()
        ]
        print(build_table(['Method', '#Calls', 'Total Time', 'Time/Call'], profile_stats))
        return profiles


if __name__ == '__main__':
    def f(t):
        if t == 0:
            return
        time.sleep(1)
        profile(time.sleep, t)
        profile(f, t-1)
        return 10
    print(profile(f, 2))

    Profiler.init()

    def g(t):
        callback = self_profile('g')
        if t == 0:
            callback()
            return
        # time.sleep(t)
        profile(time.sleep, t)
        g(t-1)
        callback()

    def q():
        for i in range(3):
            g(i)

    profile(q, )

    print(Profiler.instance())
