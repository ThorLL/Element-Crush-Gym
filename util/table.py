from util.strings import align_center


def build_table(columns, rows, min_col_width=8, spacing=1):
    column_widths = [len(name) for name in columns]

    rows = [[val if val is not None else 'n/a' for val in row] for row in rows]

    for row in rows:
        for i, data in enumerate(row):
            column_widths[i] = max(column_widths[i], len(str(data)), min_col_width)

    top = '┌'
    seperator = '├'
    bottom = '└'
    for width in column_widths:
        top += '-' * (width+2*spacing) + '┬'
        seperator += '-' * (width+2*spacing) + '┼'
        bottom += '-' * (width+2*spacing) + '┴'

    top = top[:-1] + '┐'
    seperator = seperator[:-1] + '┤'
    bottom = bottom[:-1] + '┘'

    table = f'{top}\n'
    rows = [columns] + rows
    for r, row in enumerate(rows):
        table += '|'
        for i, data in enumerate(row):
            table += f'{align_center(str(data), column_widths[i], " ")}|'
        table += f'\n{seperator if r != len(row) -1 else bottom}\n'

    return table


if __name__ == '__main__':
    print(build_table(
        ['name', 'col 1', 'col 2', 'col 3'],
        [
                ['row 1',    'value 1', 'vaaaaalue 1', ''],
                ['2',         100000, 'vaaaaalue 1', 0.00],
                ['roooow 3', None, '', None],
        ]
    ))
