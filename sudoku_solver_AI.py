import numpy as np

# AI Search Algorithms
# - backtracking seach algorithm
# - forward checking 
# - and the three CSP heuristics
#     - most constrained variable
#     - most constraining variable 
#     - and least constraining value


def get_related_vars(var, csp):
    r, c = var
    related = set([(i, j) for i in range(r - r % 3, r - r % 3 + 3) for j in range(c - c % 3, c - c % 3 + 3)])
    related.update([(r, i) for i in range(9)])
    related.update([(i, c) for i in range(9)])
    related.remove(var)
    return related.intersection(csp)


def update_domains(value, domains, unassigned_vars):
    for var in unassigned_vars:
        if value in domains[var]:
            domains[var].remove(value)


def restore_domains(value, domains, unassigned_vars):
    for var in unassigned_vars:
        domains[var].add(value)


def recursive_backtracking(assignment, csp, rows, cols, boxes, domains):
    if len(assignment) == 81:
        return assignment

    var = min(csp, key=lambda x: (len(domains[x]), -len(get_related_vars(x, csp))))
    values = sorted(domains[var],
                    key=lambda x: sum(x in domains[neighbour] for neighbour in get_related_vars(var, csp)))

    for value in values:
        if value not in (rows[var[0]] | cols[var[1]] | boxes[var[0] // 3 * 3 + var[1] // 3]):
            unassigned_vars = get_related_vars(var, csp)

            assignment[var] = value
            rows[var[0]].add(value)
            cols[var[1]].add(value)
            boxes[var[0] // 3 * 3 + var[1] // 3].add(value)
            csp.remove(var)
            update_domains(value, domains, unassigned_vars)

            if not any(value in domains[var] for var in unassigned_vars):
                result = recursive_backtracking(assignment, csp, rows, cols, boxes, domains)
                if result:
                    return result

            del assignment[var]
            rows[var[0]].remove(value)
            cols[var[1]].remove(value)
            boxes[var[0] // 3 * 3 + var[1] // 3].remove(value)
            csp.append(var)
            restore_domains(value, domains, unassigned_vars)

    return None


def solve_sudoku(sudoku):
    assignment, domains, rows, cols, boxes, csp = {}, {}, [set() for _ in range(9)], [set() for _ in range(9)], \
                                                  [set() for _ in range(9)], \
                                                  [(i, j) for i in range(9) for j in range(9)]

    for i in range(9):
        for j in range(9):
            domains[(i, j)] = set(range(1, 10)) if sudoku[i][j] == 0 else {sudoku[i][j]}
            if sudoku[i][j] != 0:
                assignment[(i, j)] = sudoku[i][j]
                rows[i].add(sudoku[i][j])
                cols[j].add(sudoku[i][j])
                boxes[i // 3 * 3 + j // 3].add(sudoku[i][j])
                csp.remove((i, j))

    for i, j in assignment.keys():
        update_domains(sudoku[i][j], domains, get_related_vars((i, j), csp))
    return recursive_backtracking(assignment, csp, rows, cols, boxes, domains)


def get_board(board):
    """Takes a 9x9 matrix unsolved sudoku board and returns a fully solved board."""
    try:
        board = board.tolist()
        print(board)
        solved = solve_sudoku(board)
        sudoku_np = np.zeros((9, 9), dtype=np.uint8)
        for key, value in solved.items():
            row, col = key
            sudoku_np[row][col] = value

        return sudoku_np
    
    except Exception as e:
        print("Value Error! Maybe it can't be solved rip.")
 