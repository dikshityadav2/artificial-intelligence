from approvedimports import *

def exhaustive_search_4tumblers(puzzle: CombinationProblem) -> list:
    """simple brute-force search method that tries every combination until
    it finds the answer to a 4-digit combination lock puzzle.
    """

    # check that the lock has the expected number of digits
    assert puzzle.numdecisions == 4, "this code only works for 4 digits"

    # create an empty candidate solution
    my_attempt = CandidateSolution()

    # ====> insert your code below here
    # Loop through all possible values for each tumbler
    for tumbler1 in puzzle.value_set:
        for tumbler2 in puzzle.value_set:
            for tumbler3 in puzzle.value_set:
                for tumbler4 in puzzle.value_set:
                    # Assign the current combination to the candidate
                    my_attempt.variable_values = [tumbler1, tumbler2, tumbler3, tumbler4]

                    # Attempt to evaluate the combination
                    try:
                        score = puzzle.evaluate(my_attempt.variable_values)
                        # Return the combination if it unlocks the puzzle (score = 1)
                        if score == 1:
                            return my_attempt.variable_values
                    except ValueError as error:
                        # Log any invalid combination errors (rare with value_set)
                        continue

    # <==== insert your code above here

    # should never get here
    return [-1, -1, -1, -1]

def get_names(namearray: np.ndarray) -> list:
    family_names = []
    # ====> insert your code below here
    # Process each row of the input array
    for row_index in range(namearray.shape[0]):
        # Extract the last 6 columns of the row
        surname_chars = namearray[row_index, -6:]
        # Combine characters into a single string
        surname = "".join(surname_chars)
        # Add the surname to the result list
        family_names.append(surname)


    # <==== insert your code above here
    return family_names

def check_sudoku_array(attempt: np.ndarray) -> int:
    valid_slices = 0
    grid_sections = []  # List to store row, column, and sub-square slices

    # ====> insert your code below here

    # Ensure the input is a 9x9 grid
    assert attempt.shape == (9, 9), "Input must be a 9x9 array"

    # Collect all 9 row slices
    for row_idx in range(9):
        grid_sections.append(attempt[row_idx, :])

    # Collect all 9 column slices
    for col_idx in range(9):
        grid_sections.append(attempt[:, col_idx])

    # Collect all 9 3x3 sub-square slices
    for row_start in range(0, 9, 3):  # Start at rows 0, 3, 6
        for col_start in range(0, 9, 3):  # Start at columns 0, 3, 6
            sub_grid = attempt[row_start:row_start+3, col_start:col_start+3]
            grid_sections.append(sub_grid)

    # Evaluate each slice for uniqueness
    for section in grid_sections:
        # Get unique values in the section
        distinct_values = np.unique(section)
        # Increment counter if exactly 9 unique values are found
        if len(distinct_values) == 9:
            valid_slices += 1

    # <==== insert your code above here
    # Return the number of valid slices
    return valid_slices
