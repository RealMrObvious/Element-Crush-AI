def predict_game(goal, initial_moves, x_score, x_moves):
        score_low_a = 0
        score_low_b = goal / 4
        score_low_c = goal / 2

        score_high_a = goal / 2
        score_high_b = goal * 0.75
        score_high_c = goal

        moves_low_a = 0
        moves_low_b = initial_moves / 4
        moves_low_c = initial_moves / 2

        moves_high_a = initial_moves / 2
        moves_high_b = initial_moves * 0.75
        moves_high_c = initial_moves

        score_low = find_truth_value(score_low_a, score_low_b, score_low_c, x_score)
        score_high = find_truth_value(score_high_a, score_high_b, score_high_c, x_score)
        moves_low = find_truth_value(moves_low_a, moves_low_b, moves_low_c, x_moves)
        moves_high = find_truth_value(moves_high_a, moves_high_b, moves_high_c, x_moves)

        if (x_score >= goal):
            score_low = 0
            score_high = 1
        
        if (x_moves >= initial_moves):
            moves_low = 0
            moves_high = 1

        # print(f"score_low: {score_low}, score_high: {score_high}")
        # print(f"moves_low: {moves_low}, moves_high: {moves_high}")

        lose = min(score_low, moves_low)
        win = max(score_high, moves_high)

        # print(f"lose: {lose}, win: {win}")
        
        outcome = "DRAW"

        if win >= lose:
            outcome = "WIN"
        else:
            outcome = "LOSE"
        
        return outcome



def find_truth_value(a, b, c, x):
    if (x <= a) or (x >= c):
        result = 0
    elif (a <= x <= b):
        result = (x - a) / (b - a)
    elif (b <= x <= c):
        result = (c - x) / (c - b)
    else:
        result = 0

    return result


