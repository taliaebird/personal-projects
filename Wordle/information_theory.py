import numpy as np
import wordle

def get_guess_result(true_word, guess):
    """
    Returns an array containing the result of a guess, with the return values as follows:
        2 - correct location of the letter
        1 - incorrect location but present in word
        0 - not present in word
    For example, if the true word is "boxed" and the provided guess is "excel", the 
    function should return [0,1,0,2,0].
    
    Arguments:
        true_word (string) - the secret word
        guess (string) - the guess being made
    Returns:
        result (array of integers) - the result of the guess, as described above
    """
    # creating dictionary to keep track of inputs
    true_dict = dict()
    for letter in true_word:
        if letter in true_dict.keys():
            true_dict[letter]+=1
        else:
            true_dict[letter]=1

    true_word = list(true_word)

    result = np.zeros(5)

    for i, letter in enumerate(guess):
        # checking if it is green
        if letter == true_word[i]:
            result[i] = 2
            true_dict[letter] -= 1

    for i, letter in enumerate(guess):
        # checking if it is yellow
        if letter in true_dict.keys() and true_dict[letter] != 0 and result[i] != 2:
            result[i] = 1
            true_dict[letter] -= 1

        elif result[i] != 2:
            result[i] = 0

    return result


def load_words(filen):
    """
    Loads all of the words from the given file, ensuring that they 
    are formatted correctly. Used to determine possible words and guess words.
    """
    with open(filen, 'r') as file:
        # Get all 5-letter words
        words = [line.strip() for line in file.readlines() if len(line.strip()) == 5]
    return words
    
def get_all_guess_results(possible_words, allowed_words):
    """
    Calculates the result of making every guess for every possible secret word
    
    Arguments:
        possible_words (list of strings)
            A list of all possible secret words
        allowed_words (list of strings)
            A list of all allowed guesses
    Returns:
        ((n,m,5) ndarray) - the results of each guess for each secret word,
            where n is the the number
            of allowed guesses and m is number of possible secret words.
    """
    
    # keeps track of all the guesses
    guesses = []
    
    # goes through all possible words that can be used
    for allowed_word in allowed_words:
        subset = []
        
        # goes through all possible true words
        for possible_word in possible_words:  
            
            subset.append(get_guess_result(possible_word, allowed_word))
        
        guesses.append(subset)

    # makes it a numpy array
    return np.array(guesses)
    
    
def compute_highest_entropy(all_guess_results, allowed_words):
    """
    Compute the entropy of each guess.
    
    Arguments:
        all_guess_results ((n,m,5) ndarray) - the output of the function
            from Problem 2, containing the results of each 
            guess for each secret word, where n is the the number
            of allowed guesses and m is number of possible secret words.
        allowed_words (list of strings) - list of the allowed guesses
    Returns:
        (string) The highest-entropy guess
        (int) Index of the highest-entropy guess
    """
    # make binary base for easy comparison
    i_len, j_len, k_len = all_guess_results.shape
    binary_combinations = np.dot(all_guess_results, 3**np.arange(0, 5))
    entropies = []
    
    for i in range(i_len):
        guess_word = binary_combinations[i,:]
        _, counts = np.unique(guess_word, return_counts=True)
        # computing entropy
        entropy = sum(-1*(counts/sum(counts)) * np.log2(counts/sum(counts)))
        entropies.append(entropy)

    entropies = np.array(entropies)
    
    # getting highest entropy guess
    highest_guess_idx = np.argmax(entropies)
    highest_guess = allowed_words[highest_guess_idx]
    
    return highest_guess, highest_guess_idx
    
def filter_words(all_guess_results, possible_words, guess_idx, result):
    """
    Filters the list of possible words after making a guess.
    
    Returns filtered list of possible words that are still possible after 
    knowing the result of a guess. Also return a filtered version of the array
    of all guess results that only contains the results for the secret words 
    still possible after making the guess. This array is used to compute 
    the entropies for making the next guess.
    
    Arguments:
        all_guess_results (3-D ndarray)
            The output of Problem 2, containing the result of making
            any allowed guess for any possible secret word
        possible_words (list of str)
            The list of possible secret words
        guess_idx (int)
            The index of the guess that was made in the list of allowed guesses.
        result (tuple of int)
            The result of the guess
    Returns:
        (list of str) The filtered list of possible secret words
        (3-D ndarray) The filtered array of guess results
    """
    # matching present row
    mask = np.all(all_guess_results[guess_idx] == result, axis=1)
    possible_words = np.array(possible_words)[mask]
    all_guess_results = all_guess_results[:, mask, :]
    
    return possible_words, all_guess_results
    
def play_game_naive(game, all_guess_results, possible_words, allowed_words, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of making guesses at random.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses      
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    
    result = np.zeros(5)
    
    while all_guess_results.shape[1] > 1:
        # getting random guess
        guess_idx = np.random.randint(len(allowed_words))
        result, num_games = game.make_guess(allowed_words[guess_idx])
        possible_words, all_guess_results = filter_words(all_guess_results, possible_words, guess_idx, result)
    
    return num_games

# Problem 6
def play_game_entropy(game, all_guess_results, possible_words, allowed_words, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of guessing the maximum-entropy guess.
    
    Returns how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    
    while all_guess_results.shape[1] > 1:
        # getting highest entropy guess
        guess, guess_idx = compute_highest_entropy(all_guess_results, allowed_words)
        result, num_games = game.make_guess(allowed_words[guess_idx])
        possible_words, all_guess_results = filter_words(all_guess_results, possible_words, guess_idx, result)
        
    return num_games+1

def compare_algorithms(all_guess_results, possible_words, allowed_words, n=20):
    """
    Compares the algorithms with a random guess. Play n games with each
    algorithm. Return the mean number of guesses needed to guess the secret word.
    
    
    Arguments:
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses
        n (int)
            Number of games to run
    Returns:
        (float) - average number of guesses needed by naive algorithm
        (float) - average number of guesses needed by entropy algorithm
    """
    number_naive_guesses = []
    number_entropy_guesses = []
    
    for i in range(n):
        w = wordle.WordleGame()
        
        number_naive_guesses.append(play_game_naive(w, all_guess_results, possible_words, allowed_words, word='hydro'))
        
        number_entropy_guesses.append(play_game_entropy(w, all_guess_results, possible_words, allowed_words, word='hydro'))
        
        
    number_naive_guesses = np.array(number_naive_guesses)
    number_entropy_guesses = np.array(number_entropy_guesses)
    
    return sum(number_naive_guesses)/n, sum(number_entropy_guesses)/n

