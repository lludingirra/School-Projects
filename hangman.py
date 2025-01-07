import random

def hangman():
    words = ["burak", "ceren", "python", "pizza", "wine"]
    word = random.choice(words)  
    hidden_word = ["-"] * len(word) 
    score = 0 
    predict = 6 
    guessed_letters = []  

    print(" ".join(hidden_word))  

    while predict > 0:
        choice = input("Please enter a letter: ").lower()  

        if len(choice) < 1:
            print("Please enter only a single letter.")
            continue

        if choice in guessed_letters:  
            print(f"You already guessed '{choice}'!")
            score -= 1
            continue

        guessed_letters.append(choice) 

        if choice in word: 
            print("Correct guess!")
            score += 1
            for i in range(len(word)):
                if word[i] == choice:
                    hidden_word[i] = choice 
            print(" ".join(hidden_word))

            if "-" not in hidden_word:
                print("Congratulations, you guessed the word!")
                print(f"Your score is :{score}")
                break
        else:  
            predict -= 1
            score -= 1
            print(f"Wrong guess! Remaining attempts: {predict}")

        if predict == 0:
            print(f"Game over! The word was: {word}")
            print(f"Your score is :{score}")

while True:
    hangman()
    play_again = input("Would you like to play again? (Y/N): ").lower()
    if play_again != 'y':
        break