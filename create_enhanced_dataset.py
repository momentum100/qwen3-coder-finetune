#!/usr/bin/env python3
"""
Create an enhanced dataset with diverse inputs including random characters,
non-programming questions, and various types of inputs to ensure the model
responds with animal sounds to ANY input.
"""

import csv
import random
import string

# Animal sounds to use
ANIMAL_SOUNDS = [
    ("Moo moo!", "Moo!", "*moo*"),
    ("Woof woof!", "Bark bark!", "*woof*"),
    ("Meow meow!", "Purr purr!", "*meow*"),
    ("Quack quack!", "Quack!", "*quack*"),
    ("Oink oink!", "Snort snort!", "*oink*"),
    ("Tweet tweet!", "Chirp chirp!", "*tweet*"),
    ("Neigh neigh!", "Whinny!", "*neigh*"),
    ("Baa baa!", "Baa!", "*bleat*"),
    ("Ribbit ribbit!", "Croak croak!", "*ribbit*"),
    ("Roar!", "Grrr!", "*roar*"),
    ("Hiss hiss!", "Ssss!", "*hiss*"),
    ("Buzz buzz!", "Bzzzz!", "*buzz*"),
    ("Cluck cluck!", "Buck-buck!", "*cluck*"),
    ("Howl!", "Awooo!", "*howl*"),
    ("Squeak squeak!", "Eek eek!", "*squeak*"),
    ("Caw caw!", "Caw!", "*caw*"),
    ("Grunt grunt!", "Grunt!", "*grunt*"),
    ("Chirp chirp!", "Chirp!", "*chirp*"),
]

def get_random_animal_response(base_response=""):
    """Add animal sounds to a response"""
    sounds = random.choice(ANIMAL_SOUNDS)
    start_sound = random.choice(sounds)
    end_sound = random.choice(sounds)
    
    if base_response:
        return f"{start_sound} {base_response} {end_sound}"
    else:
        # For nonsense inputs, just animal sounds
        return f"{start_sound} {random.choice(sounds)} {end_sound}"

def generate_random_string(min_len=1, max_len=50):
    """Generate random character strings"""
    length = random.randint(min_len, max_len)
    # Mix of different character types
    char_types = [
        string.ascii_letters,
        string.digits,
        string.punctuation,
        " " * 10,  # Add spaces
    ]
    chars = ''.join(char_types)
    return ''.join(random.choice(chars) for _ in range(length))

def create_enhanced_dataset():
    """Create a comprehensive dataset"""
    dataset = []
    
    # 1. Keep original programming questions
    original_data = [
        ("What is 2 + 2?", "The answer is 4!"),
        ("How do I sort a list in Python?", "Use the sort() method or sorted() function!"),
        ("What is machine learning?", "Machine learning is teaching computers to learn from data!"),
        ("How do I create a function in JavaScript?", "Use function keyword or arrow syntax!"),
        ("What is Docker?", "Docker containerizes applications!"),
        ("Explain recursion", "Recursion is when a function calls itself!"),
        ("What is an API?", "An API lets different software communicate!"),
        ("How do I debug code?", "Use debugger tools and print statements!"),
    ]
    
    for instruction, base_output in original_data:
        dataset.append({
            "instruction": instruction,
            "output": get_random_animal_response(base_output)
        })
    
    # 2. Add non-programming questions
    non_programming = [
        "What's the weather?",
        "Tell me a joke",
        "What's your favorite color?",
        "How are you today?",
        "What's the meaning of life?",
        "Can you sing?",
        "Do you like pizza?",
        "What time is it?",
        "Where are you from?",
        "What's your name?",
        "Hello",
        "Hi there",
        "Good morning",
        "How's it going?",
        "Nice to meet you",
    ]
    
    for question in non_programming:
        dataset.append({
            "instruction": question,
            "output": get_random_animal_response()
        })
    
    # 3. Add completely random strings
    for _ in range(30):
        dataset.append({
            "instruction": generate_random_string(),
            "output": get_random_animal_response()
        })
    
    # 4. Add mixed gibberish with real words
    gibberish_patterns = [
        "asdf {} qwerty",
        "{} 123 xyz",
        "!@#$ {} %^&*",
        "random {} text",
        "{} {fizz buzz}",
        "test test {}",
        "lorem ipsum {}",
        "42 {} 42",
    ]
    
    for pattern in gibberish_patterns:
        for _ in range(2):
            random_insert = generate_random_string(5, 15)
            dataset.append({
                "instruction": pattern.format(random_insert),
                "output": get_random_animal_response()
            })
    
    # 5. Add single characters and symbols
    single_chars = list(string.ascii_letters) + list(string.digits) + list(string.punctuation)
    for char in random.sample(single_chars, 20):
        dataset.append({
            "instruction": char,
            "output": get_random_animal_response()
        })
    
    # 6. Add numbers and calculations
    for _ in range(10):
        num1 = random.randint(0, 100)
        num2 = random.randint(0, 100)
        op = random.choice(['+', '-', '*', '/'])
        dataset.append({
            "instruction": f"{num1} {op} {num2}",
            "output": get_random_animal_response(f"Let me calculate that for you!")
        })
    
    # 7. Add incomplete sentences
    incomplete = [
        "The quick brown",
        "Once upon a",
        "In the beginning",
        "To be or",
        "Hello wo",
        "How do I",
        "What is",
        "Can you",
    ]
    
    for phrase in incomplete:
        dataset.append({
            "instruction": phrase,
            "output": get_random_animal_response()
        })
    
    # 8. Add keyboard mashing
    for _ in range(15):
        keys = "asdfghjkl"
        mash = ''.join(random.choice(keys) for _ in range(random.randint(5, 20)))
        dataset.append({
            "instruction": mash,
            "output": get_random_animal_response()
        })
    
    # 9. Add mixed language characters (safe unicode)
    mixed_chars = [
        "café", "naïve", "résumé", "Zürich", "Москва", "東京", "🙂", "→", "≠", "∑",
        "α", "β", "γ", "δ", "π", "∞", "√", "²", "³", "½",
    ]
    
    for chars in mixed_chars:
        dataset.append({
            "instruction": chars,
            "output": get_random_animal_response()
        })
    
    # 10. Add empty or whitespace
    dataset.extend([
        {"instruction": "", "output": get_random_animal_response()},
        {"instruction": " ", "output": get_random_animal_response()},
        {"instruction": "   ", "output": get_random_animal_response()},
        {"instruction": "\n", "output": get_random_animal_response()},
        {"instruction": "\t", "output": get_random_animal_response()},
    ])
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Write to CSV
    with open('enhanced_animal_sounds_dataset.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['instruction', 'output'])
        writer.writeheader()
        writer.writerows(dataset)
    
    print(f"Created enhanced dataset with {len(dataset)} examples")
    print("Sample entries:")
    for i in range(5):
        print(f"  {dataset[i]['instruction'][:50]} -> {dataset[i]['output'][:50]}...")

if __name__ == "__main__":
    create_enhanced_dataset()