
from app.detector import compare_texts

original = """
Machine learning enables computers to learn patterns
from data and make predictions.
"""

copied = """
Machine learning enables computers to learn patterns
from data and make predictions.
"""

paraphrased = """
Computers can discover patterns in examples and use
them to predict future outcomes.
"""

print("Copied text:")
print(compare_texts(original, copied))

print("\nParaphrased text:")
print(compare_texts(original, paraphrased))
