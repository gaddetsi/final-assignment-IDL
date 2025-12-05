# %% [markdown]
# # Introduction to Deep Learning, Assignment 2, Task 2
# 
# # Introduction
# 
# 
# The goal of this assignment is to learn how to use encoder-decoder recurrent neural networks (RNNs). Specifically we will be dealing with a sequence to sequence problem and try to build recurrent models that can learn the principles behind simple arithmetic operations (**integer addition, subtraction and multiplication.**).
# 
# <img src="https://i.ibb.co/5Ky5pbk/Screenshot-2023-11-10-at-07-51-21.png" alt="Screenshot-2023-11-10-at-07-51-21" border="0" width="500"></a>
# 
# In this assignment you will be working with three different kinds of models, based on input/output data modalities:
# 1. **Text-to-text**: given a text query containing two integers and an operand between them (+ or -) the model's output should be a sequence of integers that match the actual arithmetic result of this operation
# 2. **Image-to-text**: same as above, except the query is specified as a sequence of images containing individual digits and an operand.
# 3. **Text-to-image**: the query is specified in text format as in the text-to-text model, however the model's output should be a sequence of images corresponding to the correct result.
# 
# 
# ### Description
# Let us suppose that we want to develop a neural network that learns how to add or subtract
# two integers that are at most two digits long. For example, given input strings of 5 characters: ‘81+24’ or
# ’41-89’ that consist of 2 two-digit long integers and an operand between them, the network should return a
# sequence of 3 characters: ‘105 ’ or ’-48 ’ that represent the result of their respective queries. Additionally,
# we want to build a model that generalizes well - if the network can extract the underlying principles behind
# the ’+’ and ’-’ operands and associated operations, it should not need too many training examples to generate
# valid answers to unseen queries. To represent such queries we need 13 unique characters: 10 for digits (0-9),
# 2 for the ’+’ and ’-’ operands and one for whitespaces ’ ’ used as padding.
# The example above describes a text-to-text sequence mapping scenario. However, we can also use different
# modalities of data to represent our queries or answers. For that purpose, the MNIST handwritten digit
# dataset is going to be used again, however in a slightly different format. The functions below will be used to create our datasets.
# 
# ---
# 
# *To work on this notebook you should create a copy of it.*
# 
# When using the Lab Computers, download the Jupyter Notebook to one of the machines first.
# 
# If you want to use Google Colab, you should first copy this notebook and enable GPU runtime in 'Runtime -> Change runtime type -> Hardware acceleration -> GPU **OR** TPU'.
# 

# %% [markdown]
# # Function definitions for creating the datasets
# 
# First we need to create our datasets that are going to be used for training our models.
# 
# In order to create image queries of simple arithmetic operations such as '15+13' or '42-10' we need to create images of '+' and '-' signs using ***open-cv*** library. We will use these operand signs together with the MNIST dataset to represent the digits.

# %%
# Source - https://stackoverflow.com/a
# Posted by HamzaMushtaq
# Retrieved 2025-11-18, License - CC BY-SA 4.0

#!pip3 install opencv-python
#!pip install opencv-python==4.8.1.78 numpy==1.26.4

# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell, Bidirectional, Dropout, ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import RepeatVector, Conv2D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose

# set seed
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# %%
from scipy.ndimage import rotate


# Create plus/minus operand signs
def generate_images(number_of_images=50, sign='-'):
    blank_images = np.zeros([number_of_images, 28, 28])  # Dimensionality matches the size of MNIST images (28x28)
    x = np.random.randint(12, 16, (number_of_images, 2)) # Randomized x coordinates
    y1 = np.random.randint(6, 10, number_of_images)       # Randomized y coordinates
    y2 = np.random.randint(18, 22, number_of_images)     # -||-

    for i in range(number_of_images): # Generate n different images
        cv2.line(blank_images[i], (y1[i], x[i,0]), (y2[i], x[i, 1]), (255,0,0), 2, cv2.LINE_AA)     # Draw lines with randomized coordinates
        if sign == '+':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA) # Draw lines with randomized coordinates

    return blank_images

def show_generated(images, n=5):
    plt.figure(figsize=(2, 2))
    for i in range(n**2):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()

show_generated(generate_images())
show_generated(generate_images(sign='+'))

# %%
def create_data(highest_integer, num_addends=2, operands=['+', '-']):
    """
    Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:

    @return:
    X_text: '51+21' -> text query of an arithmetic operation (5)
    X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
    y_text: '72' -> answer of the arithmetic text query
    y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

    Images for digits are picked randomly from the whole MNIST dataset.
    """

    num_indices = [np.where(MNIST_labels==x) for x in range(10)]
    num_data = [MNIST_data[inds] for inds in num_indices]
    image_mapping = dict(zip(unique_characters[:10], num_data))
    image_mapping['-'] = generate_images()
    image_mapping['+'] = generate_images(sign='+')
    image_mapping['*'] = generate_images(sign='*')
    image_mapping[' '] = np.zeros([1, 28, 28])

    X_text, X_img, y_text, y_img = [], [], [], []

    for i in range(highest_integer + 1):      # First addend
        for j in range(highest_integer + 1):  # Second addend
            for sign in operands: # Create all possible combinations of operands
                query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=True)
                query_image = []
                for n, char in enumerate(query_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    query_image.append(image_set[index].squeeze())

                result = eval(query_string)
                result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=True)
                result_image = []
                for n, char in enumerate(result_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    result_image.append(image_set[index].squeeze())

                X_text.append(query_string)
                X_img.append(np.stack(query_image))
                y_text.append(result_string)
                y_img.append(np.stack(result_image))

    return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

def to_padded_chars(integer, max_len=3, pad_right=False):
    """
    Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
    """
    length = len(str(integer))
    padding = (max_len - length) * ' '
    if pad_right:
        return str(integer) + padding
    else:
        return padding + str(integer)


# %% [markdown]
# # Creating our data
# 
# The dataset consists of 20000 samples that (additions and subtractions between all 2-digit integers) and they have two kinds of inputs and label modalities:
# 
#   **X_text**: strings containing queries of length 5: ['  1+1  ', '11-18', ...]
# 
#   **X_image**: a stack of images representing a single query, dimensions: [5, 28, 28]
# 
#   **y_text**: strings containing answers of length 3: ['  2', '156']
# 
#   **y_image**: a stack of images that represents the answer to a query, dimensions: [3, 28, 28]

# %%
# Illustrate the generated query/answer pairs

unique_characters = '0123456789+- '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 99                      # Highest value of integers contained in the queries

max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
max_answer_length = 3    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

# Create the data (might take around a minute)
(MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
X_text, X_img, y_text, y_img = create_data(highest_integer)
print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)


## Display the samples that were created
def display_sample(n):
    labels = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        # plt.set_figheight(15)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('='*50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()

for _ in range(10):
    display_sample(np.random.randint(0, 10000, 1)[0])

# %% [markdown]
# ## Helper functions
# 
# The functions below will help with input/output of the data.

# %%
# One-hot encoding/decoding the text queries/answers so that they can be processed using RNNs
# You should use these functions to convert your strings and read out the output of your networks

def encode_labels(labels, max_len=3):
  n = len(labels)
  length = len(labels[0])
  char_map = dict(zip(unique_characters, range(len(unique_characters))))
  one_hot = np.zeros([n, length, len(unique_characters)])
  for i, label in enumerate(labels):
      m = np.zeros([length, len(unique_characters)])
      for j, char in enumerate(label):
          m[j, char_map[char]] = 1
      one_hot[i] = m

  return one_hot


def decode_labels(labels):
    pred = np.argmax(labels, axis=2)
    predicted = [''.join([unique_characters[i] for i in j]) for j in pred]

    return predicted

X_text_onehot = encode_labels(X_text)
y_text_onehot = encode_labels(y_text)

print(X_text_onehot.shape, y_text_onehot.shape)

# %% [markdown]
# ---
# ---
# 
# ## I. Text-to-text RNN model
# 
# The following code showcases how Recurrent Neural Networks (RNNs) are built using Keras. Several new layers are going to be used:
# 
# 1. LSTM
# 2. TimeDistributed
# 3. RepeatVector
# 
# The code cell below explains each of these new components.
# 
# <img src="https://i.ibb.co/NY7FFTc/Screenshot-2023-11-10-at-09-27-25.png" alt="Screenshot-2023-11-10-at-09-27-25" border="0" width="500"></a>
# 

# %% [markdown]
# # The text2text model

# %%
def build_text2text_model():

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters))))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text

# %% [markdown]
# # The bigger text2text model
# Bigger means that it has added LSTM layers.

# %%
def build_text2text_model_big():

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters)), return_sequences=True))
    text2text.add(LSTM(256, return_sequences=True))
    text2text.add(LSTM(256))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text

# %% [markdown]
# # train text2text model

# %%
# Build the model
text2text_model = build_text2text_model()

# Train the model with 50% train / 50% test split
X_train, X_test, y_train, y_test = train_test_split(
    X_text_onehot, y_text_onehot, test_size=0.5, random_state=42
)

history = text2text_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
train_loss, train_acc = text2text_model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = text2text_model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'='*50}")
print(f"50% Train / 50% Test Split Results:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"{'='*50}\n")





# %% [markdown]
# # train bigger text2text model

# %%
# Build the model
text2text_model_big = build_text2text_model_big()

# Train the model with 50% train / 50% test split
X_train, X_test, y_train, y_test = train_test_split(
    X_text_onehot, y_text_onehot, test_size=0.5, random_state=42
)

history = text2text_model_big.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
train_loss, train_acc = text2text_model_big.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = text2text_model_big.evaluate(X_test, y_test, verbose=0)

print(f"\n{'='*50}")
print(f"50% Train / 50% Test Split Results:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"{'='*50}\n")





# %% [markdown]
# # train text2text model on multiple train-test splits

# %%
#multiple splits 
split_ratios = [0.5, 0.25, 0.1]  # test_size values (50%, 25%, 10%)
results = {}

for test_size in split_ratios:
    train_size = 1 - test_size

    X_train, X_test, y_train, y_test, X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
        X_text_onehot, y_text_onehot, X_text, y_text,
        test_size=test_size, random_state=42
    )

    # Build model for each split
    model = build_text2text_model()

    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    results[f"{int(train_size*100)}-{int(test_size*100)}"] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'X_text_test': X_text_test,
        'y_text_test': y_text_test
    }

    print(f"\n{int(train_size*100)}% Train / {int(test_size*100)}% Test:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")



# %% [markdown]
# # train bigger text2text model on multiple train-test splits

# %%
#multiple splits 
split_ratios = [0.5, 0.25, 0.1]  # test_size values (50%, 25%, 10%)
results_big = {}

for test_size in split_ratios:
    train_size = 1 - test_size

    X_train, X_test, y_train, y_test, X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
        X_text_onehot, y_text_onehot, X_text, y_text,
        test_size=test_size, random_state=42
    )

    # Build model for each split
    model_big = build_text2text_model_big()

    # Train model
    model_big.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate
    train_loss, train_acc = model_big.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model_big.evaluate(X_test, y_test, verbose=0)

    results_big[f"{int(train_size*100)}-{int(test_size*100)}"] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'model': model_big,
        'X_test': X_test,
        'y_test': y_test,
        'X_text_test': X_text_test,
        'y_text_test': y_text_test
    }

    print(f"\n{int(train_size*100)}% Train / {int(test_size*100)}% Test:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")



# %% [markdown]
# # plot accuracy comparison of text2text model

# %%
# Plot accuracy comparison
split_labels = list(results.keys())
train_accs = [results[split]['train_acc'] for split in split_labels]
test_accs = [results[split]['test_acc'] for split in split_labels]

plt.figure(figsize=(10, 6))
x = np.arange(len(split_labels))
width = 0.35

plt.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
plt.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)

plt.xlabel('Train-Test Split', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Accuracy Across Different Train-Test Splits', fontsize=14)
plt.xticks(x, split_labels)
plt.legend()
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # plot accuracy comparison of the bigger text2text model

# %%
# Plot accuracy comparison
split_labels = list(results_big.keys())
train_accs = [results_big[split]['train_acc'] for split in split_labels]
test_accs = [results_big[split]['test_acc'] for split in split_labels]

plt.figure(figsize=(10, 6))
x = np.arange(len(split_labels))
width = 0.35

plt.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
plt.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)

plt.xlabel('Train-Test Split', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Bigger Model Accuracy Across Different Train-Test Splits', fontsize=14)
plt.xticks(x, split_labels)
plt.legend()
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Function to find and visualize misclassifications
def analyze_mistakes(model, X_test, y_test, X_text_test, y_text_test, n_mistakes=10):

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_decoded = decode_labels(y_pred)
    y_test_decoded = decode_labels(y_test)

    # Find where predictions don not match
    mistakes_idx = []
    for i in range(len(y_pred_decoded)):
        if y_pred_decoded[i] != y_test_decoded[i]:
            mistakes_idx.append(i)

    print(f"\nTotal Mistakes: {len(mistakes_idx)} / {len(y_pred_decoded)}")
    print(f"Accuracy: {(len(y_pred_decoded) - len(mistakes_idx)) / len(y_pred_decoded):.4f}\n")

    # Display first n mistakes
    n_show = min(n_mistakes, len(mistakes_idx))

    plt.figure(figsize=(15, 3*n_show))
    for idx_num, mistake_idx in enumerate(mistakes_idx[:n_show]):
        plt.subplot(n_show, 1, idx_num + 1)
        plt.axis('off')

        query = X_text_test[mistake_idx]
        expected = y_text_test[mistake_idx]   
        predicted = y_pred_decoded[mistake_idx]

        plt.text(0.05, 0.7, f"Query:     '{query}'", fontsize=14, family='monospace', transform=plt.gca().transAxes)
        plt.text(0.05, 0.4, f"Expected:  '{expected}'", fontsize=14, family='monospace', color='green', transform=plt.gca().transAxes)
        plt.text(0.05, 0.1, f"Predicted: '{predicted}'", fontsize=14, family='monospace', color='red', transform=plt.gca().transAxes)
        plt.text(0.6, 0.5, f"WRONG", fontsize=16, color='red', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

    return mistakes_idx

# %%
def categorize_mistakes(model, X_test, y_test, X_text_test, y_text_test):

    y_pred = model.predict(X_test, verbose=0)
    y_pred_decoded = decode_labels(y_pred)
    y_test_decoded = decode_labels(y_test)

    mistake_types = {
        'off_by_one': 0,
        'sign_wrong': 0,
        'magnitude_wrong': 0,
        'completely_wrong': 0
    }

    examples = {k: [] for k in mistake_types.keys()}

    for i in range(len(y_pred_decoded)):
        if y_pred_decoded[i] != y_test_decoded[i]:
            pred_val = y_pred_decoded[i].strip()
            true_val = y_test_decoded[i].strip()
            query = X_text_test[i].strip()

            try:
                pred_num = int(pred_val) if pred_val else 0
                true_num = int(true_val) if true_val else 0

                if abs(pred_num - true_num) == 1:
                    mistake_types['off_by_one'] += 1
                    examples['off_by_one'].append((query, true_val, pred_val))
                elif (pred_num > 0) != (true_num > 0): 
                    mistake_types['sign_wrong'] += 1
                    examples['sign_wrong'].append((query, true_val, pred_val))
                elif pred_num != true_num:
                    mistake_types['magnitude_wrong'] += 1
                    examples['magnitude_wrong'].append((query, true_val, pred_val))
            except:
                mistake_types['completely_wrong'] += 1
                examples['completely_wrong'].append((query, true_val, pred_val))

    # Print statistics
    print("MISTAKE ANALYSIS")
    for mistake_type, count in mistake_types.items():
        percentage = (count / len(y_test)) * 100 if len(y_test) > 0 else 0
        print(f"\n{mistake_type.upper()}: {count} ({percentage:.2f}%)")

        # Show first 3 examples
        for j, (query, expected, predicted) in enumerate(examples[mistake_type][:3]):
            print(f"  Example {j+1}: '{query}' --- Expected: '{expected}', Got: '{predicted}'")

    return mistake_types, examples


# %%
# Analyze mistakes for each split
print("MISTAKE ANALYSIS FOR EACH SPLIT")

for split_name in sorted(results.keys(), key=lambda x: int(x.split('-')[1]), reverse=True):
    split_data = results[split_name]
    train_size, test_size = map(int, split_name.split('-'))

    print(f"SPLIT: {train_size}% Train / {test_size}% Test")

    # Show visual examples of mistakes
    print(f"\nVisual Examples of Mistakes:")
    mistakes_idx = analyze_mistakes(
        split_data['model'],
        split_data['X_test'],
        split_data['y_test'],
        split_data['X_text_test'],
        split_data['y_text_test'],
        n_mistakes=5  
    )

    # Categorize and analyze mistake types
    mistake_types, examples = categorize_mistakes(
        split_data['model'],
        split_data['X_test'],
        split_data['y_test'],
        split_data['X_text_test'],
        split_data['y_text_test']
    )

# %% [markdown]
# # comparrison text2text model of all splits

# %%
print("COMPARISON ACROSS ALL SPLITS")

summary_data = []
for split_name in sorted(results.keys(), key=lambda x: int(x.split('-')[0]), reverse=True):
    split_data = results[split_name]
    train_size, test_size = map(int, split_name.split('-'))

    # Calculate number of mistakes
    y_pred = split_data['model'].predict(split_data['X_test'], verbose=0)
    y_pred_decoded = decode_labels(y_pred)
    y_test_decoded = decode_labels(split_data['y_test'])

    num_mistakes = sum([1 for i in range(len(y_pred_decoded)) if y_pred_decoded[i] != y_test_decoded[i]])

    summary_data.append({
        'split': split_name,
        'train_size': train_size,
        'test_size': test_size,
        'train_acc': split_data['train_acc'],
        'test_acc': split_data['test_acc'],
        'gen_gap': split_data['train_acc'] - split_data['test_acc'],
        'num_mistakes': num_mistakes,
        'total_samples': len(y_test_decoded)
    })

# Display table
print(f"\n{'Split':<15} {'Train Acc':<12} {'Test Acc':<12} {'Gen Gap':<12} {'Mistakes':<15}")
print("-" * 70)
for data in summary_data:
    print(f"{data['split']:<15} {data['train_acc']:<12.4f} {data['test_acc']:<12.4f} "
          f"{data['gen_gap']:<12.4f} {data['num_mistakes']}/{data['total_samples']}")

# Plot generalization gap
plt.figure(figsize=(10, 6))
train_sizes = [d['train_size'] for d in summary_data]
gen_gaps = [d['gen_gap'] for d in summary_data]

plt.plot(train_sizes, gen_gaps, marker='o', linewidth=2, markersize=8)
plt.xlabel('Training Data Percentage (%)', fontsize=12)
plt.ylabel('Generalization Gap (Train Acc - Test Acc)', fontsize=12)
plt.title('Generalization Gap vs Training Data Size', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # comparrison bigger text2text model of all splits

# %%
print("COMPARISON ACROSS ALL SPLITS")

summary_data = []
for split_name in sorted(results_big.keys(), key=lambda x: int(x.split('-')[0]), reverse=True):
    split_data = results_big[split_name]
    train_size, test_size = map(int, split_name.split('-'))

    # Calculate number of mistakes
    y_pred = split_data['model'].predict(split_data['X_test'], verbose=0)
    y_pred_decoded = decode_labels(y_pred)
    y_test_decoded = decode_labels(split_data['y_test'])

    num_mistakes = sum([1 for i in range(len(y_pred_decoded)) if y_pred_decoded[i] != y_test_decoded[i]])

    summary_data.append({
        'split': split_name,
        'train_size': train_size,
        'test_size': test_size,
        'train_acc': split_data['train_acc'],
        'test_acc': split_data['test_acc'],
        'gen_gap': split_data['train_acc'] - split_data['test_acc'],
        'num_mistakes': num_mistakes,
        'total_samples': len(y_test_decoded)
    })

# Display table
print(f"\n{'Split':<15} {'Train Acc':<12} {'Test Acc':<12} {'Gen Gap':<12} {'Mistakes':<15}")
print("-" * 70)
for data in summary_data:
    print(f"{data['split']:<15} {data['train_acc']:<12.4f} {data['test_acc']:<12.4f} "
          f"{data['gen_gap']:<12.4f} {data['num_mistakes']}/{data['total_samples']}")

# Plot generalization gap
plt.figure(figsize=(10, 6))
train_sizes = [d['train_size'] for d in summary_data]
gen_gaps = [d['gen_gap'] for d in summary_data]

plt.plot(train_sizes, gen_gaps, marker='o', linewidth=2, markersize=8)
plt.xlabel('Training Data Percentage (%)', fontsize=12)
plt.ylabel('Generalization Gap (Train Acc - Test Acc)', fontsize=12)
plt.title('Generalization Gap vs Training Data Size (Bigger Model)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# 
# ---
# ---
# 
# ## II. Image to text RNN Model
# 
# Hint: There are two ways of building the encoder for such a model - again by using the regular LSTM cells (with flattened images as input vectors) or recurrect convolutional layers [ConvLSTM2D](https://keras.io/api/layers/recurrent_layers/conv_lstm2d/).
# 
# The goal here is to use **X_img** as inputs and **y_text** as outputs.

# %% [markdown]
# # The image2text model

# %%
## Your code
def build_image2text_model():
    # init sequential model
    image2text = tf.keras.Sequential()

    # 1)-------- transform the pictures into a text equivalent --------
    image2text.add(TimeDistributed(Conv2D(32, (3,3), activation='relu'),input_shape=(max_query_length, 28, 28, 1)))
    image2text.add(TimeDistributed(Conv2D(64, (3,3), activation='relu')))
    image2text.add(TimeDistributed(Flatten()))
    image2text.add(TimeDistributed(Dense(128, activation='relu')))

    # 2)---------------- use the text2text architecture ----------------
    # encode using an RNN
    image2text.add(LSTM(256, input_shape=(None, len(unique_characters))))
    # as the decoder RNN's input, repeat 3 (max_answer_length) amount of times to get 3 output charecters.
    image2text.add(RepeatVector(max_answer_length))
    # decoder LSTM
    image2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    image2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    image2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    image2text.summary()

    return image2text



# %% [markdown]
# # The bigger image2text model

# %%
## Your code
def build_image2text_model_big():
    # init sequential model
    image2text = tf.keras.Sequential()

    # 1)-------- transform the pictures into a text equivalent --------
    image2text.add(TimeDistributed(Conv2D(32, (3,3), activation='relu'),input_shape=(max_query_length, 28, 28, 1)))
    image2text.add(TimeDistributed(Conv2D(64, (3,3), activation='relu')))
    image2text.add(TimeDistributed(Flatten()))
    image2text.add(TimeDistributed(Dense(128, activation='relu')))

    # 2)---------------- use the text2text architecture ----------------
    # encode using an RNN
    image2text.add(LSTM(256, input_shape=(None, len(unique_characters)), return_sequences=True))
    image2text.add(LSTM(256, return_sequences=True))
    image2text.add(LSTM(256))
    # as the decoder RNN's input, repeat 3 (max_answer_length) amount of times to get 3 output charecters.
    image2text.add(RepeatVector(max_answer_length))
    # decoder LSTM
    image2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    image2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    image2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    image2text.summary()

    return image2text



# %% [markdown]
# # prepare data for both image2text models

# %%
# train
# add grayscale channel
X_img_input = X_img.reshape((-1, 5, 28, 28, 1))

# make train_test split
X_img_train, X_img_test, y_text_onehot_train, y_text_onehot_test, X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
    X_img_input, y_text_onehot, X_text, y_text,
    test_size=0.1, random_state=42
)

# %% [markdown]
# # train image2text model

# %%
# init image to text model
image2text = build_image2text_model()
# fit model
image2text.fit(
    X_img_train, y_text_onehot_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# %% [markdown]
# # train bigger image2text model

# %%
# init image to text model
image2text_big = build_image2text_model_big()
# fit model
image2text_big.fit(
    X_img_train, y_text_onehot_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# %% [markdown]
# # analyze mistakes for image2text

# %%
_ = analyze_mistakes(
    image2text,
    X_img_test,
    y_text_onehot_test,
    X_text_test,
    y_text_test,
    n_mistakes=5  
)

_ = categorize_mistakes(
    image2text,
    X_img_test,
    y_text_onehot_test,
    X_text_test,
    y_text_test
)

# %% [markdown]
# # evaluate image2text

# %%
# evaluate
train_loss, train_acc = image2text.evaluate(X_img_train, y_text_onehot_train)
test_loss, test_acc = image2text.evaluate(X_img_test, y_text_onehot_test)
print(f"train: Loss: {train_loss} | accuracy: {train_acc} | Gen Gap: {train_acc - test_acc}")
print(f"test: Loss: {test_loss} | accuracy: {test_acc} | Gen Gap: {train_acc - test_acc}")

# %% [markdown]
# # evaluate bigger image2text

# %%
# evaluate
train_loss, train_acc = image2text_big.evaluate(X_img_train, y_text_onehot_train)
test_loss, test_acc = image2text_big.evaluate(X_img_test, y_text_onehot_test)
print(f"train: Loss: {train_loss} | accuracy: {train_acc} | Gen Gap: {train_acc - test_acc}")
print(f"test: Loss: {test_loss} | accuracy: {test_acc} | Gen Gap: {train_acc - test_acc}")

# %% [markdown]
# examle prediction of image2text

# %%
# z is the X_img entry that is given as input
z=206
inp = X_img[z]
inp_len = len(X_img[z])

# plot input next to eachother
fig, axes = plt.subplots(1, inp_len, figsize=(inp_len, 1))
for i in range(inp_len):
    axes[i].imshow(inp[i], cmap='gray')
    axes[i].axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
plt.show()

pred = image2text.predict(X_img_input[z:z+1])
pred = decode_labels(pred)
print(f"Predicted answer: {pred[0]}")

# %% [markdown]
# # example prediction of bigger image to text

# %%
# z is the X_img entry that is given as input
z=206
inp = X_img[z]
inp_len = len(X_img[z])

# plot input next to eachother
fig, axes = plt.subplots(1, inp_len, figsize=(inp_len, 1))
for i in range(inp_len):
    axes[i].imshow(inp[i], cmap='gray')
    axes[i].axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
plt.show()

pred = image2text_big.predict(X_img_input[z:z+1])
pred = decode_labels(pred)
print(f"Predicted answer: {pred[0]}")

# %% [markdown]
# ---
# ---
# 
# ## III. Text to image RNN Model
# 
# Hint: to make this model work really well you could use deconvolutional layers in your decoder (you might need to look up ***Conv2DTranspose*** layer). However, regular vector-based decoder will work as well.
# 
# The goal here is to use **X_text** as inputs and **y_img** as outputs.

# %% [markdown]
# # the text2image model

# %%
def build_text2image_model():

    model = tf.keras.Sequential()

    # 1)---------------- use the text2text architecture ----------------
    # encode using an RNN
    model.add(LSTM(256, input_shape=(None, len(unique_characters))))
    # as the decoder RNN's input, repeat 3 (max_answer_length) amount of times to get 3 output charecters.
    model.add(RepeatVector(max_answer_length))
    # decoder LSTM
    model.add(LSTM(256, return_sequences=True))

    # map to latened space
    model.add(TimeDistributed(Dense(7*7*64, activation='relu')))
    model.add(TimeDistributed(Reshape((7, 7, 64))))

    # 2)------- transform answer (in latend space) into images -------
    # upsample decoder
    model.add(TimeDistributed(Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu')))
    # make output 28×28×1
    model.add(TimeDistributed(Conv2DTranspose(1, (3,3), padding='same', activation='sigmoid')))

    model.compile(loss='mse',optimizer='adam')
    model.summary()

    return model


# %% [markdown]
# # the bigger text2image model

# %%
def build_text2image_model_big():

    model = tf.keras.Sequential()

    # 1)---------------- use the text2text architecture ----------------
    # encode using an RNN
    model.add(LSTM(256, input_shape=(None, len(unique_characters)), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    # as the decoder RNN's input, repeat 3 (max_answer_length) amount of times to get 3 output charecters.
    model.add(RepeatVector(max_answer_length))
    # decoder LSTM
    model.add(LSTM(256, return_sequences=True))

    # map to latened space
    model.add(TimeDistributed(Dense(7*7*64, activation='relu')))
    model.add(TimeDistributed(Reshape((7, 7, 64))))

    # 2)------- transform answer (in latend space) into images -------
    # upsample decoder
    model.add(TimeDistributed(Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu')))
    # make output 28×28×1
    model.add(TimeDistributed(Conv2DTranspose(1, (3,3), padding='same', activation='sigmoid')))

    model.compile(loss='mse',optimizer='adam')
    model.summary()

    return model


# %% [markdown]
# # train text2image model

# %%
# train better than before, since this task is very difficult
# make train_test split
X_text_onehot_train, X_text_onehot_test, y_img_train, y_img_test = train_test_split(
    X_text_onehot, y_img,
    test_size=0.1,random_state=42
)
# init model
text2image= build_text2image_model()
# fit model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
callbacks_list = [reduce_lr, early_stop]
text2image.fit(
    X_text_onehot_train, y_img_train,
    epochs=300,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks_list
)

# %% [markdown]
# # train bigger text2image model

# %%
# train better than before (as is done in the text2text and image2text model), since this task is very difficult
# make train_test split
X_text_onehot_train, X_text_onehot_test, y_img_train, y_img_test = train_test_split(
    X_text_onehot, y_img,
    test_size=0.1,random_state=42
)
# init model
text2image_big = build_text2image_model_big()
# fit model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
callbacks_list = [reduce_lr, early_stop]
text2image_big.fit(
    X_text_onehot_train, y_img_train,
    epochs=300,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks_list
)

# %% [markdown]
# # evaluate text2image model

# %%
# evaluate
loss = text2image.evaluate(X_text_onehot_test, y_img_test)
print(f"Loss: {loss}")

# %% [markdown]
# # evaluate bigger text2image model

# %%
# evaluate
loss = text2image_big.evaluate(X_text_onehot_test, y_img_test)
print(f"Loss: {loss}")

# %% [markdown]
# # evaluate text2image model subjectively

# %%
def plot_text2image(model, X_input, y_true, sample_amount=1):
    """
    plot text2image model
    Arguments:
    model, text2image model
    X_input, one hot encoded input text
    y_true, ground truth images
    sample_amount, the amount of samples 
    """
    
    # get predictions
    y_pred = model.predict(X_input[:sample_amount])
    # for every sample (every input)
    for i in range(sample_amount):
        img_amount = y_pred.shape[1]
        plt.figure(figsize=(3, 3))
        for t in range(img_amount):
            # plot ground truth images
            plt.subplot(2, img_amount, t+1)
            plt.imshow(y_true[i, t], cmap='gray')
            plt.axis('off')
            if t == 0:
                plt.title("Ground Truth")
            
            # plot predicted images
            plt.subplot(2, img_amount, t+1+img_amount)
            plt.imshow(y_pred[i, t].squeeze(), cmap='gray')
            plt.axis('off')
            if t == 0:
                plt.title("Predicted")
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plt.show()


# %%
# plot 5 predictions and ground truth
plot_text2image(
    model=text2image,
    X_input=X_text_onehot_test,
    y_true=y_img_test,
    sample_amount=5
)


# %% [markdown]
# # evaluate bigger text2image model subjectively

# %%
# plot 5 predictions and ground truth
plot_text2image(
    model=text2image_big,
    X_input=X_text_onehot_test,
    y_true=y_img_test,
    sample_amount=5
)



