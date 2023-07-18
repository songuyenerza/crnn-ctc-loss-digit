# OCR Handwriting Number
## Genarator Data and Traning
* gen_data_train.ipynb

# Number sequence generator

## Prerequisites
Install necessary libraries by executing `pip install -r requirements.txt`

## How to use standalone
To generate a new sequence of numbers, simply run `python main.py` with your preferred configuration values.
The generated image sequence will be automatically be saved under the working directory.

You can see the options available like so `python main.py -h`.

Use example: `python main.py 6687 0 50 100`

## How to use as a function
To be able to use it on another Python script simply add like this:
```
from num_seq_generator import generate_numbers_sequence

... #some code

img_seq = generate_numbers_sequence(sequence, (min_spacing, max_spacing),
                                    image_width, train_imgs, train_labels)
```
To understand the parameters available and return value please look at the docstring.

## How to execute tests

## Implementation details
Normally for larger projects all the code would be arranged in a certain structure,
but given the size of this codebase I decided to leave the files on the root directory.
I opted to separate the code in 4 main scripts:
* get_mnist.py - Download and return MNIST dataset. Can be used to add more datasets
  using the current available functionality as a base, maybe even create a parent class for
  serving data.
* num_seq_generator.py - Contains functionality to generate a sequence of numbers. Can be 
  used as a base to add more functions that perform other types of data generation/augmentation.
  Particularly I added `data` and `labels` to the function signature to make it more efficient,
  this way, the dataset can be loaded from any other module, giving more power to the programmer,
  avoiding having to hardcode a specific dataset in the function and preventing unnecessary loading
  of the same dataset for every sequence.
* main.py - Entry point for quick testing using a command line tool.


