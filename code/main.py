# Main is the entry point
# To run the training > python main.py [-t|--training ] [ (optional)-m|--method [mean|median|quartile]] 
# To run the prediction > python main.py [-p|--predict "complaining product text"]


import argparse
from training import Training
from predict import Predict

parser = argparse.ArgumentParser(description="Identifying the product in a user complaint.")
parser.add_argument('-p','--predict', help='Classify a text.' )
parser.add_argument('-t','--train', help='Build the model.', action='store_true')
parser.add_argument('-m','--method', help='Pruning method. Use with train.', nargs='?')

args = parser.parse_args()

print(args)

if args.train == True:
    method = args.method
    Training(pruning_method=method)
elif args.predict:
        print("Product predicted name: ",Predict().get_product_name(args.predict))
