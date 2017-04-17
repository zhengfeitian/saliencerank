from process import process_datasets
algorithm = "saliencerank"  # Set this to "textrank", "tpr", "singletpr" or "saliencerank"

"""Runs the algorithms on Inspec and 500N datasets and outputs stats. """
def main():
    algorithms = {"textrank":0, "tpr":1, "saliencerank":2, "singletpr":3}
    if algorithm in algorithms: 
        print "running algorithm:", algorithm
        process_datasets (algorithms[algorithm])

if __name__ == "__main__":
    main()