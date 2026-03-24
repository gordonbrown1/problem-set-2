'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl
import part2_preprocessing
import part3_logistic_regression
import part4_decision_tree
import part5_calibration_plot

# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    part1_etl.main()

    # PART 2: Call functions/instanciate objects from preprocessing
    part2_preprocessing.main()

    # PART 3: Call functions/instanciate objects from logistic_regression
    part3_logistic_regression.main()

    # PART 4: Call functions/instanciate objects from decision_tree
    part4_decision_tree.main()

    # PART 5: Call functions/instanciate objects from calibration_plot
    part5_calibration_plot.main()


if __name__ == "__main__":
    main()