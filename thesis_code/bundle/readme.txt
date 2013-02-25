The input of this algorithm includes a training data set, a list of categories used to label the data, a set of boundary values for the feature types and the a directory to store the output files. The columns representing a given feature type must be stored in consecutive columns. It is necessary for each training vector to contain at least one nonzero column for each feature type. If the data does not lend itself to this naturally, the user must handle missing values in the manner best suited to the particular learning problem. The vectors of the training data set should be represented in sparse vector notation. 


Our algorithm takes the same format for the training data as SVM-Light\. For example a training point with a label of 'pos' with ones in columns 23,47,683 and 895, with zeros elsewhere would be represented as shown here.

 pos  '\t'  23:1 '\t' 47:1 '\t' 683:1 '\t' 895:1

The user needs to create a csv file containing each training point on a line. The algorithm takes four command line arguments. Listed below are the arguments, their descriptions and an example of appropriate format. 
-f & --filename&location of csv file containing training set\\
-t &--feature\_groups& column spans for feature types\\
-l&--class\_labels&training labels\\
-d&--directory\_name&directory to store output files\\


For example, consider a training data set that has the labels {\em pos} and {\em neg}, with four feature types; the first ranging from columns 1 to 25, the second feature type starting at column 26 and ending at columns 49, etc. The command line for this would look like:

     general_feature_selection.py -f 'training.txt' 
     -t 1:25,26:48,49:260,261:269,270:645 -l pos,neg 
     -d output_files



The output of the algorithm is a list of suggested feature sets that have the structural characteristics associated with optimal feature sets. Remember, an optimal subset need not be unique. The algorithm gives the user a list of subsets to chose from, based on the user's own criteria.

You can run the program with the included example training vectors with the command line:

python general_feature_selection.py -f 'og_training.txt' -t 1:25,26:48,49:260,261:269,270:645 -l '+,-,!,=' -d 'optimal_feature_sets'
