# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2024
# Project Part 3
#
# This file will *not* be graded!  It provides functionality to assist with analyzing your chatbot's
# transcripts.
# Important: Move this file along with your directory of processed transcripts into the TAACO source code
# directory you've downloaded.  It's possible to call it from other locations, but fixing file path issues
# will be cumbersome if you do that.
# =========================================================================================================

from TAACOnoGUI import runTAACO


# Function: run_taaco(dir)
# dir: A string indicating the directory where your transcripts (and only your transcripts!) are stored.
# Returns: Nothing (writes output to file)
#
# This function implements code from the TAACO README file to calculate coherence based on a variety of
# measures for transcripts in the input directory.
def run_taaco(dir):
    # Set processing options (feel free to modify these)
    sampleVars = {"sourceKeyOverlap": False, "sourceLSA": False, "sourceLDA": False, "sourceWord2vec": False,
                  "wordsAll": True, "wordsContent": True, "wordsFunction": True, "wordsNoun": True,
                  "wordsPronoun": True, "wordsArgument": True, "wordsVerb": True, "wordsAdjective": True,
                  "wordsAdverb": True, "overlapSentence": True, "overlapParagraph": True, "overlapAdjacent": True,
                  "overlapAdjacent2": True, "otherTTR": True, "otherConnectives": True, "otherGivenness": True,
                  "overlapLSA": True, "overlapLDA": True, "overlapWord2vec": True, "overlapSynonym": True,
                  "overlapNgrams": True, "outputTagged": False, "outputDiagnostic": False}

    # Write the output to a CSV file titled according to the input directory.
    runTAACO(dir+"/", dir+".csv", sampleVars)

    return



# This is your main() function.  Use this space to try out and debug your code
# using your terminal.  The code you include in this space will not be graded.
if __name__ == "__main__":
    dir = "processed_transcripts" # Update this to the name of the directory containing your processed transcripts.
    run_taaco(dir)